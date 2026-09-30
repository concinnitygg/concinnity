// The world's floor: a heightfield collider matching whichever terrain source
// the `PhysicsConfig` names. The generated one samples the "terrain" mesh
// generator's own height function, so the collided surface and the rendered one
// agree vertex for vertex.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::physics::{BodyHandle, LayerMask, Simulation};

use crate::components::ProceduralMesh;
use crate::ecs::PipelineContext;
use crate::geometry::{terrain_height, terrain_subdivisions};

/// The generated-terrain parameters a `PhysicsConfig` authors.
#[derive(Debug, Clone)]
pub(super) struct TerrainParams {
    pub(super) half_width: f32,
    pub(super) half_depth: f32,
    pub(super) subdivisions: u32,
    pub(super) amplitude: f32,
    pub(super) offset_y: f32,
}

// Build a heightfield collider for a heightfield-generator
// `ProceduralMesh` from the collider grid baked into its compiled payload. The
// build step stores the mesh's own per-vertex heights (an `n x n` row-major
// world-Y grid) as a trailer on the payload, so the collider tracks the
// rendered surface vertex-for-vertex without decoding the source image at
// runtime. The terrain mesh's blob is held resident past GraphicsSystem init
// for exactly this read (see the release sweep in `gfx::system::init`).
pub(super) fn build_heightfield_collider(
    world: &mut Simulation,
    mesh: &ProceduralMesh,
    offset_y: f32,
    mask: LayerMask,
    ctx: &mut PipelineContext,
) -> Result<(), String> {
    let locator = mesh
        .locator
        .as_ref()
        .ok_or("heightfield ProceduralMesh has no compiled payload")?;
    let bytes = ctx
        .read_payload(locator)
        .map_err(|e| format!("read terrain payload: {e}"))?;
    let grid = crate::gfx::mesh_payload::deserialize_heightfield(bytes)?
        .ok_or("terrain mesh payload has no baked heightfield collider")?;
    if grid.rows < 2 || grid.cols < 2 {
        return Err(format!(
            "heightfield collider grid too small ({}x{})",
            grid.rows, grid.cols
        ));
    }
    let width = mesh.half_width * 2.0;
    let depth = mesh.half_depth * 2.0;
    world
        .add_heightfield(
            grid.rows,
            grid.cols,
            grid.heights,
            [width, 1.0, depth],
            [0.0, offset_y, 0.0],
            mask,
        )
        .ok_or("the simulation declined the heightfield")?;
    Ok(())
}

// Build a heightfield collider matching the procedural terrain mesh.
pub(super) fn build_heightfield(
    world: &mut Simulation,
    terrain: &TerrainParams,
    mask: LayerMask,
) -> Option<BodyHandle> {
    let (n, heights) = terrain_heights(terrain);
    world.add_heightfield(
        n,
        n,
        heights,
        [terrain.half_width * 2.0, 1.0, terrain.half_depth * 2.0],
        [0.0, terrain.offset_y, 0.0],
        mask,
    )
}

// The `n x n` row-major height grid of the terrain mesh `terrain` describes,
// sampled at the mesh's own lattice points and resolution.
fn terrain_heights(terrain: &TerrainParams) -> (usize, Vec<f32>) {
    let subdivisions = terrain_subdivisions(terrain.subdivisions);
    let n = subdivisions as usize + 1;
    let mut heights = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            heights.push(terrain_height(
                col as f32,
                row as f32,
                subdivisions,
                terrain.amplitude,
            ));
        }
    }
    (n, heights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::ProceduralMesh;
    use crate::ecs::{PayloadLocator, PayloadStore, World};
    use crate::error::PayloadError;
    use crate::gfx::mesh_payload::serialize_heightfield_trailer;
    use crate::physics::{SimConfig, Simulation};
    use alloc::boxed::Box;
    use alloc::vec;

    fn terrain(amplitude: f32) -> TerrainParams {
        TerrainParams {
            half_width: 32.0,
            half_depth: 32.0,
            subdivisions: 32,
            amplitude,
            offset_y: 0.0,
        }
    }

    // The collider grid is the rendered mesh's heights, vertex for vertex,
    // including where the authored resolution needs clamping.
    #[test]
    fn the_collider_grid_matches_the_rendered_terrain_mesh() {
        for subdivisions in [2, 32, 300] {
            let t = TerrainParams {
                subdivisions,
                ..terrain(4.0)
            };
            let (n, heights) = terrain_heights(&t);
            let (verts, _) = crate::geometry::build_terrain(
                t.half_width,
                t.half_depth,
                t.subdivisions,
                t.amplitude,
            )
            .unwrap();
            assert_eq!(n * n, verts.len(), "subdivisions {subdivisions}");
            for (h, (pos, ..)) in heights.iter().zip(&verts) {
                assert_eq!(*h, pos[1], "subdivisions {subdivisions}");
            }
        }
    }

    // The generated collider is one body carrying an (n + 1) x (n + 1) grid of
    // the same samples the height function reports.
    #[test]
    fn a_generated_heightfield_is_one_body_over_the_authored_footprint() {
        let mut sim = Simulation::with_capacity(2);
        let t = terrain(4.0);
        assert!(build_heightfield(&mut sim, &t, LayerMask::ALL).is_some());
        assert_eq!(sim.body_count(), 1);
    }

    // A store handing back one fixed payload, standing in for the terrain
    // blob held resident past GraphicsSystem init for exactly this read.
    struct OnePayload(Vec<u8>);

    impl PayloadStore for OnePayload {
        fn read(&mut self, _locator: &PayloadLocator) -> Result<&[u8], PayloadError> {
            Ok(&self.0)
        }

        fn release(&mut self, _blob_index: u32) {}

        fn disk_backed(&self) -> bool {
            false
        }
    }

    // A mesh payload with no geometry and, when asked, a baked collider grid
    // trailer: the collider build reads past the vertex and index blocks to
    // reach the trailer, so empty blocks exercise the same walk.
    fn payload(grid: Option<(usize, usize, Vec<f32>)>) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        if let Some((rows, cols, heights)) = grid {
            bytes.extend_from_slice(&serialize_heightfield_trailer(rows, cols, &heights));
        }
        bytes
    }

    fn terrain_mesh(locator: Option<PayloadLocator>) -> ProceduralMesh {
        ProceduralMesh {
            generator: String::from("heightfield"),
            half_width: 8.0,
            half_depth: 4.0,
            locator,
            ..ProceduralMesh::default()
        }
    }

    fn locator() -> Option<PayloadLocator> {
        Some(PayloadLocator {
            blob_index: 0,
            offset: 0,
            len: 0,
        })
    }

    fn build(sim: &mut Simulation, mesh: &ProceduralMesh, world: &mut World) -> Result<(), String> {
        build_heightfield_collider(sim, mesh, 1.0, LayerMask::ALL, &mut world.context())
    }

    fn sim(capacity: usize) -> Simulation {
        Simulation::new(SimConfig::default(), capacity)
    }

    #[test]
    fn a_baked_grid_becomes_one_heightfield_body() {
        let mut world = World::from_payloads(Box::new(OnePayload(payload(Some((
            2,
            2,
            vec![0.0, 1.0, 2.0, 3.0],
        ))))));
        let mut sim = sim(4);
        build(&mut sim, &terrain_mesh(locator()), &mut world).expect("the collider builds");
        assert_eq!(sim.body_count(), 1);
    }

    // Every refusal names what was missing rather than building a collider
    // that does not match the rendered surface.
    #[test]
    fn a_mesh_with_no_compiled_payload_is_refused() {
        let mut world = World::new();
        let err = build(&mut sim(4), &terrain_mesh(None), &mut world)
            .expect_err("a mesh with no payload cannot be collided");
        assert!(err.contains("no compiled payload"), "{err}");
    }

    #[test]
    fn a_payload_the_store_cannot_read_is_refused() {
        let mut world = World::new();
        let err = build(&mut sim(4), &terrain_mesh(locator()), &mut world)
            .expect_err("an unreadable payload cannot be collided");
        assert!(err.contains("read terrain payload"), "{err}");
    }

    #[test]
    fn a_payload_with_no_baked_grid_is_refused() {
        let mut world = World::from_payloads(Box::new(OnePayload(payload(None))));
        let err = build(&mut sim(4), &terrain_mesh(locator()), &mut world)
            .expect_err("a payload with no trailer cannot be collided");
        assert!(err.contains("no baked heightfield collider"), "{err}");
    }

    // A grid needs two rows and two columns before it spans a single cell.
    #[test]
    fn a_grid_too_small_to_span_a_cell_is_refused() {
        let mut world =
            World::from_payloads(Box::new(OnePayload(payload(Some((1, 1, vec![0.0]))))));
        let err = build(&mut sim(4), &terrain_mesh(locator()), &mut world)
            .expect_err("a single-vertex grid cannot be collided");
        assert!(err.contains("too small"), "{err}");
    }

    #[test]
    fn a_simulation_with_no_room_declines_the_heightfield() {
        let mut world = World::from_payloads(Box::new(OnePayload(payload(Some((
            2,
            2,
            vec![0.0, 1.0, 2.0, 3.0],
        ))))));
        let err = build(&mut sim(0), &terrain_mesh(locator()), &mut world)
            .expect_err("a full simulation takes no more bodies");
        assert!(err.contains("declined the heightfield"), "{err}");
    }
}
