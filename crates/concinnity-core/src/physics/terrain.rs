// The ground every `Terrain` lays down: one heightfield collider per terrain,
// built from the same cooked height grid its render mesh and grass read, so
// the collided surface is the drawn one triangle for triangle.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::components::Terrain;
use crate::ecs::PipelineContext;
use crate::physics::{LayerMask, Simulation};
use crate::terrain::TerrainGrid;
use crate::terrain::payload::TerrainPayload;

// Add a heightfield for every terrain in the world, returning how many were
// built. A terrain whose payload cannot be read gets no collider rather than
// one that disagrees with what is drawn.
pub(super) fn build_terrain_colliders(
    world: &mut Simulation,
    mask: LayerMask,
    ctx: &mut PipelineContext,
) -> usize {
    let terrains: Vec<Terrain> = ctx.query::<Terrain>().cloned().collect();
    terrains
        .iter()
        .filter(|terrain| {
            terrain_grid(terrain, ctx)
                .and_then(|grid| add_heightfield(world, &grid, terrain.center, mask))
                .is_ok()
        })
        .count()
}

// The cooked height grid of `terrain`.
fn terrain_grid(terrain: &Terrain, ctx: &mut PipelineContext) -> Result<TerrainGrid, String> {
    let locator = terrain.locator.as_ref().ok_or("no compiled payload")?;
    let bytes = ctx
        .read_payload(locator)
        .map_err(|e| format!("read terrain payload: {e}"))?;
    Ok(TerrainPayload::decode(bytes, terrain.extent)?.grid)
}

// The heightfield body for `grid`, centered on `center`.
fn add_heightfield(
    world: &mut Simulation,
    grid: &TerrainGrid,
    center: [f32; 3],
    mask: LayerMask,
) -> Result<(), String> {
    let side = grid.side();
    let [hx, hz] = grid.extent();
    world
        .add_heightfield(
            side,
            side,
            grid.heights().to_vec(),
            [2.0 * hx, 1.0, 2.0 * hz],
            center,
            mask,
        )
        .map(|_| ())
        .ok_or_else(|| String::from("the simulation declined the heightfield"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::{PayloadLocator, PayloadStore, World};
    use crate::error::PayloadError;
    use crate::physics::SimConfig;
    use crate::terrain::noise_heights;
    use alloc::boxed::Box;
    use alloc::vec;

    // A store handing back one fixed payload, standing in for the terrain
    // blob held resident past GraphicsSystem init for this read.
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

    const CENTER: [f32; 3] = [5.0, -1.5, 3.0];
    const EXTENT: [f32; 2] = [12.0, 8.0];

    fn grid() -> TerrainGrid {
        TerrainGrid::new(24, EXTENT, noise_heights(24, 4.0, 11)).unwrap()
    }

    fn terrain(locator: Option<PayloadLocator>) -> Terrain {
        Terrain {
            center: CENTER,
            extent: EXTENT,
            resolution: 24,
            locator,
            ..Terrain::default()
        }
    }

    fn locator() -> Option<PayloadLocator> {
        Some(PayloadLocator {
            blob_index: 0,
            offset: 0,
            len: 0,
        })
    }

    fn world_with(payload: Vec<u8>, terrains: usize) -> World {
        let mut world = World::from_payloads(Box::new(OnePayload(payload)));
        for _ in 0..terrains {
            world.push(terrain(locator()));
        }
        world
    }

    fn payload() -> Vec<u8> {
        TerrainPayload {
            grid: grid(),
            masks: Vec::new(),
        }
        .encode()
    }

    fn drop_ray(sim: &Simulation, x: f32, z: f32) -> Option<f32> {
        sim.raycast([x, 100.0, z], [0.0, -1.0, 0.0], 200.0, None, LayerMask::ALL)
            .map(|hit| hit.point[1])
    }

    // The collider is the cooked grid: a ray dropped anywhere on the terrain
    // lands at the height the grid's own surface reports, which is the height
    // the render mesh and the grass placement read.
    #[test]
    fn the_collider_is_the_grid_surface() {
        let mut world = world_with(payload(), 1);
        let mut sim = Simulation::new(SimConfig::default(), 4);
        assert_eq!(
            build_terrain_colliders(&mut sim, LayerMask::ALL, &mut world.context()),
            1
        );
        let g = grid();
        for i in 0..40 {
            let x = -11.7 + i as f32 * 0.587;
            let z = -7.9 + (i * 7 % 40) as f32 * 0.395;
            let expected = CENTER[1] + g.height_at([x, z]).unwrap();
            let hit = drop_ray(&sim, CENTER[0] + x, CENTER[2] + z).expect("the ray lands");
            assert!(
                (hit - expected).abs() < 1e-3,
                "at ({x}, {z}): collider {hit}, grid {expected}"
            );
        }
        assert_eq!(drop_ray(&sim, CENTER[0] + 13.0, CENTER[2]), None);
    }

    #[test]
    fn every_terrain_gets_its_own_collider() {
        let mut world = world_with(payload(), 3);
        let mut sim = Simulation::new(SimConfig::default(), 4);
        let built = build_terrain_colliders(&mut sim, LayerMask::ALL, &mut world.context());
        assert_eq!(built, 3);
        assert_eq!(sim.body_count(), 3);
    }

    // A terrain that cannot be read is left out rather than collided wrong.
    #[test]
    fn an_unreadable_terrain_is_skipped() {
        let mut world = World::from_payloads(Box::new(OnePayload(vec![1, 2, 3])));
        world.push(terrain(locator()));
        world.push(terrain(None));
        let mut sim = Simulation::new(SimConfig::default(), 4);
        assert_eq!(
            build_terrain_colliders(&mut sim, LayerMask::ALL, &mut world.context()),
            0
        );
        assert_eq!(sim.body_count(), 0);
    }

    #[test]
    fn a_full_simulation_declines_the_heightfield() {
        let mut world = world_with(payload(), 1);
        let mut sim = Simulation::new(SimConfig::default(), 0);
        assert_eq!(
            build_terrain_colliders(&mut sim, LayerMask::ALL, &mut world.context()),
            0
        );
    }
}
