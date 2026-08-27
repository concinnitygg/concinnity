// The world's floor: a heightfield collider matching whichever terrain source
// the `PhysicsConfig` names, and the procedural noise the generated one is
// sampled from.
//
// The noise here is the same bilinear octave sum the "terrain" mesh generator
// uses, so the collided surface and the rendered one agree vertex for vertex.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use concinnity_physics::{BodyHandle, LayerMask, Simulation};

use crate::components::ProceduralMesh;
use crate::ecs::PipelineContext;
use crate::math::floor;

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
// for exactly this read (see the release sweep in `graphics_system::init`).
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
        .map_err(|e| format!("read terrain payload: {e:?}"))?;
    let grid = crate::gfx::mesh_payload::deserialise_heightfield(bytes)?
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
    let n = (terrain.subdivisions as usize) + 1;
    let width = terrain.half_width * 2.0;
    let depth = terrain.half_depth * 2.0;
    let mut heights = Vec::with_capacity(n * n);
    for i in 0..n {
        // row i spans Z
        let z = (i as f32 / (n - 1) as f32 - 0.5) * depth;
        for j in 0..n {
            // col j spans X
            let x = (j as f32 / (n - 1) as f32 - 0.5) * width;
            heights.push(terrain_height_at(x, z, terrain));
        }
    }
    world.add_heightfield(
        n,
        n,
        heights,
        [width, 1.0, depth],
        [0.0, terrain.offset_y, 0.0],
        mask,
    )
}

// Compute terrain surface height at world-space (x, z) using the same bilinear
// noise as the "terrain" mesh generator in build_mesh.rs. Converting world XZ
// to a fractional grid position and bilinearly interpolating between lattice
// samples gives a continuous height field that matches the rendered mesh exactly.
fn terrain_height_at(world_x: f32, world_z: f32, t: &TerrainParams) -> f32 {
    // clamp to terrain footprint; out-of-bounds positions use the edge height
    let x = world_x.clamp(-t.half_width, t.half_width);
    let z = world_z.clamp(-t.half_depth, t.half_depth);

    // fractional grid position in [0, subdivisions]
    let s = (x + t.half_width) / (t.half_width * 2.0) * t.subdivisions as f32;
    let g = (z + t.half_depth) / (t.half_depth * 2.0) * t.subdivisions as f32;

    let octaves: &[(u32, f32)] = &[
        (1, 1.00), // coarse hills
        (3, 0.40), // medium bumps
        (9, 0.15), // fine surface variation
    ];

    let mut sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;

    for &(divisor, weight) in octaves {
        let scale = (t.subdivisions / divisor).max(1) as f32;
        let gs = s / scale;
        let gt = g / scale;
        let gx = floor(gs) as u32;
        let gy = floor(gt) as u32;
        let fx = gs - gx as f32;
        let fy = gt - gy as f32;

        let h00 = lattice_val(gx, gy);
        let h10 = lattice_val(gx + 1, gy);
        let h01 = lattice_val(gx, gy + 1);
        let h11 = lattice_val(gx + 1, gy + 1);
        let top = h00 + (h10 - h00) * fx;
        let bot = h01 + (h11 - h01) * fx;
        sum += (top + (bot - top) * fy) * weight;
        weight_sum += weight;
    }

    let normalised = sum / weight_sum;
    (normalised - 0.05).max(0.0) * t.amplitude
}

fn lattice_val(x: u32, y: u32) -> f32 {
    let h = lcg_hash(x.wrapping_mul(1619).wrapping_add(y.wrapping_mul(31337)));
    (h & 0xFF) as f32 / 255.0
}

fn lcg_hash(mut v: u32) -> u32 {
    v = v.wrapping_mul(1664525).wrapping_add(1013904223);
    v ^= v >> 16;
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terrain(amplitude: f32) -> TerrainParams {
        TerrainParams {
            half_width: 32.0,
            half_depth: 32.0,
            subdivisions: 32,
            amplitude,
            offset_y: 0.0,
        }
    }

    #[test]
    fn flat_terrain_is_zero_height() {
        let t = terrain(0.0);
        assert_eq!(terrain_height_at(0.0, 0.0, &t), 0.0);
        assert_eq!(terrain_height_at(10.0, -5.0, &t), 0.0);
    }

    #[test]
    fn terrain_height_is_continuous_and_bounded() {
        let t = terrain(4.0);
        // Height never exceeds the amplitude and neighbouring samples are close.
        let mut prev = terrain_height_at(-32.0, 0.0, &t);
        let mut x = -32.0;
        while x <= 32.0 {
            let h = terrain_height_at(x, 0.0, &t);
            assert!((0.0..=4.0).contains(&h), "height {h} out of range at x={x}");
            assert!((h - prev).abs() < 1.0, "terrain jumped at x={x}");
            prev = h;
            x += 0.5;
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
}
