//! Deterministic procedural generation of voxel chunks for an infinite
//! `VoxelWorld`.
//!
//! A `ChunkGenerator` turns a `ChunkCoord` + a world seed into the dense block
//! array a chunk mesher consumes. Generation is a pure function of the seed and
//! the chunk coordinate, so a chunk that streams out and back in regenerates
//! byte-identically. Terrain height comes from multi-octave value noise keyed
//! on *world* block coordinates, so adjacent chunks line up seamlessly across
//! their shared edge.
//!
//! The lattice hash is integer arithmetic and interpolation is a polynomial
//! smoothstep, so generation involves no transcendentals at all.

use crate::components::BlockType;
use crate::ecs::Ref;
use crate::ecs::asset_id::AssetId;
use crate::gfx::chunk_coord::ChunkCoord;
use crate::math::{SaturatingFrom, lerp, smoothstep};
use alloc::vec;
use alloc::vec::Vec;
use hashbrown::HashMap;

/// One palette entry for the chunk mesher: solidity plus per-face atlas UVs.
///
/// The public, `geometry`-external counterpart of the private
/// `voxel::PaletteSlot`. The streaming subsystem resolves a `VoxelWorld`'s
/// `BlockType` palette into a `Vec<ChunkBlockType>` and hands it to
/// [`super::build_chunk_mesh`].
#[derive(Clone, Copy, Debug)]
pub struct ChunkBlockType {
    /// When false the block is air -- emits no geometry, occludes nothing.
    pub solid: bool,
    /// Atlas UV rect `[u_min, v_min, u_max, v_max]` for the +Y face.
    pub uv_top: [f32; 4],
    /// Atlas UV rect for the -Y face.
    pub uv_bottom: [f32; 4],
    /// Atlas UV rect for the four side faces.
    pub uv_side: [f32; 4],
}

impl From<&BlockType> for ChunkBlockType {
    /// A `BlockType`'s palette entry: per-face UV overrides fall back to the
    /// `uv_min`/`uv_max` rectangle.
    fn from(bt: &BlockType) -> Self {
        let default_rect = [bt.uv_min[0], bt.uv_min[1], bt.uv_max[0], bt.uv_max[1]];
        Self {
            solid: bt.solid,
            uv_top: bt.uv_top.unwrap_or(default_rect),
            uv_bottom: bt.uv_bottom.unwrap_or(default_rect),
            uv_side: bt.uv_side.unwrap_or(default_rect),
        }
    }
}

impl ChunkBlockType {
    /// Empty space: no geometry, occludes nothing.
    const AIR: Self = Self {
        solid: false,
        uv_top: [0.0; 4],
        uv_bottom: [0.0; 4],
        uv_side: [0.0; 4],
    };

    /// A `VoxelWorld` palette resolved against its `BlockType`s, in palette
    /// order. An entry naming no known `BlockType` is logged and becomes air
    /// rather than failing the world.
    pub fn resolve_palette(
        palette: &[Ref<BlockType>],
        block_types: &HashMap<AssetId, BlockType>,
    ) -> Vec<Self> {
        palette
            .iter()
            .map(|id| {
                if let Some(bt) = block_types.get(&id.id()) {
                    Self::from(bt)
                } else {
                    tracing::warn!("voxel world: palette entry {id} is not a known BlockType");
                    Self::AIR
                }
            })
            .collect()
    }
}

/// Deterministic terrain generator for one `VoxelWorld`.
///
/// Constructed once from the world's seed and chunk dimensions; `generate`
/// produces the block array for any chunk on demand.
pub struct ChunkGenerator {
    seed: u64,
    chunk_blocks: [u32; 3],
    // Palette index emitted for the topmost solid block of each column.
    surface_idx: u32,
    // Palette index emitted for solid blocks below the surface.
    subsurface_idx: u32,
}

// Feature size (in blocks) and weight of each value-noise octave. Larger
// features give broad hills; smaller ones add detail. Weights are normalized
// at evaluation time so the combined noise stays in [0, 1).
const OCTAVES: [(i32, f32); 3] = [(64, 1.0), (32, 0.5), (16, 0.25)];

impl ChunkGenerator {
    /// A generator for a world with the given `seed`, chunk dimensions, and
    /// palette length.
    ///
    /// By the `VoxelWorld` palette convention index 0 is air, 1 the surface
    /// block, and 2 (when the palette has it) the subsurface block; a
    /// shorter palette falls back to index 1 for subsurface.
    pub fn new(seed: u64, chunk_blocks: [u32; 3], palette_len: u32) -> Self {
        let surface_idx = u32::from(palette_len > 1);
        let subsurface_idx = if palette_len > 2 { 2 } else { surface_idx };
        Self {
            seed,
            chunk_blocks: [
                chunk_blocks[0].max(1),
                chunk_blocks[1].max(1),
                chunk_blocks[2].max(1),
            ],
            surface_idx,
            subsurface_idx,
        }
    }

    /// Generate the dense block array for chunk `coord`.
    ///
    /// The result has length `dx*dy*dz` with the layout
    /// `index = x + y*dx + z*dx*dy`, exactly what the voxel mesher expects.
    /// A column is solid up to its noise-derived surface height and air above.
    pub fn generate(&self, coord: ChunkCoord) -> Vec<u32> {
        let [dx, dy, dz] = [
            self.chunk_blocks[0] as usize,
            self.chunk_blocks[1] as usize,
            self.chunk_blocks[2] as usize,
        ];
        let mut blocks = vec![0u32; dx * dy * dz];
        // World block coordinate of this chunk's (0,0) corner.
        let base_x = coord.x * i32::saturating_from(self.chunk_blocks[0]);
        let base_z = coord.z * i32::saturating_from(self.chunk_blocks[2]);

        for z in 0..dz {
            for x in 0..dx {
                let wx = base_x + i32::saturating_from(x);
                let wz = base_z + i32::saturating_from(z);
                let height = self.surface_height(wx, wz, i32::saturating_from(dy));
                for y in 0..dy {
                    let yi = i32::saturating_from(y);
                    let id = match yi.cmp(&height) {
                        core::cmp::Ordering::Greater => 0, // air
                        core::cmp::Ordering::Equal => self.surface_idx,
                        core::cmp::Ordering::Less => self.subsurface_idx,
                    };
                    blocks[x + y * dx + z * dx * dy] = id;
                }
            }
        }
        blocks
    }

    /// Surface block height (topmost solid block index) of the column at world
    /// block coordinate `(wx, wz)`, clamped to `[0, chunk_height-1]`.
    ///
    /// Public so the distant-chunk impostor mesher can sample the terrain
    /// surface on a coarse grid without paying for a full dense block array.
    /// Because it keys on world coordinates (like [`generate`](Self::generate)),
    /// two impostor chunks sample identical heights along their shared edge, so
    /// their coarse surfaces meet watertight.
    pub fn surface_height_world(&self, wx: i32, wz: i32) -> i32 {
        self.surface_height(wx, wz, i32::saturating_from(self.chunk_blocks[1]))
    }

    /// Palette index of the surface (topmost) block: `1` when the palette has
    /// a dedicated surface block, else `0`. The impostor mesher uses it to pick
    /// the surface block's atlas UVs so impostors texture like the full chunks.
    pub fn surface_palette_index(&self) -> u32 {
        self.surface_idx
    }

    // Surface block height of the column at world coordinate `(wx, wz)`,
    // clamped to `[0, dy-1]` so every column has at least one solid block and
    // never overflows the chunk.
    fn surface_height(&self, wx: i32, wz: i32, dy: i32) -> i32 {
        let n = self.combined_noise(wx, wz); // [0, 1)
        // Center the terrain around 45% of the chunk height with a +/-30% swing.
        let base = dy as f32 * 0.45;
        let amplitude = dy as f32 * 0.30;
        let h = base + (n - 0.5) * 2.0 * amplitude;
        i32::saturating_from(h).clamp(0, dy - 1)
    }

    // Multi-octave value noise at world block coordinate `(wx, wz)`, in
    // `[0, 1)`. Octave seeds are offset so the octaves are independent.
    fn combined_noise(&self, wx: i32, wz: i32) -> f32 {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;
        for (octave, &(feature, weight)) in OCTAVES.iter().enumerate() {
            let octave_seed = self.seed.wrapping_add(octave as u64 * 0x9E37_79B9);
            sum += weight * value_noise(octave_seed, wx, wz, feature);
            weight_sum += weight;
        }
        if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            0.5
        }
    }
}

// Value noise sampled at world coordinate `(wx, wz)` on a lattice of spacing
// `feature` blocks. Bilinear interpolation of four hashed lattice values with
// a polynomial smoothstep; returns `[0, 1)`.
fn value_noise(seed: u64, wx: i32, wz: i32, feature: i32) -> f32 {
    let feature = feature.max(1);
    // Floored lattice cell + fractional position within it. `div_euclid` /
    // `rem_euclid` floor correctly for negative coordinates, unlike `/` `%`.
    let cell_x = wx.div_euclid(feature);
    let cell_z = wz.div_euclid(feature);
    let tx = wx.rem_euclid(feature) as f32 / feature as f32;
    let tz = wz.rem_euclid(feature) as f32 / feature as f32;

    let v00 = hash01(seed, cell_x, cell_z);
    let v10 = hash01(seed, cell_x + 1, cell_z);
    let v01 = hash01(seed, cell_x, cell_z + 1);
    let v11 = hash01(seed, cell_x + 1, cell_z + 1);

    let sx = smoothstep(0.0, 1.0, tx);
    let sz = smoothstep(0.0, 1.0, tz);
    lerp(lerp(v00, v10, sx), lerp(v01, v11, sx), sz)
}

// Hash an integer lattice point to a pseudo-random `f32` in `[0, 1)`.
//
// Integer-only mixing (a variant of the SplitMix64 finalizer) so the result
// is deterministic and reproducible across platforms.
fn hash01(seed: u64, x: i32, z: i32) -> f32 {
    let mut h = seed;
    h ^= i64::from(x)
        .cast_unsigned()
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= i64::from(z)
        .cast_unsigned()
        .wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 31;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    // Top 24 bits give a uniform [0, 1) without needing all 64.
    (h >> 40) as f32 / (1u64 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // A per-face override wins where set; the remaining faces fall back to the
    // uv_min/uv_max rectangle.
    #[test]
    fn block_type_to_chunk_overrides_then_falls_back() {
        let bt = BlockType {
            solid: true,
            uv_min: [0.1, 0.2],
            uv_max: [0.6, 0.7],
            uv_top: Some([0.0, 0.0, 0.25, 0.25]),
            uv_bottom: None,
            uv_side: None,
        };
        let chunk = ChunkBlockType::from(&bt);
        assert!(chunk.solid);
        let mut known = HashMap::new();
        known.insert(AssetId(5), bt);
        let ids = [Ref::new(AssetId(5)), Ref::new(AssetId(6))];
        let palette = ChunkBlockType::resolve_palette(&ids, &known);
        assert!(
            palette[0].solid && !palette[1].solid,
            "an unknown entry is air"
        );
        assert_eq!(chunk.uv_top, [0.0, 0.0, 0.25, 0.25], "override kept");
        let fallback = [0.1, 0.2, 0.6, 0.7];
        assert_eq!(
            chunk.uv_bottom, fallback,
            "unset face -> uv_min/uv_max rect"
        );
        assert_eq!(chunk.uv_side, fallback);
    }

    #[test]
    fn block_type_to_chunk_air_is_not_solid() {
        let bt = BlockType {
            solid: false,
            ..Default::default()
        };
        assert!(!ChunkBlockType::from(&bt).solid);
    }

    fn make_gen() -> ChunkGenerator {
        ChunkGenerator::new(1234, [16, 24, 16], 3)
    }

    #[test]
    fn generate_produces_the_expected_block_count() {
        let blocks = make_gen().generate(ChunkCoord::new(0, 0));
        assert_eq!(blocks.len(), 16 * 24 * 16);
    }

    #[test]
    fn generation_is_deterministic() {
        let a = make_gen().generate(ChunkCoord::new(3, -2));
        let b = make_gen().generate(ChunkCoord::new(3, -2));
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_produce_different_worlds() {
        let a = ChunkGenerator::new(1, [16, 24, 16], 3).generate(ChunkCoord::new(0, 0));
        let b = ChunkGenerator::new(2, [16, 24, 16], 3).generate(ChunkCoord::new(0, 0));
        assert_ne!(a, b);
    }

    #[test]
    fn columns_are_solid_below_the_surface_and_air_above() {
        let g = make_gen();
        let [dx, dy, dz] = [16usize, 24usize, 16usize];
        let blocks = g.generate(ChunkCoord::new(0, 0));
        for z in 0..dz {
            for x in 0..dx {
                // Find the topmost solid block in the column.
                let mut top = None;
                for y in (0..dy).rev() {
                    if blocks[x + y * dx + z * dx * dy] != 0 {
                        top = Some(y);
                        break;
                    }
                }
                let top = top.expect("every column has a solid block");
                // Everything above is air; everything at/below is solid.
                for y in 0..dy {
                    let solid = blocks[x + y * dx + z * dx * dy] != 0;
                    assert_eq!(solid, y <= top, "column ({x},{z}) y={y}");
                }
            }
        }
    }

    #[test]
    fn terrain_keys_on_world_coordinates() {
        // Each chunk's column heights are exactly the world-coordinate height
        // function sampled over its block range. Because the function depends
        // only on world coords, adjacent chunks line up seamlessly across
        // their shared edge with no per-chunk discontinuity.
        let g = make_gen();
        let [dx, dy, dz] = [16i32, 24i32, 16i32];
        let chunk = ChunkCoord::new(1, -2);
        let blocks = g.generate(chunk);
        let base_x = chunk.x * dx;
        let base_z = chunk.z * dz;
        for z in 0..usize::saturating_from(dz) {
            for x in 0..usize::saturating_from(dx) {
                let mut top = 0;
                for y in (0..usize::saturating_from(dy)).rev() {
                    if blocks
                        [x + y * usize::saturating_from(dx) + z * usize::saturating_from(dx * dy)]
                        != 0
                    {
                        top = i32::saturating_from(y);
                        break;
                    }
                }
                let expected = g.surface_height(
                    base_x + i32::saturating_from(x),
                    base_z + i32::saturating_from(z),
                    dy,
                );
                assert_eq!(top, expected, "column ({x},{z}) height");
            }
        }
    }

    #[test]
    fn value_noise_stays_in_unit_range() {
        for wx in -40..40 {
            for wz in -40..40 {
                let n = value_noise(99, wx, wz, 16);
                assert!((0.0..1.0).contains(&n), "noise {n} out of range");
            }
        }
    }

    #[test]
    fn short_palette_falls_back_to_the_surface_index() {
        // palette_len 2 -> no dedicated subsurface index; below-surface uses 1.
        let g = ChunkGenerator::new(5, [4, 8, 4], 2);
        let blocks = g.generate(ChunkCoord::new(0, 0));
        assert!(blocks.iter().all(|&b| b <= 1));
        assert!(blocks.contains(&1));
    }
}
