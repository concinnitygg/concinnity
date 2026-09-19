//! CBV/SRV/UAV heap slot layout for the DirectX backend. The shader-visible
//! descriptor heap is a flat array of fixed slots assigned in a positional
//! cascade: each block's base is the previous block's base plus its
//! reservation. Keeping the cascade in one function (instead of two dozen
//! inline `let x_slot = prev_slot + prev_extra` bindings) lets a unit test
//! assert it stays gap-free and that the total matches the created heap size,
//! so a stray offset edit fails a test instead of silently misbinding a
//! descriptor at shader time (a visual glitch or an out-of-bounds heap write
//! with no compile-time signal).
//!
//! Heap order:
//!   [0]                            shadow map array SRV (Texture2DArray)
//!   [1]                            IBL irradiance cube SRV
//!   [2]                            IBL prefilter cube SRV
//!   [atlas_base_slot..]            text atlas SRVs
//!   [hdr_srv_slot]                 HDR scene target SRV (composite pass)
//!   [bloom_srv_base_slot..]        bloom mip SRVs
//!   [lut_srv_slot]                 3D color-grading LUT SRV
//!   [post_srv_base_slot..]         POST_TARGET_SLOTS shared post-pass target SRVs
//!   [ssao_srv_base_slot..]         (SSAO) ao_raw + ao_blurred
//!   [ssao_white_srv_slot]          1x1 white occlusion fallback (always)
//!   [decal_depth_srv_slot]         main-depth SRV (decal + glass + line passes)
//!   [decal_srv_base_slot..]        MAX_DECALS per-decal albedo SRVs
//!   [particle_srv_base_slot..]     MAX_EMITTERS emitter albedo SRVs
//!   [fog_froxel_uav_slot]          froxel-volume UAV
//!   [fog_froxel_srv_slot]          froxel-volume SRV
//!   [upscale_uav_slot]             temporal-upscale output UAV
//!   [upscale_srv_slot]             temporal-upscale output SRV
//!   [raymarch_srv_base_slot..+4]   raymarch t0..t3 (shadow, irr, prefilter, scene)
//!   [hiz_srv_slot]                 Hi-Z pyramid SRV (covers every mip)
//!   [hiz_uav_base_slot..]          HIZ_MAX_MIPS per-mip UAVs
//!   [probe_capture_srv_slot]       reflection-probe capture pyramid SRV
//!   [probe_capture_uav_base_slot..] PROBE_MAX_MIPS capture per-mip UAVs
//!   [probe_cube_uav_base_slot..]   PROBE_MAX_MIPS probe-cube per-mip UAVs
//!   [probe_mip0_pair_slot..+2]     the mirror copy's (capture mip 0, probe mip 0)
//!   [transparent_scene_copy_srv_slot] pre-transparent scene snapshot SRV
//!   [glass_reflection_srv_base_slot..] GLASS_REFLECTION_SRV_SLOTS glass reflection layer windows
//!   [gbuffer_srv_base_slot..]      (G-buffer) normal+depth, roughness, velocity
//!   [rt_output_srv_slot]           (RT) reflection output
//!   [refl_composite_srv_base_slot..] (reflections) composited output + blur
//!   [planar_resolve_srv_base_slot..] planar reflector resolves
//!   [flat_pool_base_slot..]        bindless albedo + normal pool, per frame
//!   [probe_cube_base_slot..]       MAX_PROBES reflection-probe cubes
//!   [spot_shadow_srv_slot]         spot shadow depth array SRV (Texture2DArray)
//!   [ltc_srv_base_slot..+2]        area-light LTC tables (matrix, magnitude)
//!   srv_slots                      total descriptor count (heap size)
//!
//! The RTV heap follows the same cascade in `RtvHeapLayout`, and the DSV heap's
//! fixed slots are the `DSV_*` constants.

use concinnity_core::gfx::render_types::{MAX_SHADOWED_SPOTS, NUM_SHADOW_CASCADES};

use super::HIZ_MAX_MIPS;
use crate::directx::context::FRAMES;
use crate::directx::decal::MAX_DECALS;
use crate::directx::particle::MAX_EMITTERS;
use crate::directx::post::descriptors::POST_TARGET_SLOTS;
use crate::directx::probe_prefilter::PROBE_MAX_MIPS;

// Per-world feature counts that size the variable-length blocks of the SRV
// heap. The fixed-size blocks (decals, particles, raymarch, Hi-Z, fog,
// upscale) use module constants and are not parameters.
pub(in crate::directx) struct SrvHeapParams {
    pub n_atlases: usize,
    pub bloom_count: usize,
    // SSAO's SRV reservation when enabled, else 0: 2 (raw + blurred
    // occlusion). The view normal / depth / roughness / velocity all come from
    // the unified G-buffer pre-pass (`gbuffer_srv_extra`). The shared post
    // passes take theirs from a fixed block instead (`post_srv_base_slot`), so
    // no effect drawing through that seam appears here.
    pub ssao_srv_extra: usize,
    // 3 (normal+depth, roughness, velocity) when the unified G-buffer pre-pass
    // is active, else 0.
    pub gbuffer_srv_extra: usize,
    // 1 when hardware ray-traced reflections are enabled (the RT output target's
    // SRV), else 0.
    pub rt_output_srv_extra: usize,
    // 2 (composited output + reduced-res blur) when the reflection composite is
    // built (SSR resolve or RT reflections authored), else 0.
    pub refl_composite_srv_extra: usize,
    // One resolve SRV per distinct planar reflector plane (0..MAX_PLANAR_PLANES),
    // reserved when the world has glass panes assigned to a planar slot, else 0.
    // The glass pass binds these per pane.
    pub planar_resolve_srv_extra: usize,
    // Flat deduplicated bindless pool sizes: one SRV per distinct albedo-pool
    // texture (incl. emissive / ORM maps) followed by one per distinct normal
    // map. The bindless main pass and the RT hit shader address this region by a
    // flat index (`albedo = texture_slot`, `normal = albedo_count + normal_slot`),
    // mirroring Vulkan/Metal. `albedo_count` is the albedo resource count
    // (>= 1: a 1x1 white fallback stands in when no albedo textures exist);
    // `normal_count` includes the slot-0 flat-normal fallback.
    pub albedo_count: usize,
    pub normal_count: usize,
}

// Resolved slot indices into the CBV/SRV/UAV heap. Field order matches the
// heap order documented above; `srv_slots` is the total descriptor count the
// heap is created with.
pub(in crate::directx) struct SrvHeapLayout {
    pub atlas_base_slot: usize,
    pub hdr_srv_slot: usize,
    pub bloom_srv_base_slot: usize,
    pub lut_srv_slot: usize,
    // Base of the shared post passes' target SRVs. A fixed
    // `POST_TARGET_SLOTS`-wide block, always reserved, sub-allocated at runtime
    // by `post/descriptors.rs`. Fixed and unconditional on purpose: a pass that
    // draws through the shared seam should not have to add a row to this
    // cascade, and a block that appears only when one feature is on would put
    // every later block at a different offset per world.
    pub post_srv_base_slot: usize,
    pub ssao_srv_base_slot: usize,
    pub ssao_white_srv_slot: usize,
    pub decal_depth_srv_slot: usize,
    pub decal_srv_base_slot: usize,
    pub particle_srv_base_slot: usize,
    pub fog_froxel_uav_slot: usize,
    pub fog_froxel_srv_slot: usize,
    pub upscale_uav_slot: usize,
    pub upscale_srv_slot: usize,
    pub raymarch_srv_base_slot: usize,
    pub hiz_srv_slot: usize,
    pub hiz_uav_base_slot: usize,
    pub probe_capture_srv_slot: usize,
    pub probe_capture_uav_base_slot: usize,
    pub probe_cube_uav_base_slot: usize,
    pub probe_mip0_pair_slot: usize,
    pub transparent_scene_copy_srv_slot: usize,
    // The glass reflection pre-pass's three two-descriptor windows over its
    // layers (see `GLASS_REFLECTION_SRV_SLOTS`).
    pub glass_reflection_srv_base_slot: usize,
    pub gbuffer_srv_base_slot: usize,
    pub rt_output_srv_slot: usize,
    // Reflection-composite SRVs: [0] composited output, [1] reduced-res blur.
    pub refl_composite_srv_base_slot: usize,
    // Planar reflection resolve SRVs (one per distinct reflector plane).
    pub planar_resolve_srv_base_slot: usize,
    pub flat_pool_base_slot: usize,
    // Contiguous MAX_PROBES-slot block of reflection-probe cube SRVs (the bindless
    // main shader's `TextureCube probe_cubes[MAX_PROBES]` table). Filled with the sky
    // prefilter cube at init; a baked probe overwrites its slot.
    pub probe_cube_base_slot: usize,
    // Spot shadow depth array SRV. A single slot rather than one of the three
    // fixed globals: the main root signatures reach the CSM array and the IBL
    // cubes as one contiguous 3-slot table, so slots 0..3 cannot take a fourth
    // member without splitting that table.
    pub spot_shadow_srv_slot: usize,
    // The two area-light LTC lookup tables, contiguous so one 2-descriptor
    // table covers both: [0] the inverse-transform matrix (RGBA32F), [1] the
    // magnitude / Fresnel pair (RG32F).
    pub ltc_srv_base_slot: usize,
    pub srv_slots: usize,
}

// The three global SRVs (shadow array, IBL irradiance, IBL prefilter) occupy
// slots [0, 3); the first per-world block starts here.
const GLOBAL_SRV_COUNT: usize = 3;

// Reflection-probe cube array length (must equal `concinnity_core::render::uniforms::MAX_PROBES`).
const PROBE_CUBE_COUNT: usize = concinnity_core::render::uniforms::MAX_PROBES;

impl SrvHeapLayout {
    pub(in crate::directx) fn compute(p: &SrvHeapParams) -> Self {
        let atlas_base_slot = GLOBAL_SRV_COUNT;
        // Text atlases. `n_atlases.max(1)` reserves one slot even with no atlas
        // so the HDR SRV that follows always lands at a stable offset.
        let hdr_srv_slot = atlas_base_slot + p.n_atlases.max(1);
        // The composite pass binds {HDR, bloom mip 0} as one contiguous
        // 2-descriptor table, so bloom mip 0 sits right after the HDR SRV.
        let bloom_srv_base_slot = hdr_srv_slot + 1;
        let lut_srv_slot = bloom_srv_base_slot + p.bloom_count;
        let post_srv_base_slot = lut_srv_slot + 1;
        let ssao_srv_base_slot = post_srv_base_slot + POST_TARGET_SLOTS;
        // The white fallback always sits one slot past the SSAO block (present
        // whether SSAO is on or off) so the main pass can bind a pass-through
        // occlusion when SSAO is disabled.
        let ssao_white_srv_slot = ssao_srv_base_slot + p.ssao_srv_extra;
        let decal_depth_srv_slot = ssao_white_srv_slot + 1;
        let decal_srv_base_slot = decal_depth_srv_slot + 1;
        let particle_srv_base_slot = decal_srv_base_slot + MAX_DECALS;
        let fog_froxel_uav_slot = particle_srv_base_slot + MAX_EMITTERS;
        let fog_froxel_srv_slot = fog_froxel_uav_slot + 1;
        let upscale_uav_slot = fog_froxel_srv_slot + 1;
        let upscale_srv_slot = upscale_uav_slot + 1;
        let raymarch_srv_base_slot = upscale_srv_slot + 1;
        let hiz_srv_slot = raymarch_srv_base_slot + 4;
        let hiz_uav_base_slot = hiz_srv_slot + 1;
        // One bake's convolution descriptors. Rewritten per bake rather than per
        // probe slot: only one probe convolves at a time, and the install's fence
        // gate proves the prior bake's dispatches retired before the next rewrite.
        let probe_capture_srv_slot = hiz_uav_base_slot + HIZ_MAX_MIPS;
        let probe_capture_uav_base_slot = probe_capture_srv_slot + 1;
        let probe_cube_uav_base_slot = probe_capture_uav_base_slot + PROBE_MAX_MIPS;
        let probe_mip0_pair_slot = probe_cube_uav_base_slot + PROBE_MAX_MIPS;
        let transparent_scene_copy_srv_slot = probe_mip0_pair_slot + 2;
        let glass_reflection_srv_base_slot = transparent_scene_copy_srv_slot + 1;
        // Unified G-buffer SRVs (normal+depth, roughness, velocity). 3 slots
        // when any screen-space consumer drives the pre-pass, else 0.
        let gbuffer_srv_base_slot = glass_reflection_srv_base_slot + GLASS_REFLECTION_SRV_SLOTS;
        // RT-reflection output SRV: one slot at the heap tail when RT is on.
        let rt_output_srv_slot = gbuffer_srv_base_slot + p.gbuffer_srv_extra;
        // Reflection-composite SRVs (composited output + reduced-res blur): 2 slots
        // when SSR resolve or RT is authored.
        let refl_composite_srv_base_slot = rt_output_srv_slot + p.rt_output_srv_extra;
        // Planar reflection resolve SRVs: one per distinct reflector plane, bound
        // per pane by the glass pass.
        let planar_resolve_srv_base_slot =
            refl_composite_srv_base_slot + p.refl_composite_srv_extra;
        // Flat deduplicated bindless pool: [albedo SRVs..] ++ [normal SRVs..],
        // one full copy per frame in flight. The bindless main pass and the RT
        // hit shader bind the current frame's copy and index it by a flat slot.
        // Per-frame copies let a streamed texture swap rewrite the copy whose
        // frame just fence-waited (provably unreferenced) instead of draining
        // the device to rewrite one shared region while lists reference it.
        let flat_pool_base_slot = planar_resolve_srv_base_slot + p.planar_resolve_srv_extra;
        // Reflection-probe cube array at the heap tail (MAX_PROBES contiguous cube
        // SRVs); a single descriptor table covers the whole block.
        let probe_cube_base_slot = flat_pool_base_slot + FRAMES * (p.albedo_count + p.normal_count);
        // Spot shadow array SRV. Always reserved: a world with no shadowed spot
        // binds a 1x1 fallback array there so the descriptor is never unwritten.
        let spot_shadow_srv_slot = probe_cube_base_slot + PROBE_CUBE_COUNT;
        // Area-light LTC tables. Scene-independent (fitted at build time), so
        // they are always reserved and always uploaded.
        let ltc_srv_base_slot = spot_shadow_srv_slot + 1;
        let srv_slots = ltc_srv_base_slot + 2;
        Self {
            atlas_base_slot,
            hdr_srv_slot,
            bloom_srv_base_slot,
            lut_srv_slot,
            post_srv_base_slot,
            ssao_srv_base_slot,
            ssao_white_srv_slot,
            decal_depth_srv_slot,
            decal_srv_base_slot,
            particle_srv_base_slot,
            fog_froxel_uav_slot,
            fog_froxel_srv_slot,
            upscale_uav_slot,
            upscale_srv_slot,
            raymarch_srv_base_slot,
            hiz_srv_slot,
            hiz_uav_base_slot,
            probe_capture_srv_slot,
            probe_capture_uav_base_slot,
            probe_cube_uav_base_slot,
            probe_mip0_pair_slot,
            transparent_scene_copy_srv_slot,
            glass_reflection_srv_base_slot,
            gbuffer_srv_base_slot,
            rt_output_srv_slot,
            refl_composite_srv_base_slot,
            planar_resolve_srv_base_slot,
            flat_pool_base_slot,
            probe_cube_base_slot,
            spot_shadow_srv_slot,
            ltc_srv_base_slot,
            srv_slots,
        }
    }
}

// The live-toggleable Quality features (TAA, SSAO, SSR, SSGI, and the unified
// G-buffer pre-pass they share) reserve their RTV / DSV / SRV slots
// UNCONDITIONALLY, independent of the world's init-time gates. The slots are
// fixed positions the passes bind by absolute index, so a live toggle
// (`apply_quality_settings`) can build a feature that launched off and write
// into its pre-reserved slot without shifting any other feature's slots. A
// reserved-but-unbuilt feature leaves its slots unwritten; that is safe because
// no always-running pass binds them (each feature's own pass runs only when the
// feature is on, and the main pass's SSAO occlusion binding falls back to the
// 1x1 white slot).
//
// SSAO: ao_raw + ao. View normal + depth come from the G-buffer pre-pass, so no
// DSV.
pub(super) const SSAO_TARGETS: usize = 2;
// Unified G-buffer pre-pass: normal+depth, roughness, velocity, plus one DSV
// (`DSV_GBUFFER_DEPTH_SLOT`) for its private depth.
pub(super) const GBUFFER_TARGETS: usize = 3;
// RT-reflection output: the trace writes the RTV, the post stack samples the SRV.
pub(super) const RT_OUTPUT_TARGETS: usize = 1;
// Reflection composite: composited output + reduced-res blur.
pub(super) const REFL_COMPOSITE_TARGETS: usize = 2;
// The glass reflection pre-pass's two layers. Its SRVs are three contiguous
// two-descriptor windows the RT transparent signature's t11..t12 table points
// at: the first layer's pass (both null), the second layer's (the first layer,
// null) and the scene pass's (both layers). A null SRV reads as zero, which is
// the empty layer the first one peels behind.
pub(in crate::directx) const GLASS_REFLECTION_TARGETS: usize = 2;
pub(in crate::directx) const GLASS_REFLECTION_SRV_SLOTS: usize = 6;

// DSV heap slots: the main depth, one per shadow cascade (a slice each into the
// shadow map array), one per shadowed spot slice, the unified G-buffer
// pre-pass's private depth buffer, then the glass reflection pre-pass's.
pub(super) const DSV_MAIN_DEPTH_SLOT: usize = 0;
pub(super) const DSV_SHADOW_BASE_SLOT: usize = DSV_MAIN_DEPTH_SLOT + 1;
pub(super) const DSV_SPOT_SHADOW_BASE_SLOT: usize = DSV_SHADOW_BASE_SLOT + NUM_SHADOW_CASCADES;
pub(super) const DSV_GBUFFER_DEPTH_SLOT: usize = DSV_SPOT_SHADOW_BASE_SLOT + MAX_SHADOWED_SPOTS;
pub(super) const DSV_GLASS_REFLECTION_DEPTH_SLOT: usize = DSV_GBUFFER_DEPTH_SLOT + 1;
pub(super) const DSV_SLOTS: usize = DSV_GLASS_REFLECTION_DEPTH_SLOT + 1;

// Resolved slot indices into the RTV heap, after the back-buffer views at
// `[0, FRAMES)`. `rtv_slots` is the total descriptor count the heap is created
// with.
pub(super) struct RtvHeapLayout {
    // HDR scene target.
    pub hdr_slot: usize,
    pub bloom_base_slot: usize,
    // The shared fullscreen post passes' target RTVs: a fixed
    // `POST_TARGET_SLOTS` block sub-allocated at runtime by
    // `post/descriptors.rs`. A post target is color only, so it reserves no DSV.
    pub post_base_slot: usize,
    pub ssao_base_slot: usize,
    // `hdr_resolve`, which the projected-decal pass renders into. Reserved only
    // under MSAA; the MSAA-off path writes through the HDR scene RTV.
    pub decal_resolve_slot: usize,
    pub gbuffer_base_slot: usize,
    pub rt_output_slot: usize,
    pub refl_composite_base_slot: usize,
    pub glass_reflection_base_slot: usize,
    pub rtv_slots: usize,
}

impl RtvHeapLayout {
    pub(super) fn compute(bloom_count: usize, msaa_samples: u32) -> Self {
        let hdr_slot = FRAMES;
        let bloom_base_slot = hdr_slot + 1;
        let post_base_slot = bloom_base_slot + bloom_count;
        let ssao_base_slot = post_base_slot + POST_TARGET_SLOTS;
        let decal_resolve_slot = ssao_base_slot + SSAO_TARGETS;
        let gbuffer_base_slot = decal_resolve_slot + usize::from(msaa_samples > 1);
        let rt_output_slot = gbuffer_base_slot + GBUFFER_TARGETS;
        let refl_composite_base_slot = rt_output_slot + RT_OUTPUT_TARGETS;
        let glass_reflection_base_slot = refl_composite_base_slot + REFL_COMPOSITE_TARGETS;
        let rtv_slots = glass_reflection_base_slot + GLASS_REFLECTION_TARGETS;
        Self {
            hdr_slot,
            bloom_base_slot,
            post_base_slot,
            ssao_base_slot,
            decal_resolve_slot,
            gbuffer_base_slot,
            rt_output_slot,
            refl_composite_base_slot,
            glass_reflection_base_slot,
            rtv_slots,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Walk the cascade for a given feature set: pair each block's reported
    // base with the size that block is independently known to occupy, then
    // assert every base equals the running total of all earlier reservations
    // (gap-free, no overlap) and that `srv_slots` covers the whole chain.
    //
    // The reservations here are derived independently of `compute`'s
    // arithmetic, so an offset slip in `compute` (a `+ 2` where `+ 1` was
    // meant, or a block sized off the wrong constant) makes a base disagree
    // with the running total and fails the assert.
    fn assert_gap_free(p: &SrvHeapParams) {
        let l = SrvHeapLayout::compute(p);
        let blocks: [(usize, usize); 31] = [
            (l.atlas_base_slot, p.n_atlases.max(1)),
            (l.hdr_srv_slot, 1),
            (l.bloom_srv_base_slot, p.bloom_count),
            (l.lut_srv_slot, 1),
            (l.post_srv_base_slot, POST_TARGET_SLOTS),
            (l.ssao_srv_base_slot, p.ssao_srv_extra),
            (l.ssao_white_srv_slot, 1),
            (l.decal_depth_srv_slot, 1),
            (l.decal_srv_base_slot, MAX_DECALS),
            (l.particle_srv_base_slot, MAX_EMITTERS),
            (l.fog_froxel_uav_slot, 1),
            (l.fog_froxel_srv_slot, 1),
            (l.upscale_uav_slot, 1),
            (l.upscale_srv_slot, 1),
            (l.raymarch_srv_base_slot, 4),
            (l.hiz_srv_slot, 1),
            (l.hiz_uav_base_slot, HIZ_MAX_MIPS),
            (l.probe_capture_srv_slot, 1),
            (l.probe_capture_uav_base_slot, PROBE_MAX_MIPS),
            (l.probe_cube_uav_base_slot, PROBE_MAX_MIPS),
            (l.probe_mip0_pair_slot, 2),
            (l.transparent_scene_copy_srv_slot, 1),
            (l.glass_reflection_srv_base_slot, 6),
            (l.gbuffer_srv_base_slot, p.gbuffer_srv_extra),
            (l.rt_output_srv_slot, p.rt_output_srv_extra),
            (l.refl_composite_srv_base_slot, p.refl_composite_srv_extra),
            (l.planar_resolve_srv_base_slot, p.planar_resolve_srv_extra),
            (
                l.flat_pool_base_slot,
                FRAMES * (p.albedo_count + p.normal_count),
            ),
            (l.probe_cube_base_slot, PROBE_CUBE_COUNT),
            (l.spot_shadow_srv_slot, 1),
            (l.ltc_srv_base_slot, 2),
        ];
        let mut expected_base = GLOBAL_SRV_COUNT;
        for (i, (base, count)) in blocks.iter().enumerate() {
            assert_eq!(
                *base, expected_base,
                "block {i} base {base} should sit at running total {expected_base}",
            );
            expected_base += count;
        }
        assert_eq!(
            l.srv_slots, expected_base,
            "srv_slots must cover every block exactly",
        );
        // The heap always reserves at least the three global SRVs.
        assert!(l.srv_slots >= GLOBAL_SRV_COUNT);
    }

    #[test]
    fn layout_gap_free_all_features_on() {
        assert_gap_free(&SrvHeapParams {
            n_atlases: 2,
            bloom_count: 6,
            ssao_srv_extra: 2,
            gbuffer_srv_extra: 3,
            rt_output_srv_extra: 1,
            refl_composite_srv_extra: 2,
            planar_resolve_srv_extra: 2,
            albedo_count: 9,
            normal_count: 4,
        });
    }

    #[test]
    fn layout_gap_free_all_features_off() {
        assert_gap_free(&SrvHeapParams {
            n_atlases: 0,
            bloom_count: 0,
            ssao_srv_extra: 0,
            gbuffer_srv_extra: 0,
            rt_output_srv_extra: 0,
            refl_composite_srv_extra: 0,
            planar_resolve_srv_extra: 0,
            albedo_count: 1,
            normal_count: 1,
        });
    }

    #[test]
    fn layout_gap_free_mixed_features() {
        assert_gap_free(&SrvHeapParams {
            n_atlases: 1,
            bloom_count: 5,
            ssao_srv_extra: 0,
            gbuffer_srv_extra: 3,
            rt_output_srv_extra: 1,
            refl_composite_srv_extra: 2,
            planar_resolve_srv_extra: 1,
            albedo_count: 50,
            normal_count: 12,
        });
    }

    // The per-world blocks must start past the three fixed global SRVs
    // regardless of feature set, so slot 0/1/2 are never reused.
    #[test]
    fn first_block_clears_the_global_srvs() {
        let l = SrvHeapLayout::compute(&SrvHeapParams {
            n_atlases: 0,
            bloom_count: 0,
            ssao_srv_extra: 0,
            gbuffer_srv_extra: 0,
            rt_output_srv_extra: 0,
            refl_composite_srv_extra: 0,
            planar_resolve_srv_extra: 0,
            albedo_count: 1,
            normal_count: 1,
        });
        assert_eq!(l.atlas_base_slot, GLOBAL_SRV_COUNT);
        assert!(l.hdr_srv_slot >= GLOBAL_SRV_COUNT);
    }

    // The RTV blocks must follow the back buffers gap-free, with the decal
    // resolve slot present only under MSAA. Sizes are restated independently of
    // `compute` so an offset slip there fails the assert.
    fn assert_rtv_gap_free(bloom_count: usize, msaa_samples: u32) {
        let l = RtvHeapLayout::compute(bloom_count, msaa_samples);
        let decal_resolve = if msaa_samples > 1 { 1 } else { 0 };
        let blocks: [(usize, usize); 9] = [
            (l.hdr_slot, 1),
            (l.bloom_base_slot, bloom_count),
            (l.post_base_slot, POST_TARGET_SLOTS),
            (l.ssao_base_slot, 2),
            (l.decal_resolve_slot, decal_resolve),
            (l.gbuffer_base_slot, 3),
            (l.rt_output_slot, 1),
            (l.refl_composite_base_slot, 2),
            (l.glass_reflection_base_slot, 2),
        ];
        let mut expected_base = FRAMES;
        for (i, (base, count)) in blocks.iter().enumerate() {
            assert_eq!(
                *base, expected_base,
                "RTV block {i} base {base} should sit at running total {expected_base}",
            );
            expected_base += count;
        }
        assert_eq!(l.rtv_slots, expected_base);
    }

    #[test]
    fn rtv_layout_gap_free_without_msaa() {
        assert_rtv_gap_free(6, 1);
    }

    #[test]
    fn rtv_layout_gap_free_with_msaa() {
        assert_rtv_gap_free(5, 4);
    }

    #[test]
    fn rtv_layout_without_bloom_starts_post_after_hdr() {
        let l = RtvHeapLayout::compute(0, 1);
        assert_eq!(l.post_base_slot, FRAMES + 1);
        assert_eq!(l.decal_resolve_slot, l.gbuffer_base_slot);
    }

    // Main depth first, then a view per cascade and per shadowed spot slice,
    // then the G-buffer and glass reflection depths, with the heap sized to cover
    // the last one.
    #[test]
    fn dsv_layout_covers_every_depth_view() {
        assert_eq!(DSV_MAIN_DEPTH_SLOT, 0);
        assert_eq!(
            DSV_SPOT_SHADOW_BASE_SLOT - DSV_SHADOW_BASE_SLOT,
            NUM_SHADOW_CASCADES
        );
        assert_eq!(
            DSV_GBUFFER_DEPTH_SLOT - DSV_SPOT_SHADOW_BASE_SLOT,
            MAX_SHADOWED_SPOTS
        );
        assert_eq!(DSV_GLASS_REFLECTION_DEPTH_SLOT, DSV_GBUFFER_DEPTH_SLOT + 1);
        assert_eq!(DSV_SLOTS, 3 + NUM_SHADOW_CASCADES + MAX_SHADOWED_SPOTS);
    }
}
