// The hand-written gates the pass table replaced, kept verbatim as the oracle
// the table's selection is checked against.

use crate::gfx::view_modes::{ShowFlags, ViewMode};
use crate::render::planar_reflection::planar_pass_needed;
use crate::render::post::reflection_path::ReflectionPath;
use crate::render::render_graph::FrameGraphInputs;
use crate::render::render_graph::gates::{FeatureGates, FrameFacts};

// This frame's graph inputs, before the view mode and show flags mask them.
pub(super) fn frame_graph_inputs(gates: &FeatureGates, facts: &FrameFacts) -> FrameGraphInputs {
    let path = ReflectionPath::new(gates.ssr, gates.rt_pass, facts.bvh_live);
    let taa = gates.taa && !facts.upscaler;
    let velocity = taa || facts.upscaler || gates.ssgi || path.rt_trace();
    let bindless = facts.cull_ready && facts.cull_records > 0 && !facts.world_hidden;
    let rt_transparent = path.rt_trace() && facts.rt_transparent_ready;
    let mesh_glass = facts.visible_see_through && rt_transparent;
    FrameGraphInputs {
        shadow_enabled: facts.shadow,
        shadow_map_size: facts.shadow_map_size,
        far_shadow_map_size: if facts.shadow {
            facts.far_shadow_map_size
        } else {
            0
        },
        far_shadow_drawn: facts.shadow && facts.far_shadow_drawn,
        hdr_width: facts.hdr_width,
        hdr_height: facts.hdr_height,
        hdr_sample_count: facts.hdr_sample_count,
        bindless_cull_enabled: bindless,
        auto_exposure_enabled: gates.auto_exposure,
        bloom_enabled: facts.bloom_chain && facts.bloom_intensity > 0.0,
        velocity_enabled: velocity,
        taa_enabled: taa,
        ssr_enabled: path.ssr_resolve(),
        particles_enabled: facts.particles,
        late_particles_enabled: facts.late_particles,
        fog_enabled: facts.fog,
        decals_enabled: facts.decals,
        ssr_prepass_enabled: gates.ssr || gates.ssgi || gates.ssgi_resolve || gates.rt_pass,
        ssao_enabled: gates.ssao,
        upscale_enabled: facts.upscaler,
        transparent_enabled: facts.visible_surfaces || mesh_glass,
        planar_reflection_enabled: planar_pass_needed(
            facts.planar_targets,
            facts.water_planar_slot,
            rt_transparent,
        ),
        lines_enabled: facts.lines,
        raymarch_enabled: facts.visible_volumes,
        two_pass_occlusion_enabled: facts.two_pass_ready && facts.cull_records > 0,
        hiz_build_enabled: facts.hiz,
        shadow_receivers_enabled: facts.shadow && facts.hiz && facts.shadow_receivers,
        atmosphere_enabled: facts.atmosphere,
        clouds_enabled: facts.atmosphere && facts.clouds,
        water_sim_enabled: facts.water_sim,
        reactive_mask_enabled: gates.traits.reactive_mask,
        upscale_reads_reactive: facts.upscaler && facts.upscaler_reads_reactive,
        disocclusion_enabled: facts.disocclusion,
        ssgi_enabled: gates.ssgi,
        ssgi_resolve_enabled: gates.ssgi_resolve,
        rt_reflections_enabled: if gates.traits.composite_scene_fixed {
            path.rt_node()
        } else {
            path.rt_trace()
        },
        grass_enabled: facts.grass && bindless,
        grass_shadow_enabled: facts.grass_casts,
        gbuffer_prepass_enabled: gates.gbuffer,
        world_hidden: facts.world_hidden,
        clustering_enabled: facts.clustered,
        composite_reads_ao: false,
        composite_reads_motion: false,
        composite_reads_reactive: false,
        shadowed_spot_count: facts.shadowed_spot_count,
        spot_shadow_slice_size: facts.spot_shadow_slice_size,
        // A traced reflection already shows the prop's surroundings.
        prop_probe_enabled: facts.prop_probe && bindless && !path.rt_trace(),
    }
}

// The frame's effective graph inputs under the view state. Cleared show flags
// gate their passes off; the flat modes (Unlit, Wireframe) additionally drop
// the surface-effect stack; the G-buffer channel modes keep the prepass alive
// and, for the occlusion view, route `ao_output` into the composite.
pub(super) fn apply_view(
    inputs: &FrameGraphInputs,
    mode: ViewMode,
    show: ShowFlags,
) -> FrameGraphInputs {
    let mut out = *inputs;
    out.composite_reads_ao = false;
    out.composite_reads_motion = false;
    out.composite_reads_reactive = false;
    if !show.contains(ShowFlags::SHADOWS) {
        out.shadow_enabled = false;
        out.shadowed_spot_count = 0;
    }
    if !show.contains(ShowFlags::FOG) {
        out.fog_enabled = false;
    }
    if !show.contains(ShowFlags::BLOOM) {
        out.bloom_enabled = false;
    }
    if !show.contains(ShowFlags::SSGI) {
        out.ssgi_enabled = false;
    }
    if !show.contains(ShowFlags::SSR) {
        out.ssr_enabled = false;
        out.rt_reflections_enabled = false;
    }
    if !show.contains(ShowFlags::LINES) {
        out.lines_enabled = false;
    }
    if !show.contains(ShowFlags::REACTIVE) {
        out.reactive_mask_enabled = false;
    }
    if mode.is_flat() {
        out.ssao_enabled = false;
        out.ssr_enabled = false;
        out.rt_reflections_enabled = false;
        out.ssgi_enabled = false;
        out.ssgi_resolve_enabled = false;
        out.bloom_enabled = false;
        out.fog_enabled = false;
        out.decals_enabled = false;
        out.taa_enabled = false;
        out.upscale_enabled = false;
        out.velocity_enabled = false;
        out.ssr_prepass_enabled = false;
    }
    if mode.is_gbuffer_channel() {
        // The prepass must run for its channels to be current this frame.
        out.ssr_prepass_enabled = true;
        // The composite presents the channel, not the scene, so bloom is
        // wasted; keeping it off also keeps its pooled target from sharing a
        // heap slot lifetime with the AO output the occlusion view extends.
        out.bloom_enabled = false;
        out.composite_reads_ao = mode == ViewMode::Occlusion && out.ssao_enabled;
        out.composite_reads_motion = mode == ViewMode::Motion;
        out.composite_reads_reactive = mode == ViewMode::Reactive;
    }
    out
}
