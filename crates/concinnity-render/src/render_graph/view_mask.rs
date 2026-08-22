// src/render_graph/view_mask.rs
//
// Per-frame masking of the graph inputs by the viewport's view mode + show
// flags: the runtime counterpart to the init-time requirement trims. Pure
// like `build_frame_graph`, so every backend makes identical decisions, and
// the compiled-graph cache (keyed on the masked inputs) absorbs the cost of a
// toggle as one rebuild.

use super::frame::FrameGraphInputs;
use concinnity_core::gfx::view_modes::{ShowFlags, ViewMode};

/// The frame's effective graph inputs under the view state. Cleared show flags
/// gate their passes off; the flat modes (Unlit, Wireframe) additionally drop
/// the surface-effect stack; the G-buffer channel modes keep the prepass alive
/// and, for the occlusion view, route `ao_output` into the composite.
pub fn apply_view(inputs: &FrameGraphInputs, mode: ViewMode, show: ShowFlags) -> FrameGraphInputs {
    let mut out = *inputs;
    out.composite_reads_ao = false;
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
    if mode.is_flat() {
        out.ssao_enabled = false;
        out.ssr_enabled = false;
        out.rt_reflections_enabled = false;
        out.ssgi_enabled = false;
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
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::frame::build_frame_graph;
    use super::super::passes::PassId;
    use super::*;

    fn everything_on() -> FrameGraphInputs {
        FrameGraphInputs {
            shadow_enabled: true,
            auto_exposure_enabled: true,
            bloom_enabled: true,
            velocity_enabled: true,
            taa_enabled: true,
            ssr_enabled: true,
            fog_enabled: true,
            decals_enabled: true,
            ssr_prepass_enabled: true,
            ssao_enabled: true,
            transparent_enabled: true,
            lines_enabled: true,
            ssgi_enabled: true,
            shadowed_spot_count: 2,
            ..FrameGraphInputs::all_off()
        }
    }

    #[test]
    fn lit_with_all_flags_is_identity() {
        let inputs = everything_on();
        assert_eq!(apply_view(&inputs, ViewMode::Lit, ShowFlags::all()), inputs);
    }

    #[test]
    fn each_flag_masks_its_passes() {
        let inputs = everything_on();
        let m = |f| apply_view(&inputs, ViewMode::Lit, ShowFlags::all().toggled(f));
        let no_shadows = m(ShowFlags::SHADOWS);
        assert!(!no_shadows.shadow_enabled);
        assert_eq!(no_shadows.shadowed_spot_count, 0);
        assert!(!m(ShowFlags::FOG).fog_enabled);
        assert!(!m(ShowFlags::BLOOM).bloom_enabled);
        assert!(!m(ShowFlags::SSGI).ssgi_enabled);
        let no_ssr = m(ShowFlags::SSR);
        assert!(!no_ssr.ssr_enabled && !no_ssr.rt_reflections_enabled);
        assert!(!m(ShowFlags::LINES).lines_enabled);
    }

    #[test]
    fn flat_modes_drop_the_surface_effect_stack() {
        let inputs = everything_on();
        for mode in [ViewMode::Unlit, ViewMode::Wireframe] {
            let out = apply_view(&inputs, mode, ShowFlags::all());
            assert!(!out.ssao_enabled && !out.ssr_enabled && !out.ssgi_enabled);
            assert!(!out.bloom_enabled && !out.fog_enabled && !out.decals_enabled);
            assert!(!out.taa_enabled && !out.upscale_enabled && !out.velocity_enabled);
            // Shadows follow their show flag, not the mode.
            assert!(out.shadow_enabled);
        }
    }

    #[test]
    fn occlusion_view_routes_ao_into_the_composite() {
        let inputs = everything_on();
        let out = apply_view(&inputs, ViewMode::Occlusion, ShowFlags::all());
        assert!(out.composite_reads_ao && out.ssao_enabled);
        assert!(!out.bloom_enabled, "bloom never overlaps the extended AO");
        // With SSAO unavailable there is nothing to route.
        let mut no_ssao = everything_on();
        no_ssao.ssao_enabled = false;
        let out = apply_view(&no_ssao, ViewMode::Occlusion, ShowFlags::all());
        assert!(!out.composite_reads_ao);
    }

    #[test]
    fn channel_modes_keep_the_prepass_alive() {
        let mut inputs = everything_on();
        inputs.ssr_prepass_enabled = false;
        for mode in [
            ViewMode::Normals,
            ViewMode::Roughness,
            ViewMode::Occlusion,
            ViewMode::Depth,
        ] {
            assert!(apply_view(&inputs, mode, ShowFlags::all()).ssr_prepass_enabled);
        }
    }

    #[test]
    fn masked_inputs_still_compile_and_ao_read_orders_composite_last() {
        let inputs = everything_on();
        for mode in ViewMode::ALL {
            let out = apply_view(&inputs, mode, ShowFlags::all());
            let graph = build_frame_graph(&out).expect("masked graph compiles");
            assert_eq!(
                graph.passes.last().map(|p| p.id),
                Some(PassId::Composite),
                "{mode:?}"
            );
        }
        // The occlusion view's composite read extends ao_output's lifetime to
        // the last pass.
        let out = apply_view(&inputs, ViewMode::Occlusion, ShowFlags::all());
        let graph = build_frame_graph(&out).unwrap();
        let composite_idx = graph.passes.len() - 1;
        let ao = graph
            .resources
            .iter()
            .find(|r| r.label == "ao_output")
            .expect("ao_output present");
        assert_eq!(ao.lifetime.last, composite_idx);
    }
}
