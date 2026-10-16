//! The fullscreen post stack and the two compute helpers that feed it. These
//! params blocks carry no host-shape gate, so each is checked against all three
//! targets.

use concinnity_core::gfx::render_types::{
    CompositeParams, PostProcessParams, SsaoParams, SsgiParams, SsrParams,
};
use concinnity_core::render::atmosphere::AtmosphereParams;
use concinnity_core::render::uniforms::{AutoExposureParams, HizParams, TaaParams};

use crate::shader_layout::mirror::{Case, everywhere, mirror};

pub(in crate::shader_layout) fn atmosphere() -> Vec<Case> {
    vec![everywhere(mirror!(AtmosphereParams => "AtmosphereParams" {
        inv_vp,
        rayleigh,
        mie_scattering,
        mie_extinction,
        ozone,
        ground_albedo,
        radii,
        camera,
        view_ground,
        sun_direction,
        sun_illuminance,
        sun_disc,
        cam_pos,
        viewport,
        sky_rot,
        night,
        galactic_pole,
        galactic_center,
        moon_direction,
        moon_face,
        moon_up,
        galaxies,
    }))]
}

pub(in crate::shader_layout) fn taa() -> Vec<Case> {
    vec![everywhere(mirror!(TaaParams => "TaaParams" {
        history_valid,
        reactive_max,
        lock_decay,
        history_scale,
    }))]
}

pub(in crate::shader_layout) fn bloom() -> Vec<Case> {
    vec![everywhere(
        mirror!(PostProcessParams => "PostProcessParams" {
            bloom_intensity,
            bloom_threshold,
            bloom_knee,
            exposure,
            vignette,
            lut_strength,
            hdr_output,
            pq_output,
            fxaa,
            pre_exposure,
        }),
    )]
}

pub(in crate::shader_layout) fn composite() -> Vec<Case> {
    // The shader flattens the nested post block into its own leading floats, so
    // each member of it is checked through its path.
    vec![everywhere(mirror!(CompositeParams => "CompositeParams" {
        [post.bloom_intensity] => ["bloom_intensity"],
        [post.bloom_threshold] => ["bloom_threshold"],
        [post.bloom_knee] => ["bloom_knee"],
        [post.exposure] => ["exposure"],
        [post.vignette] => ["vignette"],
        [post.lut_strength] => ["lut_strength"],
        [post.hdr_output] => ["hdr_output"],
        [post.pq_output] => ["pq_output"],
        [post.fxaa] => ["fxaa"],
        [post.pre_exposure] => ["pre_exposure"],
        fade,
        view_mode,
        depth_near,
        depth_far,
        reactive_valid,
        night_vision,
    }))]
}

pub(in crate::shader_layout) fn ssao() -> Vec<Case> {
    vec![everywhere(mirror!(SsaoParams => "SsaoParams" {
        radius,
        intensity,
        tan_half_fov_y,
        aspect,
        reach,
        fade_scale,
        fade_bias,
        [_pad] => [],
    }))]
}

pub(in crate::shader_layout) fn ssr() -> Vec<Case> {
    vec![everywhere(mirror!(SsrParams => "SsrParams" {
        intensity,
        max_distance,
        tan_half_fov_y,
        aspect,
        stride,
        thickness,
        prefilter_mip_count,
        pre_exposure,
        inv_view,
        sky_rot,
    }))]
}

pub(in crate::shader_layout) fn ssgi() -> Vec<Case> {
    vec![everywhere(mirror!(SsgiParams => "SsgiParams" {
        intensity,
        max_distance,
        tan_half_fov_y,
        aspect,
        thickness,
        history_valid,
        rays,
        frame,
        levels,
        gi_scale,
        history_scale,
        [_pad] => [],
    }))]
}

pub(in crate::shader_layout) fn auto_exposure() -> Vec<Case> {
    vec![everywhere(
        mirror!(AutoExposureParams => "AutoExposureParams" {
            lum_log2_min,
            lum_log2_range,
            lum_to_bin_scale,
            pre_exposure,
        }),
    )]
}

pub(in crate::shader_layout) fn hiz() -> Vec<Case> {
    vec![everywhere(mirror!(HizParams => "HizParams" {
        dst_width,
        dst_height,
        src_mip,
        sample_count,
    }))]
}
