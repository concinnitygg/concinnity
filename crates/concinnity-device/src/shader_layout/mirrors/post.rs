// src/shader_layout/mirrors/post.rs
//
// The fullscreen post stack and the two compute helpers that feed it. These
// params blocks carry no host-shape gate, so each is checked against all three
// targets.

use concinnity_core::gfx::render_types::{
    CompositeParams, PostProcessParams, SsaoParams, SsgiParams, SsrParams,
};
use concinnity_render::uniforms::{AutoExposureParams, HizParams, TaaParams};

use crate::shader_layout::mirror::{Case, everywhere, mirror};

pub(in crate::shader_layout) fn taa() -> Vec<Case> {
    vec![everywhere(
        mirror!(TaaParams => "TaaParams" { history_valid, }),
    )]
}

pub(in crate::shader_layout) fn bloom() -> Vec<Case> {
    // The prefilter declares only the six floats it reads; the tone-map fields
    // the composite adds are uploaded but never touched here.
    vec![everywhere(
        mirror!(PostProcessParams => "PostProcessParams" {
            bloom_intensity,
            bloom_threshold,
            bloom_knee,
            exposure,
            vignette,
            lut_strength,
            [hdr_output, pq_output, fxaa] => [],
        }),
    )]
}

pub(in crate::shader_layout) fn composite() -> Vec<Case> {
    // The shader flattens the nested post block into its own leading floats, so
    // each member of it is checked through its path. This is the only
    // declaration that spells out all nine: the bloom prefilter declares six.
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
        fade,
        view_mode,
        [far] => ["far_plane"],
    }))]
}

pub(in crate::shader_layout) fn ssao() -> Vec<Case> {
    vec![everywhere(mirror!(SsaoParams => "SsaoParams" {
        radius,
        intensity,
        tan_half_fov_y,
        aspect,
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
        _pad,
        inv_view,
    }))]
}

pub(in crate::shader_layout) fn ssgi() -> Vec<Case> {
    vec![everywhere(mirror!(SsgiParams => "SsgiParams" {
        intensity,
        max_distance,
        tan_half_fov_y,
        aspect,
        stride,
        thickness,
        rays,
        steps,
    }))]
}

pub(in crate::shader_layout) fn auto_exposure() -> Vec<Case> {
    vec![everywhere(
        mirror!(AutoExposureParams => "AutoExposureParams" {
            lum_log2_min,
            lum_log2_range,
            lum_to_bin_scale,
            _pad,
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
