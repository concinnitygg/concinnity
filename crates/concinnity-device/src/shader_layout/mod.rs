// src/shader_layout/mod.rs
//
// Layout drift guard for the `#[repr(C)]` structs the CPU uploads into the
// single-source `.slang` shaders. The expected offsets and sizes are not
// written down here: they come from `slangc -reflection-json` over the same
// source the renderer compiles, per target, so an edit on either side of the
// boundary fails the check.
//
// That is the difference from a hand-written assert. A hand assert pins the
// Rust struct against a number and a comment describing what the shader is
// believed to do, and nothing checks the comment; a shader-side edit -- the
// direction that actually broke things during the single-source migration --
// slides straight past it.
//
// The reflection is taken per target because the three do not agree. MSL sizes
// a `float3` at 16 bytes -- in a structured buffer as much as in a constant
// buffer -- where SPIR-V and DXIL pack a scalar after it at 12, and SPIR-V
// aligns a following `float2` to 16 where neither of the others does. The
// engine's `.slang` sources avoid both shapes on purpose (`float4` lanes
// instead of `float3` + scalar), so today the three agree on every mirrored
// struct -- which is itself worth asserting rather than assuming. The block
// sizes already differ: DirectX reports a 276-byte `ShadowUniforms` block where
// Metal and SPIR-V round it to 288.
//
// Not every layout assert can move here. `cull`, `rt_skin`, the legacy per-draw
// main pass and the raymarch templates are still hand-written per backend, and
// so a struct used only by those has no `.slang` to reflect and its hand assert
// is the only check it has. The reverse gap exists too: slangc rejects `TraceRayInline` on the Metal
// target, so `rt_reflections.slang` reflects on Vulkan and DirectX only and
// `RtParams` / `RtGeomEntry` have no check against the Metal RT sources. Vertex payloads are the other exclusion: slangc binds a vertex input by
// attribute index, not byte offset, so `Vertex` / `SkinnedVertex` /
// `MorphEntry` / `TextVertex` / `LineVertex` reflect no layout at all. The
// surviving asserts are listed in private/docs/shader-single-source.md.
//
// The runtime Metal reflection validator (concinnity-shader) stays as well: it
// checks world-authored shaders, which reflection over engine `.slang` files
// cannot see.

mod mirror;
mod mirrors;
mod programs;
mod reflect;

use concinnity_slang as slang;

use mirror::Case;
use programs::{Program, Target};

// The probe-array length the programs bake in, pinned to the constant the
// mirrored `ProbeSet` array uses.
const _: () = assert!(concinnity_render::uniforms::MAX_PROBES == 8);

// Reflect `program` on every target its mirrors name and compare each against
// what that target's layout rules produced. Skipped when slangc is absent, the
// way concinnity-slang's own round-trip tests are.
//
// A target no mirror names is not compiled at all. That is how a program opts
// out of a target it cannot build on, which is otherwise indistinguishable from
// a layout failure.
fn check(program: &Program, cases: &[Case]) {
    if slang::slangc_path().is_none() {
        return;
    }
    let mut drift = Vec::new();
    for target in Target::ALL {
        if !cases.iter().any(|case| case.targets.contains(&target)) {
            continue;
        }
        let layouts = match programs::layouts(program, target) {
            Ok(layouts) => layouts,
            Err(e) => {
                drift.push(e);
                continue;
            }
        };
        for case in cases.iter().filter(|c| c.targets.contains(&target)) {
            let name = case.mirror.shader_name;
            let Some(shader) = layouts.get(name) else {
                drift.push(format!(
                    "{} ({}): the reflection of {} declares no `{name}`; the mirror names a \
                     struct this variant does not compile",
                    case.mirror.rust_name,
                    target.label(),
                    program.entry,
                ));
                continue;
            };
            drift.extend(
                mirror::drift(&case.mirror, shader)
                    .into_iter()
                    .map(|line| format!("[{}] {line}", target.label())),
            );
        }
    }
    assert!(
        drift.is_empty(),
        "{} layouts drifted from the shader:\n  {}",
        program.entry,
        drift.join("\n  "),
    );
}

#[test]
fn main_bindless_layouts_match_the_shader() {
    check(
        &programs::MAIN_BINDLESS_VERT,
        &mirrors::forward::main_bindless(),
    );
}

#[test]
fn light_cull_layouts_match_the_shader() {
    check(
        &programs::LIGHT_CULL_KERNEL,
        &mirrors::forward::light_cull(),
    );
}

#[test]
fn gbuffer_prepass_vertex_layouts_match_the_shader() {
    check(
        &programs::GBUFFER_PREPASS_VERT,
        &mirrors::geometry::gbuffer_vertex(),
    );
}

#[test]
fn gbuffer_prepass_fragment_layouts_match_the_shader() {
    check(
        &programs::GBUFFER_PREPASS_FRAG,
        &mirrors::geometry::gbuffer_fragment(),
    );
}

#[test]
fn shadow_layouts_match_the_shader() {
    check(&programs::SHADOW_VERT, &mirrors::geometry::shadow());
}

#[test]
fn decal_layouts_match_the_shader() {
    check(&programs::DECAL_VERT, &mirrors::geometry::decal());
}

#[test]
fn line_layouts_match_the_shader() {
    check(&programs::LINE_VERT, &mirrors::geometry::line());
}

#[test]
fn particle_layouts_match_the_shader() {
    check(&programs::PARTICLE_VERT, &mirrors::geometry::particle());
}

#[test]
fn text_layouts_match_the_shader() {
    check(&programs::TEXT_VERT, &mirrors::geometry::text());
}

#[test]
fn glass_layouts_match_the_shader() {
    check(&programs::GLASS_VERT, &mirrors::transparent::glass());
}

#[test]
fn glass_mesh_layouts_match_the_shader() {
    check(
        &programs::GLASS_MESH_VERT,
        &mirrors::transparent::glass_mesh(),
    );
}

#[test]
fn water_layouts_match_the_shader() {
    check(&programs::WATER_VERT, &mirrors::transparent::water());
}

#[test]
fn rt_reflections_layouts_match_the_shader() {
    check(
        &programs::RT_REFLECTIONS_FRAG,
        &mirrors::transparent::rt_reflections(),
    );
}

#[test]
fn fog_layouts_match_the_shader() {
    check(&programs::FOG_FROXEL, &mirrors::transparent::fog());
}

#[test]
fn taa_layouts_match_the_shader() {
    check(&programs::TAA_FRAG, &mirrors::post::taa());
}

#[test]
fn bloom_layouts_match_the_shader() {
    check(&programs::BLOOM_PREFILTER, &mirrors::post::bloom());
}

#[test]
fn composite_layouts_match_the_shader() {
    check(&programs::COMPOSITE_FRAG, &mirrors::post::composite());
}

#[test]
fn ssao_layouts_match_the_shader() {
    check(&programs::SSAO_KERNEL, &mirrors::post::ssao());
}

#[test]
fn ssr_layouts_match_the_shader() {
    check(&programs::SSR_RESOLVE, &mirrors::post::ssr());
}

#[test]
fn ssgi_layouts_match_the_shader() {
    check(&programs::SSGI_GATHER, &mirrors::post::ssgi());
}

#[test]
fn auto_exposure_layouts_match_the_shader() {
    check(
        &programs::AUTO_EXPOSURE_BUILD,
        &mirrors::post::auto_exposure(),
    );
}

#[test]
fn hiz_layouts_match_the_shader() {
    check(&programs::HIZ_INIT_SINGLE, &mirrors::post::hiz());
}
