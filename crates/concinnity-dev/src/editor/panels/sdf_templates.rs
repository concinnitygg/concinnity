//! What a new distance field starts as: a few starters per kind, each placing
//! its shape by `volume_center()` and sizing it by `volume_extent()`, so it
//! shows in whichever volume reads it. Every starter is pinned compilable by
//! test.

use super::shader_templates::Template;

pub(crate) const SURFACE: [Template; 3] = [
    Template {
        name: "Lit sphere",
        text: "\
// A sphere filling the volume's box, lit by the engine.
float map(float3 p, SdfParams params, float time)
{
    float3 e = volume_extent();
    return sdSphere(p - volume_center(), 0.8 * min(e.x, min(e.y, e.z)));
}

SdfSurface shade(float3 p, float3 normal, SdfParams params, float time, float2 frag_uv)
{
    SdfSurface s;
    s.albedo = float3(0.8, 0.8, 0.8);
    s.roughness = 0.5;
    s.metallic = 0.0;
    s.emissive = float3(0.0, 0.0, 0.0);
    s.transmitted = float3(0.0, 0.0, 0.0);
    return s;
}
",
    },
    Template {
        name: "Rounded box",
        text: "\
// A box with rounded edges inside the volume's box, colored by its params:
// sdf_param(params, 0..2) is the albedo, 3 the roughness.
float map(float3 p, SdfParams params, float time)
{
    float3 e = volume_extent();
    float r = 0.15 * min(e.x, min(e.y, e.z));
    return sdRoundBox(p - volume_center(), 0.7 * e, r);
}

SdfSurface shade(float3 p, float3 normal, SdfParams params, float time, float2 frag_uv)
{
    SdfSurface s;
    s.albedo = float3(sdf_param(params, 0u), sdf_param(params, 1u), sdf_param(params, 2u));
    s.roughness = clamp(sdf_param(params, 3u), 0.05, 1.0);
    s.metallic = 0.0;
    s.emissive = float3(0.0, 0.0, 0.0);
    s.transmitted = float3(0.0, 0.0, 0.0);
    return s;
}
",
    },
    Template {
        name: "Spinning blob",
        text: "\
// A sphere melted into a torus that spins about the volume's vertical axis,
// shaded as polished metal.
float map(float3 p, SdfParams params, float time)
{
    float3 e = volume_extent();
    float r = min(e.x, min(e.y, e.z));
    float3 q = p - volume_center();
    float c = cos(time);
    float s = sin(time);
    q = float3(c * q.x + s * q.z, q.y, c * q.z - s * q.x);
    float ball = sdSphere(q, 0.45 * r);
    float ring = sdTorus(q, float2(0.6 * r, 0.12 * r));
    return opSmoothUnion(ball, ring, 0.2 * r);
}

SdfSurface shade(float3 p, float3 normal, SdfParams params, float time, float2 frag_uv)
{
    SdfSurface s;
    s.albedo = float3(0.95, 0.93, 0.88);
    s.roughness = 0.15;
    s.metallic = 1.0;
    s.emissive = float3(0.0, 0.0, 0.0);
    s.transmitted = float3(0.0, 0.0, 0.0);
    return s;
}
",
    },
];

pub(crate) const VOLUMETRIC: [Template; 2] = [
    Template {
        name: "Fog ball",
        text: "\
// A ball of fog, densest at the volume's center and clear at its edge.
VolumeSample sampleVolume(float3 p, SdfParams params, float time)
{
    float3 e = volume_extent();
    float d = length(p - volume_center()) / min(e.x, min(e.y, e.z));
    VolumeSample v;
    v.density = 2.0 * saturate(1.0 - d);
    v.scattering = float3(0.9, 0.9, 0.9);
    v.emission = float3(0.0, 0.0, 0.0);
    return v;
}
",
    },
    Template {
        name: "Glowing core",
        text: "\
// A thin medium that glows from within, brightest at the center and pulsing
// over time.
VolumeSample sampleVolume(float3 p, SdfParams params, float time)
{
    float3 e = volume_extent();
    float d = length(p - volume_center()) / min(e.x, min(e.y, e.z));
    float core = saturate(1.0 - d);
    float pulse = 0.75 + 0.25 * sin(time * 2.0);
    VolumeSample v;
    v.density = 0.6 * core;
    v.scattering = float3(0.2, 0.2, 0.2);
    v.emission = float3(1.0, 0.45, 0.15) * core * core * pulse * 4.0;
    return v;
}
",
    },
];

// The starters for a surface or a volumetric field.
pub(crate) fn of(volumetric: bool) -> &'static [Template] {
    match volumetric {
        true => &VOLUMETRIC,
        false => &SURFACE,
    }
}

// Starter `i` of its kind, or the first past the end.
pub(crate) fn text(volumetric: bool, i: usize) -> &'static str {
    let set = of(volumetric);
    set.get(i).unwrap_or(&set[0]).text
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::render::shader_source::SourceFile;

    // Compile `text` as a volume's field would, cleanly. A starter that fails
    // here would fail its world's build instead.
    fn compiles_clean(text: &str, volumetric: bool, cast_shadows: bool) {
        let compiled = concinnity_cook::compile::sdf_field::compile_sdf_field(
            "starter",
            SourceFile {
                path: "shaders/starter.hlsl",
                text,
            },
            crate::cook_platform(),
            volumetric,
            cast_shadows,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert!(compiled.warnings.is_empty(), "{:?}", compiled.warnings);
    }

    // A surface starter compiles for the shadow caster too, since the volume's
    // own form can turn shadows on.
    #[test]
    fn every_surface_starter_compiles_with_its_shadow() {
        concinnity_shader::require_dxc!();
        for t in SURFACE {
            compiles_clean(t.text, false, true);
        }
    }

    #[test]
    fn every_volumetric_starter_compiles() {
        concinnity_shader::require_dxc!();
        for t in VOLUMETRIC {
            compiles_clean(t.text, true, false);
        }
    }

    #[test]
    fn the_first_surface_starter_is_a_plain_lit_sphere() {
        assert_eq!(SURFACE[0].name, "Lit sphere");
        assert!(SURFACE[0].text.contains("sdSphere(p - volume_center()"));
        assert_eq!(text(true, 9), VOLUMETRIC[0].text);
        assert_eq!(text(false, 2), SURFACE[2].text);
    }
}
