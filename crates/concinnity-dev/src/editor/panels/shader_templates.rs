//! What a new Shader file starts as: a small set of starters per stage, the
//! first of each the engine's own behavior, so a Shader made from the
//! defaults renders exactly as the engine would without one. Every starter is
//! pinned compilable by test.

use concinnity_core::components::ShaderStage;

// One starter file: what the picker calls it and the text it writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Template {
    pub(crate) name: &'static str,
    pub(crate) text: &'static str,
}

pub(crate) const FRAGMENT: [Template; 4] = [
    Template {
        name: "Lit",
        text: "\
// The surface's color. shade_surface is the engine's own lighting.
float4 shade(VertexOut v, GpuObjectData od)
{
    return shade_surface(v, od);
}
",
    },
    Template {
        name: "Unlit color",
        text: "\
// The material's tint and the vertex color, with no lighting, shadow or
// environment.
float4 shade(VertexOut v, GpuObjectData od)
{
    return float4(od.tint_roughness.rgb * v.color, 1.0);
}
",
    },
    Template {
        name: "Tinted lit",
        text: "\
// The engine's own lighting, recolored by TINT.
static const float3 TINT = float3(1.0, 0.8, 0.6);

float4 shade(VertexOut v, GpuObjectData od)
{
    float4 lit = shade_surface(v, od);
    return float4(lit.rgb * TINT, lit.a);
}
",
    },
    Template {
        name: "Textured",
        text: "\
// The material's albedo texture times its tint, unlit. pool_sample reads a
// texture from the world's pool by the index the material record carries.
float4 shade(VertexOut v, GpuObjectData od)
{
    float4 albedo = pool_sample(od.albedo_index, v.uv);
    return float4(albedo.rgb * od.tint_roughness.rgb, albedo.a);
}
",
    },
];

pub(crate) const VERTEX: [Template; 2] = [
    Template {
        name: "Engine projection",
        text: "\
// Where the vertex lands and what it hands shade. project_vertex is the
// engine's own projection.
VertexOut transform(float4x4 model, float3 pos, float3 normal, float3 tangent,
                    float3 color, float2 uv)
{
    return project_vertex(model, pos, normal, tangent, color, uv);
}
",
    },
    Template {
        name: "Sway",
        text: "\
// Sways the vertex sideways over time, further the higher it sits above the
// model's origin, then projects it as the engine does.
VertexOut transform(float4x4 model, float3 pos, float3 normal, float3 tangent,
                    float3 color, float2 uv)
{
    float sway = sin(VIEW.elapsed * 1.5 + pos.x + pos.z) * 0.1 * max(pos.y, 0.0);
    return project_vertex(model, pos + float3(sway, 0.0, 0.0), normal, tangent, color, uv);
}
",
    },
];

// The starters for `stage`.
pub(crate) fn of(stage: ShaderStage) -> &'static [Template] {
    match stage {
        ShaderStage::Fragment => &FRAGMENT,
        ShaderStage::Vertex => &VERTEX,
    }
}

// The starter `i` of `stage`, or its first past the end.
pub(crate) fn text(stage: ShaderStage, i: usize) -> &'static str {
    let set = of(stage);
    set.get(i).unwrap_or(&set[0]).text
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::render::shader_programs::surface::{SourceFile, Sources};

    // Compile `fragment` with `vertex` as a real Shader, cleanly. A starter
    // that fails here would fail its first save instead.
    fn compiles_clean(fragment: &str, vertex: Option<&str>) {
        let sources = Sources {
            vertex: vertex.map(|text| SourceFile {
                path: "shaders/starter_vertex.hlsl",
                text,
            }),
            fragment: SourceFile {
                path: "shaders/starter.hlsl",
                text: fragment,
            },
        };
        let compiled = concinnity_cook::compile::shader::compile_world_shader(
            "starter",
            &sources,
            crate::cook_platform(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert!(compiled.warnings.is_empty(), "{:?}", compiled.warnings);
    }

    #[test]
    fn every_fragment_starter_compiles() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        for t in FRAGMENT {
            compiles_clean(t.text, None);
        }
    }

    #[test]
    fn every_vertex_starter_compiles_beside_the_first_fragment() {
        if !concinnity_shader::dxc_available() {
            return;
        }
        for t in VERTEX {
            compiles_clean(FRAGMENT[0].text, Some(t.text));
        }
    }

    #[test]
    fn a_starter_past_the_end_is_the_first() {
        assert_eq!(text(ShaderStage::Vertex, 9), VERTEX[0].text);
        assert_eq!(text(ShaderStage::Fragment, 3), FRAGMENT[3].text);
    }
}
