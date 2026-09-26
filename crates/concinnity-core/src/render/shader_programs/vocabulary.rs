//! What a world Shader's files can read from the engine: the helpers the
//! main-pass template defines ahead of the hooks, the fields of the uniform
//! blocks it binds, and the fields of the two structs the hooks receive. One
//! list, which authoring tools show and highlight from and which the `Shader`
//! rustdoc and the template are both tested against.

use alloc::string::String;

/// One of the engine's uniform blocks a Shader reads by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Block {
    /// `VIEW`: the camera and clock.
    View,
    /// `LIGHTS`: the scene's lights.
    Lights,
}

impl Block {
    /// Every block.
    pub const ALL: [Block; 2] = [Block::View, Block::Lights];

    /// The name a Shader reads the block by.
    pub const fn name(self) -> &'static str {
        match self {
            Block::View => "VIEW",
            Block::Lights => "LIGHTS",
        }
    }

    /// The HLSL struct the block is declared as.
    pub const fn declared_as(self) -> &'static str {
        match self {
            Block::View => "ViewUniforms",
            Block::Lights => "LightUniforms",
        }
    }
}

/// What an [`Entry`] is, which decides how a Shader reaches it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A function (or function-like macro) the file calls.
    Helper,
    /// A field of a uniform block, read as `BLOCK.field`.
    BlockField(Block),
    /// A field of the surface's material record, read as `od.field`.
    RecordField,
    /// A field of the varying block, read as `v.field`.
    Varying,
}

/// The struct `shade` receives its material record as.
pub const RECORD_STRUCT: &str = "GpuObjectData";
/// The struct `shade` receives its varyings as, and `transform` returns.
pub const VARYING_STRUCT: &str = "VertexOut";

impl Kind {
    /// The HLSL struct declaring a field kind; `None` for a helper.
    pub const fn declared_in(self) -> Option<&'static str> {
        match self {
            Kind::Helper => None,
            Kind::BlockField(b) => Some(b.declared_as()),
            Kind::RecordField => Some(RECORD_STRUCT),
            Kind::Varying => Some(VARYING_STRUCT),
        }
    }

    /// What a file writes ahead of a field's name to reach it: the block's
    /// name, or the parameter the `shade` hook names the struct by.
    pub const fn owner(self) -> Option<&'static str> {
        match self {
            Kind::Helper => None,
            Kind::BlockField(b) => Some(b.name()),
            Kind::RecordField => Some("od"),
            Kind::Varying => Some("v"),
        }
    }
}

/// One name a Shader can read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    /// The name as the source spells it.
    pub name: &'static str,
    /// What it is.
    pub kind: Kind,
    /// Its HLSL declaration: a helper's prototype, or a field's type and name.
    pub signature: &'static str,
    /// One line on what it gives.
    pub summary: &'static str,
}

impl Entry {
    /// The text a file writes to use the entry: `VIEW.elapsed`, or a
    /// helper's call with its parameters named as the signature names them.
    pub fn usage(&self) -> String {
        match self.kind.owner() {
            Some(owner) => alloc::format!("{owner}.{}", self.name),
            None => call_skeleton(self.name, self.signature),
        }
    }
}

// `name(a, b)` from a prototype `T name(T a, out T b)`: each parameter's last
// word.
fn call_skeleton(name: &str, signature: &str) -> String {
    let params = signature
        .split_once('(')
        .and_then(|(_, rest)| rest.rsplit_once(')'))
        .map_or("", |(params, _)| params);
    let mut out = String::from(name);
    out.push('(');
    let names = params
        .split(',')
        .filter_map(|p| p.split_whitespace().last())
        .map(|p| p.trim_end_matches("[]"));
    for (i, p) in names.enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(p);
    }
    out.push(')');
    out
}

const fn helper(name: &'static str, signature: &'static str, summary: &'static str) -> Entry {
    Entry {
        name,
        kind: Kind::Helper,
        signature,
        summary,
    }
}

const fn field(
    kind: Kind,
    name: &'static str,
    signature: &'static str,
    summary: &'static str,
) -> Entry {
    Entry {
        name,
        kind,
        signature,
        summary,
    }
}

const VIEW: Kind = Kind::BlockField(Block::View);
const LIGHTS: Kind = Kind::BlockField(Block::Lights);
const OD: Kind = Kind::RecordField;
const V: Kind = Kind::Varying;

/// Every name a world Shader's files can read, grouped by kind.
pub const ENTRIES: &[Entry] = &[
    helper(
        "shade_surface",
        "float4 shade_surface(VertexOut v, GpuObjectData od)",
        "The engine's own PBR lighting of the surface.",
    ),
    helper(
        "project_vertex",
        "VertexOut project_vertex(float4x4 model, float3 pos, float3 normal, float3 tangent, float3 color, float2 uv)",
        "The engine's own projection of a model-space vertex.",
    ),
    helper(
        "pool_sample",
        "float4 pool_sample(uint index, float2 uv)",
        "A texel of the world texture a record index names.",
    ),
    helper(
        "decode_normal_map",
        "float3 decode_normal_map(float2 rg)",
        "A tangent-space normal from a normal-map texel.",
    ),
    helper(
        "shadow_factor_cascaded",
        "float shadow_factor_cascaded(float3 world_pos, float view_depth, float2 screen_xy)",
        "The sun's cascaded shadow term: 0 in shadow, 1 lit.",
    ),
    helper(
        "environment_specular",
        "bool environment_specular(ProbeMask probes, float3 world_pos, float3 reflected, float roughness, out float3 radiance)",
        "The reflection environment into radiance; false with no probe or environment map.",
    ),
    helper(
        "probe_mask_all",
        "ProbeMask probe_mask_all()",
        "Every reflection probe, for environment_specular.",
    ),
    helper(
        "irradiance_sample",
        "float3 irradiance_sample(float3 normal)",
        "The diffuse environment light arriving along a normal.",
    ),
    helper(
        "SKY_DIR",
        "float3 SKY_DIR(float3 d)",
        "A world direction in the environment map's frame.",
    ),
    field(VIEW, "vp", "float4x4 vp", "World to clip space."),
    field(
        VIEW,
        "view_mat",
        "float4x4 view_mat",
        "World to view space.",
    ),
    field(
        VIEW,
        "elapsed",
        "float elapsed",
        "Seconds elapsed, for animation.",
    ),
    field(VIEW, "cam_x", "float cam_x", "Camera position, x."),
    field(VIEW, "cam_y", "float cam_y", "Camera position, y."),
    field(VIEW, "cam_z", "float cam_z", "Camera position, z."),
    field(
        VIEW,
        "sky_rot",
        "float4 sky_rot[3]",
        "Rows of the world-to-environment rotation.",
    ),
    field(
        LIGHTS,
        "dir",
        "DirLight dir[4]",
        "Directional lights: dir_i (direction, intensity) and col.",
    ),
    field(
        LIGHTS,
        "pt",
        "PointLight pt[8]",
        "Point lights: pos_r (position, range) and col_i (color, intensity).",
    ),
    field(LIGHTS, "num_dir", "int num_dir", "How many of dir are lit."),
    field(LIGHTS, "num_pt", "int num_pt", "How many of pt are lit."),
    field(
        LIGHTS,
        "ambient_intensity",
        "float ambient_intensity",
        "Multiplier on the indirect ambient light.",
    ),
    field(
        OD,
        "tint_roughness",
        "float4 tint_roughness",
        "xyz: the material's tint; w: its roughness.",
    ),
    field(
        OD,
        "emissive_metallic",
        "float4 emissive_metallic",
        "xyz: emissive color; w: metallic.",
    ),
    field(
        OD,
        "albedo_index",
        "uint albedo_index",
        "The albedo texture, for pool_sample.",
    ),
    field(
        OD,
        "normal_index",
        "uint normal_index",
        "The normal map, for pool_sample.",
    ),
    field(
        OD,
        "emissive_map_index",
        "uint emissive_map_index",
        "The emissive map, for pool_sample; 0 when there is none.",
    ),
    field(
        OD,
        "orm_map_index",
        "uint orm_map_index",
        "The occlusion-roughness-metallic map; 0 when there is none.",
    ),
    field(
        OD,
        "bb_max_alpha_cutoff",
        "float4 bb_max_alpha_cutoff",
        "w: the alpha cutoff, 0 when the material cuts nothing out.",
    ),
    field(V, "position", "float4 position", "Clip-space position."),
    field(V, "world_pos", "float3 world_pos", "World-space position."),
    field(V, "normal", "float3 normal", "World-space normal."),
    field(V, "tangent", "float3 tangent", "World-space tangent."),
    field(V, "bitangent", "float3 bitangent", "World-space bitangent."),
    field(V, "uv", "float2 uv", "Texture coordinates."),
    field(
        V,
        "view_depth",
        "float view_depth",
        "Depth along the camera's view direction.",
    ),
    field(V, "color", "float3 color", "Vertex color."),
];

#[cfg(test)]
mod tests;
