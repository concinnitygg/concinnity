//! What an `SdfVolume`'s distance field names from the engine: the functions
//! the file defines for the raymarch template to call, the helpers the
//! template defines ahead of it, the structs it passes and returns, and the
//! fields of the returned ones.

use super::{Entry, Kind, field, helper};

/// The struct `shade` returns.
pub const SURFACE_STRUCT: &str = "SdfSurface";
/// The struct `sampleVolume` returns.
pub const SAMPLE_STRUCT: &str = "VolumeSample";

const fn hook(name: &'static str, signature: &'static str, summary: &'static str) -> Entry {
    field(Kind::Hook, name, signature, summary)
}

const fn ty(name: &'static str, signature: &'static str, summary: &'static str) -> Entry {
    field(Kind::Type, name, signature, summary)
}

const SURFACE: Kind = Kind::ReturnField(SURFACE_STRUCT);
const SAMPLE: Kind = Kind::ReturnField(SAMPLE_STRUCT);

/// Every name a distance field defines or reads, grouped by kind.
pub const ENTRIES: &[Entry] = &[
    hook(
        "map",
        "float map(float3 p, SdfParams params, float time)",
        "The signed distance from world point p to the surface. A surface field defines it.",
    ),
    hook(
        "shade",
        "SdfSurface shade(float3 p, float3 normal, SdfParams params, float time, float2 frag_uv)",
        "The material at a surface point, which the engine lights. A surface field defines it.",
    ),
    hook(
        "sampleVolume",
        "VolumeSample sampleVolume(float3 p, SdfParams params, float time)",
        "The medium at world point p. A volumetric field defines it.",
    ),
    helper(
        "sdf_param",
        "float sdf_param(SdfParams p, uint i)",
        "Parameter i (0-31) of the volume's params.",
    ),
    helper(
        "volume_center",
        "float3 volume_center()",
        "The world-space center of the volume being drawn.",
    ),
    helper(
        "volume_extent",
        "float3 volume_extent()",
        "The half-widths of the volume being drawn.",
    ),
    helper(
        "sdSphere",
        "float sdSphere(float3 p, float r)",
        "A sphere of radius r around the origin.",
    ),
    helper(
        "sdBox",
        "float sdBox(float3 p, float3 b)",
        "A box of half-widths b around the origin.",
    ),
    helper(
        "sdRoundBox",
        "float sdRoundBox(float3 p, float3 b, float r)",
        "A box of half-widths b with edges rounded by r.",
    ),
    helper(
        "sdTorus",
        "float sdTorus(float3 p, float2 t)",
        "A torus in the XZ plane: t.x the ring radius, t.y the tube radius.",
    ),
    helper(
        "sdCapsule",
        "float sdCapsule(float3 p, float3 a, float3 b, float r)",
        "A capsule of radius r from a to b.",
    ),
    helper(
        "sdPlane",
        "float sdPlane(float3 p, float3 n, float h)",
        "A plane of unit normal n, offset h along it.",
    ),
    helper(
        "opSmoothUnion",
        "float opSmoothUnion(float a, float b, float k)",
        "Two distances joined, blended over k.",
    ),
    helper(
        "opSmoothSubtraction",
        "float opSmoothSubtraction(float d1, float d2, float k)",
        "d1 carved out of d2, blended over k.",
    ),
    helper(
        "opSmoothIntersection",
        "float opSmoothIntersection(float a, float b, float k)",
        "The overlap of two distances, blended over k.",
    ),
    helper(
        "sampleSceneRefracted",
        "float3 sampleSceneRefracted(float2 frag_uv, float3 normal, float strength)",
        "The scene behind the surface, bent by the normal; for SdfSurface.transmitted.",
    ),
    ty(
        "SdfParams",
        "struct SdfParams",
        "The volume's 32 params, read with sdf_param.",
    ),
    ty(
        "SdfSurface",
        "struct SdfSurface",
        "What shade returns: the material at the point.",
    ),
    ty(
        "VolumeSample",
        "struct VolumeSample",
        "What sampleVolume returns: the medium at the point.",
    ),
    field(SURFACE, "albedo", "float3 albedo", "Base color."),
    field(
        SURFACE,
        "roughness",
        "float roughness",
        "0 mirror to 1 matte.",
    ),
    field(
        SURFACE,
        "metallic",
        "float metallic",
        "0 dielectric to 1 metal.",
    ),
    field(
        SURFACE,
        "emissive",
        "float3 emissive",
        "Light the surface gives off.",
    ),
    field(
        SURFACE,
        "transmitted",
        "float3 transmitted",
        "Color shown through the surface, added after lighting; 0 when opaque.",
    ),
    field(
        SAMPLE,
        "density",
        "float density",
        "Extinction at the point; 0 is empty.",
    ),
    field(
        SAMPLE,
        "scattering",
        "float3 scattering",
        "Single-scatter albedo, times the sun's light.",
    ),
    field(
        SAMPLE,
        "emission",
        "float3 emission",
        "Light the medium gives off.",
    ),
];

#[cfg(test)]
mod tests;
