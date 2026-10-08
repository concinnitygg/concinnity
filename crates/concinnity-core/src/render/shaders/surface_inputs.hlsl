// The material inputs read off a surface before any lighting: alpha cutout,
// the normal-mapped shading normal, and roughness / metallic with the ORM map
// applied. Spliced into `main_shading.hlsl`, so the forward shading and the
// G-buffer pre-pass entries in `main_bindless.hlsl` describe one surface.
// Reads the bindless pool through `pool_sample`.

// Decode a tangent-space normal map texel. Only X and Y are read; Z is
// reconstructed from them, so a two-channel source (BC5) decodes the same as
// an RGBA8 one and normal maps can ship as BC5 blocks.
float3 decode_normal_map(float2 encoded)
{
    float2 nxy = encoded * 2.0 - 1.0;
    return float3(nxy, sqrt(clamp(1.0 - dot(nxy, nxy), 0.0, 1.0)));
}

{SPECULAR_AA}

// Alpha cutout: punch the texel out entirely so foliage and decal cards stay
// in the opaque pass. Disabled at cutoff 0.
void surface_cutout(GpuObjectData od, float albedo_alpha)
{
    float alpha_cutoff = od.bb_max_alpha_cutoff.w;
    if (alpha_cutoff > 0.0 && albedo_alpha < alpha_cutoff)
    {
        discard;
    }
}

// The shading normal: the normal map's texel carried out of the interpolated
// tangent frame.
float3 surface_normal(GpuObjectData od, float2 uv, float3 normal, float3 tangent, float3 bitangent)
{
    float3 norm_samp = decode_normal_map(pool_sample(od.normal_index, uv).rg);
    // Tangent frame as rows so mul(v, M) applies the column-basis transform.
    float3x3 TBN = float3x3(
        normalize(tangent),
        normalize(bitangent),
        normalize(normal));
    return normalize(mul(norm_samp, TBN));
}

// Perceptual roughness (x) and metallic (y): the record's scalars, or the
// occlusion-roughness-metallic map's green and blue (glTF convention) where
// one is bound. Slot 0 is the "no map" sentinel.
float2 surface_roughness_metallic(GpuObjectData od, float2 uv)
{
    if (od.orm_map_index != 0u)
    {
        return pool_sample(od.orm_map_index, uv).gb;
    }
    return float2(od.tint_roughness.w, od.emissive_metallic.w);
}
