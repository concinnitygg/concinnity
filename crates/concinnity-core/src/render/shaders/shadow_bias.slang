// SHADOW_BIAS marker: the depth-compare offset a cascade sample is taken with,
// shared by the rasterised main pass and the raymarched surfaces so both take
// the shadow the other would at the same point.
//
// Needs `SHADOW_UNI` and `shadow_map_size()` from the including shader.

// How many of the cascade's own shadow texels the compare offset spans. Sized
// to clear the depth the 5x5 PCF kernel walks across without detaching contact
// shadows.
static const float SHADOW_BIAS_TEXELS = 2.0;

// Compare offset for one cascade, in that cascade's NDC depth units.
//
// Row 0 of the cascade's VP is the ortho x scale (1 / radius), so the shadow
// texel measures `2 / (|row0| * map size)` metres; row 2 is the ortho z scale,
// which converts a metre along the light back into NDC depth. A distant
// cascade covers more world per texel, so its offset grows with it.
//
// Slang indexes HLSL-style (`m[i][j]` is row i, column j), unlike the GLSL and
// MSL counterparts where `m[i]` is column i.
float cascade_depth_bias(int cascade)
{
    float3 row0 = float3(SHADOW_UNI.light_vps[cascade][0][0],
                         SHADOW_UNI.light_vps[cascade][0][1],
                         SHADOW_UNI.light_vps[cascade][0][2]);
    float3 row2 = float3(SHADOW_UNI.light_vps[cascade][2][0],
                         SHADOW_UNI.light_vps[cascade][2][1],
                         SHADOW_UNI.light_vps[cascade][2][2]);
    float texel_world = 2.0 / max(length(row0) * shadow_map_size().x, 1e-6);
    return SHADOW_BIAS_TEXELS * texel_world * length(row2);
}
