// REFLECTION_CUT marker: the roughness past which a surface takes no SSR or RT
// reflection, shared by the forward fade, the SSR and RT resolves, and the
// reflection blur so they never disagree. Locked to REFLECTION_ROUGHNESS_CUT
// in concinnity_engine::render::post::ssr::settings by unit test.
static const float REFLECTION_ROUGHNESS_CUT = 0.6;

// A resolve's output for `reflected` radiance off a surface of head-on
// reflectance `f0` seen at `ndv`: the radiance tinted by the Schlick Fresnel
// (.rgb) over the weight the composite blends it in by (.a). A metal's tinted
// F0 makes the traced reflection its whole specular; a dielectric's 0.04 keeps
// it a rim.
float4 reflection_resolve_out(float3 reflected, float3 f0, float ndv, float gloss, float intensity)
{
    float3 fresnel = f0 + (1.0 - f0) * pow(1.0 - ndv, 5.0);
    float reflectance = max(fresnel.r, max(fresnel.g, fresnel.b));
    float weight = saturate(reflectance * gloss * intensity);
    return float4(reflected * (fresnel / max(reflectance, 1e-4)), weight);
}
