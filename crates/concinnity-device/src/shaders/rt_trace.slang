// The inline ray-query reflection trace, spliced into a shader at its RT_TRACE
// marker (see slang_source.rs). The second half of the ray-tracing splice:
// RT_TYPES declares the records ahead of a shader's bindings, this traverses
// them. One traversal loop for every target -- `RayQuery` lowers to
// `raytracing::intersection_query` on Metal, `SPV_KHR_ray_query` on Vulkan and
// SM 6.5 DXIL on DirectX. Nothing here may spell either marker.
//
// Hooks the including shader must provide, because each differs per binding
// model or per pass:
//   rt_params, scene_tlas, geom     - the bound params / TLAS / geometry table
//   vert_float / svert_float        - one float of the static / deformed stream
//   index_at / skinned_index_word   - one u32 index of the static / skinned IB
//   prefilter_level(dir, lod)       - the IBL prefilter cube at an explicit mip
//   pool_sample_level0(idx, uv)     - the bindless pool at mip 0 (RT_TEXTURED)

// Dielectric base reflectance for the Fresnel.
static const float RT_F0 = 0.04;
// Floats per `Vertex` (56-byte stride): position, normal, tangent, colour, uv.
static const uint VERTEX_FLOATS = 14u;
// Skinned objects flag bit 31 of `normal_index`; the trace then fetches the hit
// triangle from the deformed-vertex / skinned index buffers, which mirror
// the static layout. The flag is masked off before the pool sample, so skinned
// hits shade textured like static ones. Matches render_types::RT_SKINNED_FLAG.
static const uint RT_SKINNED_FLAG = 0x80000000u;

// Attribute fetchers into the shared 14-float `Vertex` (normal at float 3,
// tangent at 6, uv at 12), addressed by absolute vertex index `vi`. The
// `skinned` set reads the deformed buffer, which carries the identical layout.
float3 rt_vertex_normal(uint vi)
{
    uint b = vi * VERTEX_FLOATS;
    return float3(vert_float(b + 3u), vert_float(b + 4u), vert_float(b + 5u));
}
float3 rt_vertex_tangent(uint vi)
{
    uint b = vi * VERTEX_FLOATS;
    return float3(vert_float(b + 6u), vert_float(b + 7u), vert_float(b + 8u));
}
float2 rt_vertex_uv(uint vi)
{
    uint b = vi * VERTEX_FLOATS;
    return float2(vert_float(b + 12u), vert_float(b + 13u));
}
float3 rt_skinned_normal(uint vi)
{
    uint b = vi * VERTEX_FLOATS;
    return float3(svert_float(b + 3u), svert_float(b + 4u), svert_float(b + 5u));
}
float3 rt_skinned_tangent(uint vi)
{
    uint b = vi * VERTEX_FLOATS;
    return float3(svert_float(b + 6u), svert_float(b + 7u), svert_float(b + 8u));
}
float2 rt_skinned_uv(uint vi)
{
    uint b = vi * VERTEX_FLOATS;
    return float2(svert_float(b + 12u), svert_float(b + 13u));
}

// One index from the skinned index buffer. The skinned BLAS bakes absolute
// indices, so no base_vertex is added.
uint rt_skinned_index(uint o)
{
    return skinned_index_word(o);
}

// Decode a tangent-space normal map texel. Only X and Y are read; Z is
// reconstructed from them, so a two-channel source (BC5) decodes the same as an
// RGBA8 one and normal maps can ship as BC5 blocks.
float3 decode_normal_map(float2 encoded)
{
    float2 nxy = encoded * 2.0 - 1.0;
    return float3(nxy, sqrt(saturate(1.0 - dot(nxy, nxy))));
}

// Sun visibility at a hit: 0 where a second ray toward the sun is occluded (a
// cast shadow inside the reflection), 1 where the sun is visible.
//
// RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH is what an occlusion query wants and
// is what the Vulkan and DirectX legs get; the Metal target drops it (slangc
// does not map the flag to `intersection_params::accept_any_intersection`), so
// there the query commits the closest hit instead of the first. Same answer,
// more traversal.
float rt_shadow(float3 hp, float3 n)
{
    RayDesc sr;
    sr.Origin = hp + n * 0.02;
    sr.Direction = normalize(rt_params.sun_dir.xyz);
    sr.TMin = 0.001;
    sr.TMax = rt_params.max_distance;

    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
    sq.TraceRayInline(scene_tlas, RAY_FLAG_NONE, 0xFFu, sr);
    // Drained rather than stepped once: with the opacity forced there is no
    // non-opaque candidate to resolve, and discarding the step's result trips an
    // unused-value warning out of the Metal compiler on slangc's emitted call.
    while (sq.Proceed()) {}
    return (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0 : 1.0;
}

// Metallic/roughness hit shading: a sun diffuse term (dielectric only, masked
// by the shadow ray) plus split IBL -- diffuse irradiance along N, and a
// specular tap along the onward reflection at a roughness-selected prefilter
// mip, tinted by F0. Metals drop the diffuse term and tint the reflected
// environment by their albedo. The material's self-emission is added on top, so
// glowing surfaces light up in reflections. With no EnvironmentMap bound the
// IBL pair degrades to a small constant ambient.
float3 rt_shade_hit(float3 n, float3 albedo, float hit_rough, float metallic, float3 emissive,
                    float3 dir, bool ibl, float max_mip, float shadow)
{
    float3 f0 = lerp(float3(RT_F0), albedo, metallic);
    float3 diff_a = albedo * (1.0 - metallic);
    float ndl = saturate(dot(n, rt_params.sun_dir.xyz));
    float3 sun = diff_a * rt_params.sun_color.xyz * ndl * shadow;
    if (!ibl)
    {
        return sun + (diff_a + f0) * 0.03 + emissive;
    }
    float3 spec = prefilter_level(reflect(dir, n), hit_rough * max_mip);
    float3 diff = prefilter_level(n, max_mip) * diff_a;
    return sun + diff + spec * f0 + emissive;
}

// Trace one reflection ray from `origin` along `dir` and shade what it hits.
// Returns false when the ray escaped the scene, leaving `shaded` untouched: the
// miss fallback is the caller's, because a screen-space resolve and a glass
// pane want different ones.
bool rt_trace_reflection(float3 origin, float3 dir, bool ibl, float max_mip, out float3 shaded)
{
    shaded = float3(0.0);

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.01;
    ray.TMax = rt_params.max_distance;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
    rq.TraceRayInline(scene_tlas, RAY_FLAG_NONE, 0xFFu, ray);
    while (rq.Proceed()) {}

    if (rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        return false;
    }

    // The instance's position in the TLAS instance array, which every host
    // builds in geometry-table order. NOT CommittedInstanceID: that is the user
    // instance id, which the Metal target reads from a descriptor field only
    // MTLAccelerationStructureUserIDInstanceDescriptor carries, and the engine
    // builds Default descriptors.
    uint inst = rq.CommittedInstanceIndex();
    uint tri = rq.CommittedPrimitiveIndex();
    float2 b = rq.CommittedTriangleBarycentrics();
    float hit_t = rq.CommittedRayT();

    RtGeomEntry e = geom[inst];
    bool skin = (e.normal_index & RT_SKINNED_FLAG) != 0u;
    uint nidx = e.normal_index & ~RT_SKINNED_FLAG;
    uint o = e.index_offset + tri * 3u;
    float w0 = 1.0 - b.x - b.y;

    float3 nl, tl;
    float2 huv;
    if (skin)
    {
        uint i0 = rt_skinned_index(o);
        uint i1 = rt_skinned_index(o + 1u);
        uint i2 = rt_skinned_index(o + 2u);
        nl = rt_skinned_normal(i0) * w0 + rt_skinned_normal(i1) * b.x + rt_skinned_normal(i2) * b.y;
        tl = rt_skinned_tangent(i0) * w0 + rt_skinned_tangent(i1) * b.x + rt_skinned_tangent(i2) * b.y;
        huv = rt_skinned_uv(i0) * w0 + rt_skinned_uv(i1) * b.x + rt_skinned_uv(i2) * b.y;
    }
    else
    {
        uint i0 = index_at(o) + e.base_vertex;
        uint i1 = index_at(o + 1u) + e.base_vertex;
        uint i2 = index_at(o + 2u) + e.base_vertex;
        nl = rt_vertex_normal(i0) * w0 + rt_vertex_normal(i1) * b.x + rt_vertex_normal(i2) * b.y;
        tl = rt_vertex_tangent(i0) * w0 + rt_vertex_tangent(i1) * b.x + rt_vertex_tangent(i2) * b.y;
        huv = rt_vertex_uv(i0) * w0 + rt_vertex_uv(i1) * b.x + rt_vertex_uv(i2) * b.y;
    }

    float3x3 m3 = (float3x3)e.model;
    float3 hit_n = normalize(mul(m3, nl));
    if (dot(hit_n, dir) > 0.0)
    {
        hit_n = -hit_n;
    }
    float3 hit_tan = mul(m3, tl);

    float3 tint = float3(e.tint_r, e.tint_g, e.tint_b);
    float3 emissive = float3(e.emissive_r, e.emissive_g, e.emissive_b);
    float shadow = rt_shadow(origin + dir * hit_t, hit_n);

#ifdef RT_TEXTURED
    // Explicit level 0: a reflected ray's screen-space UV gradients are
    // unstable (neighbouring pixels hit unrelated triangles), so sampling the
    // base mip avoids gradient-driven mip thrash. Skinned hits take the same
    // path -- their textures live in the same pool and the geometry entry
    // carries the real indices.
    float3 albedo = tint * pool_sample_level0(e.albedo_index, huv);
    // Perturb the geometric normal by the tangent-space normal map. The
    // flat-normal fallback decodes to (0, 0, 1) so N is unchanged when an object
    // has no map; a degenerate tangent (procedural meshes may carry none) skips
    // it and keeps the geometric normal.
    float3 n = hit_n;
    float tlen = length(hit_tan);
    if (tlen > 1e-4)
    {
        float3 nm = decode_normal_map(pool_sample_level0(nidx, huv).xy);
        float3 t = hit_tan / tlen;
        t = normalize(t - n * dot(n, t));  // Gram-Schmidt
        float3 bt = cross(n, t);
        n = normalize(t * nm.x + bt * nm.y + n * nm.z);
    }
    // Colour the self-emission by the emissive map (the bistro string lights),
    // mirroring the main bindless pass. Gated on a non-zero pool index; the flat
    // variant has no pool and keeps the scalar emissive.
    if (e.emissive_map_index != 0u)
    {
        emissive *= pool_sample_level0(e.emissive_map_index, huv);
    }
    shaded = rt_shade_hit(n, albedo, e.roughness, e.metallic, emissive, dir, ibl, max_mip, shadow);
#else
    shaded = rt_shade_hit(hit_n, tint, e.roughness, e.metallic, emissive, dir, ibl, max_mip, shadow);
#endif
    return true;
}
