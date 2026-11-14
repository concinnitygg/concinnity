// The inline ray-query reflection trace, spliced into a shader at its RT_TRACE
// marker (see shader_source.rs). The second half of the ray-tracing splice:
// RT_TYPES declares the records ahead of a shader's bindings, this traverses
// them. One traversal loop for every target -- `RayQuery` lowers to
// `raytracing::intersection_query` on Metal, `SPV_KHR_ray_query` on Vulkan and
// SM 6.5 DXIL on DirectX. Nothing here may spell either marker.
//
// Hooks the including shader must provide, because each differs per binding
// model or per pass:
//   rt_params, scene_tlas, geom     - the bound params / TLAS / geometry table
//   verts / sverts                  - the static / deformed vertex stream
//   indices / sidx                  - the static / skinned index buffer
//   prefilter_level(dir, lod)       - the IBL prefilter cube at an explicit mip
//   tex_pool / pool_sampler         - the bindless pool (RT_TEXTURED)
//   LEVEL_WAVE_SLOPES / _SAMPLER    - the waves' slope cascades, when the pass
//                                     binds them; without, a shell is drawn
//                                     at its distant roughness
// The streams are `ByteAddressBuffer`s on DirectX and structured buffers of
// `float` / `uint` elsewhere.

// Dielectric base reflectance for the Fresnel.
static const float RT_F0 = 0.04;
// The Lambert normalization the main pass divides a diffuse albedo by.
static const float RT_INV_PI = 0.31830988618;
// Floats per `Vertex` (56-byte stride): position, normal, tangent, color, uv,
// and the float each attribute the trace reads starts at.
static const uint VERTEX_FLOATS = 14u;
static const uint VERTEX_NORMAL_FLOAT = 3u;
static const uint VERTEX_TANGENT_FLOAT = 6u;
static const uint VERTEX_COLOR_FLOAT = 9u;
static const uint VERTEX_UV_FLOAT = 12u;
// Skinned objects flag bit 31 of `normal_index`; the trace then fetches the hit
// triangle from the deformed-vertex / skinned index buffers, which mirror
// the static layout. The flag is masked off before the pool sample, so skinned
// hits shade textured like static ones. Matches render_types::RT_SKINNED_FLAG.
static const uint RT_SKINNED_FLAG = 0x80000000u;

// One float of the static / deformed stream, one u32 of the static / skinned
// index buffer.
#ifdef CN_BACKEND_DIRECTX
float vert_float(uint i) { return asfloat(verts.Load(i * 4u)); }
float svert_float(uint i) { return asfloat(sverts.Load(i * 4u)); }
uint index_at(uint o) { return indices.Load(o * 4u); }
uint skinned_index_word(uint w) { return sidx.Load(w * 4u); }
#else
float vert_float(uint i) { return verts[i]; }
float svert_float(uint i) { return sverts[i]; }
uint index_at(uint o) { return indices[o]; }
uint skinned_index_word(uint w) { return sidx[w]; }
#endif

#ifdef RT_TEXTURED
float3 pool_sample_level0(uint idx, float2 uv)
{
    return tex_pool[NonUniformResourceIndex(idx)].SampleLevel(pool_sampler, uv, 0.0).rgb;
}
#endif

// Attribute fetchers into the shared 14-float `Vertex`, addressed by absolute
// vertex index `vi`. The `skinned` set reads the deformed buffer, which carries
// the identical layout.
float3 rt_vertex_normal(uint vi)
{
    uint b = vi * VERTEX_FLOATS + VERTEX_NORMAL_FLOAT;
    return float3(vert_float(b), vert_float(b + 1u), vert_float(b + 2u));
}
float3 rt_vertex_tangent(uint vi)
{
    uint b = vi * VERTEX_FLOATS + VERTEX_TANGENT_FLOAT;
    return float3(vert_float(b), vert_float(b + 1u), vert_float(b + 2u));
}
float3 rt_vertex_color(uint vi)
{
    uint b = vi * VERTEX_FLOATS + VERTEX_COLOR_FLOAT;
    return float3(vert_float(b), vert_float(b + 1u), vert_float(b + 2u));
}
float2 rt_vertex_uv(uint vi)
{
    uint b = vi * VERTEX_FLOATS + VERTEX_UV_FLOAT;
    return float2(vert_float(b), vert_float(b + 1u));
}
float3 rt_skinned_normal(uint vi)
{
    uint b = vi * VERTEX_FLOATS + VERTEX_NORMAL_FLOAT;
    return float3(svert_float(b), svert_float(b + 1u), svert_float(b + 2u));
}
float3 rt_skinned_tangent(uint vi)
{
    uint b = vi * VERTEX_FLOATS + VERTEX_TANGENT_FLOAT;
    return float3(svert_float(b), svert_float(b + 1u), svert_float(b + 2u));
}
float2 rt_skinned_uv(uint vi)
{
    uint b = vi * VERTEX_FLOATS + VERTEX_UV_FLOAT;
    return float2(svert_float(b), svert_float(b + 1u));
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

// True when anything lies along the ray, as an inline query.
bool rt_occluded(RaytracingAccelerationStructure tlas, float3 origin, float3 dir, float tmin, float tmax)
{
    RayDesc r;
    r.Origin = origin;
    r.Direction = dir;
    r.TMin = tmin;
    r.TMax = tmax;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFFu, r);
    // Drained rather than stepped once: with the opacity forced there is no
    // non-opaque candidate to resolve.
    while (q.Proceed()) {}
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

// Sun visibility at a hit: 0 where a second ray toward the sun is occluded (a
// cast shadow inside the reflection), 1 where the sun is visible or the
// shadow ray is switched off, by the settings or by the caller (`shadowed`).
float rt_shadow(float3 hp, float3 n, bool shadowed)
{
    if (!shadowed || rt_params.sun_shadows < 0.5)
    {
        return 1.0;
    }
    float3 origin = hp + n * 0.02;
    float3 dir = normalize(rt_params.sun_dir.xyz);
    return rt_occluded(scene_tlas, origin, dir, 0.001, rt_params.max_distance) ? 0.0 : 1.0;
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
    float3 f0 = lerp((float3)(RT_F0), albedo, metallic);
    float3 diff_a = albedo * (1.0 - metallic);
    float ndl = saturate(dot(n, rt_params.sun_dir.xyz));
    float3 sun = diff_a * RT_INV_PI * rt_params.sun_color.xyz * ndl * shadow;
    if (!ibl)
    {
        return sun + (diff_a + f0) * 0.03 + emissive;
    }
    float3 spec = prefilter_level(reflect(dir, n), hit_rough * max_mip);
    float3 diff = prefilter_level(n, max_mip) * diff_a;
    return sun + diff + spec * f0 + emissive;
}

// A reflection ray's footprint: how wide it is where it leaves the surface,
// in meters, and how fast it widens, in radians: the pixel's own angle plus
// the turn the reflector's curvature adds across it.
struct RtCone
{
    float width;
    float spread;
};

#ifdef LEVEL_WAVE_SLOPES
// The surface of `waves` at `p` with radial up `up`, met by a ray along `dir`
// whose footprint there is `width` meters across: the normal its resolved
// slopes tilt, and in `alpha2` the slope variance left unresolved, the finest
// waves' included. The footprint stretches along the ray's track over the
// surface by how grazing the ray is, and each cascade is read through it as
// the water reads a pixel's. Past the tiles' reach the slopes fade into the
// variance.
float3 rt_layer_waves(LevelWaves waves, float3 p, float3 up, float3 dir, float width, out float alpha2)
{
    float texels = float(waves.layers.z);
    uint cascades = min(waves.layers.x, LEVEL_WAVE_CASCADES);
    float top = log2(max(texels, 1.0));
    // Every slope each whole tile carries, the same on every face.
    float total = 0.0;
    for (uint c = 0u; c < cascades; c++)
    {
        float layer = float(waves.layers.y + c);
        float4 whole = LEVEL_WAVE_SLOPES.SampleLevel(LEVEL_WAVE_SAMPLER, float3((float2)0.5, layer), top);
        total += max(whole.z - whole.x * whole.x, 0.0) + max(whole.w - whole.y * whole.y, 0.0);
    }
    float3 rest = p - waves.anchor.xyz;
    float near = 1.0 - smoothstep(0.75, 1.0, length(rest) / max(waves.shape.z, 1e-3));
    if (near <= 0.0)
    {
        alpha2 = waves.shape.w + total;
        return up;
    }

    float3 track = dir - up * dot(dir, up);
    float track_len = length(track);
    float3 side = abs(up.x) < 0.9 ? float3(1.0, 0.0, 0.0) : float3(0.0, 0.0, 1.0);
    float3 along = track_len > 1e-4 ? track / track_len : normalize(cross(up, side));
    float3 across = cross(up, along);
    float3 step_along = along * (width / max(abs(dot(dir, up)), 0.05));
    float3 step_across = across * width;

    float inv_radius = waves.shape.x;
    float4 rows[LEVEL_WAVE_FACES][3];
    float3 shares;
    [unroll]
    for (uint f = 0u; f < LEVEL_WAVE_FACES; f++)
    {
        [unroll]
        for (uint r = 0u; r < 3u; r++)
        {
            rows[f][r] = waves.face_rows[f * 3u + r];
        }
        shares[f] = wave_face_share(rows[f], inv_radius, waves.shape.y, rest);
    }
    shares = wave_face_shares(shares);
    float3 slope = (float3)0.0;
    float removed = 0.0;
    [unroll]
    for (uint f = 0u; f < LEVEL_WAVE_FACES; f++)
    {
        if (shares[f] <= 0.0)
        {
            continue;
        }
        WaveFacePoint at = wave_face_point(rows[f], inv_radius, rest);
        float2 dx = at.stretch * float2(dot(step_along, at.tu), dot(step_along, at.tv));
        float2 dy = at.stretch * float2(dot(step_across, at.tu), dot(step_across, at.tv));
        float4 offsets[2] = { waves.face_offsets[f * 2u], waves.face_offsets[f * 2u + 1u] };
        float2 rise = (float2)0.0;
        for (uint c = 0u; c < cascades; c++)
        {
            float inv_length = waves.inv_lengths[c];
            float layer = float(waves.layers.y + c);
            float2 uv = wave_cascade_uv(at.s, inv_length, wave_cascade_offset(offsets, c), texels);
            float4 m = LEVEL_WAVE_SLOPES.SampleGrad(LEVEL_WAVE_SAMPLER, float3(uv, layer),
                                                    dx * inv_length, dy * inv_length);
            rise += m.xy;
            removed += shares[f] * shares[f] * (max(m.z - m.x * m.x, 0.0) + max(m.w - m.y * m.y, 0.0));
        }
        slope += shares[f] * at.stretch * (rise.x * at.tu + rise.y * at.tv);
    }
    alpha2 = waves.shape.w + lerp(total, removed, near);
    return normalize(up - slope * near);
}
#endif

// A hit on `layer` with normal `n` and GGX `alpha`, seen along `dir`: the
// environment mirrored off it by its Fresnel share, its body lit by the sun
// (through `shadow`) and the sky, and the sun's glint off its roughness.
float3 rt_shade_layer(LevelLayer layer, float3 n, float alpha, float3 dir, bool ibl, float max_mip,
                      float shadow)
{
    float3 v = -dir;
    float3 L = normalize(rt_params.sun_dir.xyz);
    float3 sun = rt_params.sun_color.xyz * shadow;
    float fresnel = level_layer_fresnel(saturate(dot(n, v)), layer.body.w);
    float3 reflected = ibl
        ? prefilter_level(reflect(dir, n), sqrt(alpha) * max_mip)
        : (float3)(0.03);
    float3 sky = ibl ? prefilter_level(n, max_mip) : (float3)(0.03);
    float3 lit = sun * saturate(dot(n, L)) + sky / RT_INV_PI;
    return level_layer_radiance(fresnel, reflected, layer.body.rgb, lit)
         + level_layer_glint(n, v, alpha, L, sun);
}

#ifdef RT_CANOPY
// A hit on ground growing grass, shaded as the main pass shades the far field:
// the sun on the blades through the leaf above them, the sky through the open
// canopy, and the ground's share lit by what reaches it under the grass.
float3 rt_shade_canopy(CanopyShade c, float3 n, float3 dir, bool ibl, float max_mip, float shadow)
{
    float3 L = normalize(rt_params.sun_dir.xyz);
    float3 sun = rt_params.sun_color.xyz * shadow;
    canopy_add_light(c, -dir, L, sun, rt_params.pre_exposure);
    float3 sky = ibl ? prefilter_level(c.along, max_mip) : (float3)(0.03);
    float3 ground_sky = ibl ? prefilter_level(n, max_mip) : (float3)(0.03);
    float3 under = sun * RT_INV_PI * saturate(dot(n, L)) * canopy_transmit(c, L);
    float3 ground = c.ground_albedo * (ground_sky * c.ground_sky + under);
    return canopy_radiance(c, sky, ground);
}
#endif

// Trace one reflection ray from `origin` along `dir`, of footprint `cone`, and
// shade what it hits, casting the hit's sun-shadow ray only when `shadowed`. A level surface
// within the ray's reach is met analytically and hides whatever lies under
// it; past the reach the ray escapes, so a surface hidden under land there
// never shows through. Returns how far along the ray the hit lies, or 0 when
// the ray escaped the scene, with `shaded` zero: the miss fallback is the
// caller's, because a screen-space resolve and a glass pane want different
// ones.
float rt_trace_reflection_cone(float3 origin, float3 dir, RtCone cone, bool ibl, float max_mip,
                               bool shadowed, out float3 shaded)
{
    shaded = (float3)(0.0);

    uint layer_i;
    float layer_t = level_layers_nearest(rt_params.level.layers, rt_params.cam_pos.xyz, origin, dir, layer_i);
    bool layer_reached = layer_t > 0.0 && layer_t <= rt_params.max_distance;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.01;
    ray.TMax = layer_reached ? max(layer_t, ray.TMin) : rt_params.max_distance;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
    rq.TraceRayInline(scene_tlas, RAY_FLAG_NONE, 0xFFu, ray);
    while (rq.Proceed()) {}

    if (rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        if (!layer_reached)
        {
            return 0.0;
        }
        LevelLayer layer = rt_params.level.layers[layer_i];
        float3 layer_p = origin + dir * layer_t;
        float3 layer_up = level_layer_normal(layer, layer_p);
        float shadow = rt_shadow(layer_p, layer_up, shadowed);
        float3 layer_n = layer_up;
        float alpha = layer.surface.y;
#ifdef LEVEL_WAVE_SLOPES
        if (layer.shape.w > 0.0 && rt_params.level.waves.anchor.w > 0.5)
        {
            float alpha2;
            float width = cone.width + cone.spread * layer_t;
            layer_n = rt_layer_waves(rt_params.level.waves, layer_p, layer_up, dir, width, alpha2);
            alpha = sqrt(alpha2);
        }
#endif
        shaded = rt_shade_layer(layer, layer_n, alpha, dir, ibl, max_mip, shadow);
        return layer_t;
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
    float3 hcolor = (float3)(1.0);
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
        hcolor = rt_vertex_color(i0) * w0 + rt_vertex_color(i1) * b.x + rt_vertex_color(i2) * b.y;
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
    float3 hit_p = origin + dir * hit_t;
    float shadow = rt_shadow(hit_p, hit_n, shadowed);

#ifdef RT_TEXTURED
    // Explicit level 0: a reflected ray's screen-space UV gradients are
    // unstable (neighboring pixels hit unrelated triangles), so sampling the
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
    // Color the self-emission by the emissive map (the bistro string lights),
    // mirroring the main bindless pass. Gated on a non-zero pool index; the flat
    // variant has no pool and keeps the scalar emissive.
    if (e.emissive_map_index != 0u)
    {
        emissive *= pool_sample_level0(e.emissive_map_index, huv);
    }
#ifdef RT_CANOPY
    // Ground growing grass reflects as the meadow over it: no acceleration
    // structure holds the blades, so the hit is shaded as the far field past
    // them, its covers and canopy seen back along the ray.
    if (e.far_canopy != 0u)
    {
        float footprint = hit_t * RT_CANOPY_SPREAD;
        GroundPlace place = (GroundPlace)0;
        if (canopy_on_planet(e.far_canopy))
        {
            place = ground_place(hit_p, hit_n, footprint, canopy_layers(e.far_canopy));
            GroundTaps taps = ground_taps(huv, (float2)(0.0), (float2)(0.0), GROUND_HEX_FAR);
            GroundSurface surface = ground_surface_at(place, nidx, tint, taps);
            place.covered = 1.0 - surface.keep;
            albedo = surface.layer.albedo;
        }
        CanopyShade canopy = canopy_at(e.far_canopy, place, hit_p, hit_n, hcolor, albedo, -dir,
                                       footprint, (float2)(0.0), RT_CANOPY_PAST_BLADES, true);
        shaded = rt_shade_canopy(canopy, hit_n, dir, ibl, max_mip, shadow) + emissive;
        return hit_t;
    }
#endif
    shaded = rt_shade_hit(n, albedo, e.roughness, e.metallic, emissive, dir, ibl, max_mip, shadow);
#else
    shaded = rt_shade_hit(hit_n, tint, e.roughness, e.metallic, emissive, dir, ibl, max_mip, shadow);
#endif
    return hit_t;
}

// `rt_trace_reflection_cone` for a ray of no footprint, for a pass that reads
// no surface detail through one: whether the ray hit anything.
bool rt_trace_reflection(float3 origin, float3 dir, bool ibl, float max_mip, bool shadowed,
                         out float3 shaded)
{
    RtCone point_ray;
    point_ray.width = 0.0;
    point_ray.spread = 0.0;
    return rt_trace_reflection_cone(origin, dir, point_ray, ibl, max_mip, shadowed, shaded) > 0.0;
}
