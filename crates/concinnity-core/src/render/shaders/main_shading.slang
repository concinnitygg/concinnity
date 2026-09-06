// The forward main pass's shading model and stage interface, spliced at a
// shader's MAIN_SHADING marker. The second half of the main-pass splice:
// `main_types.slang` declares the records a binding names, this is the body
// that reads the bound resources.
//
// `main_bindless.slang` drives it from the GPU-culled object buffer and the
// bindless texture pool. Everything a binding decides is reached through an
// accessor the including shader defines ahead of this splice, so nothing here
// spells one:
//
//   VIEW / LIGHTS / SHADOW_UNI / CLUSTER      the bound uniform blocks
//   LOCAL_LIGHTS / CLUSTER_LIST /
//   SPOT_SHADOWS / AREA_LIGHTS                the bound storage buffers
//   pool_sample                               one albedo / normal / map texel
//   shadow_map_cmp / shadow_map_size          the cascade depth array
//   spot_shadow_cmp / spot_shadow_map_size    the spot depth array
//   ssao_sample / ssao_size                   the blurred occlusion buffer
//   irradiance_sample /
//   prefilter_sample_level0 /
//   prefilter_sample_bias                     the environment cubes
//   environment_specular                      the reflection tap: probe set
//                                             where one is bound, else the
//                                             prefilter cube
//   ltc_matrix_sample / ltc_magnitude_sample  the area-light lookup tables

// ---- Constants ----

static const float PI = 3.14159265359;

// Per-cluster light-list stride: MAX_LIGHTS_PER_CLUSTER + 1 (slot 0 is the
// count). Matches CLUSTER_LIGHT_LIST_STRIDE in render_types.rs.
static const uint CLUSTER_LIGHT_LIST_STRIDE = 64u;

// Surfaces rougher than this get no SSR / RT reflection; the forward fade
// ramps in below it. Matches the resolve gloss gate (SSR_ROUGH_CUT /
// RT_ROUGH_CUT).
static const float REFLECTION_ROUGHNESS_CUT = 0.6;

// Edge of the LTC lookup tables, and the scale / bias that map [0, 1] onto
// texel centres. Must match LTC_LUT_SIZE in `core::render`'s ltc module.
static const float LTC_LUT_SIZE  = 64.0;
static const float LTC_LUT_SCALE = (LTC_LUT_SIZE - 1.0) / LTC_LUT_SIZE;
static const float LTC_LUT_BIAS  = 0.5 / LTC_LUT_SIZE;

static const float3 SKY_ZENITH  = float3(0.110, 0.322, 0.726);
static const float3 SKY_HORIZON = float3(0.765, 0.863, 0.941);

// ---- Stage interface ----

struct VertexIn
{
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 normal  : NORMAL;
    [[vk::location(2)]] float3 tangent : TANGENT;
    [[vk::location(3)]] float3 color   : COLOR0;
    [[vk::location(4)]] float2 uv      : TEXCOORD0;
};

struct VertexOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_pos  : TEXCOORD1;
    [[vk::location(1)]] float3 normal     : TEXCOORD2;
    [[vk::location(2)]] float3 tangent    : TEXCOORD3;
    [[vk::location(3)]] float3 bitangent  : TEXCOORD4;
    [[vk::location(4)]] float2 uv         : TEXCOORD5;
    [[vk::location(5)]] float view_depth  : TEXCOORD6;
    [[vk::location(6)]] float3 color      : TEXCOORD7;
    // The object id is needed in the fragment stage too (the instance index is
    // a vertex-only built-in), so it rides a flat varying.
    [[vk::location(7)]] nointerpolation uint object_id : TEXCOORD8;
};

// The whole varying block has to be read by a fragment entry, whichever
// subset the world's `shade` uses: slangc drops an unread input from the
// SPIR-V interface, and Vulkan then reports the vertex stage's matching output
// as unconsumed. An entry folds this sum into its result behind a condition no
// input can meet.
float varyings_read(VertexOut in)
{
    return in.world_pos.x + in.normal.x + in.tangent.x + in.bitangent.x + in.uv.x
         + in.view_depth + in.color.x;
}

// ---- Vertex ----

// Project one model-space vertex through `model`. Normals ride the cofactor
// matrix, so they stay perpendicular under non-uniform scale; the normalize()
// at every use site absorbs the determinant scale it carries.
VertexOut project_vertex(float4x4 model, float3 pos, float3 normal, float3 tangent,
                         float3 color, float2 uv)
{
    VertexOut o;
    float4 world = mul(model, float4(pos, 1.0));
    o.world_pos = world.xyz;

    float3x3 nm = normal_matrix(model);
    o.normal    = normalize(mul(nm, normal));
    o.tangent   = normalize(mul(nm, tangent));
    o.bitangent = cross(o.normal, o.tangent);

    o.uv    = uv;
    o.color = color;

    o.view_depth = -mul(VIEW.view_mat, world).z;
    o.position   = mul(VIEW.vp, world);

    // Skybox sentinel (blue channel 2.0): pin to the far plane so the sky is
    // never clipped by the camera far plane and always renders behind scene
    // geometry. Every forward vertex path needs it.
    if (color.b > 1.5)
    {
        o.position.z = o.position.w * (1.0 - 1e-6);
    }
    return o;
}

// ---- Fragment helpers ----

float distribution_ggx(float3 N, float3 H, float roughness)
{
    float a  = roughness * roughness;
    float a2 = a * a;
    float NdH  = max(dot(N, H), 0.0);
    float NdH2 = NdH * NdH;
    float denom = NdH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

float geometry_schlick_ggx(float NdV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdV / (NdV * (1.0 - k) + k);
}

float geometry_smith(float3 N, float3 V, float3 L, float roughness)
{
    float NdV = max(dot(N, V), 0.0);
    float NdL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdV, roughness) * geometry_schlick_ggx(NdL, roughness);
}

float3 fresnel_schlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float2 env_brdf_approx(float NdV, float rough)
{
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4( 1.0,  0.0425,  1.040, -0.040);
    float4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdV)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

// Decode a tangent-space normal map texel. Only X and Y are read; Z is
// reconstructed from them, so a two-channel source (BC5) decodes the same as
// an RGBA8 one and normal maps can ship as BC5 blocks.
float3 decode_normal_map(float2 encoded)
{
    float2 nxy = encoded * 2.0 - 1.0;
    return float3(nxy, sqrt(clamp(1.0 - dot(nxy, nxy), 0.0, 1.0)));
}

// Geometric specular antialiasing (Kaplanyan et al. 2016, as in Filament):
// widen the NDF by the screen-space variance of the shading normal so an
// undersampled high-frequency normal map at a distance does not alias into
// specular fireflies. A no-op where the normal is smooth (close up), so the
// surface detail is preserved.
float specular_aa_roughness(float3 N, float perceptual_roughness)
{
    const float VARIANCE  = 0.25;
    const float THRESHOLD = 0.18;
    float3 dndx = ddx(N);
    float3 dndy = ddy(N);
    float variance = VARIANCE * (dot(dndx, dndx) + dot(dndy, dndy));
    float alpha = perceptual_roughness * perceptual_roughness;
    float kernel = min(2.0 * variance, THRESHOLD);
    float filtered_alpha2 = clamp(alpha * alpha + kernel, 0.0, 1.0);
    return sqrt(sqrt(filtered_alpha2));
}

float hash_rotation(float2 p)
{
    float h = frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
    return h * 6.2831853;
}

{SHADOW_BIAS}

// 3x3 hash-rotated PCF of one spot shadow slice. Returns [0, 1] (1.0 fully
// lit), and 1.0 outside the cone's light frustum so an unshadowed region is
// never darkened. A smaller kernel than the cascade PCF: a spot slice covers
// far less world area per texel.
float sample_spot_shadow(int shadow_index, float3 world_pos, float3 normal, float2 screen_xy)
{
    SpotShadowData sd = SPOT_SHADOWS[shadow_index];
    // Offsetting along the normal before projecting pushes the sample off
    // surfaces near-parallel to the light, where depth slope causes acne.
    float3 biased = world_pos + normal * sd.normal_bias;
    float4 light_clip = mul(sd.light_vp, float4(biased, 1.0));
    if (light_clip.w <= 0.0)
    {
        return 1.0;
    }
    float3 ndc = light_clip.xyz / light_clip.w;
    // Flip Y to match the negative-height viewport the spot pass renders with,
    // exactly as the cascade PCF does.
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0)
    {
        return 1.0;
    }

    float ref = ndc.z - sd.depth_bias;
    float angle = hash_rotation(screen_xy);
    float ca = cos(angle);
    float sa = sin(angle);
    float2 tex_size = 1.0 / spot_shadow_map_size();

    float sum = 0.0;
    const int RADIUS = 1; // 3x3
    const float SAMPLES = float((2 * RADIUS + 1) * (2 * RADIUS + 1));
    for (int dy = -RADIUS; dy <= RADIUS; dy++)
    {
        for (int dx = -RADIUS; dx <= RADIUS; dx++)
        {
            float2 off = float2(float(dx), float(dy));
            float2 rot = float2(off.x * ca - off.y * sa, off.x * sa + off.y * ca);
            sum += spot_shadow_cmp(
                float3(uv + rot * tex_size, float(shadow_index)), ref);
        }
    }
    return sum / SAMPLES;
}

// Clip a quad against the horizon plane z = 0, keeping the part above it.
// Sutherland-Hodgman rather than the usual hardcoded 16-case table: a quad cut
// by one plane yields at most 5 vertices, and the loop form cannot be got
// wrong case by case. Mirrors clip_quad_to_horizon in `core::render`'s
// ltc::polygon, which is unit-tested against brute-force integration.
int clip_quad_to_horizon(float3 quad[4], out float3 clipped[5])
{
    clipped = { float3(0.0), float3(0.0), float3(0.0), float3(0.0), float3(0.0) };
    int n = 0;
    for (int i = 0; i < 4; i++)
    {
        float3 current  = quad[i];
        float3 previous = quad[(i + 3) % 4];
        bool current_in  = current.z > 0.0;
        bool previous_in = previous.z > 0.0;
        if (current_in != previous_in)
        {
            float t = previous.z / (previous.z - current.z);
            clipped[n++] = float3(previous.xy + t * (current.xy - previous.xy), 0.0);
        }
        if (current_in)
        {
            clipped[n++] = current;
        }
    }
    return n;
}

// Twice the contribution of one edge of the spherical polygon. The cross
// product's z carries the sign, so a reversed winding flips the whole sum,
// which is what tells a front-facing polygon from a back-facing one.
float integrate_edge(float3 v1, float3 v2)
{
    float cos_theta = clamp(dot(v1, v2), -1.0, 1.0);
    float theta     = acos(cos_theta);
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    float ratio     = (sin_theta > 1e-4) ? (theta / sin_theta) : 1.0;
    return cross(v1, v2).z * ratio;
}

// Fraction of the clamped-cosine distribution the quad covers, in [0, 1].
// `m_inv` is the LTC inverse transform (rows follow mul(v, M) convention), or
// the identity for the diffuse term.
float ltc_evaluate(float3 N, float3 V, float3 P, float3x3 m_inv, float3 corners[4], bool two_sided)
{
    // Shading frame with the normal on +z and the first tangent in the view
    // plane, matching how the table was fitted.
    float3 t1 = normalize(V - N * dot(V, N));
    float3 t2 = cross(N, t1);

    float3 quad[4];
    for (int i = 0; i < 4; i++)
    {
        float3 d = corners[i] - P;
        float3 local = float3(dot(t1, d), dot(t2, d), dot(N, d));
        quad[i] = mul(local, m_inv);
    }

    float3 clipped[5];
    int n = clip_quad_to_horizon(quad, clipped);
    if (n < 3)
    {
        return 0.0;
    }
    for (int k = 0; k < n; k++)
    {
        clipped[k] = normalize(clipped[k]);
    }

    float sum = 0.0;
    for (int e = 0; e < n; e++)
    {
        sum += integrate_edge(clipped[e], clipped[(e + 1) % n]);
    }

    // The edge sum is twice the irradiance; dividing by pi normalises the
    // clamped cosine, so the covered fraction is sum / (2 * pi).
    float form_factor = sum / (2.0 * PI);
    return two_sided ? abs(form_factor) : max(-form_factor, 0.0);
}

// 5x5 hash-rotated PCF of a single cascade. Returns the shadow factor in
// [0, 1] (1.0 fully lit), or 1.0 when the fragment lies outside this
// cascade's light frustum.
float sample_cascade_pcf(int cascade, float3 world_pos, float2 screen_xy)
{
    float4 lc = mul(SHADOW_UNI.light_vps[cascade], float4(world_pos, 1.0));
    float3 ndc = lc.xyz / lc.w;
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0)
    {
        return 1.0;
    }

    float ref = ndc.z - cascade_depth_bias(cascade);

    float angle = hash_rotation(screen_xy);
    float ca = cos(angle);
    float sa = sin(angle);

    float2 tex_size = 1.0 / shadow_map_size();

    float sum = 0.0;
    const int RADIUS = 2;
    const float SAMPLES = float((2 * RADIUS + 1) * (2 * RADIUS + 1));
    for (int dy = -RADIUS; dy <= RADIUS; dy++)
    {
        for (int dx = -RADIUS; dx <= RADIUS; dx++)
        {
            float2 off = float2(float(dx), float(dy));
            float2 rot = float2(off.x * ca - off.y * sa, off.x * sa + off.y * ca);
            float2 sample_uv = uv + rot * tex_size;
            sum += shadow_map_cmp(float3(sample_uv, float(cascade)), ref);
        }
    }
    return sum / SAMPLES;
}

// Cascade-aware PCF with cross-cascade blending. Selects the cascade whose far
// split exceeds the fragment's view-space depth, then blends into the next
// cascade across a band at the far edge of that cascade's depth range: each
// cascade places the shadow edge slightly differently, and a hard switch would
// sweep across the world as the camera moves.
float shadow_factor_cascaded(float3 world_pos, float view_depth, float2 screen_xy)
{
    int cascade = 4;
    if      (view_depth < SHADOW_UNI.cascade_splits[0]) cascade = 0;
    else if (view_depth < SHADOW_UNI.cascade_splits[1]) cascade = 1;
    else if (view_depth < SHADOW_UNI.cascade_splits[2]) cascade = 2;
    else if (view_depth < SHADOW_UNI.cascade_splits[3]) cascade = 3;
    if (cascade >= int(SHADOW_UNI.active_cascades))
    {
        return 1.0;
    }

    float shade = sample_cascade_pcf(cascade, world_pos, screen_xy);

    if (cascade + 1 < int(SHADOW_UNI.active_cascades))
    {
        float split_far  = SHADOW_UNI.cascade_splits[cascade];
        float split_near = (cascade == 0) ? 0.0 : SHADOW_UNI.cascade_splits[cascade - 1];
        float band = (split_far - split_near) * 0.15;
        float t = (view_depth - (split_far - band)) / max(band, 1e-4);
        if (t > 0.0)
        {
            float next = sample_cascade_pcf(cascade + 1, world_pos, screen_xy);
            shade = lerp(shade, next, clamp(t, 0.0, 1.0));
        }
    }
    return shade;
}

// ---- The shading model ----

// The forward shading model. `od` is the object record the GPU-driven pass
// reads straight out of the per-frame buffer.
float4 shade_surface(VertexOut in, GpuObjectData od)
{
    float roughness = od.tint_roughness.w;
    float metallic  = od.emissive_metallic.w;
    float3 tint     = od.tint_roughness.xyz;
    float3 emissive = od.emissive_metallic.xyz;

    float3 cam_pos = float3(VIEW.cam_x, VIEW.cam_y, VIEW.cam_z);
    bool ibl_enabled = VIEW.prefilter_mip_count > 0.5;

    // Skybox sentinel (blue channel 2.0): sky colour from the view direction.
    if (in.color.b > 1.5)
    {
        float3 view_dir = normalize(in.world_pos - cam_pos);
        float3 sky;
        if (ibl_enabled)
        {
            sky = prefilter_sample_level0(view_dir);
        }
        else
        {
            float t = max(0.0, view_dir.y);
            sky = lerp(SKY_HORIZON, SKY_ZENITH, t);
        }
        return float4(sky, 1.0);
    }

    // The record is per-object, so a fragment wave that straddles two objects
    // of one indirect draw carries two pool indices. That makes every pool
    // index non-uniform on the descriptor-indexing targets (pool_sample
    // annotates it there).
    float4 albedo_samp = pool_sample(od.albedo_index, in.uv);
    // Alpha cutout: punch the texel out entirely so foliage and decal cards
    // stay in the opaque pass. Disabled at cutoff 0.
    float alpha_cutoff = od.bb_max_alpha_cutoff.w;
    if (alpha_cutoff > 0.0 && albedo_samp.a < alpha_cutoff)
    {
        discard;
    }
    float3 albedo = albedo_samp.rgb * in.color * tint;

    // Unlit view mode: the surface's base color, no lighting.
    if (VIEW.shade_mode > 0.5)
    {
        return float4(albedo, 1.0);
    }

    // Per-material emissive texture carries the colour (the scalar factor is a
    // uniform strength when a map is bound). Slot 0 is the "no map" sentinel.
    if (od.emissive_map_index != 0u)
    {
        emissive *= pool_sample(od.emissive_map_index, in.uv).rgb;
    }

    // Occlusion-roughness-metallic map: green carries roughness, blue carries
    // metallic (glTF convention). Slot 0 is the "no map" sentinel.
    if (od.orm_map_index != 0u)
    {
        float3 orm = pool_sample(od.orm_map_index, in.uv).rgb;
        roughness = orm.g;
        metallic  = orm.b;
    }

    float3 norm_samp = decode_normal_map(pool_sample(od.normal_index, in.uv).rg);
    // Tangent frame as rows so mul(v, M) applies the column-basis transform.
    float3x3 TBN = float3x3(
        normalize(in.tangent),
        normalize(in.bitangent),
        normalize(in.normal));
    float3 N = normalize(mul(norm_samp, TBN));

    // Geometric specular antialiasing on the normal map. Minification aliasing
    // is handled by the texture's mip chain (trilinear + anisotropic
    // sampling); this widens the specular NDF for residual sub-pixel normal
    // variance.
    roughness = specular_aa_roughness(N, roughness);

    float3 V   = normalize(cam_pos - in.world_pos);
    float NdV  = max(dot(N, V), 0.0);

    float3 F0 = lerp(float3(0.04), albedo, metallic);

    float2 screen_xy = in.position.xy;
    float shadow = shadow_factor_cascaded(in.world_pos, in.view_depth, screen_xy);

    float2 ab        = env_brdf_approx(NdV, roughness);
    float ess        = ab.x + ab.y;
    float3 energy_ms = 1.0 + F0 * (1.0 / max(ess, 0.001) - 1.0);

    float3 Lo = float3(0.0);

    for (int i = 0; i < LIGHTS.num_dir; i++)
    {
        float3 L = normalize(LIGHTS.dir[i].dir_i.xyz);
        float intensity = LIGHTS.dir[i].dir_i.w;
        float3 radiance = LIGHTS.dir[i].col.xyz * intensity;

        float3 H = normalize(V + L);
        float NdL = max(dot(N, L), 0.0);

        float D = distribution_ggx(N, H, roughness);
        float G = geometry_smith(N, V, L, roughness);
        float3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);

        float3 kd = (1.0 - F) * (1.0 - metallic);
        float3 spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001) * energy_ms;
        float3 diff = kd * albedo / PI;

        float s = (i == 0) ? shadow : 1.0;
        Lo += (diff + spec) * radiance * NdL * s;
    }

    // Clustered light iteration: when clustering is active (the main camera),
    // map this fragment to its froxel cluster and shade only that cluster's
    // binned lights. Planar / probe re-renders bind use_clusters = 0 (their
    // viewpoint differs from the grid the main camera binned) and fall back
    // to iterating every local light.
    uint cluster_base = 0u;
    int  local_count;
    if (CLUSTER.use_clusters != 0u)
    {
        uint cx = min(uint(screen_xy.x / CLUSTER.screen_w * float(CLUSTER.grid_x)),
                      CLUSTER.grid_x - 1u);
        uint cy = min(uint(screen_xy.y / CLUSTER.screen_h * float(CLUSTER.grid_y)),
                      CLUSTER.grid_y - 1u);
        float zd = max(in.view_depth, CLUSTER.cam_pos_znear.w);
        uint cz = min(uint(log(zd / CLUSTER.cam_pos_znear.w) / log(CLUSTER.view_forward_zfar.w / CLUSTER.cam_pos_znear.w)
                           * float(CLUSTER.grid_z)),
                      CLUSTER.grid_z - 1u);
        uint cid = cx + cy * CLUSTER.grid_x + cz * CLUSTER.grid_x * CLUSTER.grid_y;
        cluster_base = cid * CLUSTER_LIGHT_LIST_STRIDE;
        local_count = int(CLUSTER_LIST[cluster_base]);
    }
    else
    {
        local_count = LIGHTS.num_local_lights;
    }

    for (int jj = 0; jj < local_count; jj++)
    {
        int i = (CLUSTER.use_clusters != 0u)
              ? int(CLUSTER_LIST[cluster_base + 1u + uint(jj)])
              : jj;
        float3 pos_w  = LOCAL_LIGHTS[i].position_range.xyz;
        float  range  = LOCAL_LIGHTS[i].position_range.w;
        float3 col    = LOCAL_LIGHTS[i].color_intensity.xyz;
        float  intens = LOCAL_LIGHTS[i].color_intensity.w;

        // Area lights integrate the whole panel rather than a single
        // direction, so they replace the point / spot BRDF evaluation.
        if (light_kind(LOCAL_LIGHTS[i]) == LIGHT_KIND_AREA)
        {
            int ai = LOCAL_LIGHTS[i].data_index;
            if (ai < 0)
            {
                continue;
            }
            float3 centre = pos_w;
            float3 right  = AREA_LIGHTS[ai].right_two_sided.xyz;
            float3 up     = AREA_LIGHTS[ai].up_pad.xyz;
            bool two_sided = asuint(AREA_LIGHTS[ai].right_two_sided.w) != 0u;

            // Range is a cutoff measured from the panel centre, matching the
            // sphere the clustered cull bins this light with. The physical
            // falloff is already in the form factor: the panel subtends a
            // smaller solid angle further away.
            float centre_dist = length(centre - in.world_pos);
            float window = clamp(1.0 - centre_dist / range, 0.0, 1.0);
            window = window * window;
            if (window <= 0.0)
            {
                continue;
            }

            float3 corners[4];
            corners[0] = centre - right - up;
            corners[1] = centre + right - up;
            corners[2] = centre + right + up;
            corners[3] = centre - right + up;

            // Diffuse needs no lookup: it is the polygon integral under the
            // plain clamped cosine, i.e. an identity transform.
            float3x3 identity = float3x3(
                float3(1.0, 0.0, 0.0),
                float3(0.0, 1.0, 0.0),
                float3(0.0, 0.0, 1.0));
            float diffuse_ff = ltc_evaluate(N, V, in.world_pos, identity, corners, two_sided);

            // Specular applies the fitted transform before the same integral.
            float2 lut_uv = float2(roughness, sqrt(clamp(1.0 - NdV, 0.0, 1.0)));
            lut_uv = lut_uv * LTC_LUT_SCALE + LTC_LUT_BIAS;
            float4 t1 = ltc_matrix_sample(lut_uv);
            float2 t2 = ltc_magnitude_sample(lut_uv);
            // The table stores the inverse normalised so its middle entry is
            // 1, packed as (m00, m20, m02, m22). Rows here follow the
            // mul(v, M) convention, matching the GLSL / MSL column form.
            float3x3 m_inv = float3x3(
                float3(t1.x, 0.0, t1.y),
                float3(0.0,  1.0, 0.0),
                float3(t1.z, 0.0, t1.w));
            float specular_ff = ltc_evaluate(N, V, in.world_pos, m_inv, corners, two_sided);
            // Schlick split baked into the table: t2.x weights the base
            // reflectance, t2.y the grazing response.
            float3 area_spec = F0 * t2.x + (1.0 - F0) * t2.y;

            float3 area_radiance = col * intens * window;
            float3 area_kd = (1.0 - F0) * (1.0 - metallic);
            Lo += area_radiance * (area_kd * albedo * diffuse_ff
                                   + area_spec * specular_ff);
            continue;
        }

        float3 L    = normalize(pos_w - in.world_pos);
        float  dist = length(pos_w - in.world_pos);
        float atten = clamp(1.0 - (dist / range), 0.0, 1.0);
        atten *= atten;
        // Spot cone: full brightness inside cos_inner, squared fade to black
        // at cos_outer. Point lights leave both at zero and skip this.
        if (light_kind(LOCAL_LIGHTS[i]) == LIGHT_KIND_SPOT)
        {
            float cd = dot(LOCAL_LIGHTS[i].direction_kind.xyz, -L);
            float ci = LOCAL_LIGHTS[i].cos_inner;
            float co = LOCAL_LIGHTS[i].cos_outer;
            float t  = clamp((cd - co) / max(ci - co, 1e-4), 0.0, 1.0);
            atten *= t * t;
            // Only spots that claimed a shadow slice sample the array; the
            // rest keep shadow_index at -1 and light without casting.
            int si = LOCAL_LIGHTS[i].shadow_index;
            if (si >= 0 && atten > 0.0)
            {
                atten *= sample_spot_shadow(si, in.world_pos, N, screen_xy);
            }
        }
        float3 radiance = col * intens * atten;

        float3 H  = normalize(V + L);
        float NdL = max(dot(N, L), 0.0);

        float D = distribution_ggx(N, H, roughness);
        float G = geometry_smith(N, V, L, roughness);
        float3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);

        float3 kd   = (1.0 - F) * (1.0 - metallic);
        float3 spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001) * energy_ms;
        float3 diff = kd * albedo / PI;
        Lo += (diff + spec) * radiance * NdL;
    }

    float3 ambient;
    if (ibl_enabled)
    {
        float3 F_ibl       = fresnel_schlick(NdV, F0);
        float3 kd_ibl      = (1.0 - F_ibl) * (1.0 - metallic);
        float3 irradiance  = irradiance_sample(N);
        float3 diffuse_ibl = kd_ibl * albedo * irradiance / PI;

        float3 R = reflect(-V, N);
        // SampleBias (not SampleLevel) so the reflection vector's screen-space
        // footprint widens the mip at grazing or distant angles. A forced LOD
        // defeats minification filtering and aliases the environment into
        // sparkle on near mirrors; flat close-up pixels have a near-zero
        // footprint, so they keep the plain roughness mip.
        float lod = roughness * (VIEW.prefilter_mip_count - 1.0);
        // Local reflection probes when the host binds a probe set and any are
        // baked (box-parallax partition of unity), else the imported
        // environment prefilter cube.
        float3 prefiltered = environment_specular(in.world_pos, R, lod);
        float3 specular_ibl = prefiltered * (F0 * ab.x + ab.y);

        // When an SSR / RT reflection composite owns the sharp specular for
        // glossy surfaces this frame, fade the forward probe specular for
        // glossy dielectrics so the two do not double-count. Metals keep
        // their full albedo-tinted forward specular (the resolve adds only a
        // faint dielectric term), and surfaces rougher than the cut (which
        // the resolve skips) keep theirs too.
        if (VIEW.reflections_enabled > 0.5)
        {
            float fade = smoothstep(REFLECTION_ROUGHNESS_CUT * 0.7,
                                    REFLECTION_ROUGHNESS_CUT, roughness);
            specular_ibl *= lerp(1.0, fade, 1.0 - metallic);
        }

        ambient = diffuse_ibl + specular_ibl;
    }
    else
    {
        ambient = float3(0.03) * albedo;
    }

    // Authored indirect-fill multiplier (PostProcessConfig.ambient_intensity);
    // 1.0 is a no-op. Lifts shadow fill without touching sun-lit surfaces.
    ambient *= LIGHTS.ambient_intensity;

    // SSAO modulates the indirect (ambient / IBL) term only: direct lighting
    // is unaffected. A 1x1 white view is bound when SSAO is disabled, so this
    // samples a constant 1.0 then.
    float2 ssao_uv = screen_xy / ssao_size();
    ambient *= ssao_sample(ssao_uv);

    float3 color = ambient + Lo + emissive;

    return float4(color, albedo_samp.a);
}

