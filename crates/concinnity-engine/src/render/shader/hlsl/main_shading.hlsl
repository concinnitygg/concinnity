// The forward main pass's shading model and stage interface, spliced at a
// shader's MAIN_SHADING marker. The second half of the main-pass splice:
// `main_types.hlsl` declares the records a binding names, this is the body
// that reads the bound resources.
//
// `main_bindless.hlsl` drives it from the GPU-culled object buffer and the
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
//   screen_indirect_at                        the screen-space indirect light
//   irradiance_sample /
//   prefilter_sample_bias                     the environment cubes
//   environment_specular                      the reflection tap: the
//                                             fragment's probes where the set
//                                             holds any, else the prefilter
//                                             cube, else none
//   cluster_at / probe_cluster_mask /
//   probe_mask_all                            the cluster grid's lookups
//                                             (probe_common.hlsl)
//   ltc_matrix_sample / ltc_magnitude_sample  the area-light lookup tables

{DEPTH_CONVENTION}

{SURFACE_INPUTS}

// ---- Constants ----

static const float PI = 3.14159265359;

// Surfaces rougher than REFLECTION_ROUGHNESS_CUT get no SSR / RT reflection;
// the forward fade ramps in below it.
{REFLECTION_CUT}

// Edge of the LTC lookup tables, and the scale / bias that map [0, 1] onto
// texel centers. Must match LTC_LUT_SIZE in `core::render`'s ltc module.
static const float LTC_LUT_SIZE = 64.0;
static const float LTC_LUT_SCALE = (LTC_LUT_SIZE - 1.0) / LTC_LUT_SIZE;
static const float LTC_LUT_BIAS  = 0.5 / LTC_LUT_SIZE;

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

{SHADOW_CASCADE}

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

    float ref = depth_offset_near(ndc.z, sd.depth_bias);
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
    float3 zero[5] = { (float3)(0.0), (float3)(0.0), (float3)(0.0), (float3)(0.0), (float3)(0.0) };
    clipped = zero;
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

    // The edge sum is twice the irradiance; dividing by pi normalizes the
    // clamped cosine, so the covered fraction is sum / (2 * pi).
    float form_factor = sum / (2.0 * PI);
    return two_sided ? abs(form_factor) : max(-form_factor, 0.0);
}

#define DAYLIGHT VIEW.daylight
{DAYLIGHT}

// A planet's ground skips a body too faint to show. Only its daylight
// variants light it from its own sky; every other surface keeps the camera's,
// and compiles without it.
#if defined(SURFACE_CANOPY) || defined(SURFACE_GRASS_GROUND)
#define FAINT_BODY_SKIP 1
#endif
#if defined(FAINT_BODY_SKIP) && defined(SURFACE_DAYLIGHT)
#define DAYLIGHT_SURFACE 1
#endif

{CANOPY}

// ---- The shading model ----

// The forward shading model. `od` is the object record the GPU-driven pass
// reads straight out of the per-frame buffer.
float4 shade_surface(VertexOut v, GpuObjectData od)
{
    float3 tint     = od.tint_roughness.xyz;
    float3 emissive = od.emissive_metallic.xyz;

    float3 cam_pos = float3(VIEW.cam_x, VIEW.cam_y, VIEW.cam_z);
    bool ibl_enabled = VIEW.irradiance > 0.5;

    // The record is per-object, so a fragment wave that straddles two objects
    // of one indirect draw carries two pool indices. That makes every pool
    // index non-uniform on the descriptor-indexing targets (pool_sample
    // annotates it there).
    float4 albedo_samp = pool_sample(od.albedo_index, v.uv);
    surface_cutout(od, albedo_samp.a);
    // A surface under grass carries its layers' densities in its vertex color,
    // and a planet tile the edits under it, rather than a multiplier on its
    // albedo.
    bool color_tints = od.far_canopy == 0u && od.tile_morph.eye_range.w <= 0.0;
    float3 albedo = albedo_samp.rgb * tint * (color_tints ? v.color : (float3)(1.0));

    // A surface growing grass has litter on its ground and the leaf above
    // shading it, and past where the blades end it is shaded as the canopy
    // they make. The pixel's footprint is taken here, in uniform control flow.
    // Only the grass variants, which every surface growing grass draws under,
    // carry the ground: SURFACE_CANOPY with the canopy, SURFACE_GRASS_GROUND
    // without it for draws wholly nearer than their band. In every other
    // program the band is a constant 0 and the grass code folds away.
    float3 V_far = cam_pos - v.world_pos;
    float far_dist = length(V_far);
    float footprint = max(length(ddx(v.world_pos)), length(ddy(v.world_pos)));
    CanopyShade canopy = (CanopyShade)0;
    float3 ground_normal = v.normal;
    // A planet's ground takes its covers' look where their rules hold.
    GroundCoverMix covers = ground_no_covers();
#if defined(SURFACE_CANOPY) || defined(SURFACE_GRASS_GROUND)
    float4 ground = canopy_ground_normal(od, v.world_pos, v.normal, footprint, v.color, v.tangent);
    ground_normal = ground.xyz;
#ifdef SURFACE_CANOPY
    float far_w = canopy_band(od, far_dist);
#else
    const float far_w = 0.0;
#endif
    GroundPlace place = (GroundPlace)0;
    GroundSwash swash = (GroundSwash)0;
    if (canopy_on_planet(od))
    {
        place = ground_place(v.world_pos, ground_normal, footprint, canopy_layers(od));
        covers = ground_covers_at(place, v.uv);
        place.covered = 1.0 - covers.keep;
        albedo = albedo * covers.keep + covers.albedo;
        // Where the sea's waves wash up the beach, the sand is darker and wet
        // and a foam line runs at the wash's front.
        swash = ground_swash_at(place, ground.w, footprint);
        albedo = lerp(albedo * lerp(1.0, GROUND_WET_DARKEN, swash.wet), GROUND_SWASH_FOAM, swash.edge);
    }
    canopy = canopy_at(od, place, v.world_pos, ground_normal, v.color, albedo,
                       V_far / max(far_dist, 1e-4), footprint, v.position.xy, far_dist,
                       far_w > 0.0);
    albedo = canopy.ground_albedo;
#else
    const float far_w = 0.0;
    canopy.ground_sky = 1.0;
#endif

    // Unlit view mode: the surface's base color, no lighting.
    if (VIEW.shade_mode > 0.5)
    {
        return float4(far_w > 0.0 ? canopy_albedo_seen(canopy, albedo) : albedo, 1.0);
    }

    // Per-material emissive texture carries the color (the scalar factor is a
    // uniform strength when a map is bound). Slot 0 is the "no map" sentinel.
    if (od.emissive_map_index != 0u)
    {
        emissive *= pool_sample(od.emissive_map_index, v.uv).rgb;
    }

    float2 roughness_metallic = surface_roughness_metallic(od, v.uv) * covers.keep
                              + covers.roughness_metallic;
    float roughness = roughness_metallic.x;
    float metallic  = roughness_metallic.y;
#if defined(SURFACE_CANOPY) || defined(SURFACE_GRASS_GROUND)
    // Wet sand is smoother than dry, and the film the wash leaves smoother
    // still, but for its foam.
    roughness = lerp(roughness, min(roughness, GROUND_WET_ROUGHNESS), swash.wet);
    roughness = lerp(roughness, GROUND_FILM_ROUGHNESS, swash.film * (1.0 - swash.edge));
#endif

    float3 N = surface_normal(od, v.uv, ground_normal, v.tangent, v.bitangent);

    // Geometric specular antialiasing on the normal map. Minification aliasing
    // is handled by the texture's mip chain (trilinear + anisotropic
    // sampling); this widens the specular NDF for residual sub-pixel normal
    // variance.
    roughness = specular_aa_roughness(N, roughness);

    float3 V   = normalize(cam_pos - v.world_pos);
    float NdV  = max(dot(N, V), 0.0);

    float3 F0 = lerp((float3)(0.04), albedo, metallic);

    float2 screen_xy = v.position.xy;
    float shadow = shadow_factor_cascaded(v.world_pos, v.view_depth, screen_xy);

    float2 ab        = env_brdf_approx(NdV, roughness);
    float ess        = ab.x + ab.y;
    float3 energy_ms = 1.0 + F0 * (1.0 / max(ess, 0.001) - 1.0);

    float3 Lo = (float3)(0.0);
    // The direct light the ground under a canopy still gets.
    float3 Lo_under = (float3)(0.0);

    // On a planet's ground, how much the light is the point's own: the
    // bodies through the air above it and its own sky, in place of the
    // camera's, as it lies further round the planet or the camera climbs.
#ifdef DAYLIGHT_SURFACE
    DaylightPoint day = daylight_point(v.world_pos);
    float3 day_up = day.up;
    float day_reach = day.reach;
#endif

    for (int i = 0; i < LIGHTS.num_dir; i++)
    {
        float3 L = normalize(LIGHTS.dir[i].dir_i.xyz);
        float intensity = LIGHTS.dir[i].dir_i.w;
        float3 radiance = LIGHTS.dir[i].col.xyz * intensity;
#ifdef DAYLIGHT_SURFACE
        if (day_reach > 0.0 && daylight_body(uint(i)))
        {
            radiance = lerp(radiance, daylight_point_light(day, uint(i)), day_reach);
        }
#endif
#ifdef FAINT_BODY_SKIP
        // A body too faint to show (the sun by night, the moon by day).
        if (daylight_on() && max(radiance.r, max(radiance.g, radiance.b)) * VIEW.pre_exposure < 1e-4)
        {
            continue;
        }
#endif

        float3 H = normalize(V + L);
        float NdL = max(dot(N, L), 0.0);

        float D = distribution_ggx(N, H, roughness);
        float G = geometry_smith(N, V, L, roughness);
        float3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);

        float3 kd = (1.0 - F) * (1.0 - metallic);
        float3 spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001) * energy_ms;
        float3 diff = kd * albedo / PI;

        float s = (i == 0) ? shadow : 1.0;
        float3 direct = (diff + spec) * radiance * NdL * s;
        Lo += direct;
        if (far_w > 0.0)
        {
            Lo_under += direct * canopy_transmit(canopy, L);
            canopy_add_light(canopy, V, L, radiance * s);
        }
    }

    // Clustered light and probe lookup: when clustering is active (the main
    // camera), map this fragment to its froxel cluster and shade only that
    // cluster's binned lights and probes. Planar / probe re-renders bind
    // use_clusters = 0 (their viewpoint differs from the grid the main camera
    // binned) and fall back to iterating every local light and probe.
    uint cluster_base = 0u;
    int  local_count;
    ProbeMask probes;
    if (CLUSTER.use_clusters != 0u)
    {
        uint cid = cluster_at(screen_xy / float2(CLUSTER.screen_w, CLUSTER.screen_h), v.view_depth);
        cluster_base = cid * CLUSTER_LIGHT_LIST_STRIDE;
        local_count = int(CLUSTER_LIST[cluster_base]);
        probes = probe_cluster_mask(cid);
    }
    else
    {
        local_count = LIGHTS.num_local_lights;
        probes = probe_mask_all();
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
            float3 center = pos_w;
            float3 right  = AREA_LIGHTS[ai].right_two_sided.xyz;
            float3 up     = AREA_LIGHTS[ai].up_pad.xyz;
            bool two_sided = asuint(AREA_LIGHTS[ai].right_two_sided.w) != 0u;

            // Range is a cutoff measured from the panel center, matching the
            // sphere the clustered cull bins this light with. The physical
            // falloff is already in the form factor: the panel subtends a
            // smaller solid angle further away.
            float center_dist = length(center - v.world_pos);
            float window = clamp(1.0 - center_dist / range, 0.0, 1.0);
            window = window * window;
            if (window <= 0.0)
            {
                continue;
            }

            float3 corners[4];
            corners[0] = center - right - up;
            corners[1] = center + right - up;
            corners[2] = center + right + up;
            corners[3] = center - right + up;

            // Diffuse needs no lookup: it is the polygon integral under the
            // plain clamped cosine, i.e. an identity transform.
            float3x3 identity = float3x3(
                float3(1.0, 0.0, 0.0),
                float3(0.0, 1.0, 0.0),
                float3(0.0, 0.0, 1.0));
            float diffuse_ff = ltc_evaluate(N, V, v.world_pos, identity, corners, two_sided);

            // Specular applies the fitted transform before the same integral.
            float2 lut_uv = float2(roughness, sqrt(clamp(1.0 - NdV, 0.0, 1.0)));
            lut_uv = lut_uv * LTC_LUT_SCALE + LTC_LUT_BIAS;
            float4 t1 = ltc_matrix_sample(lut_uv);
            float2 t2 = ltc_magnitude_sample(lut_uv);
            // The table stores the inverse normalized so its middle entry is
            // 1, packed as (m00, m20, m02, m22). Rows here follow the
            // mul(v, M) convention.
            float3x3 m_inv = float3x3(
                float3(t1.x, 0.0, t1.y),
                float3(0.0,  1.0, 0.0),
                float3(t1.z, 0.0, t1.w));
            float specular_ff = ltc_evaluate(N, V, v.world_pos, m_inv, corners, two_sided);
            // Schlick split baked into the table: t2.x weights the base
            // reflectance, t2.y the grazing response.
            float3 area_spec = F0 * t2.x + (1.0 - F0) * t2.y;

            float3 area_radiance = col * intens * window;
            float3 area_kd = (1.0 - F0) * (1.0 - metallic);
            Lo += area_radiance * (area_kd * albedo * diffuse_ff
                                   + area_spec * specular_ff);
            continue;
        }

        float3 L    = normalize(pos_w - v.world_pos);
        float  dist = length(pos_w - v.world_pos);
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
                atten *= sample_spot_shadow(si, v.world_pos, N, screen_xy);
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
        float3 direct = (diff + spec) * radiance * NdL;
        Lo += direct;
        if (far_w > 0.0)
        {
            // Local lights light the canopy from its facing toward the
            // viewer alone.
            Lo_under += direct * canopy_transmit(canopy, L);
            canopy_add_local_light(canopy, V, L, radiance);
        }
    }

#ifdef DAYLIGHT_SURFACE
    // The point's own sky on a level surface, for the ground and the canopy.
    float3 day_sky = day.irradiance;
#endif

    // What the surface reflects of diffuse light arriving from every side.
    float3 diffuse_albedo;
    float3 ambient;
    if (ibl_enabled)
    {
        float3 F_ibl       = fresnel_schlick(NdV, F0);
        diffuse_albedo     = (1.0 - F_ibl) * (1.0 - metallic) * albedo;
        float3 irradiance  = irradiance_sample(N);
#ifdef DAYLIGHT_SURFACE
        if (day_reach > 0.0)
        {
            irradiance = lerp(irradiance, day_sky * saturate(0.5 + 0.5 * dot(N, day_up)), day_reach);
        }
#endif
        ambient = diffuse_albedo * irradiance / PI;
    }
    else
    {
        diffuse_albedo = (1.0 - metallic) * albedo;
        ambient = (float3)(0.03) * albedo;
    }

    // Local reflection probes when the host binds a probe set and any are
    // baked (box-parallax partition of unity), else the imported environment
    // prefilter cube, each at a roughness-keyed mip. Both taps widen that mip
    // by the reflection vector's screen-space footprint (not a fixed level) at
    // grazing or distant angles. A forced LOD defeats
    // minification filtering and aliases the environment into sparkle on near
    // mirrors; flat close-up pixels have a near-zero footprint, so they keep
    // the plain roughness mip.
    float3 prefiltered;
    if (environment_specular(probes, v.world_pos, reflect(-V, N), roughness, prefiltered))
    {

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

        ambient += specular_ibl;
    }

    // The environment's share is held back where something nearby occludes
    // it, and what those surfaces reflect arrives instead; direct lighting is
    // unaffected. The occluders are the bounce's source, so the occlusion does
    // not dim the bounce itself.
    ScreenIndirect indirect = screen_indirect_at(screen_xy);
    ambient = ambient * indirect.visibility + diffuse_albedo * indirect.bounce;

    // Authored indirect-fill multiplier (IndirectLighting.ambient_intensity);
    // 1.0 is a no-op. Lifts shadow fill without touching sun-lit surfaces.
    // Under grass, the leaf above keeps part of the sky off the ground.
    ambient *= LIGHTS.ambient_intensity * canopy.ground_sky;

    float3 color = ambient + Lo + emissive;
    if (far_w > 0.0)
    {
        // Where this pixel shows the canopy rather than the blades, the canopy
        // over the ground, which takes the direct light the canopy lets
        // through; elsewhere the ground between the blades, as nearer in. The
        // blades see the sky their facing toward the viewer sees, as the
        // grass draws do.
        float3 sky = ibl_enabled ? irradiance_sample(canopy.along) / PI : (float3)(0.03);
#ifdef DAYLIGHT_SURFACE
        if (ibl_enabled && day_reach > 0.0)
        {
            sky = lerp(sky, day_sky * saturate(0.5 + 0.5 * dot(canopy.along, day_up)) / PI, day_reach);
        }
#endif
        sky = (sky * indirect.visibility + indirect.bounce) * LIGHTS.ambient_intensity;
        float3 ground = ambient + lerp(Lo, Lo_under, canopy.handed) + emissive;
        color = canopy_radiance(canopy, sky, ground);
    }

    return float4(color, albedo_samp.a);
}

