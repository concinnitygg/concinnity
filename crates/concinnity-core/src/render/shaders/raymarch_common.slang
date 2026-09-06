// The body of the raymarched SDF volume pass, spliced at a shader's
// RAYMARCH_COMMON marker: the distance-field primitive library an authored
// `map` composes, the cone-stepping marcher, and the PBR / IBL / shadow
// helpers an authored `shade` and the templates call.
//
// Nothing here spells a binding. Every resource is reached through an accessor
// the including file defines ahead of the splice, which is what lets the three
// hosts' binding models differ without the shading differing.

// Distance-field primitives, after https://iquilezles.org/articles/distfunctions/
// Kept small and well known; an authored `map` composes them.

float sdSphere(float3 p, float r) { return length(p) - r; }

float sdBox(float3 p, float3 b)
{
    float3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdRoundBox(float3 p, float3 b, float r)
{
    float3 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

float sdTorus(float3 p, float2 t)
{
    float2 q = float2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdCapsule(float3 p, float3 a, float3 b, float r)
{
    float3 pa = p - a;
    float3 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float sdPlane(float3 p, float3 n, float h) { return dot(p, n) + h; }

float opSmoothUnion(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0);
    return lerp(b, a, h) - k * h * (1.0 - h);
}

float opSmoothSubtraction(float d1, float d2, float k)
{
    float h = clamp(0.5 - 0.5 * (d2 + d1) / max(k, 1e-6), 0.0, 1.0);
    return lerp(d2, -d1, h) + k * h * (1.0 - h);
}

float opSmoothIntersection(float a, float b, float k)
{
    float h = clamp(0.5 - 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0);
    return lerp(b, a, h) + k * h * (1.0 - h);
}

// Slab ray-box intersection, returning (t_enter, t_exit). A miss leaves
// t_exit < max(0, t_enter).
float2 rayBox(float3 ro, float3 rd, float3 box_min, float3 box_max)
{
    float3 inv = 1.0 / rd;
    float3 t0 = (box_min - ro) * inv;
    float3 t1 = (box_max - ro) * inv;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    return float2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

// Four-tap central-difference gradient, normalised. `eps` has to be small
// enough for the linearisation to hold and large enough that the field does not
// return zero on both sides; 0.001 world units suits the library above.
float3 sdfNormal(float3 p, SdfParams params, float time, float eps)
{
    float3 ex = float3(eps, 0.0, 0.0);
    float3 ey = float3(0.0, eps, 0.0);
    float3 ez = float3(0.0, 0.0, eps);
    return normalize(float3(
        map(p + ex, params, time) - map(p - ex, params, time),
        map(p + ey, params, time) - map(p - ey, params, time),
        map(p + ez, params, time) - map(p - ez, params, time)));
}

// Cone-stepping sphere trace. Marches from `t_start` along `dir` until the
// field returns under the surface epsilon, `t` passes `t_max`, or the volume's
// step cap fires. `cone_ratio` is the reciprocal Lipschitz constant, 1 for the
// primitives above.
RayHit coneRaymarch(float3 origin, float3 dir, float t_start, float t_max, float time)
{
    RayHit r;
    r.t = t_start;
    r.hit = false;
    r.steps = 0;
    float t = t_start;
    int cap = min(VOL.max_steps, 256);
    float ratio = max(VOL.cone_ratio, 0.01);
    const float surface_eps = 0.001;
    for (int i = 0; i < cap; ++i)
    {
        if (t >= t_max) break;
        float3 p = origin + dir * t;
        float d = map(p, VOL.params, time);
        r.steps = i + 1;
        if (abs(d) < surface_eps)
        {
            r.t = t;
            r.hit = true;
            return r;
        }
        t += max(abs(d) * ratio, 0.001);
    }
    return r;
}

// Sample the pre-raymarch scene through a normal-perturbed screen UV. The
// perturbation is the world normal's XZ tilt scaled by `strength` (0.02 to 0.10
// suits a water surface). The result is linear-light RGB; attenuate it in the
// authored shader before writing it into `SdfSurface.transmitted`.
float3 sampleSceneRefracted(float2 frag_uv, float3 normal, float strength)
{
    return scene_sample(clamp(frag_uv + normal.xz * strength, 0.0, 1.0));
}

// Cook-Torrance GGX with Smith G and Schlick F, the same math the forward main
// pass runs, so a raymarched surface and a rasterised one agree under one sun.
// `shadow` is 1 for fully lit.
float3 shadePbrSun(SdfSurface s, float3 normal, float3 viewDir, DirLight sun, float shadow)
{
    float3 L = normalize(sun.dir_i.xyz);
    float3 H = normalize(viewDir + L);
    float NdotL = max(0.0, dot(normal, L));
    float NdotV = max(1e-3, dot(normal, viewDir));
    float NdotH = max(0.0, dot(normal, H));
    float VdotH = max(0.0, dot(viewDir, H));

    float a = max(s.roughness * s.roughness, 1e-3);
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (3.14159265 * denom * denom);

    float k = (s.roughness + 1.0) * (s.roughness + 1.0) / 8.0;
    float G = (NdotL / (NdotL * (1.0 - k) + k)) * (NdotV / (NdotV * (1.0 - k) + k));

    float3 F0 = lerp(float3(0.04), s.albedo, s.metallic);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

    float3 spec = (D * G * F) / max(4.0 * NdotL * NdotV, 1e-3);
    float3 diff = (1.0 - F) * (1.0 - s.metallic) * s.albedo / 3.14159265;
    float3 light = sun.col.xyz * sun.dir_i.w * shadow;
    return (diff + spec) * light * NdotL;
}

// Hemispheric ambient fallback, used when no EnvironmentMap is bound.
float3 shadeAmbient(SdfSurface s, float3 normal)
{
    float3 sky = float3(0.45, 0.52, 0.62);
    float3 ground = float3(0.07, 0.06, 0.05);
    float t = clamp(0.5 + 0.5 * normal.y, 0.0, 1.0);
    return s.albedo * lerp(ground, sky, t) * 0.35 + s.emissive;
}

{SHADOW_BIAS}

float raymarchHashRotation(float2 p)
{
    return frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
}

// Cascade-shadow PCF, mirroring `shadow_factor_cascaded` in
// `main_shading.slang` so a raymarched surface takes the shadow a rasterised
// one at the same point would.
float sampleSunShadow(float3 world_pos, float view_depth, float2 screen_xy)
{
    int cascade = 4;
    if (view_depth < SHADOW_UNI.cascade_splits[0]) cascade = 0;
    else if (view_depth < SHADOW_UNI.cascade_splits[1]) cascade = 1;
    else if (view_depth < SHADOW_UNI.cascade_splits[2]) cascade = 2;
    else if (view_depth < SHADOW_UNI.cascade_splits[3]) cascade = 3;
    if (cascade >= int(SHADOW_UNI.active_cascades)) return 1.0;

    float4 lc = mul(SHADOW_UNI.light_vps[cascade], float4(world_pos, 1.0));
    float3 ndc = lc.xyz / max(lc.w, 1e-6);
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0)
    {
        return 1.0;
    }

    float ref = ndc.z - cascade_depth_bias(cascade);

    // A per-pixel rotation breaks the 5x5 kernel's banding.
    float angle = raymarchHashRotation(screen_xy);
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
            sum += shadow_map_cmp(float3(uv + rot * tex_size, float(cascade)), ref);
        }
    }
    return sum / SAMPLES;
}

float2 raymarchEnvBrdfApprox(float NdV, float rough)
{
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.040, -0.040);
    float4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdV)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

float3 raymarchFresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Image-based ambient, mirroring the main pass's ambient term: the irradiance
// cube for diffuse and the prefilter cube at a roughness-selected mip for
// specular, combined through the Karis split-sum. Falls back to the hemispheric
// fill when no EnvironmentMap is bound.
float3 shadeAmbientIbl(SdfSurface s, float3 normal, float3 view_dir)
{
    if (VIEW.prefilter_mip_count <= 0.5)
    {
        return shadeAmbient(s, normal);
    }
    float NdV = max(dot(normal, view_dir), 0.0);
    float3 F0 = lerp(float3(0.04), s.albedo, s.metallic);
    float3 F_ibl = raymarchFresnelSchlick(NdV, F0);
    float3 kd_ibl = (1.0 - F_ibl) * (1.0 - s.metallic);

    float3 irradiance = irradiance_sample(RM_SKY_DIR(normal));
    float3 diffuse_ibl = kd_ibl * s.albedo * irradiance / 3.14159265;

    float3 R = reflect(-view_dir, normal);
    float lod = s.roughness * (VIEW.prefilter_mip_count - 1.0);
    float3 prefiltered = prefilter_sample_lod(RM_SKY_DIR(R), lod);
    float2 ab = raymarchEnvBrdfApprox(NdV, s.roughness);

    return diffuse_ibl + prefiltered * (F0 * ab.x + ab.y) + s.emissive;
}
