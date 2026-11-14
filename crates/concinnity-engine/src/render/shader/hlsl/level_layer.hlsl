// A level surface a ray meets analytically: the shell of a sphere about a
// center far below it, or a horizontal rectangle. What it returns toward a
// viewer is a Fresnel mirror over a body that scatters the light reaching it
// back out as a Lambertian reflector, plus the glints of the lights off its
// roughness. Pure functions of their arguments, so any pass may splice them
// whatever it binds.

#ifndef LEVEL_LAYER
#define LEVEL_LAYER

// Base reflectance of a level surface head-on.
static const float LEVEL_LAYER_F0 = 0.02;
static const float LEVEL_LAYER_INV_PI = 0.31830988618;

// The share a level surface reflects at cosine `cos_v` between its normal and
// the view, rising toward grazing by the falloff exponent `power`.
float level_layer_fresnel(float cos_v, float power)
{
    return LEVEL_LAYER_F0 + (1.0 - LEVEL_LAYER_F0) * pow(saturate(1.0 - cos_v), power);
}

// What a level surface returns toward its viewer: `reflected` by its
// `fresnel` share, and the irradiance `lit` reaching it scattered back out of
// a body of reflectance `body` by the rest.
float3 level_layer_radiance(float fresnel, float3 reflected, float3 body, float3 lit)
{
    return fresnel * reflected + (1.0 - fresnel) * body * lit * LEVEL_LAYER_INV_PI;
}

// The specular lobe of a light toward `light_dir` of `light_color` off normal
// `normal` at GGX `alpha`, seen along `view_dir`: GGX with a height-correlated
// Smith visibility term. A high light draws a compact disc; a low one spreads
// the same lobe into a path running toward the viewer.
float3 level_layer_glint(float3 normal, float3 view_dir, float alpha, float3 light_dir, float3 light_color)
{
    float n_dot_l = dot(normal, light_dir);
    if (n_dot_l <= 0.0)
    {
        return (float3)(0.0);
    }
    float3 h = normalize(light_dir + view_dir);
    float n_dot_v = max(dot(normal, view_dir), 1e-4);
    float n_dot_h = saturate(dot(normal, h));
    float v_dot_h = saturate(dot(view_dir, h));

    float a2 = alpha * alpha;
    float denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    float d = a2 / max(3.14159265358979 * denom * denom, 1e-6);

    float lambda_v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    float lambda_l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    float vis = 0.5 / max(lambda_v + lambda_l, 1e-6);

    float f = LEVEL_LAYER_F0 + (1.0 - LEVEL_LAYER_F0) * pow(1.0 - v_dot_h, 5.0);
    return light_color * (d * vis * f * n_dot_l);
}

// The most level surfaces a ray meets. Mirrors `MAX_LEVEL_LAYERS`.
static const uint LEVEL_LAYERS = 4u;

// One level surface: the shell of a sphere about a center far below it, or a
// horizontal rectangle. Layout matches `LevelLayerGpu` in render_types.
struct LevelLayer
{
    // xyz: the shell's or the rectangle's center; w: the shell's radius, 0 for
    // a rectangle.
    float4 shape;
    // xy: a rectangle's half extents along x and z; z: the camera's height
    // over the surface, formed in double precision; w: 1 when in use.
    float4 extent;
    // rgb: the reflectance of the body under the surface; w: its Fresnel
    // falloff exponent.
    float4 body;
    // x: perceptual roughness; y: GGX alpha.
    float4 surface;
};

// How far along unit `dir` a ray from `origin` travels before it meets
// `layer` from above, or a negative distance when it does not: a layer out
// of use, an origin on or under it, or a ray that passes it by. Heights are
// taken from the camera at `camera`, whose own height over the layer came
// from the host in double precision, so a shell the size of a planet still
// resolves near the camera: the shell's squared distances are formed from
// that height and the origin's small offset from the camera, never by
// subtracting two squared radii.
float level_layer_distance(LevelLayer layer, float3 camera, float3 origin, float3 dir)
{
    if (layer.extent.w < 0.5)
    {
        return -1.0;
    }
    float3 offset = origin - camera;
    float camera_height = layer.extent.z;
    float radius = layer.shape.w;
    if (radius <= 0.0)
    {
        float height = camera_height + offset.y;
        if (height <= 0.0 || dir.y >= 0.0)
        {
            return -1.0;
        }
        float t = height / -dir.y;
        float2 at = origin.xz + dir.xz * t - layer.shape.xz;
        return all(abs(at) <= layer.extent.xy) ? t : -1.0;
    }
    float3 from_center = camera - layer.shape.xyz;
    // |origin - center|^2 - radius^2, about the camera.
    float c = camera_height * (2.0 * radius + camera_height)
            + 2.0 * dot(from_center, offset) + dot(offset, offset);
    float b = dot(from_center + offset, dir);
    if (c <= 0.0 || b >= 0.0)
    {
        return -1.0;
    }
    float disc = b * b - c;
    if (disc < 0.0)
    {
        return -1.0;
    }
    return c / (sqrt(disc) - b);
}

// The nearest of `layers` a ray from `origin` along `dir` meets (see
// `level_layer_distance`), its index in `hit`; a negative distance when it
// meets none.
float level_layers_nearest(LevelLayer layers[LEVEL_LAYERS], float3 camera, float3 origin, float3 dir,
                           out uint hit)
{
    hit = 0u;
    float nearest = -1.0;
    [unroll]
    for (uint i = 0u; i < LEVEL_LAYERS; i++)
    {
        float t = level_layer_distance(layers[i], camera, origin, dir);
        if (t > 0.0 && (nearest < 0.0 || t < nearest))
        {
            nearest = t;
            hit = i;
        }
    }
    return nearest;
}

// The surface normal of `layer` at a point `p` on it.
float3 level_layer_normal(LevelLayer layer, float3 p)
{
    return layer.shape.w > 0.0 ? normalize(p - layer.shape.xyz) : float3(0.0, 1.0, 0.0);
}

#endif
