// Procedural grass: single source for every backend.
//
// GRASS_GENERATE compiles the kernel. One dispatch grows every layer in reach:
// the group's z picks the layer, its y a tile of that layer's block, and its x
// a run of 64 of the tile's cells. The group's first thread decides whether
// the tile can show a blade at all: its box, grown by how far a blade reaches,
// must meet the view frustum and the draw distance, and must not have been
// hidden behind last frame's depth (the Hi-Z pyramid, read through last
// frame's view-projection). Each thread then places its cell's blade (a hash
// of the cell's world coordinates and the layer's seed jitters it, and the
// nearest of a jittered clump lattice gives it the height, facing and lean it
// shares with its clump), roots it on the terrain's own triangles, and drops it
// where the layer's mask thins it out, past the terrain's edge, or out of view.
// With distance the field thins: a blade survives while the keep fraction
// there exceeds its hash, shrinks into the ground over a band before it is
// dropped, and the survivors widen to keep the field's coverage. Survivors
// append to their detail level's region of the visible-blade buffer. The same
// dispatch fills this frame's draw-argument slot and resets the other one for
// the next frame.
//
// On a planet (grass.ground == GRASS_GROUND_PLANET) the group's y walks the
// grass tiles of the drawn planet tiles in reach instead: each record names a
// tile's mesh in the shared vertex buffer, its model matrix in the simulated
// frame, and the block of its grass tiles this run covers. A blade's cell is
// counted across its cube face, so its hash is the planet's, never the
// frame's; it roots on the tile's own triangles, stands along the radial up,
// and is thinned by the area its cell covers and by the meadow noise laid
// over the sphere (render::grass::planet). The blade carries its up to the
// draws, which stand the curve along it. The wind's gusts and last frame's
// bend field are read through maps the host carries across every move of the
// simulated frame, so the same blade in the same wind reads the same values
// on either side of a move.
//
// GRASS_GENERATE also runs a second time for the nearest shadow cascade, under
// its own block: the frustum is the cascade's light frustum, the cull distance
// is the cast distance, there is no occlusion test, and every blade it keeps
// goes to the coarsest level's region of its own buffer. The blades it keeps
// are the ones the view keeps, so the shadow and the blades agree.
//
// GRASS_BEND compiles the bend pass: one thread per cell of the camera-centered
// field trampled blades are pressed by (render::grass::bend). Each cell keeps
// last frame's bend, relaxed toward upright, when last frame's window held it,
// and takes the strongest press of any footprint stamped this frame. The kernel
// reads the field at each blade's root, this frame's and last frame's, into the
// blade, so every draw bends it the same way and its motion includes the
// trampling.
//
// The other entries draw the blades with one indirect draw per detail level: a
// strip of 15, 9 or 5 vertices per blade, bent along a quadratic Bezier by the
// blade's lean and the wind. Level k's draw starts at vertex id
// k * GRASS_LOD_VERTEX_STRIDE, which is how the vertex stage knows its level;
// approaching the next level, the vertices that level drops fold onto its
// strip, so the switch moves nothing. (DirectX passes the first vertex in its
// b0 root constant instead, see grass_draw_vertex.) The pre-pass pair writes the G-buffer,
// the blade's motion included, and runs the bend a second time at the previous
// frame's clock and camera so a swaying blade reprojects to where it was
// drawn. The lit pair shades the blade into the main pass. Both splice the
// main pass's resources, so they draw inside those passes on the bindings
// already there; the grass block and the blades add one vertex-stage pair of
// their own:
//
//   CN_BACKEND_METAL   buffer(19) params, buffer(20) blades
//   CN_BACKEND_DIRECTX b7 params, t23 blades in both root signatures
//   Vulkan             set 2 (main) or set 3 (pre-pass), bindings 0 and 1
//
// GRASS_SHADOW compiles the depth-only draw of the cascade's blades with the
// coarsest strip, through the cascade's light view-projection, swaying at the
// view's clock. It binds only its block and its blades, at b0 and t1.

{WIND}

// Mirrors `GrassLayerGpu` in render::uniforms::grass (144 B).
struct GrassLayer
{
    // The terrain's min x, min z, max x, max z.
    float4 rect;
    int2 tile_origin;
    uint2 tile_count;
    float base_y;
    float min_y;
    float max_y;
    uint grid_resolution;
    uint heights_offset;
    uint mask_offset;
    // Width in the low 16 bits, height in the high 16; 0 for no mask.
    uint mask_size;
    uint seed;
    float height;
    float height_variance;
    float width;
    float clump_size;
    float stiffness;
    float color_variation;
    float cell_size;
    uint cells_per_side;
    // xyz = linear RGB at the root, w = how far a blade reaches from its root.
    float4 root_color_reach;
    // xyz = linear RGB at the tip, w = how far the field's color drifts.
    float4 tip_color_macro;
    float clumping;
    // The side-on and top-down axes of the ellipse a blade's projection along
    // a light follows (see grass_light_through).
    float2 light_projection;
    float _pad0;
};

// Matches MAX_GRASS_LAYERS in render::uniforms::grass.
#define GRASS_MAX_LAYERS 16

// Matches GRASS_LOD_COUNT and GRASS_LOD_VERTEX_STRIDE in render::grass::lod.
#define GRASS_LOD_COUNT 3
#define GRASS_LOD_VERTEX_STRIDE 16

// Mirrors `GrassParams` in render::uniforms::grass.
struct GrassParams
{
    // xyz = camera position, w = draw distance.
    float4 cam_pos_distance;
    // (normal, d) world-space planes; inside is dot(normal, p) + d >= 0.
    float4 frustum[6];
    float4 wind;
    float4 wind_gust;
    // The view-projection last frame's depth, and so the pyramid, was drawn
    // through.
    float4x4 prev_vp;
    float2 hiz_size;
    uint hiz_mip_count;
    uint hiz_enabled;
    // Each detail level's first blade and blade count in the visible-blade
    // buffer; w unused.
    uint4 lod_base;
    uint4 lod_capacity;
    // x, y = where the second and third levels start, z = the share of a
    // level's reach its vertices fold over, w = where thinning starts.
    float4 lod_distances;
    // x = the fewest blades thinning keeps, y = the shrink band, z = where the
    // fade starts, w = the widest a blade grows.
    float4 thinning;
    uint layer_count;
    uint args_slot;
    float tile_size;
    // How far from the camera a blade may be placed.
    float cull_distance;
    // xy = the bend field window's corner cell this frame, zw = last frame.
    int4 bend_window;
    float bend_cell_size;
    uint bend_resolution;
    // The half of the bend buffer this frame's field is in.
    uint bend_half;
    uint bend_prev_valid;
    // The shadow block's light view-projection, and the direction toward the
    // light (xyz) with the clock the shadow draw sways at (w).
    float4x4 shadow_vp;
    float4 shadow_light;
    // xyz = the planet's center in the simulated frame, w = the meadow noise's
    // lattice cells per unit of direction.
    float4 planet_center_scale;
    // GRASS_GROUND_TERRAIN or GRASS_GROUND_PLANET.
    uint ground;
    uint planet_tile_base;
    uint planet_tile_count;
    // The planet's radius, in meters.
    float planet_radius;
    // Rows over (x, y, z, 1): the ground-plane coordinates the wind's gusts
    // are laid out in.
    float4 wind_map[2];
    // Rows over (x, y, z, 1): a point's x and z in last frame's frame.
    float4 bend_prev_map[2];
    // Each layer's density on a planet, four layers to a row.
    float4 planet_density[4];
    GrassLayer layers[GRASS_MAX_LAYERS];
};

// Matches GRASS_GROUND_TERRAIN and GRASS_GROUND_PLANET in
// render::uniforms::grass.
#define GRASS_GROUND_TERRAIN 0u
#define GRASS_GROUND_PLANET 1u

{TILE_MORPH_TYPES}

// Mirrors `PlanetGrassTileGpu` in render::uniforms::grass (144 B).
struct PlanetGrassTile
{
    // The model matrix's first three rows: local = dot(row.xyz, v) + row.w.
    float4 model[3];
    uint vertex_base;
    // Face in the low 8 bits, quadtree level in the next 8.
    uint face_level;
    uint2 tile;
    // The run's first group row on this tile.
    uint first_group;
    // The block of grass tiles: origin and count, x low 16 bits, y high.
    uint block_origin;
    uint block_count;
    // Grass tiles along each side of the whole tile.
    uint grass_tiles;
    // The lowest and highest the surface lies along the up at the origin.
    float2 radial;
    float2 _pad;
    // How the tile's vertices morph, as its draw's object record carries it.
    GpuTileMorph tile_morph;
};

// Mirrors `GpuGrassBlade` in render::uniforms::grass (48 B).
struct GrassBlade
{
    // xyz = root, w = facing angle in radians.
    float4 root_facing;
    // x = height and y = root width (f32 bits), z = static lean [x, z] as two
    // halves, w = the blade's bits (see grass_blade_bits).
    uint4 shape;
    // How far it is trampled, [x, z] as two halves: x this frame, y last frame;
    // z = the x and z of the up it stands along as two halves; w = the field's
    // drift where it stands (see grass_macro) as two halves.
    uint4 bend;
};

#if defined(GRASS_BEND) || defined(GRASS_GENERATE) || defined(GRASS_SHADOW)
{GRASS_FIELD}
#else
{MAIN_RESOURCES}
#endif

// Strip pairs at detail level `lod`: every level keeps every other pair of the
// last, so its pairs sit at LOD0 pair indices that are multiples of 2^lod.
uint grass_lod_pairs(uint lod)
{
    return lod == 0u ? 7u : (8u >> lod);
}

uint grass_lod_vertices(uint lod)
{
    return 2u * grass_lod_pairs(lod) + 1u;
}

// The detail level of a blade `dist` meters from the camera. Mirrors
// render::grass::lod::lod_at.
uint grass_lod_at(GrassParams p, float dist)
{
    return dist < p.lod_distances.x ? 0u : (dist < p.lod_distances.y ? 1u : 2u);
}

// The leaf area a blade's bits resolve, in square meters of blade per square
// meter of ground.
static const float GRASS_LEAF_AREA_STEP = 0.25;

// A blade's bits: the leaf area around it in GRASS_LEAF_AREA_STEPs (4 bits),
// 12 of its clump's hash, 12 of its own, and the slot of the layer it belongs
// to, so the draws can vary hue per clump and per blade, read the layer's look
// and shade the blade by the leaf around it.
uint grass_blade_bits(uint clump_hash, uint blade_hash, uint layer_slot, float leaf_area)
{
    uint leaf = min(uint(max(leaf_area, 0.0) / GRASS_LEAF_AREA_STEP + 0.5), 15u);
    return (leaf << 28u) | ((clump_hash & 0xfffu) << 16u) | ((blade_hash & 0xfffu) << 4u)
         | (layer_slot & 0xfu);
}

uint grass_bits_layer(uint bits)
{
    return bits & 0xfu;
}

float grass_bits_leaf_area(uint bits)
{
    return float(bits >> 28u) * GRASS_LEAF_AREA_STEP;
}

uint grass_pack_half2(float2 v)
{
    return f32tof16(v.x) | (f32tof16(v.y) << 16u);
}

float2 grass_unpack_half2(uint v)
{
    return float2(f16tof32(v & 0xffffu), f16tof32(v >> 16u));
}

// `v` turned by the shortest arc that takes +Y onto the unit `up`.
float3 grass_turn_up(float3 up, float3 v)
{
    float3 k = float3(up.z, 0.0, -up.x);
    return v * up.y + cross(k, v) + k * (dot(k, v) / (1.0 + up.y));
}

// The x and z of a point `p` of this frame through `rows`.
float2 grass_map(float4 rows[2], float3 p)
{
    return float2(dot(rows[0], float4(p, 1.0)), dot(rows[1], float4(p, 1.0)));
}

#if defined(GRASS_BEND)

// Matches MAX_GRASS_STAMPS in render::uniforms::grass.
#define GRASS_MAX_STAMPS 64

// Matches GRASS_STAMP_CORE, GRASS_STAMP_REACH and GRASS_TRAMPLE_BEND in
// render::grass::bend.
static const float GRASS_STAMP_CORE = 0.8;
static const float GRASS_STAMP_REACH = 1.5;
static const float GRASS_TRAMPLE_BEND = 0.85;

// Mirrors `GrassBendParams` in render::uniforms::grass.
struct GrassBendParams
{
    // xy = the window's corner cell this frame, zw = last frame.
    int4 window;
    uint resolution;
    float cell_size;
    float decay;
    uint stamp_count;
    uint write_half;
    uint prev_valid;
    uint2 _pad;
    // Rows over a cell center's (x, z, 1): its x and z in last frame's frame.
    float4 prev_map[2];
    // Center x, z, radius and strength of each footprint.
    float4 stamps[GRASS_MAX_STAMPS];
};

[[vk::binding(0, 0)]] ConstantBuffer<GrassBendParams> bend : register(b0);
// Two halves of one packed bend per cell, one per frame parity.
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> bend_field : register(u1);

// The press footprint `stamp` gives the blades at world `xz`. Mirrors
// render::grass::bend::stamp_bend.
float2 grass_stamp_bend(float4 stamp, float2 xz)
{
    float2 d = xz - stamp.xy;
    float dist = length(d);
    float inner = stamp.z * GRASS_STAMP_CORE;
    float outer = stamp.z * GRASS_STAMP_REACH;
    float falloff = 1.0 - smoothstep(inner, max(outer, inner + 1e-4), dist);
    float2 dir = dist > 1e-4 ? d / dist : float2(1.0, 0.0);
    return dir * (GRASS_TRAMPLE_BEND * stamp.w * falloff);
}

// Last frame's bend of cell `c`; a cell outside last frame's window stands
// upright.
float2 grass_bend_prev_cell(int2 c)
{
    uint n = bend.resolution;
    int2 prev = bend.window.zw;
    if (any(c < prev) || any(c >= prev + int(n)))
    {
        return (float2)(0.0);
    }
    uint2 s = uint2(c & int(n - 1u));
    return grass_unpack_half2(bend_field[(1u - bend.write_half) * n * n + s.y * n + s.x]);
}

// Last frame's field at last frame's `xz`, bilinear between cell centers: at a
// cell's center, exactly that cell.
float2 grass_bend_prev(float2 xz)
{
    float2 g = xz / bend.cell_size - 0.5;
    int2 c = int2(floor(g));
    float2 f = g - float2(c);
    float2 b00 = grass_bend_prev_cell(c);
    float2 b10 = grass_bend_prev_cell(c + int2(1, 0));
    float2 b01 = grass_bend_prev_cell(c + int2(0, 1));
    float2 b11 = grass_bend_prev_cell(c + int2(1, 1));
    return lerp(lerp(b00, b10, f.x), lerp(b01, b11, f.x), f.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void grass_bend(uint3 tid : SV_DispatchThreadID)
{
    uint n = bend.resolution;
    uint slot = tid.x;
    if (slot >= n * n)
    {
        return;
    }
    // The world cell this slot holds: the window's cells wrap onto the slots,
    // so a cell keeps its slot while the window scrolls over it. Mirrors
    // BendWindow::cell_at_slot.
    int2 s = int2(slot % n, slot / n);
    int2 origin = bend.window.xy;
    int2 cell = origin + ((s - origin) & int(n - 1u));
    float2 value = (float2)(0.0);
    float2 xz = (float2(cell) + 0.5) * bend.cell_size;
    if (bend.prev_valid != 0u)
    {
        // Where this cell's center was last frame: itself, unless the
        // simulated frame moved in between.
        float2 was = float2(dot(bend.prev_map[0].xyz, float3(xz, 1.0)),
                            dot(bend.prev_map[1].xyz, float3(xz, 1.0)));
        value = grass_bend_prev(was) * bend.decay;
    }
    for (uint i = 0u; i < bend.stamp_count; i++)
    {
        float2 pressed = grass_stamp_bend(bend.stamps[i], xz);
        if (dot(pressed, pressed) > dot(value, value))
        {
            value = pressed;
        }
    }
    bend_field[bend.write_half * n * n + slot] = grass_pack_half2(value);
}

#elif defined(GRASS_GENERATE)

{DEPTH_CONVENTION}

[[vk::binding(0, 0)]] ConstantBuffer<GrassParams> grass : register(b0);
[[vk::binding(1, 0)]] RWStructuredBuffer<GrassBlade> blades_out : register(u1);
// Two slots of one draw per detail level, each (vertex count, instance count,
// first vertex, first instance).
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> draw_args : register(u2);
// Every terrain's heights, each run row-major (see GrassGroundBuffers).
[[vk::binding(3, 0)]] StructuredBuffer<float> terrain_heights : register(t3);
// Every density mask's texels, four to a word, low byte first.
[[vk::binding(4, 0)]] StructuredBuffer<uint> grass_masks : register(t4);
// Last frame's depth pyramid, read by texel coordinate only. Vulkan reads it
// through the Hi-Z read set the draw cull binds as set 1.
[[vk::binding(0, 1)]] Texture2D<float> grass_hiz : register(t5);
// The bend field, both halves (see grass_bend).
[[vk::binding(5, 0)]] StructuredBuffer<uint> grass_bend_field : register(t6);
// The drawn planet tiles this frame's runs read (see PlanetGrassTile).
[[vk::binding(6, 0)]] StructuredBuffer<PlanetGrassTile> planet_tiles : register(t7);
// The shared vertex stream every mesh draws from: 14 floats a vertex,
// position first, then normal. DirectX binds it as a raw view.
#ifdef CN_BACKEND_DIRECTX
ByteAddressBuffer mesh_vertices : register(t8);
float mesh_float(uint i) { return asfloat(mesh_vertices.Load(i * 4u)); }
#else
[[vk::binding(7, 0)]] StructuredBuffer<float> mesh_vertices : register(t8);
float mesh_float(uint i) { return mesh_vertices[i]; }
#endif

{TILE_MORPH}

// Matches GRASS_GROUP_SIZE in render::grass::tiles.
#define GRASS_GROUP_SIZE 64

// Words in one slot of the draw arguments: four per detail level.
#define GRASS_ARGS_SLOT_WORDS (4u * GRASS_LOD_COUNT)

// How far a blade leans downhill per unit of the ground normal's tilt.
static const float GRASS_SLOPE_LEAN = 0.6;

groupshared uint gs_count[GRASS_LOD_COUNT];
groupshared uint gs_base[GRASS_LOD_COUNT];
groupshared uint gs_tile_visible;
// On a planet: the record of the drawn tile this group grows on, and the
// grass tile within it.
groupshared uint gs_planet_tile;
groupshared uint2 gs_grass_tile;

float hiz_load(int3 texel_mip)
{
    return grass_hiz.Load(texel_mip);
}

{HIZ_TEST}

bool grass_outside(float3 center, float radius)
{
    [unroll]
    for (uint i = 0u; i < 6u; i++)
    {
        if (dot(grass.frustum[i].xyz, center) + grass.frustum[i].w < -radius)
        {
            return true;
        }
    }
    return false;
}

// Whether every blade of a tile whose box is `bmin..bmax` was hidden last
// frame: the box, seen through last frame's view-projection, lies wholly in
// last frame's view and in front of its camera, and behind the farthest depth
// the pyramid holds over it. A box any part of which last frame did not see is
// kept, so turning the camera, or a tile entering at a screen edge, never
// drops a blade.
bool grass_tile_occluded(float3 bmin, float3 bmax)
{
    if (grass.hiz_enabled == 0u)
    {
        return false;
    }
    float2 ndc_min = (float2)(1e30);
    float2 ndc_max = (float2)(-1e30);
    float nearest = DEPTH_FAR;
    [unroll]
    for (uint i = 0u; i < 8u; i++)
    {
        float3 corner = float3((i & 1u) != 0u ? bmax.x : bmin.x,
                               (i & 2u) != 0u ? bmax.y : bmin.y,
                               (i & 4u) != 0u ? bmax.z : bmin.z);
        float4 clip = mul(grass.prev_vp, float4(corner, 1.0));
        if (clip.w <= 0.0)
        {
            return false;
        }
        float3 ndc = clip.xyz / clip.w;
        ndc_min = min(ndc_min, ndc.xy);
        ndc_max = max(ndc_max, ndc.xy);
        nearest = depth_closer(nearest, ndc.z);
    }
    if (any(ndc_min < -1.0) || any(ndc_max > 1.0))
    {
        return false;
    }
    return hiz_rect_occluded(ndc_min, ndc_max, nearest, grass.hiz_size, grass.hiz_mip_count);
}

// Whether anything in the box `bmin..bmax` can be seen: it meets the frustum
// and the draw distance, and was not hidden last frame.
bool grass_box_visible(float3 bmin, float3 bmax)
{
    [unroll]
    for (uint i = 0u; i < 6u; i++)
    {
        float3 n = grass.frustum[i].xyz;
        float3 p = float3(n.x >= 0.0 ? bmax.x : bmin.x,
                          n.y >= 0.0 ? bmax.y : bmin.y,
                          n.z >= 0.0 ? bmax.z : bmin.z);
        if (dot(n, p) + grass.frustum[i].w < 0.0)
        {
            return false;
        }
    }
    float3 cam = grass.cam_pos_distance.xyz;
    float3 nearest = clamp(cam, bmin, bmax);
    if (distance(nearest, cam) > grass.cull_distance)
    {
        return false;
    }
    return !grass_tile_occluded(bmin, bmax);
}

// Whether any blade rooted in `tile` can be seen: its box, spanning the
// terrain's heights and grown by how far a blade reaches, is visible. Mirrors
// render::grass::tile_bounds.
bool grass_tile_visible(GrassLayer L, int2 tile)
{
    float reach = L.root_color_reach.w;
    float2 lo = float2(tile) * grass.tile_size - reach;
    float2 hi = float2(tile + 1) * grass.tile_size + reach;
    return grass_box_visible(float3(lo.x, L.min_y - reach, lo.y),
                             float3(hi.x, L.max_y + reach, hi.y));
}

// Where a blade grows, whatever ground it grows on: its root, the ground's
// normal and the up it stands along, its place on the ground's own plane (for
// the clump lattice) with the plane's two axes in the frame, the clump it
// belongs to, its fixed ground point (for the field's slow drift), its hash,
// the share of the layer's density that grows there, and the leaf area around
// it.
struct GrassSite
{
    float3 root;
    float3 normal;
    float3 up;
    float2 plane;
    float3 plane_u;
    float3 plane_v;
    uint hash;
    uint clump_hash;
    float2 clump_center;
    float3 ground_point;
    float density;
    float leaf_area;
};

// The ground under world `xz` on layer `L`'s terrain: the unit normal (xyz)
// blended across the cell, and the world height (w) on the grid triangle the
// point stands over, which is the triangle the terrain mesh draws and its
// collider holds. Mirrors GrassGround::surface_at.
float4 grass_ground(GrassLayer L, float2 xz)
{
    float n = float(L.grid_resolution);
    float2 size = L.rect.zw - L.rect.xy;
    float2 g = clamp((xz - L.rect.xy) / size * n, 0.0, n);
    float2 cell = min(floor(g), n - 1.0);
    float2 f = g - cell;
    uint side = L.grid_resolution + 1u;
    uint i = L.heights_offset + uint(cell.y) * side + uint(cell.x);
    float h00 = terrain_heights[i];
    float h10 = terrain_heights[i + 1u];
    float h01 = terrain_heights[i + side];
    float h11 = terrain_heights[i + side + 1u];
    float h = f.x + f.y <= 1.0
            ? h00 + (h10 - h00) * f.x + (h01 - h00) * f.y
            : h11 + (h01 - h11) * (1.0 - f.x) + (h10 - h11) * (1.0 - f.y);
    float dx = ((h10 - h00) * (1.0 - f.y) + (h11 - h01) * f.y) * n / size.x;
    float dz = ((h01 - h00) * (1.0 - f.x) + (h11 - h10) * f.x) * n / size.y;
    return float4(normalize(float3(-dx, 1.0, -dz)), L.base_y + h);
}

// Mask texel `i`, in [0, 1].
float grass_mask_texel(uint i)
{
    return float((grass_masks[i >> 2u] >> (8u * (i & 3u))) & 0xffu) / 255.0;
}

// How densely layer `L` grows at normalized `st` across its terrain, in
// [0, 1]: its mask bilinearly filtered, or 1 with no mask. Mirrors
// GrassMask::density_at.
float grass_mask_density(GrassLayer L, float2 st)
{
    if (L.mask_size == 0u)
    {
        return 1.0;
    }
    uint2 size = uint2(L.mask_size & 0xffffu, L.mask_size >> 16u);
    float2 f = saturate(st) * float2(size - 1u);
    uint2 p0 = uint2(floor(f));
    uint2 p1 = min(p0 + 1u, size - 1u);
    float2 s = f - float2(p0);
    uint base = L.mask_offset;
    float t00 = grass_mask_texel(base + p0.y * size.x + p0.x);
    float t10 = grass_mask_texel(base + p0.y * size.x + p1.x);
    float t01 = grass_mask_texel(base + p1.y * size.x + p0.x);
    float t11 = grass_mask_texel(base + p1.y * size.x + p1.x);
    float top = t00 + (t10 - t00) * s.x;
    float bottom = t01 + (t11 - t01) * s.x;
    return top + (bottom - top) * s.y;
}

// The clump `xz` belongs to on layer `L`, salted by `salt`, and where a blade
// placed at `xz` grows once drawn toward its center: returns the clump's hash
// and writes its center, the blade's place and how crowded the blades grow
// there, as a multiple of the layer's mean.
uint grass_clump(GrassLayer L, float2 xz, uint salt, out float2 center, out float2 gathered,
                 out float crowding)
{
    float spot;
    uint h = grass_clump_at(xz, L.clump_size, L.seed, salt, center, spot);
    gathered = center + (xz - center) * grass_gather_scale(spot, L.clumping);
    crowding = grass_crowding(spot, L.clumping);
    return h;
}

// The leaf area around blades of layer `L` growing at `density` blades per
// square meter, crowded by `crowding`.
float grass_site_leaf_area(GrassLayer L, float density, float crowding)
{
    return density * crowding * L.height * L.width * (1.0 - 1.0 / (1.0 + GRASS_WIDTH_TAPER));
}

// The bend of the field's cell `cell` in half `half_index` of the window whose
// corner cell is `origin`; a cell outside the window stands upright.
float2 grass_field_cell(uint half_index, int2 origin, int2 cell)
{
    uint n = grass.bend_resolution;
    if (any(cell < origin) || any(cell >= origin + int(n)))
    {
        return (float2)(0.0);
    }
    uint2 s = uint2(cell & int(n - 1u));
    return grass_unpack_half2(grass_bend_field[half_index * n * n + s.y * n + s.x]);
}

// The field's bend at world `xz`, bilinear between cell centers.
float2 grass_field_bend(uint half_index, int2 origin, float2 xz)
{
    float2 g = xz / grass.bend_cell_size - 0.5;
    int2 c = int2(floor(g));
    float2 f = g - float2(c);
    float2 b00 = grass_field_cell(half_index, origin, c);
    float2 b10 = grass_field_cell(half_index, origin, c + int2(1, 0));
    float2 b01 = grass_field_cell(half_index, origin, c + int2(0, 1));
    float2 b11 = grass_field_cell(half_index, origin, c + int2(1, 1));
    return lerp(lerp(b00, b10, f.x), lerp(b01, b11, f.x), f.y);
}

// The fraction of the field distance thinning keeps at `dist`. Mirrors
// render::grass::lod::thinning.
float grass_thinning(float dist)
{
    float full = grass.lod_distances.w;
    return dist <= full ? 1.0 : clamp(full / dist, grass.thinning.x, 1.0);
}

// How much of the field is left at `dist` as it fades out toward the draw
// distance. Mirrors render::grass::lod::fade.
float grass_fade(float dist)
{
    float far = grass.cam_pos_distance.w;
    return saturate((far - dist) / max(far - grass.thinning.z, 1e-3));
}

// The ramp's antiderivative render::grass::lod::coverage integrates with.
float grass_shrink_ramp(float t, float band)
{
    return t <= 0.0 ? 0.0 : (t <= band ? t * t / (2.0 * band) : t - 0.5 * band);
}

// The mean blade height scale over every candidate at keep fraction `keep`.
// Mirrors render::grass::lod::coverage.
float grass_coverage(float keep)
{
    if (keep >= 1.0)
    {
        return 1.0;
    }
    float span = 1.0 - grass.thinning.y;
    float band = max(grass.thinning.y * keep, 1e-6);
    return (grass_shrink_ramp(keep, band) - grass_shrink_ramp(keep - span, band)) / span;
}

// The site of world cell `cell` on layer `L`'s terrain. False when it falls
// off the terrain.
bool grass_terrain_site(GrassLayer L, int2 cell, out GrassSite site)
{
    site = (GrassSite)0;
    uint h = grass_hash(grass_hash2(cell) ^ L.seed);
    float2 placed = (float2(cell) + float2(grass_unit(h, 3u), grass_unit(h, 4u))) * L.cell_size;
    float2 xz;
    float crowding;
    site.clump_hash = grass_clump(L, placed, 0u, site.clump_center, xz, crowding);
    if (any(xz < L.rect.xy) || any(xz >= L.rect.zw))
    {
        return false;
    }
    float2 st = (xz - L.rect.xy) / (L.rect.zw - L.rect.xy);
    float4 ground = grass_ground(L, xz);
    site.root = float3(xz.x, ground.w, xz.y);
    site.normal = ground.xyz;
    site.up = float3(0.0, 1.0, 0.0);
    site.plane = xz;
    site.plane_u = float3(1.0, 0.0, 0.0);
    site.plane_v = float3(0.0, 0.0, 1.0);
    site.hash = h;
    site.ground_point = float3(xz.x, 0.0, xz.y);
    site.density = grass_mask_density(L, st);
    site.leaf_area = grass_site_leaf_area(L, site.density / (L.cell_size * L.cell_size), crowding);
    return true;
}

// ---- A planet's ground ----

float3 grass_tile_point(PlanetGrassTile T, float3 v)
{
    return float3(dot(T.model[0], float4(v, 1.0)), dot(T.model[1], float4(v, 1.0)),
                  dot(T.model[2], float4(v, 1.0)));
}

float3 grass_tile_vector(PlanetGrassTile T, float3 v)
{
    return float3(dot(T.model[0].xyz, v), dot(T.model[1].xyz, v), dot(T.model[2].xyz, v));
}

// Grid corner `c` of planet tile `T`, at `p` on its raw mesh, morphed toward
// the tile's parent as the tile's draw morphs it, `distance` meters from the
// camera. The morph is a fraction of a cell's curvature, so the grass takes
// one distance for the four corners of a cell and leaves their normals as
// they are.
float3 grass_tile_corner(PlanetGrassTile T, uint2 c, float3 p, float distance)
{
    if (((c.x | c.y) & 1u) == 0u)
    {
        return p;
    }
    float k = tile_morph_factor(T.tile_morph, c, distance);
    return k > 0.0 ? lerp(p, tile_parent_position(T.vertex_base, c), k) : p;
}

// A point of planet tile `T`'s surface: the tile's own triangle under (fs, ft)
// across it, each in [0, 1], interpolated in the mesh's frame; a little past
// an edge, the edge's triangle runs on. The cell splits along the same
// diagonal the mesh's indices do.
struct GrassMeshPoint
{
    float3 position;
    float3 normal;
    // The triangle's edges along the tile's s and t, one mesh cell long.
    float3 edge_s;
    float3 edge_t;
};

GrassMeshPoint grass_mesh_point(PlanetGrassTile T, float2 f)
{
    float n = float(PLANET_TILE_CELLS);
    float2 g = f * n;
    float2 cell = clamp(floor(g), 0.0, n - 1.0);
    float2 r = g - cell;
    uint side = PLANET_TILE_CELLS + 1u;
    uint i00 = T.vertex_base + uint(cell.y) * side + uint(cell.x);
    TileVertex c00 = tile_vertex(i00);
    TileVertex c10 = tile_vertex(i00 + 1u);
    TileVertex c01 = tile_vertex(i00 + side);
    TileVertex c11 = tile_vertex(i00 + side + 1u);
    float3 p00 = c00.position;
    float3 p10 = c10.position;
    float3 p01 = c01.position;
    float3 p11 = c11.position;
    if (T.tile_morph.eye_range.w > 0.0)
    {
        float d = length(grass_tile_point(T, p00) - T.tile_morph.eye_range.xyz);
        uint2 c = uint2(cell);
        p00 = grass_tile_corner(T, c, p00, d);
        p10 = grass_tile_corner(T, c + uint2(1u, 0u), p10, d);
        p01 = grass_tile_corner(T, c + uint2(0u, 1u), p01, d);
        p11 = grass_tile_corner(T, c + uint2(1u, 1u), p11, d);
    }
    GrassMeshPoint m;
    if (r.x + r.y <= 1.0)
    {
        m.position = p00 + (p10 - p00) * r.x + (p01 - p00) * r.y;
        m.edge_s = p10 - p00;
        m.edge_t = p01 - p00;
    }
    else
    {
        m.position = p11 + (p01 - p11) * (1.0 - r.x) + (p10 - p11) * (1.0 - r.y);
        m.edge_s = p11 - p01;
        m.edge_t = p11 - p10;
    }
    m.normal = lerp(lerp(c00.normal, c10.normal, r.x), lerp(c01.normal, c11.normal, r.x), r.y);
    return m;
}

// Whether any blade rooted in grass tile `g` of planet tile `T` can be seen:
// the box over the grass tile's corners on the surface, spanning the whole
// tile's height range along its origin's up, grown by how far a blade reaches.
bool grass_planet_tile_visible(PlanetGrassTile T, uint2 g, float reach)
{
    float3 origin = float3(T.model[0].w, T.model[1].w, T.model[2].w);
    float3 up = normalize(origin - grass.planet_center_scale.xyz);
    float3 bmin = (float3)(1e30);
    float3 bmax = (float3)(-1e30);
    [unroll]
    for (uint i = 0u; i < 4u; i++)
    {
        float2 f = (float2(g) + float2(i & 1u, i >> 1u)) / float(T.grass_tiles);
        float3 c = grass_tile_point(T, grass_mesh_point(T, f).position);
        float h = dot(c - origin, up);
        float3 lo = c + up * (T.radial.x - h);
        float3 hi = c + up * (T.radial.y - h);
        bmin = min(bmin, min(lo, hi));
        bmax = max(bmax, max(lo, hi));
    }
    return grass_box_visible(bmin - reach, bmax + reach);
}

// The site of cell `cell` of grass tile `g` of planet tile `T` for layer `L`,
// the layer's density on the planet being `density`. Mirrors
// PlanetGrassGrid::face_cell for the cell's place on its face.
GrassSite grass_planet_site(GrassLayer L, PlanetGrassTile T, uint2 g, uint2 cell, float density)
{
    GrassSite site = (GrassSite)0;
    uint k = L.cells_per_side;
    uint face = T.face_level & 0xffu;
    uint level = (T.face_level >> 8u) & 0xffu;
    int2 face_cell = (int2(T.tile * T.grass_tiles + g)) * int(k) + int2(cell);
    uint face_salt = grass_face_salt(face);
    uint h = grass_hash(grass_hash2(face_cell) ^ face_salt ^ L.seed);
    float2 jitter = float2(grass_unit(h, 3u), grass_unit(h, 4u));
    float cells = float(T.grass_tiles * k);
    GrassMeshPoint m = grass_mesh_point(T, (float2(g * k + cell) + jitter) / cells);

    site.root = grass_tile_point(T, m.position);
    site.normal = normalize(grass_tile_vector(T, m.normal));
    site.up = normalize(site.root - grass.planet_center_scale.xyz);
    float3 edge_s = grass_tile_vector(T, m.edge_s);
    float3 edge_t = grass_tile_vector(T, m.edge_t);
    site.plane_u = normalize(edge_s);
    site.plane_v = normalize(edge_t);
    site.hash = h;

    // Most far candidates fall to distance thinning, so that is tested before
    // the clump lattice and the meadow noise: a dropped candidate keeps
    // density 0 and never pays for them. grass_grow tests again at the root
    // the blade is drawn to.
    float dist = distance(site.root, grass.cam_pos_distance.xyz);
    float keep = grass_thinning(dist) * grass_fade(dist);
    if (dist > grass.cull_distance || grass_unit(h, 5u) * (1.0 - grass.thinning.y) >= keep)
    {
        return site;
    }

    // Drawn toward its clump's center, the blade roots on the tile's surface
    // there instead; the surface runs on past the tile's edge.
    float2 placed = (float2(face_cell) + jitter) * L.cell_size;
    float crowding;
    site.clump_hash = grass_clump(L, placed, face_salt, site.clump_center, site.plane, crowding);
    if (L.clumping > 0.0)
    {
        jitter += (site.plane - placed) / L.cell_size;
        m = grass_mesh_point(T, (float2(g * k + cell) + jitter) / cells);
        site.root = grass_tile_point(T, m.position);
        site.normal = normalize(grass_tile_vector(T, m.normal));
        site.up = normalize(site.root - grass.planet_center_scale.xyz);
    }

    // The share of a blade's ground this cell holds, then the meadow.
    float per_mesh_cell = cells / float(PLANET_TILE_CELLS);
    float area = length(cross(edge_s, edge_t)) / (per_mesh_cell * per_mesh_cell);
    float face_cells = float((T.grass_tiles << level) * k);
    float2 st = (float2(face_cell) + jitter) / face_cells * 2.0 - 1.0;
    float3 dir = grass_face_direction(face, st.x, st.y);
    site.ground_point = dir * grass.planet_radius;
    site.density = saturate(density * area)
                 * grass_meadow_density(dir, grass.planet_center_scale.w, L.seed);
    site.leaf_area = grass_site_leaf_area(L, site.density / max(area, 1e-6), crowding);
    return site;
}

// Place the blade at `site` on layer `L` (frame slot `slot`) and pick its
// detail level. False where the site's density thins it out, where distance
// thins it out, or out of view.
bool grass_grow(GrassLayer L, uint slot, GrassSite site, out GrassBlade blade, out uint lod)
{
    blade = (GrassBlade)0;
    lod = 0u;
    uint h = site.hash;
    if (grass_unit(h, 17u) >= site.density)
    {
        return false;
    }
    float3 root = site.root;
    float dist = distance(root, grass.cam_pos_distance.xyz);
    if (dist > grass.cull_distance)
    {
        return false;
    }
    // A blade survives while the keep fraction at its distance exceeds its
    // hash, and sinks into the ground over the shrink band before it goes.
    float thin = grass_thinning(dist);
    float keep = thin * grass_fade(dist);
    float band = grass.thinning.y;
    float threshold = grass_unit(h, 5u) * (1.0 - band);
    if (threshold >= keep)
    {
        return false;
    }
    float shrink = saturate((keep - threshold) / max(band * keep, 1e-6));

    float2 clump_center = site.clump_center;
    uint ch = site.clump_hash;
    float clump_r = grass_unit(ch, 6u);
    float height = L.height
                 * (1.0 + L.height_variance * ((clump_r - 0.5) * 1.2 + (grass_unit(h, 7u) - 0.5) * 0.8));
    height = max(height, L.height * 0.1) * shrink;
    // The survivors widen by the coverage thinning took, so the field's
    // coverage and color hold steady with distance.
    float width = L.width * (0.75 + 0.5 * grass_unit(h, 13u)) / grass_coverage(thin);
    if (grass_outside(root + site.up * (0.5 * height), height + width))
    {
        return false;
    }

    // A clump's blades lean the clump's way and splay away from its middle,
    // and every blade leans downhill with the ground. The directions are laid
    // out on the ground's plane, then read in the blade's own frame: +X and +Z
    // turned onto its up, the frame the draws bend it in.
    float clump_angle = grass_unit(ch, 8u) * GRASS_TAU;
    float2 shared_dir = float2(cos(clump_angle), sin(clump_angle));
    float2 out_dir = site.plane - clump_center;
    float out_len = length(out_dir);
    out_dir = out_len > 1e-4 ? out_dir / out_len : shared_dir;
    float2 front_plane = normalize(shared_dir * 0.6 + out_dir * 0.8
                                   + (float2(grass_unit(h, 9u), grass_unit(h, 10u)) - 0.5) * 0.9);
    float3 front3 = front_plane.x * site.plane_u + front_plane.y * site.plane_v;
    float3 axis_x = grass_turn_up(site.up, float3(1.0, 0.0, 0.0));
    float3 axis_z = grass_turn_up(site.up, float3(0.0, 0.0, 1.0));
    float2 front = normalize(float2(dot(front3, axis_x), dot(front3, axis_z)));
    // front = (-sin, cos) of the facing angle.
    float facing = atan2(-front.x, front.y);
    float lean = (0.15 + 0.35 * grass_unit(ch, 11u)) * (0.6 + 0.8 * grass_unit(h, 12u));
    float2 downhill = float2(dot(site.normal, axis_x), dot(site.normal, axis_z));
    float2 lean2 = front * lean + downhill * GRASS_SLOPE_LEAN;

    // Last frame's field is read where the blade was last frame.
    float2 bend_now = grass_field_bend(grass.bend_half, grass.bend_window.xy, root.xz);
    float2 bend_prev = grass.bend_prev_valid != 0u
                     ? grass_field_bend(1u - grass.bend_half, grass.bend_window.zw,
                                        grass_map(grass.bend_prev_map, root))
                     : bend_now;

    blade.root_facing = float4(root, facing);
    blade.shape = uint4(asuint(height), asuint(width), grass_pack_half2(lean2),
                        grass_blade_bits(ch, h, slot, site.leaf_area));
    // The field's slow drift where the blade stands, which the draws tint it
    // by, as the far field past the blades does.
    float slope = 1.0 - saturate(dot(site.normal, site.up));
    float2 drift = grass_macro(site.ground_point, L.seed, slope, L.tip_color_macro.w, 1.0);
    blade.bend = uint4(grass_pack_half2(bend_now), grass_pack_half2(bend_prev),
                       grass_pack_half2(site.up.xz), grass_pack_half2(drift));
    lod = grass_lod_at(grass, dist);
    return true;
}

// Layer `L`'s density on a planet, in blades per square meter.
float grass_planet_density(uint layer_slot)
{
    return grass.planet_density[layer_slot >> 2u][layer_slot & 3u];
}

[shader("compute")]
[numthreads(GRASS_GROUP_SIZE, 1, 1)]
void grass_generate(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    uint slot = (grass.args_slot & 1u) * GRASS_ARGS_SLOT_WORDS;
    uint next = GRASS_ARGS_SLOT_WORDS - slot;
    if (gi < GRASS_LOD_COUNT && all(gid == 0u))
    {
        uint vertices = grass_lod_vertices(gi);
        uint first = gi * GRASS_LOD_VERTEX_STRIDE;
        uint here = slot + 4u * gi;
        uint there = next + 4u * gi;
        draw_args[there + 0u] = vertices;
        draw_args[there + 1u] = 0u;
        draw_args[there + 2u] = first;
        draw_args[there + 3u] = 0u;
        draw_args[here + 0u] = vertices;
        draw_args[here + 2u] = first;
        draw_args[here + 3u] = 0u;
    }

    uint layer_slot = gid.z;
    bool in_layer = layer_slot < grass.layer_count;
    GrassLayer L = grass.layers[min(layer_slot, GRASS_MAX_LAYERS - 1u)];
    bool planet = grass.ground == GRASS_GROUND_PLANET;
    uint tiles_x = max(L.tile_count.x, 1u);
    int2 tile = L.tile_origin + int2(gid.y % tiles_x, gid.y / tiles_x);
    if (gi < GRASS_LOD_COUNT)
    {
        gs_count[gi] = 0u;
    }
    if (gi == 0u)
    {
        if (planet)
        {
            // The record whose block holds this group row: the last whose
            // first row is not past it.
            uint lo = 0u;
            uint hi = grass.planet_tile_count;
            while (hi - lo > 1u)
            {
                uint mid = (lo + hi) / 2u;
                if (planet_tiles[grass.planet_tile_base + mid].first_group <= gid.y)
                {
                    lo = mid;
                }
                else
                {
                    hi = mid;
                }
            }
            uint record = grass.planet_tile_base + lo;
            gs_planet_tile = record;
            bool visible = false;
            if (in_layer && grass.planet_tile_count > 0u)
            {
                PlanetGrassTile T = planet_tiles[record];
                uint row = gid.y - T.first_group;
                uint2 count = uint2(T.block_count & 0xffffu, T.block_count >> 16u);
                uint2 origin = uint2(T.block_origin & 0xffffu, T.block_origin >> 16u);
                uint2 g = origin + uint2(row % max(count.x, 1u), row / max(count.x, 1u));
                gs_grass_tile = g;
                visible = row < count.x * count.y
                       && grass_planet_tile_visible(T, g, L.root_color_reach.w);
            }
            gs_tile_visible = visible ? 1u : 0u;
        }
        else
        {
            bool in_block = in_layer && gid.y < L.tile_count.x * L.tile_count.y;
            gs_tile_visible = in_block && grass_tile_visible(L, tile) ? 1u : 0u;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    GrassBlade blade = (GrassBlade)0;
    uint lod = 0u;
    bool keep = false;
    uint n = L.cells_per_side;
    uint index = gid.x * GRASS_GROUP_SIZE + gi;
    if (gs_tile_visible != 0u && index < n * n)
    {
        uint2 at = uint2(index % n, index / n);
        GrassSite site;
        bool grows;
        if (planet)
        {
            site = grass_planet_site(L, planet_tiles[gs_planet_tile], gs_grass_tile, at,
                                     grass_planet_density(layer_slot));
            grows = true;
        }
        else
        {
            grows = grass_terrain_site(L, tile * int(n) + int2(at), site);
        }
        keep = grows && grass_grow(L, layer_slot, site, blade, lod);
    }

    uint local = 0u;
    if (keep)
    {
        InterlockedAdd(gs_count[lod], 1u, local);
    }
    GroupMemoryBarrierWithGroupSync();

    // One reservation per level per group. A group that overflows a level's
    // region keeps what fits and hands the rest of its reservation back; the
    // count never drops below the region's capacity once it has reached it,
    // so every later group finds the region full and the final count is
    // exactly what was written.
    if (gi < GRASS_LOD_COUNT && gs_count[gi] > 0u)
    {
        uint capacity = grass.lod_capacity[gi];
        uint count = gs_count[gi];
        uint base;
        InterlockedAdd(draw_args[slot + 4u * gi + 1u], count, base);
        uint room = base < capacity ? capacity - base : 0u;
        uint over = count - min(room, count);
        if (over > 0u)
        {
            uint ignored;
            InterlockedAdd(draw_args[slot + 4u * gi + 1u], 0u - over, ignored);
        }
        gs_base[gi] = base;
    }
    GroupMemoryBarrierWithGroupSync();

    if (keep)
    {
        uint i = gs_base[lod] + local;
        if (i < grass.lod_capacity[lod])
        {
            blades_out[grass.lod_base[lod] + i] = blade;
        }
    }
}

#else // the draws

#ifdef GRASS_SHADOW
[[vk::binding(0, 0)]] ConstantBuffer<GrassParams> grass : register(b0);
[[vk::binding(1, 0)]] StructuredBuffer<GrassBlade> grass_blades : register(t1);
#endif

#ifdef GRASS_SHADOW
#elif defined(CN_BACKEND_METAL)
[[vk::binding(20, 0)]] ConstantBuffer<GrassParams> grass : register(b19);
[[vk::binding(21, 0)]] StructuredBuffer<GrassBlade> grass_blades : register(t20);
#elif defined(CN_BACKEND_DIRECTX)
ConstantBuffer<GrassParams> grass : register(b7);
StructuredBuffer<GrassBlade> grass_blades : register(t23);
#elif defined(SURFACE_PREPASS)
[[vk::binding(0, 3)]] ConstantBuffer<GrassParams> grass : register(b0, space3);
[[vk::binding(1, 3)]] StructuredBuffer<GrassBlade> grass_blades : register(t1, space3);
#else
[[vk::binding(0, 2)]] ConstantBuffer<GrassParams> grass : register(b0, space2);
[[vk::binding(1, 2)]] StructuredBuffer<GrassBlade> grass_blades : register(t1, space2);
#endif

// One vertex of a bent blade.
struct GrassVertex
{
    float3 world_pos;
    // The blade's front, perpendicular to its width and its curve.
    float3 normal;
    // Across the blade, root to tip unchanged.
    float3 side;
    // 0 at the root to 1 at the tip.
    float t;
    // -1 to 1 across the blade.
    float u;
};

// How far the wind bends a blade of `stiffness` at `root_xz` on the wind's own
// ground plane (see GrassParams::wind_map) at time `time`, as a horizontal tip
// offset per meter of blade. The blade's own phase makes neighbors flutter out
// of step.
float2 grass_wind_bend(float2 root_xz, float time, float phase, float stiffness)
{
    Wind w = wind_unpack(grass.wind, grass.wind_gust);
    float speed = wind_speed(w, root_xz, time);
    float give = lerp(3.0, 28.0, stiffness);
    float push = speed / (speed + give);
    float flutter = sin(time * (4.0 + 3.0 * phase) + phase * GRASS_TAU) * 0.12 * push;
    float2 across = float2(-w.dir.y, w.dir.x);
    return w.dir * (push * 0.9 + flutter) + across * (flutter * 0.5);
}

// A blade bent for one moment: its quadratic Bezier from the root, its width
// and the side its width spans.
struct GrassCurve
{
    float3 root;
    float3 d1;
    float3 d2;
    float3 side;
    float width;
};

// Matches GRASS_TRAMPLE_BEND in render::grass::bend.
static const float GRASS_TRAMPLE_FULL = 0.85;

// `blade` bent by its lean and the wind at time `time`, pressed over by
// `trample`: a fully pressed blade lies nearly flat its way, giving up its own
// lean and its sway, and a stiff one springs up a little sooner.
GrassCurve grass_curve(GrassBlade blade, float time, float2 trample)
{
    float3 root = blade.root_facing.xyz;
    float facing = blade.root_facing.w;
    float height = asfloat(blade.shape.x);
    float2 lean = float2(f16tof32(blade.shape.z & 0xffffu), f16tof32(blade.shape.z >> 16u));
    uint bits = blade.shape.w;
    float stiffness = grass.layers[grass_bits_layer(bits)].stiffness;
    float phase = grass_unit(bits & 0x0fffffffu, 14u);

    float trample_len = length(trample);
    float pressed = saturate(trample_len / GRASS_TRAMPLE_FULL) * lerp(1.0, 0.8, stiffness);
    float2 flat = trample_len > 1e-4 ? trample * (0.95 / trample_len) : (float2)(0.0);
    float2 wind_at = grass_map(grass.wind_map, root);
    float2 bend = lerp(lean + grass_wind_bend(wind_at, time, phase, stiffness), flat, pressed);
    float bend_len = length(bend);
    if (bend_len > 0.95)
    {
        bend *= 0.95 / bend_len;
    }
    float rise = sqrt(1.0 - dot(bend, bend));

    // The control point sits above the root at the tip's height, so the blade
    // leaves the ground upright and arcs over. Scaling both offsets by the
    // curve's approximate length keeps every blade its own height long.
    float3 d1 = float3(0.0, rise, 0.0) * height;
    float3 d2 = float3(bend.x, rise, bend.y) * height;
    float chord = length(d2);
    float approx_len = (2.0 * chord + length(d1) + length(d2 - d1)) / 3.0;
    float scale = height / max(approx_len, 1e-4);

    // The curve is built about +Y, then stood along the blade's own up.
    float2 up_xz = grass_unpack_half2(blade.bend.z);
    float3 up = float3(up_xz.x, sqrt(saturate(1.0 - dot(up_xz, up_xz))), up_xz.y);
    GrassCurve c;
    c.root = root;
    c.d1 = grass_turn_up(up, d1 * scale);
    c.d2 = grass_turn_up(up, d2 * scale);
    c.side = grass_turn_up(up, float3(cos(facing), 0.0, sin(facing)));
    c.width = asfloat(blade.shape.y);
    return c;
}

// The direction a blade point `pos` is seen along, toward the viewer: `eye` is
// the viewer's position (w = 1) or, for a light at infinity, the direction
// toward it (w = 0).
float3 grass_view_dir(float4 eye, float3 pos)
{
    return eye.w > 0.5 ? normalize(eye.xyz - pos) : eye.xyz;
}

// Strip pair `q` of the full-detail strip (7 is the tip), on side `u` of the
// blade, seen from `eye`.
GrassVertex grass_curve_vertex(GrassCurve c, uint q, float u, float4 eye)
{
    float t = float(min(q, 7u)) / 7.0;
    float s = 1.0 - t;
    float3 pos = c.root + 2.0 * s * t * c.d1 + t * t * c.d2;
    float3 tangent = normalize(2.0 * s * c.d1 + 2.0 * t * (c.d2 - c.d1) + float3(0.0, 1e-5, 0.0));
    float3 normal = normalize(cross(c.side, tangent));

    // A blade seen edge-on would thin to nothing and shimmer, so its width
    // swings toward the screen as it turns away.
    float3 view = grass_view_dir(eye, pos);
    float edge = 1.0 - abs(dot(view, normal));
    float3 toward = cross(view, tangent);
    float toward_len = length(toward);
    toward = toward_len > 1e-4 ? toward / toward_len : c.side;
    toward *= dot(toward, c.side) < 0.0 ? -1.0 : 1.0;
    float3 across = normalize(lerp(c.side, toward, edge * edge * 0.6));

    float half_width = 0.5 * c.width * (1.0 - pow(t, GRASS_WIDTH_TAPER)) * (1.0 + 0.3 * edge);
    GrassVertex v;
    v.world_pos = pos + across * (u * half_width);
    v.normal = normal;
    v.side = c.side;
    v.t = t;
    v.u = u;
    return v;
}

#ifndef GRASS_SHADOW
// The vertex id within the level ranges: D3D numbers a draw's vertices from 0
// whatever its first vertex, so the host passes each level's first vertex in
// the pass's b0 root constant; elsewhere the id already counts from it.
uint grass_draw_vertex(uint vid)
{
#ifdef CN_BACKEND_DIRECTX
    return vid + objid_cb.value;
#else
    return vid;
#endif
}
#endif

// Vertex `vid` of a detail-level draw: the level its id range names, the pair
// of the full-detail strip it stands for (7 for the tip) and its side.
void grass_lod_vertex(uint vid, out uint lod, out uint q, out float u)
{
    lod = min(vid / GRASS_LOD_VERTEX_STRIDE, GRASS_LOD_COUNT - 1u);
    uint local = vid - lod * GRASS_LOD_VERTEX_STRIDE;
    bool tip = local >= 2u * grass_lod_pairs(lod);
    q = tip ? 7u : ((local >> 1u) << lod);
    u = tip ? 0.0 : ((local & 1u) != 0u ? 1.0 : -1.0);
}

// Vertex `vid` of `blade` at time `time`, pressed over by `trample`, seen from
// `eye` (see grass_view_dir). Pairs climb the
// blade root to tip and the last vertex is the tip, so the strip narrows to a
// point. A vertex the next level drops folds onto the midpoint of its two
// neighbors over the last stretch of its level's reach, so the next level's
// strip takes over exactly where this one has become it. Mirrors
// render::grass::lod::morph.
GrassVertex grass_blade_vertex(GrassBlade blade, uint vid, float time, float4 eye,
                               float2 trample)
{
    uint lod;
    uint q;
    float u;
    grass_lod_vertex(vid, lod, q, u);
    GrassCurve c = grass_curve(blade, time, trample);
    GrassVertex v = grass_curve_vertex(c, q, u, eye);
    uint step = 1u << lod;
    if (eye.w > 0.5 && lod + 1u < GRASS_LOD_COUNT && q < 7u && (q & step) != 0u)
    {
        float end = lod == 0u ? grass.lod_distances.x : grass.lod_distances.y;
        float band = end * grass.lod_distances.z;
        float m = saturate((distance(c.root, eye.xyz) - (end - band)) / band);
        if (m > 0.0)
        {
            uint qb = q + step;
            GrassVertex a = grass_curve_vertex(c, q - step, u, eye);
            GrassVertex b = grass_curve_vertex(c, qb, qb >= 7u ? 0.0 : u, eye);
            v.world_pos = lerp(v.world_pos, 0.5 * (a.world_pos + b.world_pos), m);
            v.t = lerp(v.t, 0.5 * (a.t + b.t), m);
            v.u = lerp(v.u, 0.5 * (a.u + b.u), m);
        }
    }
    return v;
}

// How far `blade` is trampled this frame and how far it was last frame.
float2 grass_trample_now(GrassBlade blade)
{
    return grass_unpack_half2(blade.bend.x);
}

float2 grass_trample_prev(GrassBlade blade)
{
    return grass_unpack_half2(blade.bend.y);
}

// The blade drawn by instance `iid` of the draw vertex `vid` belongs to: the
// instance indexes that draw's detail level's region.
GrassBlade grass_drawn_blade(uint vid, uint iid)
{
    uint lod = min(vid / GRASS_LOD_VERTEX_STRIDE, GRASS_LOD_COUNT - 1u);
    return grass_blades[grass.lod_base[lod] + iid];
}

// The blade's colors at its root and its tip: its layer's, with hue and
// brightness shifted per clump and per blade and tinted by the field's
// `drift` where it stands. The fragment ramps between them by its own height,
// so a coarse strip shows the same ramp as a fine one.
void grass_colors(uint bits, float2 drift, out float3 root, out float3 tip)
{
    GrassLayer L = grass.layers[grass_bits_layer(bits)];
    float blade_r = grass_unit((bits >> 4u) & 0xfffu, 16u);
    float v = L.color_variation;
    // Drier clumps lean yellow; every blade jitters in brightness.
    float3 tint = lerp((float3)(1.0), GRASS_DRY_TINT, v * grass_clump_dryness(bits >> 16u))
                * grass_macro_tint(drift) * (1.0 + v * (blade_r - 0.5) * 0.8);
    root = L.root_color_reach.rgb * tint;
    tip = L.tip_color_macro.rgb * tint;
}

// The normal the light sees: the blade's front turned toward the viewer, then
// rolled across the width so the flat strip shades as a rounded blade.
float3 grass_shading_normal(float3 normal, float3 side, float u, float3 to_camera)
{
    float3 n = dot(normal, to_camera) < 0.0 ? -normal : normal;
    return normalize(n + side * (u * 0.45));
}

#if defined(GRASS_SHADOW)

// The cascade's blades, every one with the coarsest strip: its draw starts at
// that level's first vertex, which D3D leaves out of the id, so the id is
// taken within the level's range on every backend.
[shader("vertex")]
float4 grass_shadow_vertex(uint vid : SV_VertexID, uint iid : SV_InstanceID) : SV_Position
{
    uint lod = GRASS_LOD_COUNT - 1u;
    uint v = lod * GRASS_LOD_VERTEX_STRIDE + (vid & (GRASS_LOD_VERTEX_STRIDE - 1u));
    GrassBlade blade = grass_blades[grass.lod_base[lod] + iid];
    float4 toward_light = float4(grass.shadow_light.xyz, 0.0);
    GrassVertex gv = grass_blade_vertex(blade, v, grass.shadow_light.w, toward_light,
                                        grass_trample_now(blade));
    return mul(grass.shadow_vp, float4(gv.world_pos, 1.0));
}

#elif !defined(SURFACE_PREPASS)

struct GrassVertexOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_pos : TEXCOORD0;
    [[vk::location(1)]] float3 normal : TEXCOORD1;
    [[vk::location(2)]] float3 side : TEXCOORD2;
    [[vk::location(3)]] float3 to_camera : TEXCOORD3;
    // The blade's colors at its root and its tip (see grass_colors).
    [[vk::location(4)]] float3 root_color : TEXCOORD4;
    // x = t up the blade, y = u across it, z = view depth.
    [[vk::location(5)]] float3 t_u_depth : TEXCOORD5;
    [[vk::location(6)]] float3 tip_color : TEXCOORD6;
    // The leaf above the point that a direct light crosses (see
    // grass_light_through).
    [[vk::location(7)]] float4 light_depth : TEXCOORD7;
};

// The leaf a direct light crosses down to height `t` along `blade`: the leaf
// around it, the share of it standing above that height, and the projection
// ellipse of its layer's blades, as grass_light_through reads them.
float4 grass_light_depth(GrassBlade blade, float t)
{
    GrassLayer L = grass.layers[grass_bits_layer(blade.shape.w)];
    float z = t * asfloat(blade.shape.x) / max(L.height, 1e-4);
    float leaf = grass_bits_leaf_area(blade.shape.w) * grass_leaf_above(z, L.height_variance);
    float2 up_xz = grass_unpack_half2(blade.bend.z);
    float3 up = float3(up_xz.x, sqrt(saturate(1.0 - dot(up_xz, up_xz))), up_xz.y);
    return float4(up * (leaf * L.light_projection.y), leaf * L.light_projection.x);
}

[shader("vertex")]
GrassVertexOut grass_vertex(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    vid = grass_draw_vertex(vid);
    GrassBlade blade = grass_drawn_blade(vid, iid);
    float3 cam = float3(VIEW.cam_x, VIEW.cam_y, VIEW.cam_z);
    GrassVertex v = grass_blade_vertex(blade, vid, VIEW.elapsed, float4(cam, 1.0),
                                       grass_trample_now(blade));
    GrassVertexOut o;
    o.position = mul(VIEW.vp, float4(v.world_pos, 1.0));
    o.world_pos = v.world_pos;
    o.normal = v.normal;
    o.side = v.side;
    o.to_camera = cam - v.world_pos;
    grass_colors(blade.shape.w, grass_unpack_half2(blade.bend.w), o.root_color, o.tip_color);
    o.t_u_depth = float3(v.t, v.u, -mul(VIEW.view_mat, float4(v.world_pos, 1.0)).z);
    o.light_depth = grass_light_depth(blade, v.t);
    return o;
}

// The blade's lit color in absolute radiance.
float4 grass_radiance(GrassVertexOut p)
{
    float t = p.t_u_depth.x;
    float view_depth = p.t_u_depth.z;
    float3 albedo = lerp(p.root_color, p.tip_color, smoothstep(0.0, 1.0, t));
    if (VIEW.shade_mode > 0.5)
    {
        return float4(albedo, 1.0);
    }

    float3 V = normalize(p.to_camera);
    float3 N = grass_shading_normal(normalize(p.normal), normalize(p.side), p.t_u_depth.y, V);
    float occlusion = grass_root_occlusion(t);
    float2 screen_xy = p.position.xy;

    float3 Lo = (float3)(0.0);
    float shadow = shadow_factor_cascaded(p.world_pos, view_depth, screen_xy);
    for (int i = 0; i < LIGHTS.num_dir; i++)
    {
        float3 L = normalize(LIGHTS.dir[i].dir_i.xyz);
        float3 radiance = LIGHTS.dir[i].col.xyz * LIGHTS.dir[i].dir_i.w;
        float s = (i == 0) ? shadow : 1.0;
        // Blades deep in the field get the light the leaf above lets through.
        Lo += grass_light(albedo, N, V, L, radiance) * s * grass_light_through(p.light_depth, L);
    }

    // Point and spot lights from the fragment's cluster; area lights are left
    // to the surfaces the panel faces.
    uint cluster_base = 0u;
    int local_count;
    if (CLUSTER.use_clusters != 0u)
    {
        uint cid = cluster_at(screen_xy / float2(CLUSTER.screen_w, CLUSTER.screen_h), view_depth);
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
        if (light_kind(LOCAL_LIGHTS[i]) == LIGHT_KIND_AREA)
        {
            continue;
        }
        float3 to_light = LOCAL_LIGHTS[i].position_range.xyz - p.world_pos;
        float dist = length(to_light);
        float3 L = to_light / max(dist, 1e-4);
        float atten = saturate(1.0 - dist / LOCAL_LIGHTS[i].position_range.w);
        atten *= atten;
        if (light_kind(LOCAL_LIGHTS[i]) == LIGHT_KIND_SPOT)
        {
            float cd = dot(LOCAL_LIGHTS[i].direction_kind.xyz, -L);
            float ci = LOCAL_LIGHTS[i].cos_inner;
            float co = LOCAL_LIGHTS[i].cos_outer;
            float k = saturate((cd - co) / max(ci - co, 1e-4));
            atten *= k * k;
            int si = LOCAL_LIGHTS[i].shadow_index;
            if (si >= 0 && atten > 0.0)
            {
                atten *= sample_spot_shadow(si, p.world_pos, N, screen_xy);
            }
        }
        float3 radiance = LOCAL_LIGHTS[i].color_intensity.xyz * LOCAL_LIGHTS[i].color_intensity.w * atten;
        Lo += grass_light(albedo, N, V, L, radiance);
    }

    float3 ambient = VIEW.irradiance > 0.5
                   ? albedo * irradiance_sample(N) / PI
                   : (float3)(0.03) * albedo;
    ScreenIndirect indirect = screen_indirect_at(screen_xy);
    ambient = ambient * indirect.visibility + albedo * indirect.bounce;
    ambient *= LIGHTS.ambient_intensity * occlusion;
    return float4(ambient + Lo * occlusion, 1.0);
}

[shader("pixel")]
float4 grass_fragment(GrassVertexOut p) : SV_Target
{
    float4 color = grass_radiance(p);
    return float4(hdr_store_cap(color.rgb * VIEW.pre_exposure), color.a);
}

#else // SURFACE_PREPASS

struct GrassPrepassOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 normal : TEXCOORD0;
    [[vk::location(1)]] float3 side : TEXCOORD1;
    [[vk::location(2)]] float3 to_camera : TEXCOORD2;
    // x = u across the blade, y = view depth.
    [[vk::location(3)]] float2 u_depth : TEXCOORD3;
    [[vk::location(4)]] float4 cur_clip : TEXCOORD4;
    [[vk::location(5)]] float4 prev_clip : TEXCOORD5;
};

// The blade rasterized through the main pass's jittered view, and bent again
// at the previous frame's clock and camera for its motion. Both positions
// reproject through the unjittered matrices, so jitter never reaches the
// motion vector.
[shader("vertex")]
GrassPrepassOut grass_prepass_vertex(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    vid = grass_draw_vertex(vid);
    GrassBlade blade = grass_drawn_blade(vid, iid);
    float3 cam = float3(view_cb.cam_x, view_cb.cam_y, view_cb.cam_z);
    GrassVertex v = grass_blade_vertex(blade, vid, view_cb.elapsed, float4(cam, 1.0),
                                       grass_trample_now(blade));
    float3 prev_world = v.world_pos;
    if (gb_view.motion != 0u)
    {
        float3 prev_cam = float3(gb_view.prev_cam_x, gb_view.prev_cam_y, gb_view.prev_cam_z);
        prev_world = grass_blade_vertex(blade, vid, gb_view.prev_elapsed, float4(prev_cam, 1.0),
                                        grass_trample_prev(blade)).world_pos;
    }
    GrassPrepassOut o;
    o.position = mul(view_cb.vp, float4(v.world_pos, 1.0));
    o.normal = v.normal;
    o.side = v.side;
    o.to_camera = cam - v.world_pos;
    o.u_depth = float2(v.u, -mul(view_cb.view_mat, float4(v.world_pos, 1.0)).z);
    o.cur_clip = mul(gb_view.cur_vp, float4(v.world_pos, 1.0));
    o.prev_clip = mul(gb_view.prev_vp, float4(prev_world, 1.0));
    return o;
}

// Grass is matte: the roughness the reflection resolves skip it at.
static const float GRASS_ROUGHNESS = 0.6;

[shader("pixel")]
GbFragmentOut grass_prepass_fragment(GrassPrepassOut p)
{
    float3 V = normalize(p.to_camera);
    float3 N = grass_shading_normal(normalize(p.normal), normalize(p.side), p.u_depth.x, V);
    GbFragmentOut o;
    o.nd = float4(normalize(mul((float3x3)gb_view.view_mat, N)), p.u_depth.y);
    o.rough = GRASS_ROUGHNESS;
    o.vel = gb_motion(p.cur_clip, p.prev_clip);
    return o;
}

#endif // GRASS_SHADOW / SURFACE_PREPASS

#endif // GRASS_BEND / GRASS_GENERATE
