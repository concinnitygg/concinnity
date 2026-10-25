// The far field's canopy block: its grass layers, the planet ground they grow
// on, its covers and the wash up its beaches, spliced at a shader's
// CANOPY_TYPES marker after GROUND_RULE_TYPES. The main pass and the
// ray-traced reflections both read it.

// Mirrors `CanopyLayerGpu` in concinnity-engine/src/render/uniforms/canopy.rs
// (112 B): one grass layer as the far field past its blades reads it.
struct CanopyLayer
{
    float4 root_color_variation;
    float4 tip_color_macro;
    // x = clump size, y = height variance, z = leaf area, w = mean bend.
    float4 shape;
    uint seed;
    float sky_projection;
    float _pad0;
    float _pad1;
    // xyz = linear RGB of the litter under the blades, w = clumping.
    float4 litter_clumping;
    // Where on a planet the layer grows.
    GroundRule rule;
};

// Matches MAX_CANOPY_LAYERS in render::uniforms::canopy.
#define CANOPY_MAX_LAYERS 16

// Matches MAX_GROUND_COVERS in planet::cover.
#define GROUND_MAX_COVERS 4

// The wash of a planet sea's waves up its beaches. Mirrors `ShoreSwashGpu`
// (96 B); `runup` 0 where no sea has shore waves.
struct ShoreSwash
{
    float sea_level;
    float runup;
    float reach;
    float _pad;
    float4 omegas;
    float4 phases;
    float4 amplitudes;
    uint4 noise_salts;
    float4 noise; // [phase cells per unit of direction, radians, swell cells, share]
};

// Mirrors `CanopyParams` in concinnity-engine/src/render/uniforms/canopy.rs
// (2464 B): every far field's layers, the planet's axes in the frame, the
// noise its height is summed from, its coasts, the covers laid over it,
// where the blades hand the field to it, and the wash up its beaches.
struct CanopyParams
{
    float4 planet_rows[3];
    float planet_radius;
    float meadow_scale;
    float meadow_mean;
    uint layer_count;
    CanopyLayer layers[CANOPY_MAX_LAYERS];
    float4 ground_turn[3];
    uint4 ground_salts[4];
    float ground_frequency;
    float ground_height_scale;
    float ground_cell;
    uint ground_octaves;
    uint4 continent_salts[2];
    float continent_frequency;
    float continent_scale;
    float continent_offset;
    uint continent_octaves;
    float beach_width;
    float beach_rise;
    float hill_rise;
    float ocean_depth;
    float shelf_length;
    float seabed_lift;
    uint cover_count;
    float _pad0;
    // xyz = where ground altitudes are measured from, w = its altitude.
    float4 ground_origin;
    // xyz = the up there, w = its distance from the planet's center.
    float4 ground_up;
    GroundCover covers[GROUND_MAX_COVERS];
    // x = where the blades start handing the field over, y = where they end,
    // z = the hand-off's phase (see grass_crossfade_shows_blades); w unused.
    float4 band;
    ShoreSwash swash;
};
