// A ground rule and a planet's ground covers, declared once for the canopy
// block and the grass kernel's block alike. Guarded because a pass can reach
// it through both.
#ifndef CN_GROUND_RULE_TYPES
#define CN_GROUND_RULE_TYPES

// Mirrors `GroundRuleGpu` in concinnity-engine/src/render/uniforms/ground.rs
// (32 B): each bound's ramp is x * k + c over its measure.
struct GroundRule
{
    // k and c of the lowest altitude's ramp (xy), then of the highest's (zw).
    float4 altitude;
    // k and c of the gentlest slope's ramp, then of the steepest's, over the
    // ground's steepness (sqrt(1 - cos) of its normal's angle from up).
    float4 slope;
};

// Matches GROUND_COVER_UNTEXTURED in render::environment::ground: a cover
// whose material has no albedo texture.
#define GROUND_COVER_UNTEXTURED 0xffffffffu

// Mirrors `GroundCoverGpu` in concinnity-engine/src/render/uniforms/ground.rs
// (64 B).
struct GroundCover
{
    GroundRule rule;
    float3 tint;
    uint albedo_index;
    float roughness;
    float metallic;
    float _pad0;
    float _pad1;
};

#endif
