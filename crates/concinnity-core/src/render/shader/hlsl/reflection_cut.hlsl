// REFLECTION_CUT marker: the roughness past which a surface takes no SSR or RT
// reflection, shared by the forward fade, the SSR and RT resolves, and the
// reflection blur so they never disagree. Locked to REFLECTION_ROUGHNESS_CUT
// in concinnity_core::render::post::ssr::settings by unit test.
static const float REFLECTION_ROUGHNESS_CUT = 0.6;
