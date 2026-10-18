//! The client's render layer: the renderer driver and the asset-streaming
//! drivers. The decisions they act on (settings, the draw list, the 2D overlay,
//! the live previews, and the backend-agnostic render-prep) live in
//! `concinnity_core::settings` and `concinnity_core::render`.

// The bundled glyph atlas baked into the binary: the face the startup error
// screen draws with, and the fallback for a world whose labels name no Font.
pub(crate) mod builtin_font;

/// Asset-streaming home: the re-exported `no_std` policy core (`StreamPlanner`),
/// the `std` texture / mesh / chunk drivers it schedules, and the system that
/// drives them. `pub` for that system alone; everything else inside is
/// crate-private.
pub mod streaming;
/// The renderer driver. An internal system (not a declarable asset), constructed
/// by `World::start` when the run resolved to a windowed one.
pub(crate) mod system;
// The GraphicsSystem test-injection hooks over core's recording backend.
#[cfg(test)]
pub(crate) mod test_hooks;
// A planet's grass layers, resolved at graphics init.
pub(crate) mod planet_grass;
