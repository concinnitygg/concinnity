//! The graphics driver: GraphicsSystem, the load threads and payload sources
//! behind core's streaming pools, and the standalone startup-error window. The
//! decisions they act on live in `concinnity_core::settings` and
//! `concinnity_core::render`.

// The bundled glyph atlas baked into the binary: the face the startup error
// screen draws with, and the fallback for a world whose labels name no Font.
pub(crate) mod builtin_font;
// The standalone startup-error window, shown when a fatal startup failure
// happens before any world exists.
pub(crate) mod error_screen;
pub(crate) mod streaming;
pub(crate) mod system;
