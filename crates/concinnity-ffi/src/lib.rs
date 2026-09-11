//! The C ABI a host application links to embed the engine.
//!
//! The surface is a world's lifecycle inside a view the host owns, because
//! that is the shape a platform whose OS owns the run loop needs: the host
//! creates the view, opens a world into it, and calls one step per display
//! refresh. Authoring lives in the dev tooling and is deliberately absent
//! here.

/// The `extern "C"` functions themselves, and the header cbindgen writes
/// from them.
pub mod ffi;
