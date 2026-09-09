//! Fullscreen post-processing passes written once, over a small backend seam.
//!
//! A screen-space post pass is the same three operations on every backend:
//! build a fullscreen-triangle pipeline from a shader program plus an output
//! format, create the persistent targets it accumulates into, and encode one
//! draw that binds N textures and a constants blob. [`device::PostPassDevice`]
//! is those three operations and nothing else, so a pass written against it
//! carries its whole implementation here rather than three times over.
//!
//! This is narrower than [`super::fullscreen`], the earlier seam: those traits
//! share a pass's *orchestration* (the begin/draw/end order) while every
//! resource, layout and bind stays per backend. Here the resources and the
//! binds cross the seam too, which is what lets a pass's state machine and its
//! target lifecycle live in this crate.

/// The backend seam: pipelines, persistent targets, and one fullscreen draw.
pub mod device;

/// The ping-pong ring and history-validity gate a temporal pass accumulates
/// through, as a pure state machine.
pub mod history;

/// Which single-source program a post pass runs, and the binding count it
/// declares.
pub mod program;

/// Temporal anti-aliasing: the resolve pass, written once.
pub mod taa;
