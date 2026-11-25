//! The stores behind the engine's persistence seams: files written off the sim
//! thread under the host's state tree, and a preview session's in-memory saves.

pub(crate) mod behavior;
pub(crate) mod settings;
pub(crate) mod story;
