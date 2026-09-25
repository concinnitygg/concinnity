//! Backend-agnostic window vocabulary: the display modes a window can take, the
//! system clipboard it reaches, and the process-wide policy on whether one may
//! open at all.

pub mod clipboard;
pub mod display_mode;
pub mod policy;
