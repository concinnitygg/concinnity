// src/assets/hidden.rs

/// Runtime-only tag: this entity's draw slots are switched off by a `hide`
/// action (VisibilityRequest). The entity keeps simulating; scene-driven
/// visibility switches leave tagged entities dark, and a `show` request
/// removes the tag and relights the slots. Never authored directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct Hidden;
