// src/assets/lifetime.rs

// Runtime-only countdown on an entity: seconds remaining before it is
// automatically removed. A spawn carries it onto short-lived instances
// (projectiles, debris, timed effects); the graphics step decrements it each
// frame and despawns the entity (and its descendants) when it reaches zero,
// reclaiming the GPU draw slots for reuse. World authors never declare this
// type directly; it is set at spawn time.
#[derive(Debug, Default, Clone, Copy)]
pub struct Lifetime {
    pub remaining: f32,
}
