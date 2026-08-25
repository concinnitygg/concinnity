// src/components/audio_occlusion_probe.rs

/// Runtime-only occlusion probe for a positional audio emitter.
///
/// The audio system attaches one to each emitter's entity and refreshes
/// `from` (the listener) and `to` (the emitter) every frame; `PhysicsSystem`
/// (which steps earlier) raycasts the segment against scene geometry and
/// writes back `blocked`. The audio system then muffles the emitter when the
/// path is blocked. The exchange is one frame behind the listener, which is
/// inaudible. Riding the emitter's entity, the probe despawns with it.
///
/// Not authored in world files: it has no `args`.
#[derive(Debug, Clone, Default)]
pub struct AudioOcclusionProbe {
    /// Listener position in world space (the ray origin).
    pub from: [f32; 3],
    /// Emitter position in world space (the ray target).
    pub to: [f32; 3],
    /// Whether scene geometry blocks the segment, or `None` when unanswered
    /// (no physics in the world, or the first frame).
    pub blocked: Option<bool>,
}
