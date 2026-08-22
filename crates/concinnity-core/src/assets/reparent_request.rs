// src/assets/reparent_request.rs

use crate::assets::EntityTarget;

/// Runtime-only event requesting that an authored placement be re-parented at
/// runtime: `child` is moved under `parent`, or detached to a root when
/// `parent` is None. GraphicsSystem reads these from its
/// `Events<ReparentRequest>` queue each step, resolves the nameto their entities,
/// re-points the child's Parent edge, and recomposes world matrices. World
/// authors never declare this type directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReparentRequest {
    /// The placement to move.
    pub child: EntityTarget,
    /// The new parent, or `None` to detach the child to a root.
    pub parent: Option<EntityTarget>,
}
