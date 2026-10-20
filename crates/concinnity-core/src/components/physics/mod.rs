//! Physics components: bodies and their dynamics, colliders, joints, trigger
//! volumes, the world's physics configuration, and contact and volume events.

// Per-instance components an entity is composed from: its placement, render
// description, collision, hierarchy, and gameplay tags.
pub(super) mod body_dynamics;
pub(super) mod collider;
pub(super) mod config;
pub(super) mod contact_event;
pub(super) mod joint;
pub(super) mod prop_body;
pub(super) mod rigid_body;
pub(super) mod trigger_volume;
pub(super) mod volume_event;
