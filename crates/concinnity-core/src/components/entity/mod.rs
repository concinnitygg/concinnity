//! Per-entity components: placement and hierarchy, identity and visibility,
//! props and their instances, render handles, lifetimes, and targets.

pub(super) mod children;
pub(super) mod entity_target;
pub(super) mod global_transform;
pub(super) mod hidden;
pub(super) mod identity;
pub(super) mod instanced_prop;
pub(super) mod lifetime;
pub(super) mod mesh_renderer;
pub(super) mod model_renderer;
pub(super) mod parent;
pub(super) mod prop;
pub(super) mod prop_instance;
pub(super) mod render_handle;
pub(super) mod scene_member;
pub(super) mod transform;
