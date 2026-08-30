//! Asset templates: typed builders that produce one `AssetSpec` each.
//!
//! These are the reusable element constructors, expressed as data so they can be
//! consumed either as world.jsonl lines (`crate::authoring::template`, `cn add`) or
//! materialized straight into a live component (the editor HUD), without any JSON
//! string in between. Each builder sets only the fields it means to change; every
//! other field takes the asset type's serde default when the spec is materialized.

mod interaction;
mod layout;
mod scene;
mod sprite;
mod text;

pub(crate) use interaction::hit_region;
pub(crate) use layout::screen;
pub(crate) use scene::{camera, directional_light, environment_map_sky, room};
pub use sprite::sprite;
pub(crate) use text::{font, menu_label};
pub use text::{text_input, text_label};
