// Asset templates: typed builders that produce one `AssetSpec` each.
//
// These are the reusable element constructors, expressed as data so they can be
// consumed either as world.jsonl lines (the world templates, `cn add`) or
// materialized straight into a live component (the editor HUD), without any JSON
// string in between. Each builder sets only the fields it means to change; every
// other field takes the asset type's serde default when the spec is materialized.

mod interaction;
mod layout;
mod scene;
mod sprite;
mod text;

pub use interaction::hit_region;
pub use layout::{panel, view};
pub use scene::{
    directional_light, environment_map_sky, point_light, post_process, volumetric_fog,
};
pub use sprite::sprite;
pub use text::{text_input, text_label};
