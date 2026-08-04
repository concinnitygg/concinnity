// Asset templates: typed builders that produce one `AssetSpec` each.
//
// These are the reusable element constructors, expressed as data so they can be
// consumed either as world.jsonl lines (`crate::template`, `cn add`) or
// materialized straight into a live component (the editor HUD), without any JSON
// string in between. Each builder sets only the fields it means to change; every
// other field takes the asset type's serde default when the spec is materialized.

mod interaction;
mod layout;
mod scene;
mod sprite;
mod text;

pub use interaction::hit_region;
pub use layout::{panel, screen};
pub use scene::{
    camera, directional_light, environment_map_sky, point_light, post_process, room, spot_light,
    volumetric_fog,
};
pub use sprite::sprite;
pub use text::{font, menu_label, text_input, text_label};
