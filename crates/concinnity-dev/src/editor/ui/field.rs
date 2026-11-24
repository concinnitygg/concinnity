//! A built panel's text fields, read and written by id.

use concinnity_engine::components::TextField;
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

// The field's text, empty when there is none.
pub(crate) fn text(world: &World, field: Option<AssetId>) -> String {
    field
        .and_then(|id| world.get_by_id::<TextField>(id))
        .map_or_default(|f| f.content.clone())
}

// Replace the field's text, the caret at its end.
pub(crate) fn set_text(world: &mut World, field: Option<AssetId>, text: &str) {
    if let Some(field) = field.and_then(|id| world.get_mut_by_id::<TextField>(id)) {
        field.content.clear();
        field.content.push_str(text);
        field.caret = text.chars().count();
    }
}
