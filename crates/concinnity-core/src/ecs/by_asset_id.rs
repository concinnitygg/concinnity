//! Address a UI component by the asset id it was built from. The overlay
//! components are authored as named assets and referenced by id long after the
//! world is decomposed, so every HUD / menu / story update resolves an id to the
//! live component before writing it.

use alloc::string::ToString;

use crate::components::TextLabel;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{ComponentSlot, PipelineContext};

/// Mutate the `C` of the entity the asset `id` was loaded into. An unset
/// reference or an asset with no `C` is a silent no-op.
pub fn update<C: ComponentSlot>(
    ctx: &mut PipelineContext,
    id: Option<AssetId>,
    apply: impl FnOnce(&mut C),
) {
    if let Some(c) = id.and_then(|id| ctx.get_mut_by_id::<C>(id)) {
        apply(c);
    }
}

/// Overwrite the text of the TextLabel asset `id`, if present.
pub fn set_text(ctx: &mut PipelineContext, id: AssetId, text: &str) {
    update::<TextLabel>(ctx, Some(id), |l| l.content = text.to_string());
}
