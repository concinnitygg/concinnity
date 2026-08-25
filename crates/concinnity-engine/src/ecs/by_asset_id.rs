// Address a UI component by the asset id it was built from. The overlay
// components are authored as named assets and referenced by id long after the
// world is decomposed, so every HUD / menu / story update resolves an id to the
// live component before writing it.

use crate::components::{Sprite, TextInput, TextLabel};
use crate::ecs::{ComponentSlot, PipelineContext, asset_id::AssetId};

// A component that carries the id of the asset it was built from.
pub(crate) trait Identified: ComponentSlot {
    fn asset_id(&self) -> AssetId;
}

macro_rules! impl_identified {
    ($($t:ty),+ $(,)?) => {
        $(impl Identified for $t {
            fn asset_id(&self) -> AssetId {
                self.asset_id
            }
        })+
    };
}

impl_identified!(Sprite, TextInput, TextLabel);

// The first component of type `C` carrying `id`, or `None` when the world never
// declared it.
pub(crate) fn find_mut<'a, C: Identified>(
    ctx: &'a mut PipelineContext,
    id: AssetId,
) -> Option<&'a mut C> {
    ctx.query_mut::<C>().find(|c| c.asset_id() == id)
}

// Mutate the component of type `C` carrying `id`. An unset reference or a
// component the world never declared is a silent no-op.
pub(crate) fn update<C: Identified>(
    ctx: &mut PipelineContext,
    id: Option<AssetId>,
    apply: impl FnOnce(&mut C),
) {
    let Some(id) = id else { return };
    if let Some(c) = find_mut::<C>(ctx, id) {
        apply(c);
    }
}

// Overwrite the text of the TextLabel carrying `id`, if present.
pub(crate) fn set_text(ctx: &mut PipelineContext, id: AssetId, text: &str) {
    if let Some(l) = find_mut::<TextLabel>(ctx, id) {
        l.content = text.to_string();
    }
}
