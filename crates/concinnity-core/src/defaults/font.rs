// The font the injected HUD chips and the loading label draw with: the face
// bundled in the binary, rasterised at start and appended to the world's font
// table. Appending leaves every handle the build assigned where it was.

use crate::bake;
use crate::ecs::{FontHandle, PipelineContext};
use crate::resource::{FontTable, ResourceEntry};
use crate::result::CnResult;

/// Pixel size the injected HUD face is rasterised at. Chips draw it minified,
/// so the atlas is supersampled from here rather than authored larger.
pub const HUD_FONT_SIZE_PX: u32 = 20;

/// The face baked for this world, so a second caller shares it rather than
/// paying for the atlas again.
#[derive(Debug, Clone, Copy)]
pub struct HudFont(pub FontHandle);

/// The font the engine's own HUD text draws with, baked into `ctx`'s font table
/// on the first call and returned as it stands afterwards.
///
/// A host that draws HUD text of its own before the world starts (the editor's
/// panels) reaches it here, so one atlas serves both.
pub fn hud_font(ctx: &mut PipelineContext) -> Result<FontHandle, CnResult> {
    if let Some(HudFont(handle)) = ctx.resource::<HudFont>().copied() {
        return Ok(handle);
    }
    let payload = bake::font::compile(
        bake::font::BUILTIN_FONT_BYTES,
        HUD_FONT_SIZE_PX,
        bake::font::BUILTIN_FONT_FILE,
    )
    .map_err(|_| CnResult::InvalidArgument)?;
    // The table is created when the world carries none: one assembled in code
    // rather than loaded from a blob.
    if ctx.resource::<FontTable>().is_none() {
        ctx.insert_resource(FontTable::default());
    }
    let table = ctx
        .resource_mut::<FontTable>()
        .ok_or(CnResult::InvalidState)?;
    let handle = FontHandle(table.append(ResourceEntry::baked(payload)));
    ctx.insert_resource(HudFont(handle));
    Ok(handle)
}
