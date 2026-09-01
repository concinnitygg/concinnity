// The font the injected HUD chips and the loading label draw with: the face
// bundled in the binary, rasterised once for the process and appended to each
// world's font table. Appending leaves every handle the build assigned where it
// was.

use alloc::vec::Vec;

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

// The bundled face at one fixed size compiles to one fixed atlas, so the
// signed-distance pass over its glyphs runs once for the process rather than
// once per world. A host that rebuilds a world repeatedly -- the editor's live
// preview, and every test that injects the HUD -- is otherwise paying for the
// same bytes each time.
//
// A racing caller waits for the first rather than compiling its own copy: the
// pass is long enough that the duplicates are the whole cost worth avoiding.
static PAYLOAD: spin::Once<Vec<u8>> = spin::Once::new();

// The compiled atlas for the bundled face, computed on the first call. A
// failure is not stored: it can only mean the bundled face itself is broken,
// and the caller decides what to do about that.
fn payload() -> Result<&'static [u8], CnResult> {
    PAYLOAD
        .try_call_once(|| {
            bake::font::compile(
                bake::font::BUILTIN_FONT_BYTES,
                HUD_FONT_SIZE_PX,
                bake::font::BUILTIN_FONT_FILE,
            )
            .map_err(|_| CnResult::InvalidArgument)
        })
        .map(Vec::as_slice)
}

/// The font the engine's own HUD text draws with, baked into `ctx`'s font table
/// on the first call and returned as it stands afterwards.
///
/// A host that draws HUD text of its own before the world starts (the editor's
/// panels) reaches it here, so one atlas serves both.
pub fn hud_font(ctx: &mut PipelineContext) -> Result<FontHandle, CnResult> {
    if let Some(HudFont(handle)) = ctx.resource::<HudFont>().copied() {
        return Ok(handle);
    }
    let payload = payload()?.to_vec();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::World;

    // The signed-distance pass over the bundled glyphs is the expensive half of
    // a bake, and it runs on a constant: the same face at the same size. A
    // second call reads what the first stored rather than repeating it.
    #[test]
    fn the_bundled_atlas_compiles_once_for_the_process() {
        let first = payload().expect("the bundled face compiles");
        let second = payload().expect("the bundled face compiles");

        assert!(!first.is_empty());
        assert_eq!(
            first.as_ptr(),
            second.as_ptr(),
            "the second call reads the stored atlas"
        );
    }

    // Sharing the compiled bytes must not share the table entry: each world
    // owns its own, at whatever handle its own table hands out.
    #[test]
    fn each_world_gets_its_own_entry_holding_the_same_atlas() {
        let mut first = World::new();
        let mut second = World::new();
        hud_font(&mut first.context()).expect("a face for the first world");
        hud_font(&mut second.context()).expect("a face for the second world");

        let bytes = |world: &World| {
            world
                .resource::<FontTable>()
                .expect("a font table")
                .0
                .first()
                .expect("the appended face")
                .baked_bytes()
                .expect("the face is baked")
                .to_vec()
        };
        assert_eq!(bytes(&first), bytes(&second));
        assert!(!bytes(&first).is_empty());
    }

    // A world that already carries the face reuses its handle rather than
    // appending a second entry for the same atlas.
    #[test]
    fn a_second_call_on_one_world_reuses_the_handle() {
        let mut world = World::new();
        let first = hud_font(&mut world.context()).expect("a face");
        let second = hud_font(&mut world.context()).expect("the same face");

        assert_eq!(first, second);
        assert_eq!(
            world.resource::<FontTable>().expect("a font table").len(),
            1
        );
    }
}
