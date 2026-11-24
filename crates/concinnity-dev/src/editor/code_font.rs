//! The monospace face code text draws with: `JetBrains` Mono Regular, under the
//! SIL Open Font License 1.1 shipped beside the font file. It lives in this
//! crate, so only the editor carries it; the atlas is baked once for the
//! process and appended to each world the editor sets up, like the HUD face.

use std::sync::OnceLock;

use concinnity_engine::bake;
use concinnity_engine::ecs::{FontHandle, World};

const FONT_BYTES: &[u8] = include_bytes!("fonts/JetBrainsMono-Regular.ttf");
const FONT_FILE: &str = "JetBrainsMono-Regular.ttf";

// The size the atlas is rasterized at; labels scale it to `TEXT_PX`.
const BAKE_PX: u32 = 20;

// The on-screen size code text draws at.
pub(crate) const TEXT_PX: f32 = 14.0;

// The `TextLabel.scale` that draws the baked face at `TEXT_PX`.
pub(crate) const SCALE: f32 = TEXT_PX / BAKE_PX as f32;

// The compiled atlas, computed on the first call. A failure is kept: it can
// only mean the bundled face is broken, and retrying would pay for the same
// failure again.
struct Baked {
    payload: Vec<u8>,
}

fn baked() -> Result<&'static Baked, &'static str> {
    static BAKED: OnceLock<Result<Baked, String>> = OnceLock::new();
    BAKED
        .get_or_init(bake_face)
        .as_ref()
        .map_err(String::as_str)
}

fn bake_face() -> Result<Baked, String> {
    let payload = bake::font::compile(FONT_BYTES, BAKE_PX, FONT_FILE)?;
    Ok(Baked { payload })
}

// Append the face to `world`'s font table. `None` (logged) leaves code labels
// on the built-in face: misaligned columns beat invisible text.
pub(crate) fn inject(world: &mut World) -> Option<FontHandle> {
    match baked() {
        Ok(b) => Some(concinnity_engine::resource::append_font(
            &mut world.context(),
            b.payload.clone(),
        )),
        Err(e) => {
            tracing::error!("the editor's code font failed to bake: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_face_bakes_with_a_monospace_advance() {
        let b = baked().expect("the bundled face bakes");
        let (_, _, _, _, _, metrics) = bake::font::deserialize(&b.payload).unwrap();
        let glyph_advance = |c: char| {
            metrics
                .iter()
                .find(|m| m.char_code == c as u32)
                .map(|m| m.advance_px)
                .unwrap()
        };
        assert_eq!(
            glyph_advance('i'),
            glyph_advance('W'),
            "every glyph shares one advance"
        );
        assert_eq!(glyph_advance('.'), glyph_advance('m'));
    }

    #[test]
    fn inject_appends_a_face_to_the_world() {
        let mut world = World::new();
        let first = inject(&mut world).expect("a handle");
        let second = inject(&mut world).expect("a handle");
        assert_ne!(first, second, "each call appends");
    }
}
