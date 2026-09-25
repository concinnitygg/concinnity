//! The monospace face code text draws with: JetBrains Mono Regular, under the
//! SIL Open Font License 1.1 shipped beside the font file. It lives in this
//! crate, so only the editor carries it; the atlas is baked once for the
//! process and appended to each world the editor injects, like the HUD face.

use std::sync::OnceLock;

use concinnity_core::bake;
use concinnity_core::ecs::{FontHandle, World};

const FONT_BYTES: &[u8] = include_bytes!("../../assets/fonts/JetBrainsMono-Regular.ttf");
const FONT_FILE: &str = "JetBrainsMono-Regular.ttf";

// The size the atlas is rasterized at; labels scale it to `TEXT_PX`.
const BAKE_PX: u32 = 20;

// The on-screen size code text draws at, and the row pitch it sits on.
pub(crate) const TEXT_PX: f32 = 14.0;
pub(crate) const LINE_H: f32 = 19.0;

// The `TextLabel.scale` that draws the baked face at `TEXT_PX`.
pub(crate) const SCALE: f32 = TEXT_PX / BAKE_PX as f32;

// The compiled atlas and the face's advance per pixel of size, computed on the
// first call. A failure is kept: it can only mean the bundled face is broken,
// and retrying would pay for the same failure again.
struct Baked {
    payload: Vec<u8>,
    advance_per_px: f32,
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
    let (_, _, _, size_px, _, metrics) = bake::font::deserialize(&payload)?;
    let space = metrics
        .iter()
        .find(|m| m.char_code == u32::from(b' '))
        .ok_or("the monospace face has no space glyph")?;
    Ok(Baked {
        advance_per_px: space.advance_px / size_px as f32,
        payload,
    })
}

// The horizontal advance of one character at `TEXT_PX`. Every glyph of a
// monospace face shares it, which is what lets a column map to a pixel.
pub(crate) fn advance() -> f32 {
    match baked() {
        Ok(b) => b.advance_per_px * TEXT_PX,
        // The face's design advance, so layout stays usable if the bake failed.
        Err(_) => 0.6 * TEXT_PX,
    }
}

// Append the face to `world`'s font table. `None` (logged) leaves code labels
// on the built-in face: misaligned columns beat invisible text.
pub(crate) fn inject(world: &mut World) -> Option<FontHandle> {
    match baked() {
        Ok(b) => Some(concinnity_core::resource::append_font(
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
        let (_, _, _, size_px, _, metrics) = bake::font::deserialize(&b.payload).unwrap();
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
        assert!((glyph_advance(' ') / size_px as f32 - b.advance_per_px).abs() < 1e-6);
        assert!(advance() > 0.4 * TEXT_PX && advance() < 0.8 * TEXT_PX);
    }

    #[test]
    fn inject_appends_a_face_to_the_world() {
        let mut world = World::new();
        let first = inject(&mut world).expect("a handle");
        let second = inject(&mut world).expect("a handle");
        assert_ne!(first, second, "each call appends");
    }
}
