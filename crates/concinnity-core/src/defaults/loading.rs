// The loading overlay a streamed world waits behind: the screen it shows, a
// black backdrop, a progress track and its fill, and the label above them.
// Synthesized for a world that jumps between streamed scenes; an overlay the
// world declares is completed the same way wherever it appears, since
// declaring one is the opt-in.

use alloc::string::ToString;

use crate::components::{LoadingOverlay, Scene, Screen, Sprite, StreamingConfig, TextLabel};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::result::CnResult;

use super::Minter;

// The reference canvas the injected pieces are laid out on.
const CANVAS_WIDTH: f32 = 1280.0;
const CANVAS_HEIGHT: f32 = 720.0;
// The progress bar: a centred strip near the bottom of the canvas.
const BAR_X: f32 = 400.0;
const BAR_Y: f32 = 600.0;
const BAR_WIDTH: f32 = 480.0;
const BAR_HEIGHT: f32 = 8.0;
const BAR_CORNER: f32 = 4.0;

pub(super) fn inject(ctx: &mut PipelineContext, minter: &mut Minter) -> Result<(), CnResult> {
    let declared = ctx.query::<LoadingOverlay>().next().is_some();
    if !declared && !streams_scenes(ctx) {
        return Ok(());
    }
    let mut overlay = ctx
        .query::<LoadingOverlay>()
        .next()
        .cloned()
        .unwrap_or_default();

    // The screen first: every piece minted below belongs to it, whether it was
    // authored or minted here.
    let screen = match overlay.screen {
        Some(screen) => screen,
        None => {
            let id = minter.id();
            ctx.push(Screen {
                asset_id: id,
                fade_in_secs: 0.15,
                ..Default::default()
            });
            overlay.screen = Some(id);
            id
        }
    };

    if overlay.backdrop.is_none() {
        overlay.backdrop = Some(sprite(
            ctx,
            minter,
            screen,
            Sprite {
                width: CANVAS_WIDTH,
                height: CANVAS_HEIGHT,
                tint: [0.0, 0.0, 0.0, 1.0],
                fit: crate::components::SpriteFit::Cover,
                ..Default::default()
            },
        ));
    }
    if overlay.track.is_none() {
        overlay.track = Some(sprite(
            ctx,
            minter,
            screen,
            Sprite {
                x: BAR_X,
                y: BAR_Y,
                width: BAR_WIDTH,
                height: BAR_HEIGHT,
                tint: [0.25, 0.25, 0.25, 1.0],
                corner_radius: BAR_CORNER,
                ..Default::default()
            },
        ));
    }
    if overlay.fill.is_none() {
        // Zero-width and hidden until the overlay drives it.
        overlay.fill = Some(sprite(
            ctx,
            minter,
            screen,
            Sprite {
                x: BAR_X,
                y: BAR_Y,
                width: 0.0,
                height: BAR_HEIGHT,
                tint: [0.92, 0.92, 0.92, 1.0],
                corner_radius: BAR_CORNER,
                visible: false,
                ..Default::default()
            },
        ));
    }
    if overlay.label.is_none() {
        let font = minter.hud_font(ctx)?;
        let id = minter.id();
        ctx.push(TextLabel {
            asset_id: id,
            font: Some(font),
            content: "Loading".to_string(),
            x: CANVAS_WIDTH / 2.0,
            y: BAR_Y - 34.0,
            align: crate::components::TextAlign::Center,
            screen: Some(screen),
            ..Default::default()
        });
        overlay.label = Some(id);
    }

    match ctx.query_mut::<LoadingOverlay>().next() {
        Some(existing) => *existing = overlay,
        None => ctx.push(overlay),
    }
    Ok(())
}

// A world with streamed scenes to wait for: without both there is nothing the
// overlay would cover.
fn streams_scenes(ctx: &PipelineContext) -> bool {
    ctx.query::<Scene>().next().is_some() && ctx.query::<StreamingConfig>().next().is_some()
}

// Push one overlay sprite onto the overlay's screen and return its name.
fn sprite(
    ctx: &mut PipelineContext,
    minter: &mut Minter,
    screen: AssetId,
    sprite: Sprite,
) -> AssetId {
    let id = minter.id();
    ctx.push(Sprite {
        asset_id: id,
        screen: Some(screen),
        ..sprite
    });
    id
}
