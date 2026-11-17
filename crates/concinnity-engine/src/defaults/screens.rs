// Every overlay element belongs to a screen. One the world leaves on none, an
// injected HUD chip or a label added in code, joins a persistent window-space
// screen minted for them, which draws in the HUD band and takes input while no
// capturing screen is open.

use crate::components::{
    HitRegion, Layout, Screen, ScreenInput, ScreenSpace, ScrollPanel, Sprite, TextInput, TextLabel,
};
use crate::ecs::{PipelineContext, Ref};
use crate::error::WorldError;

use super::Minter;

pub(super) fn adopt_screenless(
    ctx: &mut PipelineContext<'_>,
    minter: &mut Minter,
) -> Result<(), WorldError> {
    let screenless = ctx.query::<Sprite>().any(|e| e.screen.is_none())
        || ctx.query::<TextLabel>().any(|e| e.screen.is_none())
        || ctx.query::<TextInput>().any(|e| e.screen.is_none())
        || ctx.query::<HitRegion>().any(|e| e.screen.is_none())
        || ctx.query::<ScrollPanel>().any(|e| e.screen.is_none())
        || ctx.query::<Layout>().any(|e| e.screen.is_none());
    if !screenless {
        return Ok(());
    }
    let id = minter.id()?;
    ctx.push_identified(
        id,
        Screen {
            persistent: true,
            space: ScreenSpace::Window,
            input: ScreenInput::Passthrough,
            pauses_world: false,
            ..Default::default()
        },
    );
    let hud = Some(Ref::new(id));
    for e in ctx.query_mut::<Sprite>() {
        e.screen = e.screen.or(hud);
    }
    for e in ctx.query_mut::<TextLabel>() {
        e.screen = e.screen.or(hud);
    }
    for e in ctx.query_mut::<TextInput>() {
        e.screen = e.screen.or(hud);
    }
    for e in ctx.query_mut::<HitRegion>() {
        e.screen = e.screen.or(hud);
    }
    for e in ctx.query_mut::<ScrollPanel>() {
        e.screen = e.screen.or(hud);
    }
    for e in ctx.query_mut::<Layout>() {
        e.screen = e.screen.or(hud);
    }
    Ok(())
}
