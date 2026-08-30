//! Engine defaults: complete a loaded world with the standard components it
//! does not declare itself.
//!
//! [`run`] is the [`SystemTable`](crate::ecs::SystemTable) completion pass, so
//! it happens before the gates read the world and a HUD or overlay it injects
//! brings its own system into the schedule. Nothing it adds is compiled: the
//! chip font and the sky mesh are baked here out of [`crate::bake`], and the
//! names they cross-reference each other by come from the range
//! [`AssetId::MINTED_BASE`] reserves.
//!
//! A world that renders -- one declaring a
//! [`GraphicsConfig`] -- receives the HUD,
//! sky, and loading defaults. The physics default is gated on physics content
//! instead, so the headless tier, which has no renderer and needs no HUD, gets
//! the [`PhysicsConfig`](crate::components::PhysicsConfig) its simulation
//! already runs on and nothing else.
//!
//! Each default yields to what the world declares: an authored HUD keeps every
//! label it names and receives chips only for the slots it leaves unset, and a
//! world with its own skybox geometry gets no sky mesh. An
//! [`EngineDefaults`] turns individual
//! defaults off entirely; the world holds at most one, and its column is
//! drained here.

mod font;
mod hud;
mod loading;
mod physics;
mod sky;

use crate::components::{EngineDefaults, GraphicsConfig};
use crate::ecs::asset_id::{AssetId, MintedIds};
use crate::ecs::{FontHandle, PipelineContext};
use crate::result::CnResult;

pub use font::{HUD_FONT_SIZE_PX, HudFont, hud_font};

/// Complete `world`'s content with the engine defaults it does not opt out of.
///
/// Errors when the world declares more than one `EngineDefaults` (which one
/// applies would be arbitrary), or when baking an injected payload fails.
pub fn run(ctx: &mut PipelineContext) -> Result<(), CnResult> {
    let toggles = take_toggles(ctx)?;
    let mut minter = Minter::resume(ctx);
    let result = inject_defaults(ctx, &toggles, &mut minter);
    minter.store(ctx);
    result
}

fn inject_defaults(
    ctx: &mut PipelineContext,
    toggles: &EngineDefaults,
    minter: &mut Minter,
) -> Result<(), CnResult> {
    if toggles.physics_config {
        physics::inject(ctx);
    }
    // The rest exists to be drawn, and a world with no GraphicsConfig has no
    // renderer to draw it.
    if ctx.query::<GraphicsConfig>().next().is_none() {
        return Ok(());
    }
    if toggles.sky {
        sky::inject(ctx, minter)?;
    }
    if toggles.hud {
        hud::complete_stat_hud(ctx, minter)?;
    }
    if toggles.debug_hud {
        hud::inject_debug_hud(ctx, minter)?;
    }
    if toggles.loading_overlay {
        loading::inject(ctx, minter)?;
    }
    Ok(())
}

// Drain the world's EngineDefaults column into the one set of toggles that
// applies. The type is a build directive rather than something a system reads,
// so it holds nothing past this pass.
fn take_toggles(ctx: &mut PipelineContext) -> Result<EngineDefaults, CnResult> {
    let mut declared = ctx.drain::<EngineDefaults>();
    if declared.len() > 1 {
        return Err(CnResult::InvalidState);
    }
    Ok(declared.pop().unwrap_or_default())
}

/// The source of names for what the defaults inject.
#[derive(Default)]
pub(crate) struct Minter {
    ids: MintedIds,
}

impl Minter {
    // Continue from the world's own counter, so nothing minted before this
    // pass (a mesh handed over through `World::add_mesh`) is renamed.
    fn resume(ctx: &PipelineContext) -> Self {
        Self {
            ids: ctx.resource::<MintedIds>().cloned().unwrap_or_default(),
        }
    }

    // Hand the counter back to the world once the pass is done with it.
    fn store(self, ctx: &mut PipelineContext) {
        ctx.insert_resource(self.ids);
    }

    // The next name for an injected component.
    fn id(&mut self) -> AssetId {
        self.ids.next_id()
    }

    // The font every injected chip and label draws with. Baked into the world
    // on first use and shared from there, so the chips of both HUDs and the
    // loading label land on one atlas.
    fn hud_font(&mut self, ctx: &mut PipelineContext) -> Result<FontHandle, CnResult> {
        font::hud_font(ctx)
    }
}

#[cfg(test)]
mod tests;
