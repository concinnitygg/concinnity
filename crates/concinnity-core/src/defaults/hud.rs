// The HUD chips: one TextLabel per readout a HUD names, and the font they draw
// with. A HUD the world declares keeps every label it set and receives chips
// only for the slots it left unset; the debug HUD is synthesized outright when
// the world declares none.

use alloc::string::String;

use crate::components::{DebugHud, StatHud, TextLabel};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{ComponentSlot, FontHandle};
use crate::result::CnResult;

use super::Minter;

// A HUD's label slots, in chip order. Function pointers rather than field
// offsets so the two HUD types share one filling pass.
type Slot<H> = fn(&mut H) -> &mut Option<AssetId>;

const DEBUG_SLOTS: [Slot<DebugHud>; 4] = [
    |h| &mut h.passes_label,
    |h| &mut h.mouse_label,
    |h| &mut h.camera_label,
    |h| &mut h.sys_label,
];

const STAT_SLOTS: [Slot<StatHud>; 5] = [
    |h| &mut h.fps_label,
    |h| &mut h.vram_label,
    |h| &mut h.ram_label,
    |h| &mut h.ev_label,
    |h| &mut h.edr_label,
];

// Every rendering world gets the developer HUD; only a world that declares a
// StatHud gets its chips, since the stats strip is driven by menu toggles the
// runtime cannot see.
pub(super) fn inject_debug_hud(
    ctx: &mut PipelineContext,
    minter: &mut Minter,
) -> Result<(), CnResult> {
    complete(ctx, minter, &DEBUG_SLOTS, true)
}

pub(super) fn complete_stat_hud(
    ctx: &mut PipelineContext,
    minter: &mut Minter,
) -> Result<(), CnResult> {
    complete(ctx, minter, &STAT_SLOTS, false)
}

// Fill the unset label slots of the world's HUD with freshly minted chips.
// `synthesize` adds the HUD itself when the world declares none.
fn complete<H>(
    ctx: &mut PipelineContext,
    minter: &mut Minter,
    slots: &[Slot<H>],
    synthesize: bool,
) -> Result<(), CnResult>
where
    H: ComponentSlot + Clone + Default,
{
    let declared = ctx.query::<H>().next().is_some();
    if !declared && !synthesize {
        return Ok(());
    }
    let mut hud = ctx.query::<H>().next().cloned().unwrap_or_default();

    let unset: alloc::vec::Vec<usize> = (0..slots.len())
        .filter(|&i| slots[i](&mut hud).is_none())
        .collect();
    if !unset.is_empty() {
        let font = minter.hud_font(ctx)?;
        for i in unset {
            let id = minter.id();
            *slots[i](&mut hud) = Some(id);
            ctx.push(chip(id, font));
        }
    }

    match ctx.query_mut::<H>().next() {
        Some(existing) => *existing = hud,
        None => ctx.push(hud),
    }
    Ok(())
}

// The chip a HUD readout writes into: small, light-on-dark, with a padded
// background box so it stays legible over any scene.
fn chip(id: AssetId, font: FontHandle) -> TextLabel {
    TextLabel {
        asset_id: id,
        font: Some(font),
        content: String::new(),
        scale: 0.7,
        color: [1.0, 1.0, 1.0],
        background: [0.0, 0.18, 0.32, 0.85],
        padding: 5.0,
        ..Default::default()
    }
}
