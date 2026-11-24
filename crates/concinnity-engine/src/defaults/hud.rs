// The HUD chips: one TextLabel per readout a HUD names, and the font they draw
// with. A HUD the world declares keeps every label it set and receives chips
// only for the slots it left unset; the debug HUD is synthesized outright when
// the world declares none.

use alloc::string::String;

use crate::components::{
    DebugHud, Layout, LayoutAlign, LayoutAnchor, LayoutChild, StatHud, TextLabel,
};
use crate::ecs::{ComponentSlot, FontHandle, PipelineContext, Ref};
use crate::error::WorldError;

use super::Minter;

// A HUD's label slots, in chip order. Function pointers rather than field
// offsets so the two HUD types share one filling pass.
type Slot<H> = fn(&mut H) -> &mut Option<Ref<TextLabel>>;

const DEBUG_SLOTS: [Slot<DebugHud>; 4] = [
    |h| &mut h.passes_label,
    |h| &mut h.mouse_label,
    |h| &mut h.camera_label,
    |h| &mut h.sys_label,
];

const STAT_SLOTS: [Slot<StatHud>; 17] = [
    |h| &mut h.fps_label,
    |h| &mut h.worst_frame_label,
    |h| &mut h.cpu_label,
    |h| &mut h.gpu_label,
    |h| &mut h.allocs_label,
    |h| &mut h.vram_label,
    |h| &mut h.ram_label,
    |h| &mut h.streaming_label,
    |h| &mut h.ev_label,
    |h| &mut h.edr_label,
    |h| &mut h.time_label,
    |h| &mut h.phase_label,
    |h| &mut h.api_label,
    |h| &mut h.resolution_label,
    |h| &mut h.objects_label,
    |h| &mut h.draws_label,
    |h| &mut h.speed_label,
];

// Every rendering world gets the developer HUD; only a world that declares a
// StatHud gets its chips, since the stats strip is driven by menu toggles the
// runtime cannot see.
pub(super) fn inject_debug_hud(
    ctx: &mut PipelineContext<'_>,
    minter: &mut Minter,
) -> Result<(), WorldError> {
    complete(ctx, minter, &DEBUG_SLOTS, true)?;
    let Some(hud) = ctx.query::<DebugHud>().next().cloned() else {
        return Ok(());
    };
    // Stacked down the top-right corner, each chip flush right; passes last,
    // since its height follows the frame's step count.
    let chips = [
        hud.mouse_label,
        hud.camera_label,
        hud.sys_label,
        hud.passes_label,
    ];
    if laid_out(ctx, &chips) {
        return Ok(());
    }
    ctx.push(Layout {
        children: fit_children(&chips),
        gap: 6.0,
        align: LayoutAlign::End,
        anchor: LayoutAnchor::TopRight,
        margin: [MARGIN, MARGIN],
        ..Layout::default()
    });
    Ok(())
}

pub(super) fn complete_stat_hud(
    ctx: &mut PipelineContext<'_>,
    minter: &mut Minter,
) -> Result<(), WorldError> {
    complete(ctx, minter, &STAT_SLOTS, false)?;
    let Some(hud) = ctx.query::<StatHud>().next().cloned() else {
        return Ok(());
    };
    // Tight rows down the top-left corner: the top row, the benchmark row,
    // then the subject row. A row with nothing to show takes no space.
    const GAP: f32 = 4.0;
    if laid_out(ctx, &hud.top_row())
        || laid_out(ctx, &hud.bench_row())
        || laid_out(ctx, &hud.subject_row())
    {
        return Ok(());
    }
    let mut rows = alloc::vec::Vec::new();
    for chips in [&hud.top_row()[..], &hud.bench_row(), &hud.subject_row()] {
        let id = minter.id()?;
        ctx.push_identified(
            id,
            Layout {
                direction: crate::components::LayoutDirection::Row,
                children: fit_children(chips),
                gap: GAP,
                ..Layout::default()
            },
        );
        rows.push(LayoutChild::fit(id));
    }
    ctx.push(Layout {
        children: rows,
        gap: GAP,
        margin: [MARGIN, MARGIN],
        ..Layout::default()
    });
    Ok(())
}

// The distance (window pixels) the HUD strips keep from the window corners.
const MARGIN: f32 = 10.0;

// Whether a layout already holds any of `chips`, as on a second completion of
// the same world.
fn laid_out(ctx: &PipelineContext<'_>, chips: &[Option<Ref<TextLabel>>]) -> bool {
    let held = |item: Option<Ref<crate::components::LayoutItem>>| {
        item.is_some_and(|i| chips.iter().flatten().any(|c| c.id() == i.id()))
    };
    ctx.query::<Layout>()
        .any(|l| l.children.iter().any(|c| held(c.item)))
}

// Each chip a HUD names, fitted to its text.
fn fit_children(chips: &[Option<Ref<TextLabel>>]) -> alloc::vec::Vec<LayoutChild> {
    chips
        .iter()
        .flatten()
        .map(|chip| LayoutChild::fit(chip.id()))
        .collect()
}

// Fill the unset label slots of the world's HUD with freshly minted chips.
// `synthesize` adds the HUD itself when the world declares none.
fn complete<H>(
    ctx: &mut PipelineContext<'_>,
    minter: &mut Minter,
    slots: &[Slot<H>],
    synthesize: bool,
) -> Result<(), WorldError>
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
        let font = super::hud_font(ctx)?;
        for i in unset {
            let id = minter.id()?;
            *slots[i](&mut hud) = Some(Ref::new(id));
            ctx.push_identified(id, chip(font));
        }
    }

    match ctx.query_mut::<H>().next() {
        Some(existing) => *existing = hud,
        None => {
            ctx.push(hud);
        }
    }
    Ok(())
}

// The chip a HUD readout writes into: small, light-on-dark, with a padded
// background box so it stays legible over any scene.
fn chip(font: FontHandle) -> TextLabel {
    TextLabel {
        font: Some(font),
        content: String::new(),
        scale: 0.7,
        color: [1.0, 1.0, 1.0],
        background: [0.0, 0.18, 0.32, 0.85],
        padding: 5.0,
        ..Default::default()
    }
}
