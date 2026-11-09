// The menu's scroll panels: their rows and collapsible groups, the wheel and
// thumb input that scrolls them, and the per-frame layout that moves their
// elements.

use super::region_ops::point_in_rect;
use super::scroll_layout::RowSpec;
use super::{UiInputSystem, scroll_layout};
use crate::components::{FrameInput, ScrollPanel, Sprite, TextLabel, UiAction};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, Ref};
use crate::gfx::overlay::OverlayTransform;
use alloc::string::String;
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;
use hashbrown::HashMap;

// How many reference-space pixels one unit of scroll-wheel delta moves a panel.
pub(super) const WHEEL_SCROLL_SPEED: f32 = 2.0;

// One row of a scroll panel: the elements that move together, their authored
// y's (snapshot at init so the reflow is `base + dy`), the row's top + height
// (for bucketing regions to rows), and the collapsible group it belongs to.
#[derive(Debug)]
pub(super) struct RowState {
    pub(super) elements: Vec<AssetId>,
    base_ys: Vec<f32>,
    pub(super) base_y: f32,
    pub(super) height: f32,
    pub(super) group: Option<usize>,
}

// A collapsible group's runtime state: whether it is collapsed and the header
// label whose `+`/`-` prefix reflects it.
#[derive(Debug)]
pub(super) struct GroupState {
    pub(super) collapsed: bool,
    pub(super) header: Option<AssetId>,
    pub(super) title: String,
}

// Runtime state for one scroll panel, drained from a `ScrollPanel` at init.
#[derive(Debug)]
pub(super) struct PanelState {
    pub(super) screen: Option<AssetId>,
    // Content band [x, y, width, height] in reference space.
    pub(super) band: [f32; 4],
    pub(super) rows: Vec<RowState>,
    pub(super) groups: Vec<GroupState>,
    pub(super) thumb: Option<AssetId>,
    pub(super) track: Option<AssetId>,
    pub(super) track_x: f32,
    pub(super) track_y: f32,
    pub(super) track_w: f32,
    pub(super) track_h: f32,
    // Current scroll offset (reference pixels), clamped by the solver.
    pub(super) scroll: f32,
    // Last solve outputs, kept for the thumb-drag cursor->scroll mapping.
    pub(super) content_height: f32,
    pub(super) thumb_h: f32,
}

// One sprite's accumulated scroll-layout write; only the set fields apply.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct SpriteUpdate {
    pub(super) y: Option<f32>,
    pub(super) height: Option<f32>,
    pub(super) visible: Option<bool>,
}

// One label's accumulated scroll-layout write; only the set fields apply.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct LabelUpdate {
    pub(super) y: Option<f32>,
    pub(super) visible: Option<bool>,
    // `(panel, group)` of the collapsible group whose `+`/`-` prefixed title
    // this label shows; the content is written in place at apply time.
    pub(super) header: Option<(usize, usize)>,
}

// One panel's solve output, kept for the region reflow and reused (the
// placement buffer keeps its capacity) across frames.
#[derive(Debug, Default)]
pub(super) struct PanelSolve {
    pub(super) active: bool,
    pub(super) rows: Vec<scroll_layout::RowPlacement>,
}

// Scratch buffers for the per-frame scroll-layout solve/apply, kept on the
// system so the pass reuses their capacity instead of reallocating each frame.
#[derive(Debug, Default)]
pub(super) struct LayoutScratch {
    pub(super) sprites: HashMap<AssetId, SpriteUpdate>,
    pub(super) labels: HashMap<AssetId, LabelUpdate>,
    specs: Vec<RowSpec>,
    pub(super) collapsed: Vec<bool>,
    // Per-panel solve output, indexed like `panels`.
    pub(super) solved: Vec<PanelSolve>,
}

impl UiInputSystem {
    // Drain the world's ScrollPanels into runtime state: snapshot each row
    // element's authored y (so the reflow is `base + dy`), translate the
    // `i32` group index into an `Option`, and bucket each HitRegion into the
    // panel row whose band its center falls in (so the region reflows + gates
    // with that row). Runs once at init, after HitRegions are drained.
    pub(super) fn init_panels(&mut self, ctx: &mut PipelineContext<'_>) {
        let panels = ctx.drain::<ScrollPanel>();
        if panels.is_empty() {
            return;
        }
        // Snapshot the authored y of every element any panel row references.
        let wanted: hashbrown::HashSet<AssetId> = panels
            .iter()
            .flat_map(|p| {
                p.rows
                    .iter()
                    .flat_map(|r| r.elements.iter().map(|e| e.id()))
            })
            .collect();
        let elem_y: HashMap<AssetId, f32> = wanted
            .iter()
            .filter_map(|&id| {
                let y = ctx
                    .get_by_id::<TextLabel>(id)
                    .map(|l| l.y)
                    .or_else(|| ctx.get_by_id::<Sprite>(id).map(|s| s.y))?;
                Some((id, y))
            })
            .collect();

        for p in panels {
            let rows = p
                .rows
                .iter()
                .map(|r| {
                    let base_ys = r
                        .elements
                        .iter()
                        .map(|id| elem_y.get(&id.id()).copied().unwrap_or(r.base_y))
                        .collect();
                    RowState {
                        elements: r.elements.iter().map(|e| e.id()).collect(),
                        base_ys,
                        base_y: r.base_y,
                        height: r.height,
                        group: (r.group >= 0).then_some(usize::saturating_from(r.group)),
                    }
                })
                .collect();
            let groups = p
                .groups
                .iter()
                .map(|g| GroupState {
                    collapsed: g.collapsed,
                    header: g.header.map(Ref::id),
                    title: g.title.clone(),
                })
                .collect();
            self.panels.push(PanelState {
                screen: p.screen.map(Ref::id),
                band: [p.x, p.y, p.width, p.height],
                rows,
                groups,
                thumb: p.thumb.map(Ref::id),
                track: p.track.map(Ref::id),
                track_x: p.track_x,
                track_y: p.track_y,
                track_w: p.track_w,
                track_h: p.track_h,
                scroll: 0.0,
                content_height: 0.0,
                thumb_h: 0.0,
            });
        }

        // Bucket each panel-content region into its row by center y. Only
        // content regions (a settings action or a group toggle) are bucketed;
        // chrome regions (tabs, and Back, which is a `show`) are left fixed even when an
        // overflow row's authored y reaches their position. Panels read
        // immutably while the regions are mutated (disjoint fields).
        let panels = &self.panels;
        for entry in &mut self.regions {
            let is_content = matches!(
                entry.region.action,
                Some(UiAction::Setting { .. } | UiAction::GroupToggle(_))
            );
            if !is_content {
                continue;
            }
            let cy = entry.region.y + entry.region.height * 0.5;
            'find: for (pi, panel) in panels.iter().enumerate() {
                if panel.screen != entry.screen {
                    continue;
                }
                for (ri, row) in panel.rows.iter().enumerate() {
                    if cy >= row.base_y && cy < row.base_y + row.height {
                        entry.scroll_row = Some((pi, ri));
                        break 'find;
                    }
                }
            }
        }
    }

    // The reference-space rectangle of a panel's scrollbar thumb at its current
    // scroll, or `None` if the panel does not overflow (no thumb to grab).
    pub(super) fn thumb_rect(panel: &PanelState) -> Option<[f32; 4]> {
        if panel.content_height <= 0.0 || panel.thumb_h >= panel.track_h {
            return None;
        }
        let offset_frac = (panel.scroll / panel.content_height).clamp(0.0, 1.0);
        let thumb_y = panel.track_y + offset_frac * panel.track_h;
        Some([panel.track_x, thumb_y, panel.track_w, panel.thumb_h])
    }

    // Apply scroll-wheel + scrollbar-thumb input to the active screen's panel.
    // Returns true while the thumb is being dragged so the caller suppresses
    // the slider + click passes. The solver clamps the resulting scroll
    // offset. (The arrow keys move the focus cursor instead, which scrolls its
    // own row into view.)
    pub(super) fn handle_scroll_input(
        &mut self,
        input: &FrameInput,
        active_screen: Option<AssetId>,
        overlay: &OverlayTransform,
    ) -> bool {
        let (qx, qy) = overlay.inverse(input.mouse_x, input.mouse_y);
        let active_panel = self.panels.iter().position(|p| p.screen == active_screen);

        // Wheel: scroll the active panel while the cursor is over its band.
        if input.scroll_delta != 0.0
            && let Some(pi) = active_panel
            && point_in_rect(qx, qy, self.panels[pi].band)
        {
            self.panels[pi].scroll += input.scroll_delta * WHEEL_SCROLL_SPEED;
        }

        // Thumb drag: begin on the press edge over the thumb, then map the
        // cursor's y to a scroll offset for the rest of the press.
        if input.left_button_down {
            if self.thumb_drag.is_none()
                && input.left_click
                && let Some(pi) = active_panel
                && let Some(rect) = Self::thumb_rect(&self.panels[pi])
                && point_in_rect(qx, qy, rect)
            {
                self.thumb_drag = Some((pi, qy - rect[1]));
            }
            if let Some((pi, grab)) = self.thumb_drag {
                let panel = &mut self.panels[pi];
                let travel = (panel.track_h - panel.thumb_h).max(0.0);
                let max_scroll = (panel.content_height - panel.band[3]).max(0.0);
                if travel > 0.0 && max_scroll > 0.0 {
                    let thumb_top = (qy - grab).clamp(panel.track_y, panel.track_y + travel);
                    let frac = (thumb_top - panel.track_y) / travel;
                    panel.scroll = frac * max_scroll;
                }
            }
        } else {
            self.thumb_drag = None;
        }
        self.thumb_drag.is_some()
    }

    // Flip one collapsible group on the active screen's panel.
    pub(super) fn toggle_group(&mut self, active_screen: Option<AssetId>, gid: usize) {
        if let Some(panel) = self.panels.iter_mut().find(|p| p.screen == active_screen)
            && let Some(g) = panel.groups.get_mut(gid)
        {
            g.collapsed = !g.collapsed;
        }
    }

    // Solve every panel's vertical layout and write the result back: element y +
    // visibility, region reflow + hidden flag, the scrollbar thumb position +
    // size, and each group header's `+`/`-` prefix. Only the active screen's panel
    // writes (an inactive screen's elements stay hidden by the screen system). Runs
    // at init and at the end of each step so the next frame draws + hit-tests the
    // reflowed positions consistently.
    pub(super) fn apply_scroll_layout(&mut self, ctx: &mut PipelineContext<'_>) {
        if self.panels.is_empty() {
            return;
        }
        let active = self.screens.top_capture();

        // Accumulate component writes into the reused scratch maps, then apply
        // in one pass per component type.
        self.layout.sprites.clear();
        self.layout.labels.clear();
        self.layout
            .solved
            .resize_with(self.panels.len(), Default::default);

        for (pi, panel) in self.panels.iter_mut().enumerate() {
            let panel_active = panel.screen == active;
            self.layout.collapsed.clear();
            self.layout
                .collapsed
                .extend(panel.groups.iter().map(|g| g.collapsed));
            self.layout.specs.clear();
            self.layout.specs.extend(panel.rows.iter().map(|r| RowSpec {
                height: r.height,
                group: r.group,
            }));
            self.layout.solved[pi].active = panel_active;
            let solved = scroll_layout::solve_into(
                &self.layout.specs,
                &self.layout.collapsed,
                panel.band[3],
                panel.scroll,
                &mut self.layout.solved[pi].rows,
            );
            panel.scroll = solved.scroll;
            panel.content_height = solved.content_height;
            panel.thumb_h = solved.thumb_frac * panel.track_h;

            if panel_active {
                for (ri, row) in panel.rows.iter().enumerate() {
                    let pl = self.layout.solved[pi].rows[ri];
                    for (k, id) in row.elements.iter().enumerate() {
                        let y = row.base_ys[k] + pl.dy;
                        let s = self.layout.sprites.entry(*id).or_default();
                        s.y = Some(y);
                        s.visible = Some(pl.visible);
                        let l = self.layout.labels.entry(*id).or_default();
                        l.y = Some(y);
                        l.visible = Some(pl.visible);
                    }
                }
                let scrollable = solved.scrollable();
                if let Some(thumb) = panel.thumb {
                    let thumb_y = panel.track_y + solved.thumb_offset_frac * panel.track_h;
                    let s = self.layout.sprites.entry(thumb).or_default();
                    s.y = Some(thumb_y);
                    s.height = Some(panel.thumb_h);
                    s.visible = Some(scrollable);
                }
                if let Some(track) = panel.track {
                    self.layout.sprites.entry(track).or_default().visible = Some(scrollable);
                }
                for (gi, g) in panel.groups.iter().enumerate() {
                    if let Some(h) = g.header {
                        self.layout.labels.entry(h).or_default().header = Some((pi, gi));
                    }
                }
            }
        }

        // Reflow each panel-owned region in memory (positions the click loop
        // hit-tests against next frame).
        for entry in &mut self.regions {
            if let Some((pi, ri)) = entry.scroll_row
                && let Some(solve) = self.layout.solved.get(pi)
                && solve.active
            {
                let pl = solve.rows[ri];
                entry.region.y = entry.region_base_y + pl.dy;
                entry.hidden = !pl.visible;
            }
        }

        // Apply the accumulated component writes.
        for (&id, u) in &self.layout.sprites {
            if let Some(s) = ctx.get_mut_by_id::<Sprite>(id) {
                if let Some(y) = u.y {
                    s.y = y;
                }
                if let Some(h) = u.height {
                    s.height = h;
                }
                if let Some(vis) = u.visible {
                    s.visible = vis;
                }
            }
        }
        for (&id, u) in &self.layout.labels {
            if let Some(l) = ctx.get_mut_by_id::<TextLabel>(id) {
                if let Some(y) = u.y {
                    l.y = y;
                }
                if let Some(vis) = u.visible {
                    l.visible = vis;
                }
                // A group header's `+`/`-` prefixed title, rewritten in place
                // so the label's String keeps its capacity.
                if let Some((pi, gi)) = u.header {
                    let g = &self.panels[pi].groups[gi];
                    l.content.clear();
                    l.content.push_str(if g.collapsed { "+ " } else { "- " });
                    l.content.push_str(&g.title);
                }
            }
        }
    }
}
