//! `EditorHook`: the panels built from engine widgets. The engine drags, sizes,
//! raises, and drives them; the hook keeps its own model of each (open, where,
//! how big, its place in the focus stack) in step with the world it is driving,
//! so a rebuilt world opens them where the user left them, and handles the
//! events their widgets send.

use concinnity_engine::components::{FrameInput, Layout, LayoutSize, UiEvent, UiValue};
use concinnity_engine::ecs::World;

use super::EditorHook;
use crate::editor::hud;
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::ui::{BuiltUi, CLOSE_EVENT};

impl EditorHook {
    // The rect panel `key` was last laid out at, when it is built and shown.
    pub(super) fn built_rect(world: &World, key: PanelKey) -> Option<[f32; 4]> {
        let built = world.resource::<BuiltUi>()?.panel(key)?;
        let layout = world.get_by_id::<Layout>(built.root)?;
        layout.visible.then_some(layout.rect).flatten()
    }

    // Sync every built panel to the hook's model: visibility, focus order,
    // content size, and on screen below the top bar. User moves and presses
    // since last frame update the model first; freshly built panels take it
    // whole.
    pub(super) fn sync_built_panels(&mut self, world: &mut World, shown: bool, vp: [f32; 2]) {
        let Some(mut built) = world.remove_resource::<BuiltUi>() else {
            return;
        };
        let fresh = std::mem::take(&mut built.fresh);
        if let Some(bar) = &built.top_bar {
            hud::sync(world, bar, self.hud_state());
        }
        if let Some(dialog) = &built.dialog {
            crate::editor::modal::sync(world, dialog, self.modal.as_ref().filter(|_| shown));
            if let Some(d) = &mut self.modal {
                d.shown = shown;
            }
        }
        for key in PanelKey::ALL {
            let Some(panel) = built.panel(key) else {
                continue;
            };
            let Some(layout) = world.get_by_id::<Layout>(panel.root) else {
                continue;
            };
            if !fresh {
                if let Some(rect) = layout.rect.filter(|_| layout.visible) {
                    self.positions[key] = Some([rect[0], rect[1]]);
                }
                if layout.size.is_some() {
                    self.sizes[key] = layout.size;
                }
                if layout.rank > self.built_ranks[key] {
                    self.focus_panel(key);
                }
            }
        }
        for key in PanelKey::ALL {
            let Some(root) = built.panel(key).map(|p| p.root) else {
                continue;
            };
            let visible = shown && self.panel_shown(key);
            let origin = self.origin(key, vp);
            let rank = self.panel_order.iter().position(|&k| k == key).unwrap_or(0);
            let rank = u32::try_from(rank + 1).unwrap_or(u32::MAX);
            self.built_ranks[key] = rank;
            // A dragged size, held inside what the panel's content allows.
            let size = self.sizes[key].map(|_| self.effective_size(key));
            let content = self.default_size(key);
            if let Some(layout) = world.get_mut_by_id::<Layout>(root) {
                layout.visible = visible;
                layout.rank = rank;
                // Where the hook keeps it: a drag under the top bar, or off a
                // shrunk window, is put back on screen below the bar.
                layout.offset = origin;
                if fresh || layout.rect.is_none() || layout.size.is_some() {
                    layout.size = size;
                }
                let fixed = [LayoutSize::Fixed(content[0]), LayoutSize::Fixed(content[1])];
                if [layout.width, layout.height] != fixed {
                    [layout.width, layout.height] = fixed;
                }
                if layout.resizable {
                    layout.min_size = content;
                }
            }
            if visible && let Some(panel) = built.panels[key].as_mut() {
                let p = registry::panel(key);
                // An editor still out on loan is taken back first, so sync
                // reads it and a second loan cannot hand the hook a copy.
                if std::mem::take(&mut self.lent[key]) {
                    p.reclaim(self, world, panel);
                }
                p.sync(self, world, panel);
                self.lent[key] = p.lend(self, world, panel);
            }
        }
        world.insert_resource(built);
    }

    // Take in what the user did to the shown built panels' widgets last frame
    // that no event reports. Runs before anything this frame changes the
    // hook's model, so a panel compares the widgets with what it last showed.
    pub(super) fn observe_built_panels(&mut self, world: &mut World) {
        let Some(built) = world.remove_resource::<BuiltUi>() else {
            return;
        };
        // A world the panels were just built into has a queue of its own, and
        // fields that start empty.
        if built.fresh {
            self.ui_events = concinnity_engine::ecs::EventCursor::default();
            for key in PanelKey::ALL {
                if let Some(panel) = built.panel(key) {
                    registry::panel(key).restore(self, world, panel);
                }
            }
            self.replay_toasts(world);
            if let Some(d) = &mut self.modal {
                d.shown = false;
            }
            self.create_items = None;
        } else {
            for key in PanelKey::ALL {
                let Some(panel) = built.panel(key) else {
                    continue;
                };
                let shown = world
                    .get_by_id::<Layout>(panel.root)
                    .is_some_and(|l| l.visible);
                if shown {
                    registry::panel(key).observe(self, world, panel);
                }
            }
        }
        world.insert_resource(built);
    }

    // Take back what the built panels lent `world`, before the hook works with
    // it again or the world is replaced.
    pub(super) fn reclaim_built(&mut self, world: &mut World) {
        let Some(built) = world.remove_resource::<BuiltUi>() else {
            return;
        };
        for key in PanelKey::ALL {
            if !std::mem::take(&mut self.lent[key]) {
                continue;
            }
            if let Some(panel) = built.panel(key) {
                registry::panel(key).reclaim(self, world, panel);
            }
        }
        world.insert_resource(built);
    }

    // Handle the events the built panels' widgets sent last frame.
    pub(super) fn read_ui_events(&mut self, world: &mut World) {
        let events: Vec<UiEvent> = match world.events::<UiEvent>() {
            Some(queue) => queue.read(&mut self.ui_events).cloned().collect(),
            None => return,
        };
        for event in &events {
            if event.name == CLOSE_EVENT {
                let key = match event.value {
                    Some(UiValue::Int(i)) => usize::try_from(i)
                        .ok()
                        .and_then(|i| PanelKey::ALL.get(i).copied()),
                    _ => None,
                };
                if let Some(key) = key {
                    registry::close(key, self, world);
                }
                continue;
            }
            if self.on_dialog_event(world, event)
                || self.on_toast_event(world, event)
                || self.on_bar_event(event)
                || self.on_display_menu_event(event)
                || self.on_create_menu_event(event)
            {
                continue;
            }
            for key in PanelKey::ALL {
                if registry::panel(key).on_event(self, world, event) {
                    break;
                }
            }
        }
    }

    // Keep a press an editor element took from the engine's UI as well, so a
    // built panel under the element does not act on it too.
    pub(super) fn withhold_taken_press(&mut self, world: &mut World) {
        if !std::mem::take(&mut self.editor_took_press) {
            return;
        }
        if let Some(input) = world.query_mut::<FrameInput>().last() {
            input.left_click = false;
            input.right_click = false;
            input.scroll_delta = 0.0;
        }
    }
}
