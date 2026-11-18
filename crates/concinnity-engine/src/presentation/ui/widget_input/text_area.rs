// A TextArea's input: its text loaded from its binding, a press focusing it
// and placing the caret, a drag selecting, the wheel scrolling, and the keys
// editing while it holds the keyboard. Each edit is written back to the
// binding.

use alloc::vec::Vec;

use crate::components::{TextArea, TextField, TextLanguage, UiValue};
use crate::ecs::{Entity, PipelineContext};
use crate::presentation::text_edit::{
    HostClipboard, KeyConvention, LocalClipboard, TextEditor, geometry, highlight,
};
use crate::window::clipboard::Clipboard;

use super::super::bound::Values;
use super::{Effect, Pointer, WidgetOutcome, emit, set};

// The text-area state the pass keeps across frames.
pub(super) struct AreaInput<'a> {
    pub(super) clock: f64,
    pub(super) dragging: &'a mut Option<Entity>,
    pub(super) clipboard: &'a mut LocalClipboard,
}

// A fresh editor holding `text`, colored as `language`.
fn editor_for(text: &str, language: TextLanguage) -> TextEditor {
    let editor = TextEditor::from_text(text);
    match language {
        TextLanguage::Plain => editor,
        TextLanguage::Hlsl => editor.highlighted(&highlight::HLSL),
        TextLanguage::Markdown => editor.highlighted(&highlight::MARKDOWN),
    }
}

pub(super) fn step_text_areas(
    ctx: &mut PipelineContext<'_>,
    pointer: &Pointer<'_>,
    values: Values<'_>,
    state: AreaInput<'_>,
    effects: &mut Vec<Effect>,
    out: &mut WidgetOutcome,
) {
    let AreaInput {
        clock,
        dragging,
        clipboard,
    } = state;
    let keys = ctx.resource::<KeyConvention>().copied().unwrap_or_default();
    // The host's clipboard is lent out only to a frame that types into a
    // focused editor, the one that can copy or paste.
    let typing = !pointer.input.key_events.is_empty() && ctx.query::<TextArea>().any(|a| a.focused);
    let mut host = if typing {
        ctx.remove_resource::<HostClipboard>()
    } else {
        None
    };
    let mut focused_any = false;
    let mut pressed_any = false;
    for (entity, area) in ctx.query_mut_with_entity::<TextArea>() {
        load(area, values);
        let Some(m) = area.metrics else {
            continue;
        };
        let rect = [area.x, area.y, area.width, area.height];
        let g = geometry(rect, m, area.editor.line_count());
        area.editor.set_view(g.view());
        if !area.visible || !pointer.live(area.screen) {
            area.focused = false;
            continue;
        }
        let (x, y) = pointer.at(area.screen);
        let input = pointer.input;
        if *dragging == Some(entity) {
            area.editor.pointer(&g, x, y, input.left_button_down);
            if !area.editor.pointer_busy() {
                *dragging = None;
            }
        }
        if g.contains(x, y) && input.scroll_delta != 0.0 {
            area.editor.wheel(-input.scroll_delta, input.shift);
        }
        if pointer.click() {
            if area.editor.press_at(&g, x, y, input.shift, clock) {
                area.focused = true;
                pressed_any = true;
                out.click_taken = true;
                *dragging = Some(entity);
            } else {
                area.focused = false;
            }
        }
        if !area.focused {
            continue;
        }
        focused_any = true;
        if pointer.escape {
            area.focused = false;
            out.escape_taken = true;
            continue;
        }
        let before = area.editor.revision();
        let board: &mut dyn Clipboard = match host.as_mut() {
            Some(host) => &mut *host.0,
            None => &mut *clipboard,
        };
        if area.read_only {
            area.editor
                .handle_view_events(&input.key_events, keys, board);
        } else {
            area.editor.handle_events(&input.key_events, keys, board);
        }
        if area.editor.revision() != before {
            if area.text.is_some() {
                let text = area.editor.text();
                set(effects, area.text.as_ref(), UiValue::Text(text));
            }
            emit(effects, &area.on_change, entity, None);
        }
    }
    if let Some(host) = host {
        ctx.insert_resource(host);
    }
    // A text area taking the keyboard takes it from every field.
    if pressed_any {
        for field in ctx.query_mut::<TextField>() {
            field.focused = false;
        }
    } else if focused_any && ctx.query::<TextField>().any(|t| t.focused) {
        for area in ctx.query_mut::<TextArea>() {
            area.focused = false;
        }
    }
}

// Fill the editor from the bound text the first time, and again whenever the
// binding holds text the editor does not. A change while the editor holds the
// keyboard waits until it lets go.
fn load(area: &mut TextArea, values: Values<'_>) {
    let bound = area.text.as_ref().and_then(|b| values.text(b));
    let revisions = (
        values.state.map(crate::components::UiState::revision),
        values.vars.map(|v| v.revision),
    );
    if area.read_at.is_none() {
        let text = bound.unwrap_or(&area.content);
        *area.editor = editor_for(text, area.language);
        area.read_at = Some(revisions);
        return;
    }
    if area.focused || area.read_at == Some(revisions) {
        return;
    }
    area.read_at = Some(revisions);
    if let Some(text) = bound
        && !area.editor.matches(text)
    {
        *area.editor = editor_for(text, area.language);
    }
}
