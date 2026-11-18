//! Building one floating panel at a time from engine elements and widgets, each
//! under a freshly minted id on the editor's screen, in the editor's theme.

use concinnity_engine::components::{
    Binding, Button, Checkbox, Grid, Layout, LayoutAlign, LayoutAnchor, LayoutChild,
    LayoutDirection, LayoutSize, List, Menu, Screen, Sprite, TextAlign, TextArea, TextField,
    TextLabel, UiAction, UiEmit, UiValue, WidgetStyle,
};
use concinnity_engine::ecs::asset_id::AssetId;
use concinnity_engine::ecs::{ComponentSlot, FontHandle, Ref, World};

use super::{BuiltPanel, CLOSE_EVENT};
use crate::editor::panels::registry::PanelKey;
use crate::editor::{theme, widget};

// A row of a panel's list or a checkbox row.
pub(crate) const ROW_H: f32 = 28.0;
// Space between a panel's edge and its body.
pub(crate) const BODY_PAD: f32 = 8.0;
// The close button's side.
const CLOSE_SIDE: f32 = 22.0;
// Inset of the title from the panel's left edge.
const TITLE_INSET: f32 = 10.0;

// What a panel is called and how it may be sized.
pub(crate) struct PanelSpec<'a> {
    pub(crate) key: PanelKey,
    pub(crate) title: &'a str,
    // A `state:` value the title shows in place of `title`.
    pub(crate) title_state: Option<&'a str>,
    // The size it opens at.
    pub(crate) size: [f32; 2],
    // `Some` lets the user size it, never below this.
    pub(crate) min_size: Option<[f32; 2]>,
}

// Pushes elements into a world for one panel at a time.
pub(crate) struct UiBuilder<'w> {
    world: &'w mut World,
    screen: Option<Ref<Screen>>,
    font: Option<FontHandle>,
    // The monospace face a text area's code draws in.
    code_font: Option<FontHandle>,
    // Every id pushed for the panel under construction.
    pushed: Vec<AssetId>,
}

impl<'w> UiBuilder<'w> {
    pub(crate) fn new(world: &'w mut World, screen: AssetId, font: Option<FontHandle>) -> Self {
        Self {
            world,
            screen: Some(Ref::new(screen)),
            font,
            code_font: None,
            pushed: Vec::new(),
        }
    }

    // Draw text areas' code in `font`.
    pub(crate) fn with_code_font(mut self, font: Option<FontHandle>) -> Self {
        self.code_font = font;
        self
    }

    // The editor's look for a widget.
    pub(crate) fn style(&self) -> WidgetStyle {
        WidgetStyle {
            font: self.font,
            scale: theme::TEXT_SCALE,
            text: theme::LABEL,
            dim_text: theme::LABEL_DIM,
            background: [0.0; 4],
            border: [0.0; 4],
            hover: theme::HOVER_TINT,
            selected: theme::SELECTED_TINT,
            accent: theme::CHECK_ON_TINT,
            corner_radius: theme::CONTROL_RADIUS,
        }
    }

    fn push<C: ComponentSlot>(&mut self, c: C) -> Option<AssetId> {
        let (id, _) = self.world.context().push_minted(c).ok()?;
        self.pushed.push(id);
        Some(id)
    }

    // A line of text that fits its content.
    pub(crate) fn label(&mut self, text: &str, color: [f32; 3]) -> Option<LayoutChild> {
        self.clipped_label(text, color, 0.0)
    }

    // A line of text cut with an ellipsis past `width` when it is positive.
    pub(crate) fn clipped_label(
        &mut self,
        text: &str,
        color: [f32; 3],
        width: f32,
    ) -> Option<LayoutChild> {
        let id = self.push(TextLabel {
            font: self.font,
            content: text.to_string(),
            color,
            scale: theme::TEXT_SCALE,
            wrap_width: width,
            max_lines: u32::from(width > 0.0),
            screen: self.screen,
            ..Default::default()
        })?;
        Some(fit(id))
    }

    // A button sending `event` with `value`, `size` big.
    pub(crate) fn button(
        &mut self,
        caption: &str,
        event: &str,
        value: Option<UiValue>,
        size: [f32; 2],
    ) -> Option<LayoutChild> {
        let button = Button {
            label: caption.to_string(),
            action: Some(emit(event, value)),
            ..Default::default()
        };
        self.button_with(button, size)
    }

    // `button`, filled in and `size` big.
    pub(crate) fn button_with(&mut self, button: Button, size: [f32; 2]) -> Option<LayoutChild> {
        let style = WidgetStyle {
            background: theme::BUTTON_TINT,
            ..self.style()
        };
        let id = self.push(Button {
            width: size[0],
            height: size[1],
            style,
            screen: self.screen,
            ..button
        })?;
        Some(fixed(id, size))
    }

    // `children` side by side, as wide as their column and `height` tall.
    pub(crate) fn row(&mut self, children: Vec<LayoutChild>, height: f32) -> Option<LayoutChild> {
        let id = self.stack(LayoutDirection::Row, children, 4.0, 0.0)?;
        Some(LayoutChild {
            item: Some(Ref::new(id)),
            width: LayoutSize::Fill(1.0),
            height: LayoutSize::Fixed(height),
        })
    }

    // A checkbox row as wide as its column.
    pub(crate) fn checkbox(&mut self, check: Checkbox, height: f32) -> Option<LayoutChild> {
        let id = self.push(Checkbox {
            style: self.style(),
            screen: self.screen,
            height,
            ..check
        })?;
        Some(LayoutChild {
            item: Some(Ref::new(id)),
            width: LayoutSize::Fill(1.0),
            height: LayoutSize::Fixed(height),
        })
    }

    // A sprite on the panel's screen.
    pub(crate) fn sprite(&mut self, mut sprite: Sprite) -> Option<AssetId> {
        sprite.screen = self.screen;
        self.push(sprite)
    }

    // A label showing the `state:` value `name`, as wide as its column and
    // aligned in it by `align`.
    pub(crate) fn bound_label(
        &mut self,
        name: &str,
        color: [f32; 3],
        align: TextAlign,
    ) -> Option<LayoutChild> {
        let id = self.push(TextLabel {
            font: self.font,
            text_binding: Some(Binding::State(name.to_string())),
            align,
            color,
            scale: theme::TEXT_SCALE,
            screen: self.screen,
            ..Default::default()
        })?;
        Some(LayoutChild {
            item: Some(Ref::new(id)),
            width: LayoutSize::Fill(1.0),
            height: LayoutSize::Fit,
        })
    }

    // A text field as wide as its column, in the editor's look.
    pub(crate) fn text_field(&mut self, field: TextField) -> Option<LayoutChild> {
        let height = field.height;
        let id = self.push(TextField {
            font: self.font,
            scale: theme::TEXT_SCALE,
            text_color: theme::LABEL,
            placeholder_color: theme::LABEL_DIM,
            background: theme::BUTTON_TINT,
            caret_color: theme::LABEL,
            corner_radius: theme::CONTROL_RADIUS,
            screen: self.screen,
            ..field
        })?;
        Some(LayoutChild {
            item: Some(Ref::new(id)),
            width: LayoutSize::Fill(1.0),
            height: LayoutSize::Fixed(height),
        })
    }

    // A text area filling the space its siblings leave, its code in the
    // monospace face. Its editor is the host's to fill, so the area never
    // reads its own `content`.
    pub(crate) fn text_area(&mut self, area: TextArea) -> Option<LayoutChild> {
        let style = WidgetStyle {
            font: self.code_font.or(self.font),
            scale: if self.code_font.is_some() {
                crate::editor::code_font::SCALE
            } else {
                theme::TEXT_SCALE
            },
            background: theme::CODE_WELL_TINT,
            ..self.style()
        };
        let id = self.push(TextArea {
            style,
            screen: self.screen,
            read_at: Some((None, None)),
            ..area
        })?;
        Some(fill(id))
    }

    // A menu on the panel's screen; it pops up over the panel rather than
    // taking a place in its layout.
    pub(crate) fn menu(&mut self, menu: Menu) -> Option<AssetId> {
        let style = WidgetStyle {
            background: theme::CHROME_TINT,
            border: theme::PANEL_BORDER_TINT,
            ..self.style()
        };
        self.push(Menu {
            style,
            screen: self.screen,
            ..menu
        })
    }

    // A list filling the space its siblings leave.
    pub(crate) fn list(&mut self, list: List) -> Option<LayoutChild> {
        let id = self.push(List {
            style: self.style(),
            screen: self.screen,
            ..list
        })?;
        Some(fill(id))
    }

    // A grid filling the space its siblings leave.
    pub(crate) fn grid(&mut self, grid: Grid) -> Option<LayoutChild> {
        let id = self.push(Grid {
            style: self.style(),
            screen: self.screen,
            ..grid
        })?;
        Some(fill(id))
    }

    // Children laid out in `direction`, `gap` apart, inside `padding`.
    pub(crate) fn stack(
        &mut self,
        direction: LayoutDirection,
        children: Vec<LayoutChild>,
        gap: f32,
        padding: f32,
    ) -> Option<AssetId> {
        self.push(Layout {
            direction,
            children,
            gap,
            padding,
            align: LayoutAlign::Center,
            screen: self.screen,
            ..Default::default()
        })
    }

    // The panel `spec` names around `body`, a column under its title bar.
    // Hands back the panel and starts the next one afresh.
    pub(crate) fn panel(
        &mut self,
        spec: &PanelSpec<'_>,
        body: Vec<LayoutChild>,
    ) -> Option<BuiltPanel> {
        let surface = self.push(Sprite {
            tint: theme::CHROME_TINT,
            corner_radius: theme::PANEL_RADIUS,
            border_width: theme::PANEL_BORDER_WIDTH,
            border_color: theme::panel_border(theme::CHROME_TINT[3]),
            screen: self.screen,
            ..Default::default()
        })?;
        let title = self.push(TextLabel {
            font: self.font,
            content: spec.title.to_string(),
            text_binding: spec.title_state.map(|s| Binding::State(s.to_string())),
            color: theme::HEADING,
            scale: theme::TEXT_SCALE,
            align: TextAlign::Left,
            padding: TITLE_INSET,
            screen: self.screen,
            ..Default::default()
        })?;
        let close_style = WidgetStyle {
            hover: theme::CLOSE_HOVER_TINT,
            ..self.style()
        };
        let close = self.push(Button {
            label: "x".to_string(),
            action: Some(emit(
                CLOSE_EVENT,
                Some(UiValue::Int(i32::try_from(spec.key.index()).unwrap_or(0))),
            )),
            width: CLOSE_SIDE,
            height: CLOSE_SIDE,
            style: close_style,
            screen: self.screen,
            ..Default::default()
        })?;
        let bar = vec![
            LayoutChild {
                item: Some(Ref::new(title)),
                width: LayoutSize::Fill(1.0),
                height: LayoutSize::Fit,
            },
            fixed(close, [CLOSE_SIDE, CLOSE_SIDE]),
        ];
        let bar = self.stack(LayoutDirection::Row, bar, 0.0, 3.0)?;
        let body = self.stack(LayoutDirection::Column, body, 0.0, BODY_PAD)?;
        let column = vec![
            LayoutChild {
                item: Some(Ref::new(bar)),
                width: LayoutSize::Fill(1.0),
                height: LayoutSize::Fixed(widget::TITLE_H),
            },
            fill(body),
        ];
        let column = self.push(Layout {
            direction: LayoutDirection::Column,
            children: column,
            screen: self.screen,
            ..Default::default()
        })?;
        let root = self.push(Layout {
            direction: LayoutDirection::Stack,
            children: vec![fill(surface), fill(column)],
            width: LayoutSize::Fixed(spec.size[0]),
            height: LayoutSize::Fixed(spec.size[1]),
            anchor: LayoutAnchor::TopLeft,
            movable: true,
            resizable: spec.min_size.is_some(),
            grab: widget::TITLE_H,
            min_size: spec.min_size.unwrap_or(spec.size),
            screen: self.screen,
            ..Default::default()
        })?;
        Some(BuiltPanel {
            root,
            surface,
            bar,
            body,
            elements: std::mem::take(&mut self.pushed),
            slots: Vec::new(),
        })
    }
}

// The height of a panel whose body's content is `content` tall.
pub(crate) fn panel_height(content: f32) -> f32 {
    widget::TITLE_H + 2.0 * BODY_PAD + content
}

// An action sending `event` with `value`.
pub(crate) fn emit(event: &str, value: Option<UiValue>) -> UiAction {
    UiAction::Emit(UiEmit {
        name: event.to_string(),
        value,
    })
}

// A child sized to its content.
pub(crate) fn fit(id: AssetId) -> LayoutChild {
    LayoutChild {
        item: Some(Ref::new(id)),
        width: LayoutSize::Fit,
        height: LayoutSize::Fit,
    }
}

// A child `size` big.
pub(crate) fn fixed(id: AssetId, size: [f32; 2]) -> LayoutChild {
    LayoutChild {
        item: Some(Ref::new(id)),
        width: LayoutSize::Fixed(size[0]),
        height: LayoutSize::Fixed(size[1]),
    }
}

// A child filling the space its siblings leave.
pub(crate) fn fill(id: AssetId) -> LayoutChild {
    LayoutChild {
        item: Some(Ref::new(id)),
        width: LayoutSize::Fill(1.0),
        height: LayoutSize::Fill(1.0),
    }
}

#[cfg(test)]
mod tests;
