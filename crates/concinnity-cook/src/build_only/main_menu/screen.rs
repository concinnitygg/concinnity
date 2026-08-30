// The menu's own Screen: the backdrop, the heading, and one label + HitRegion
// per item, laid out as a single column.

use super::super::ui_spec::{centered_label, sprite};
use super::{TOP_MARGIN_FRAC, cursor_sprite};
use crate::authoring::registry::build_only::MainMenu;
use crate::authoring::spec::{asset, spec_to_value};

// Window dimensions and font pixel size used to lay out a menu screen.
#[derive(Clone, Copy)]
pub(super) struct MenuMetrics {
    pub(super) win_w: f32,
    pub(super) win_h: f32,
    pub(super) font_px: f32,
}

// Emit the assets for one menu layer: a Screen, an optional dim backdrop, an
// optional heading, a TextLabel + HitRegion per item, and an optional cursor.
pub(super) fn emit_menu_screen(
    screen: &str,
    title: &str,
    items: &[(String, String)],
    style: &MainMenu,
    font: &str,
    metrics: MenuMetrics,
    initial: bool,
) -> Vec<serde_json::Value> {
    let MenuMetrics {
        win_w,
        win_h,
        font_px,
    } = metrics;
    let mut out = Vec::new();

    out.push(spec_to_value(&asset::screen(screen, initial)));

    if style.dim[3] > 0.0 {
        out.push(sprite(
            &format!("{}_dim", screen),
            0.0,
            0.0,
            win_w,
            win_h,
            style.dim,
        ));
    }

    let line_h = font_px * style.text_scale;
    let center_x = if style.centered { win_w / 2.0 } else { style.x };

    let has_title = !title.is_empty();
    let pitch = style.button_height + style.row_gap;
    // Top-aligned from a fixed margin (not vertically centered), per the
    // settings-tab layout, so menu text hugs the top of the overlay.
    let start_y = if style.centered {
        win_h * TOP_MARGIN_FRAC
    } else {
        style.y
    };

    let mut row = 0usize;
    if has_title {
        let title_scale = style.text_scale * 1.4;
        out.push(centered_label(
            &format!("{}_title", screen),
            title,
            font,
            center_x,
            start_y + (style.button_height - font_px * title_scale) / 2.0,
            style.text_color,
            title_scale,
        ));
        row += 1;
    }

    for (i, (label, action)) in items.iter().enumerate() {
        let row_y = start_y + row as f32 * pitch;
        let label_name = format!("{}_label_{}", screen, i);

        out.push(centered_label(
            &label_name,
            label,
            font,
            center_x,
            row_y + (style.button_height - line_h) / 2.0,
            style.text_color,
            style.text_scale,
        ));

        out.push(spec_to_value(
            &asset::hit_region(
                format!("{}_btn_{}", screen, i),
                [
                    center_x - style.button_width / 2.0,
                    row_y,
                    style.button_width,
                    style.button_height,
                ],
                action.clone(),
            )
            .set("label", label_name)
            .set("hover_color", style.hover_color)
            .set("hover_scale", style.text_scale * style.hover_scale),
        ));
        row += 1;
    }

    if style.cursor {
        out.push(cursor_sprite(&format!("{}_cursor", screen), style));
    }

    out
}
