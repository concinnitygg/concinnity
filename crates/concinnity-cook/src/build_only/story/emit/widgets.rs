use concinnity_core::gfx::overlay::UI_REFERENCE_SIZE;

use super::names::ButtonNames;
use crate::authoring::spec::{AssetSpec, asset, spec_to_value};

// A stage-owned sprite the story system mutates: cover fit (full-bleed stage
// imagery reaches the window edges without distorting) with an explicit
// initial visibility.
pub(super) fn stage_sprite(
    name: &str,
    rect: [f32; 4],
    tint: [f32; 4],
    visible: bool,
) -> serde_json::Value {
    spec_to_value(
        &asset::sprite(name, rect, tint)
            .set("fit", "cover")
            .set("visible", visible),
    )
}

pub(super) fn font(name: &str, size_px: u32) -> serde_json::Value {
    spec_to_value(&asset::font(name, size_px))
}

pub(super) fn screen(name: &str, initial: bool) -> serde_json::Value {
    spec_to_value(&asset::screen(name, initial))
}

pub(super) fn rounded_sprite(
    name: &str,
    rect: (f32, f32, f32, f32),
    tint: [f32; 4],
    radius: f32,
) -> serde_json::Value {
    rounded_sprite_fit(name, rect, tint, radius, None)
}

// A rounded sprite with an explicit `fit`. `Some("bottom")` pins the sprite to
// the window bottom (the dialog box and its marker) instead of the letterbox.
pub(super) fn rounded_sprite_fit(
    name: &str,
    rect: (f32, f32, f32, f32),
    tint: [f32; 4],
    radius: f32,
    fit: Option<&'static str>,
) -> serde_json::Value {
    let mut spec =
        asset::sprite(name, [rect.0, rect.1, rect.2, rect.3], tint).set("corner_radius", radius);
    if let Some(fit) = fit {
        spec = spec.set("fit", fit);
    }
    spec_to_value(&spec)
}

// A full-bleed textured sprite (cover fit) for a menu backdrop image. `tint`
// multiplies the sampled texture, so a gray tint darkens the image (used to
// keep light menu text readable) while [1, 1, 1, 1] leaves it at full color.
pub(super) fn textured_cover_sprite(
    name: &str,
    rect: [f32; 4],
    texture: &str,
    tint: [f32; 4],
) -> serde_json::Value {
    spec_to_value(
        &asset::sprite(name, rect, tint)
            .set("texture", texture)
            .set("fit", "cover"),
    )
}

#[derive(Default)]
pub(super) struct LabelStyle {
    pub(super) x: f32,
    pub(super) y: f32,
    pub(super) color: [f32; 3],
    pub(super) background: Option<[f32; 4]>,
    // Horizontal alignment relative to `x` ("center" centers text around it,
    // measured with real metrics at draw time). `None` = left, the default.
    pub(super) align: Option<&'static str>,
    // Reference-to-window mapping ("bottom" hugs the window bottom). `None` =
    // fit, the default.
    pub(super) fit: Option<&'static str>,
}

pub(super) fn label(name: &str, font: &str, content: &str, style: LabelStyle) -> serde_json::Value {
    let mut spec = AssetSpec::new(name, "TextLabel")
        .set("font", font)
        .set("content", content)
        .set("x", style.x)
        .set("y", style.y)
        .set("color", style.color)
        .set("scale", 1.0f32);
    if let Some(bg) = style.background {
        spec = spec.set("background", bg).set("padding", 20.0f32);
    }
    if let Some(align) = style.align {
        spec = spec.set("align", align);
    }
    if let Some(fit) = style.fit {
        spec = spec.set("fit", fit);
    }
    spec_to_value(&spec)
}

pub(super) fn hit_region(
    name: &str,
    rect: (f32, f32, f32, f32),
    label: Option<&str>,
    action: &str,
) -> serde_json::Value {
    hit_region_fit(name, rect, label, action, None)
}

// A runtime-filled overlay label: empty and hidden at build time (centered,
// native scale), shown and filled by the story system per page (the quick-row
// controls, choice options, and save slots). `fit` bottom-anchors the quick row
// to the window bottom like the dialog box it sits on.
pub(super) fn hidden_label(
    name: &str,
    font: &str,
    x: f32,
    y: f32,
    color: [f32; 3],
    fit: Option<&'static str>,
) -> serde_json::Value {
    let mut spec = AssetSpec::new(name, "TextLabel")
        .set("font", font)
        .set("content", "")
        .set("x", x)
        .set("y", y)
        .set("color", color)
        .set("scale", 1.0f32)
        .set("align", "center")
        .set("visible", false);
    if let Some(fit) = fit {
        spec = spec.set("fit", fit);
    }
    spec_to_value(&spec)
}

// A hit region with an explicit `fit` (reference-to-window mapping). `Some`
// keeps a region aligned with bottom-anchored furniture it covers.
pub(super) fn hit_region_fit(
    name: &str,
    rect: (f32, f32, f32, f32),
    label: Option<&str>,
    action: &str,
    fit: Option<&'static str>,
) -> serde_json::Value {
    let mut spec = asset::hit_region(name, [rect.0, rect.1, rect.2, rect.3], action);
    if let Some(l) = label {
        spec = spec
            .set("label", l)
            .set("hover_color", [1.0f32, 0.85, 0.3])
            .set("hover_scale", 1.06f32);
    }
    if let Some(fit) = fit {
        spec = spec.set("fit", fit);
    }
    spec_to_value(&spec)
}

// A clickable menu row: a TextLabel and the HitRegion that styles and fires
// it. The label is centered in the region with real metrics (align center on
// the box center); buttons always use the menu font.
pub(super) fn button(
    names: &ButtonNames,
    font: &str,
    text: &str,
    rect: (f32, f32, f32),
    action: &str,
) -> Vec<serde_json::Value> {
    let (x, y, w) = rect;
    vec![
        label(
            &names.label,
            font,
            text,
            LabelStyle {
                x: x + w / 2.0,
                y: y + 6.0,
                color: [0.92, 0.92, 0.92],
                align: Some("center"),
                ..LabelStyle::default()
            },
        ),
        hit_region(&names.region, (x, y, w, 40.0), Some(&names.label), action),
    ]
}

// A title-menu button: like `button`, but its hit region follows the label the
// story lays out at runtime (and goes inert while the label is empty), so the
// menu keeps only the applicable buttons contiguous with no dead click zones.
pub(super) fn title_button(
    names: &ButtonNames,
    font: &str,
    text: &str,
    y: f32,
    action: &str,
) -> Vec<serde_json::Value> {
    let win_w = UI_REFERENCE_SIZE[0];
    let x = win_w / 2.0 - 120.0;
    let mut region = hit_region(
        &names.region,
        (x, y, 240.0, 40.0),
        Some(&names.label),
        action,
    );
    region["args"]["follow_label"] = serde_json::json!(true);
    vec![
        label(
            &names.label,
            font,
            text,
            LabelStyle {
                x: win_w / 2.0,
                y: y + 6.0,
                color: [0.92, 0.92, 0.92],
                align: Some("center"),
                ..LabelStyle::default()
            },
        ),
        region,
    ]
}
