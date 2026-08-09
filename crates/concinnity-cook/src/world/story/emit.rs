use crate::world::ui_spec::sprite;
use concinnity_core::gfx::overlay::UI_REFERENCE_SIZE;
use concinnity_world::spec::{AssetSpec, asset, spec_to_value};

use super::helpers::wrap_text;
use super::model::{FlagOp, Gate, ImageDims, Stage, Story};

// Dialog paragraphs wrap at a fixed column because TextLabel only honors
// explicit newlines and font metrics are not available at this stage. The
// column is conservative for the dialog font size on the reference canvas.
const WRAP_COLUMNS: usize = 72;

const TITLE_FONT_PX: u32 = 56;
const MENU_FONT_PX: u32 = 28;
const DIALOG_FONT_PX: u32 = 22;

// Multiplied into the title backdrop image so the light menu text keeps its
// contrast on a bright photo. The gray value dims the image to this fraction
// of its brightness; alpha stays opaque so the backdrop still fully covers.
const TITLE_BACKDROP_DIM: [f32; 4] = [0.35, 0.35, 0.35, 1.0];

// The fixed dialog box the stage's name plate and dialog text sit on: nearly
// flush with the canvas bottom, tall enough for the name plate to sit inside
// against the box's dark backdrop.
const DIALOG_BOX: (f32, f32, f32, f32) = (140.0, 500.0, 1000.0, 210.0);
pub(super) const DIALOG_BOX_RADIUS: f32 = 14.0;
// Slot rows the save / load overlay shows at once. The story scrolls this
// fixed window over its larger set of logical slots (the auto-save resumed by
// Continue is separate), so each row's click action carries its row index, not
// a fixed slot number.
const VISIBLE_SLOTS: usize = 5;
// Choice option rows: each option gets its own rounded box behind the label
// so the menu stands apart from the dialog box's dark backdrop. The color
// must match the story system's shown tint (it re-tints the boxes to show
// and hide them at runtime).
const CHOICE_BOX_COLOR: [f32; 3] = [0.16, 0.20, 0.35];
pub(super) const CHOICE_BOX_RADIUS: f32 = 10.0;

// The compiled stage entry for a page or choice menu: the backdrop and
// portrait images with their on-canvas rectangles, ready for the story
// system to apply without any probing of its own. Portraits show at the
// image's own pixel size against the reference canvas (scaled down only if
// taller than the canvas), anchored to the canvas bottom; with cover fit the
// canvas bottom sits at or below the window bottom at any aspect ratio, so
// the image's bottom edge is never visibly cut off mid-air.
const PORTRAIT_LEFT_CENTER_X: f32 = 320.0;
const PORTRAIT_CENTER_X: f32 = 640.0;
const PORTRAIT_RIGHT_CENTER_X: f32 = 960.0;

// Emit the runtime assets for one parsed story: the compiled Story graph
// plus the stage scaffolding the story system drives at runtime. Instead of
// a Screen per page, the whole story plays inside one stage screen whose labels
// and sprites are mutated page by page; the title and ending screens stay
// build-generated. `prefix` is the sanitized import name; every generated
// name starts with it. `image_dims` reads an image file's pixel size
// (portrait layout needs the aspect ratio); tests stub it so emission stays
// free of file IO.
pub(crate) fn emit_story(
    prefix: &str,
    story: &Story,
    title_screen: bool,
    text_speed: f32,
    image_dims: ImageDims,
) -> Result<Vec<serde_json::Value>, String> {
    let (win_w, win_h) = (UI_REFERENCE_SIZE[0], UI_REFERENCE_SIZE[1]);

    let font_title = format!("{}_font_title", prefix);
    let font_menu = format!("{}_font_menu", prefix);
    let font_dialog = format!("{}_font_dialog", prefix);
    let title_name = format!("{}_title", prefix);
    let stage_name = format!("{}_stage", prefix);
    let ending_name = format!("{}_ending", prefix);

    let mut out = vec![
        font(&font_title, TITLE_FONT_PX),
        font(&font_menu, MENU_FONT_PX),
        font(&font_dialog, DIALOG_FONT_PX),
    ];

    // Audio files referenced by media directives, deduplicated by path in
    // first-use order; each becomes one AudioClip entry. Same for stage
    // images (and the optional menu backdrop), which become Texture entries the
    // sprites sample.
    let mut clips: Vec<(String, String)> = Vec::new();
    let mut images: Vec<(String, String)> = Vec::new();

    if title_screen {
        out.push(screen(&title_name, true));
        // The menu backdrop: a full-bleed image when the frontmatter set one,
        // else a flat dark fill.
        match &story.background {
            Some(path) => {
                let texture = image_asset(prefix, &mut images, path);
                // Darken the backdrop image (its tint multiplies the texture)
                // so the light title and menu text stay readable over a bright
                // photo. The image still shows through; the flat-fill case
                // below is already dark enough to need no dimming.
                out.push(textured_cover_sprite(
                    &format!("{}_bg", title_name),
                    [0.0, 0.0, win_w, win_h],
                    &texture,
                    TITLE_BACKDROP_DIM,
                ));
            }
            None => out.push(sprite(
                &format!("{}_bg", title_name),
                0.0,
                0.0,
                win_w,
                win_h,
                [0.05, 0.06, 0.12, 1.0],
            )),
        }
        out.push(label(
            &format!("{}_heading", title_name),
            &font_title,
            &story.title,
            LabelStyle {
                x: win_w / 2.0,
                y: 180.0,
                color: [1.0, 0.92, 0.78],
                align: Some("center"),
                ..LabelStyle::default()
            },
        ));
        // The menu buttons at default contiguous positions; the story re-lays
        // them out at runtime (Continue and Load only when a save exists,
        // Settings only when a settings screen exists), so their hit regions
        // follow their labels.
        for (key, text, y, action) in [
            ("start", "Start", 400.0, "story:start"),
            ("continue", "Continue", 452.0, "story:continue"),
            ("load", "Load", 504.0, "story:load"),
            ("settings", "Settings", 556.0, "story:settings"),
            ("quit", "Quit", 608.0, "quit"),
        ] {
            out.extend(title_button(
                &format!("{}_{}", title_name, key),
                &font_menu,
                text,
                y,
                action,
            ));
        }
    }

    // Compile the node graph. Jump and choice targets become node indices
    // (validated against slugs during parse); media paths become the
    // deduplicated asset names; speakers resolve to their display name and
    // color; dialog text is pre-wrapped.
    let node_index = |slug: &str| -> u32 {
        story
            .nodes
            .iter()
            .position(|n| n.slug == slug)
            .expect("targets validated against node slugs") as u32
    };
    let mut nodes_json = Vec::new();
    for node in &story.nodes {
        let mut pages_json = Vec::new();
        for page in &node.pages {
            let speaker = page.speaker.as_ref().map(|id| {
                let character = &story.characters[id];
                serde_json::json!({ "name": character.name, "color": character.color })
            });
            let music = page
                .music
                .as_ref()
                .map(|p| clip_asset(prefix, &mut clips, p));
            let sounds: Vec<String> = page
                .sounds
                .iter()
                .map(|p| clip_asset(prefix, &mut clips, p))
                .collect();
            pages_json.push(serde_json::json!({
                "speaker": speaker,
                "text": wrap_text(&page.text, WRAP_COLUMNS),
                "jump": page.jump.as_deref().map(&node_index),
                "music": music,
                "sounds": sounds,
                "stage": stage_entry(&page.stage, prefix, &mut images, image_dims)?,
                "ops": ops_entries(&page.ops),
                "gates": gate_entries(&page.gates, &node_index),
            }));
        }
        let choices: Vec<serde_json::Value> = node
            .choices
            .iter()
            .map(|c| {
                let condition = c.condition.as_ref().map(|cond| {
                    serde_json::json!({
                        "name": cond.name,
                        "op": cond.op,
                        "value": cond.value,
                    })
                });
                serde_json::json!({
                    "label": c.label,
                    "target": node_index(&c.target),
                    "condition": condition,
                })
            })
            .collect();
        let choice_music = node
            .choice_music
            .as_ref()
            .map(|p| clip_asset(prefix, &mut clips, p));
        let choice_sounds: Vec<String> = node
            .choice_sounds
            .iter()
            .map(|p| clip_asset(prefix, &mut clips, p))
            .collect();
        nodes_json.push(serde_json::json!({
            "slug": node.slug,
            "pages": pages_json,
            "choices": choices,
            "choice_stage": stage_entry(&node.choice_stage, prefix, &mut images, image_dims)?,
            "choice_music": choice_music,
            "choice_sounds": choice_sounds,
            "choice_ops": ops_entries(&node.choice_ops),
            "choice_gates": gate_entries(&node.choice_gates, &node_index),
        }));
    }
    // The compiled graph takes the import's own name: the one declaration the
    // author wrote stays the one asset that carries the story. The scaffold
    // block references the generated stage assets by name; the build resolves
    // them to ids like every other cross-reference, so the runtime never
    // needs the names.
    let max_choices = story
        .nodes
        .iter()
        .map(|n| n.choices.len())
        .max()
        .unwrap_or(0);
    let option_labels: Vec<String> = (0..max_choices)
        .map(|i| format!("{}_opt{}_lbl", stage_name, i))
        .collect();
    let option_boxes: Vec<String> = (0..max_choices)
        .map(|i| format!("{}_opt{}_box", stage_name, i))
        .collect();
    let start_label = title_screen.then(|| format!("{}_start_lbl", title_name));
    let quit_label = title_screen.then(|| format!("{}_quit_lbl", title_name));
    let continue_label = title_screen.then(|| format!("{}_continue_lbl", title_name));
    let load_label = title_screen.then(|| format!("{}_load_lbl", title_name));
    let settings_label = title_screen.then(|| format!("{}_settings_lbl", title_name));
    let slot_boxes: Vec<String> = (0..VISIBLE_SLOTS)
        .map(|i| format!("{}_slot{}_box", stage_name, i))
        .collect();
    let slot_labels: Vec<String> = (0..VISIBLE_SLOTS)
        .map(|i| format!("{}_slot{}_lbl", stage_name, i))
        .collect();
    out.push(serde_json::json!({
        "name": prefix,
        "type": "Story",
        "args": {
            "title": story.title,
            "nodes": nodes_json,
            "text_speed": text_speed,
            "save_key": prefix,
            "scaffold": {
                "screen": &stage_name,
                "ending": &ending_name,
                "bg": format!("{}_bg", stage_name),
                "left": format!("{}_left", stage_name),
                "center": format!("{}_center", stage_name),
                "right": format!("{}_right", stage_name),
                "dialog_box": format!("{}_box", stage_name),
                "name_label": format!("{}_name", stage_name),
                "text_label": format!("{}_text", stage_name),
                "option_boxes": option_boxes,
                "options": option_labels,
                "start_label": start_label,
                "quit_label": quit_label,
                "continue_label": continue_label,
                "title": title_screen.then(|| title_name.clone()),
                "load_label": load_label,
                "settings_label": settings_label,
                "advance_marker": format!("{}_marker", stage_name),
                "log_label": format!("{}_qlog_lbl", stage_name),
                "auto_label": format!("{}_qauto_lbl", stage_name),
                "skip_label": format!("{}_qskip_lbl", stage_name),
                "save_label": format!("{}_qsave_lbl", stage_name),
                "overlay_dim": format!("{}_dim", stage_name),
                "backlog_label": format!("{}_history", stage_name),
                "slot_title": format!("{}_slot_title", stage_name),
                "slot_boxes": slot_boxes,
                "slot_labels": slot_labels,
            },
        }
    }));

    // The stage: one screen the story system drives. Sprites and labels are
    // placeholders here; the system fills text, swaps textures, and toggles
    // visibility page by page. Declaration order is draw order.
    out.push(screen(&stage_name, !title_screen));
    out.push(stage_sprite(
        &format!("{}_bg", stage_name),
        [0.0, 0.0, win_w, win_h],
        [0.05, 0.06, 0.09, 1.0],
        true,
    ));
    for side in ["left", "center", "right"] {
        out.push(stage_sprite(
            &format!("{}_{}", stage_name, side),
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            false,
        ));
    }
    // The dialog box and everything on it are bottom-anchored (fit scale, but
    // pinned to the window bottom) so the box hugs the bottom edge at any
    // aspect ratio instead of floating above the letterbox margin.
    out.push(rounded_sprite_fit(
        &format!("{}_box", stage_name),
        DIALOG_BOX,
        [0.0, 0.0, 0.0, 0.55],
        DIALOG_BOX_RADIUS,
        Some("bottom"),
    ));
    // The name plate sits inside the dialog box so the speaker reads against
    // its dark backdrop, with the dialogue below it.
    out.push(label(
        &format!("{}_name", stage_name),
        &font_menu,
        "",
        LabelStyle {
            x: DIALOG_BOX.0 + 30.0,
            y: DIALOG_BOX.1 + 14.0,
            color: [1.0, 1.0, 1.0],
            fit: Some("bottom"),
            ..LabelStyle::default()
        },
    ));
    out.push(label(
        &format!("{}_text", stage_name),
        &font_dialog,
        "",
        LabelStyle {
            x: DIALOG_BOX.0 + 30.0,
            y: DIALOG_BOX.1 + 58.0,
            color: [1.0, 0.95, 0.85],
            fit: Some("bottom"),
            ..LabelStyle::default()
        },
    ));
    out.push(hit_region(
        &format!("{}_advance", stage_name),
        (0.0, 0.0, win_w, win_h),
        None,
        "story:advance",
    ));
    // Space and Enter both advance the dialogue (in addition to a click). Each
    // is its own KeyBinding; the UI fires whichever key was pressed.
    for (suffix, key) in [("advance_key", "Space"), ("advance_key_enter", "Enter")] {
        out.push(serde_json::json!({
            "name": format!("{}_{}", prefix, suffix),
            "type": "KeyBinding",
            "args": { "key": key, "action": "story:advance" }
        }));
    }

    // The advance marker: a small rounded square at the dialog box's lower
    // right that the story system pulses while a fully revealed page waits
    // for a click.
    out.push(rounded_sprite_fit(
        &format!("{}_marker", stage_name),
        (
            DIALOG_BOX.0 + DIALOG_BOX.2 - 50.0,
            DIALOG_BOX.1 + DIALOG_BOX.3 - 70.0,
            14.0,
            14.0,
        ),
        [1.0, 0.95, 0.85, 0.0],
        4.0,
        Some("bottom"),
    ));

    // The quick row: small always-clickable controls along the dialog box's
    // bottom edge (Log / Auto / Skip / Save). Labels are filled by the story
    // system in page mode and cleared elsewhere; the hit regions stay active
    // the whole time and out-of-mode commands are ignored, like the choice
    // buttons.
    // The story fills each label's text (Log / Auto / Skip / Save) at runtime.
    let quick = [
        ("qlog", "story:log"),
        ("qauto", "story:auto"),
        ("qskip", "story:skip"),
        ("qsave", "story:save"),
    ];
    let quick_y = DIALOG_BOX.1 + DIALOG_BOX.3 - 38.0;
    let quick_w = 80.0;
    let quick_x0 = DIALOG_BOX.0 + DIALOG_BOX.2 - 30.0 - quick.len() as f32 * 90.0;
    for (i, (key, action)) in quick.iter().enumerate() {
        let x = quick_x0 + i as f32 * 90.0;
        let lbl = format!("{}_{}_lbl", stage_name, key);
        out.push(hidden_label(
            &lbl,
            &font_dialog,
            x + quick_w / 2.0,
            quick_y + 2.0,
            [0.75, 0.75, 0.75],
            Some("bottom"),
        ));
        out.push(hit_region_fit(
            &format!("{}_{}_btn", stage_name, key),
            (x, quick_y, quick_w, 30.0),
            Some(&lbl),
            action,
            Some("bottom"),
        ));
    }

    // Choice furniture, sized for the widest menu in the story: one rounded
    // box + label per option, hidden until the story system reaches a
    // choice (each box is re-tinted visible with its slot). The buttons stay
    // hit-active the whole time; the story system ignores a choose action
    // outside a menu (and an advance inside one), so the overlap with the
    // full-canvas advance region resolves by mode.
    if max_choices > 0 {
        let y0 = win_h / 2.0 - max_choices as f32 * 30.0;
        for ci in 0..max_choices {
            let lbl = format!("{}_opt{}_lbl", stage_name, ci);
            let y = y0 + ci as f32 * 60.0;
            out.push(rounded_sprite(
                &format!("{}_opt{}_box", stage_name, ci),
                (280.0, y, win_w - 560.0, 44.0),
                [
                    CHOICE_BOX_COLOR[0],
                    CHOICE_BOX_COLOR[1],
                    CHOICE_BOX_COLOR[2],
                    0.0,
                ],
                CHOICE_BOX_RADIUS,
            ));
            out.push(hidden_label(
                &lbl,
                &font_menu,
                win_w / 2.0,
                y + 8.0,
                [0.92, 0.92, 0.92],
                None,
            ));
            out.push(spec_to_value(
                &asset::hit_region(
                    format!("{}_opt{}_btn", stage_name, ci),
                    [280.0, y, win_w - 560.0, 44.0],
                    format!("story:choose:{}", ci),
                )
                .set("label", lbl)
                .set("hover_color", [1.0f32, 0.85, 0.3])
                .set("hover_scale", 1.06f32),
            ));
        }
    }

    // Overlay furniture, declared after the rest of the stage so the dim
    // draws over it: the shared full-canvas dim, the backlog history text,
    // and the save/load slot rows. All hidden by rendering nothing (zero
    // alpha, empty content); the story system fills them per overlay.
    out.push(sprite(
        &format!("{}_dim", stage_name),
        0.0,
        0.0,
        win_w,
        win_h,
        [0.02, 0.02, 0.04, 0.0],
    ));
    out.push(label(
        &format!("{}_history", stage_name),
        &font_dialog,
        "",
        LabelStyle {
            x: 100.0,
            y: 70.0,
            color: [0.92, 0.92, 0.92],
            ..LabelStyle::default()
        },
    ));
    out.push(label(
        &format!("{}_slot_title", stage_name),
        &font_menu,
        "",
        LabelStyle {
            x: win_w / 2.0,
            y: 160.0,
            color: [1.0, 0.92, 0.78],
            align: Some("center"),
            ..LabelStyle::default()
        },
    ));
    for i in 0..VISIBLE_SLOTS {
        let y = 230.0 + i as f32 * 80.0;
        let lbl = format!("{}_slot{}_lbl", stage_name, i);
        out.push(rounded_sprite(
            &format!("{}_slot{}_box", stage_name, i),
            (280.0, y, win_w - 560.0, 56.0),
            [
                CHOICE_BOX_COLOR[0],
                CHOICE_BOX_COLOR[1],
                CHOICE_BOX_COLOR[2],
                0.0,
            ],
            CHOICE_BOX_RADIUS,
        ));
        out.push(hidden_label(
            &lbl,
            &font_menu,
            win_w / 2.0,
            y + 14.0,
            [0.92, 0.92, 0.92],
            None,
        ));
        out.push(hit_region(
            &format!("{}_slot{}_btn", stage_name, i),
            (280.0, y, win_w - 560.0, 56.0),
            Some(&lbl),
            &format!("story:slot:{}", i),
        ));
    }

    // The ending screen, shown by the story system when the last node runs
    // out of pages.
    out.push(screen(&ending_name, false));
    out.push(sprite(
        &format!("{}_bg", ending_name),
        0.0,
        0.0,
        win_w,
        win_h,
        [0.03, 0.03, 0.05, 1.0],
    ));
    out.push(label(
        &format!("{}_fin", ending_name),
        &font_title,
        "~ fin ~",
        LabelStyle {
            x: win_w / 2.0,
            y: 260.0,
            color: [0.95, 0.88, 0.7],
            align: Some("center"),
            ..LabelStyle::default()
        },
    ));
    let (back_label, back_action) = if title_screen {
        ("Back to title", format!("screen:show:{}", title_name))
    } else {
        ("Restart", "story:start".to_string())
    };
    out.extend(button(
        &format!("{}_back", ending_name),
        &font_menu,
        back_label,
        win_w / 2.0 - 160.0,
        490.0,
        320.0,
        &back_action,
    ));

    for (path, name) in &clips {
        out.push(serde_json::json!({
            "name": name,
            "type": "AudioClip",
            "args": { "source": path }
        }));
    }
    for (path, name) in &images {
        out.push(serde_json::json!({
            "name": name,
            "type": "Texture",
            "args": { "source": path }
        }));
    }

    // UI assets attach to a Screen by name prefix, so one generated screen name
    // must never be a `_`-extension of another or the members of the longer
    // screen would be ambiguous.
    let mut screen_names = vec![stage_name, ending_name];
    if title_screen {
        screen_names.push(title_name);
    }
    screen_names.sort();
    for pair in screen_names.windows(2) {
        if pair[1].starts_with(&format!("{}_", pair[0])) {
            return Err(format!(
                "generated screen '{}' is a name-prefix of '{}'",
                pair[0], pair[1]
            ));
        }
    }

    Ok(out)
}

fn stage_entry(
    stage: &Stage,
    prefix: &str,
    images: &mut Vec<(String, String)>,
    image_dims: ImageDims,
) -> Result<serde_json::Value, String> {
    let (win_w, win_h) = (UI_REFERENCE_SIZE[0], UI_REFERENCE_SIZE[1]);
    let mut entry = serde_json::json!({});
    if let Some(path) = &stage.bg {
        entry["bg"] = serde_json::json!({
            "texture": image_asset(prefix, images, path),
            "x": 0.0, "y": 0.0, "width": win_w, "height": win_h,
        });
    }
    for (side, path, center_x) in [
        ("left", &stage.left, PORTRAIT_LEFT_CENTER_X),
        ("center", &stage.center, PORTRAIT_CENTER_X),
        ("right", &stage.right, PORTRAIT_RIGHT_CENTER_X),
    ] {
        let Some(path) = path else { continue };
        let (iw, ih) = image_dims(path)?;
        if iw == 0 || ih == 0 {
            return Err(format!("portrait '{}' has a zero dimension", path));
        }
        let h = (ih as f32).min(win_h);
        let w = h * iw as f32 / ih as f32;
        entry[side] = serde_json::json!({
            "texture": image_asset(prefix, images, path),
            "x": center_x - w / 2.0,
            "y": win_h - h,
            "width": w,
            "height": h,
        });
    }
    Ok(entry)
}

// The compiled variable operations for a page or choice menu.
fn ops_entries(ops: &[FlagOp]) -> Vec<serde_json::Value> {
    ops.iter()
        .map(|op| serde_json::json!({ "name": op.name, "value": op.value, "add": op.add }))
        .collect()
}

// The compiled conditional jumps for a page or choice menu; targets become
// node indices like every other jump.
fn gate_entries(gates: &[Gate], node_index: &dyn Fn(&str) -> u32) -> Vec<serde_json::Value> {
    gates
        .iter()
        .map(|g| {
            serde_json::json!({
                "name": g.condition.name,
                "op": g.condition.op,
                "value": g.condition.value,
                "target": node_index(&g.target),
            })
        })
        .collect()
}

// A stage-owned sprite the story system mutates: cover fit (full-bleed stage
// imagery reaches the window edges without distorting) with an explicit
// initial visibility.
fn stage_sprite(name: &str, rect: [f32; 4], tint: [f32; 4], visible: bool) -> serde_json::Value {
    spec_to_value(
        &asset::sprite(name, rect, tint)
            .set("fit", "cover")
            .set("visible", visible),
    )
}

// The Texture asset name for a backdrop image path, allocating one on the
// path's first use.
fn image_asset(prefix: &str, images: &mut Vec<(String, String)>, path: &str) -> String {
    if let Some((_, name)) = images.iter().find(|(p, _)| p == path) {
        return name.clone();
    }
    let name = format!("{}_img{}", prefix, images.len());
    images.push((path.to_string(), name.clone()));
    name
}

// The AudioClip asset name for an audio file path, allocating one on the
// path's first use.
fn clip_asset(prefix: &str, clips: &mut Vec<(String, String)>, path: &str) -> String {
    if let Some((_, name)) = clips.iter().find(|(p, _)| p == path) {
        return name.clone();
    }
    let name = format!("{}_clip{}", prefix, clips.len());
    clips.push((path.to_string(), name.clone()));
    name
}

fn font(name: &str, size_px: u32) -> serde_json::Value {
    spec_to_value(&asset::font(name, size_px))
}

fn screen(name: &str, initial: bool) -> serde_json::Value {
    spec_to_value(&asset::screen(name, initial))
}

fn rounded_sprite(
    name: &str,
    rect: (f32, f32, f32, f32),
    tint: [f32; 4],
    radius: f32,
) -> serde_json::Value {
    rounded_sprite_fit(name, rect, tint, radius, None)
}

// A rounded sprite with an explicit `fit`. `Some("bottom")` pins the sprite to
// the window bottom (the dialog box and its marker) instead of the letterbox.
fn rounded_sprite_fit(
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
fn textured_cover_sprite(
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
struct LabelStyle {
    x: f32,
    y: f32,
    color: [f32; 3],
    background: Option<[f32; 4]>,
    // Horizontal alignment relative to `x` ("center" centers text around it,
    // measured with real metrics at draw time). `None` = left, the default.
    align: Option<&'static str>,
    // Reference-to-window mapping ("bottom" hugs the window bottom). `None` =
    // fit, the default.
    fit: Option<&'static str>,
}

fn label(name: &str, font: &str, content: &str, style: LabelStyle) -> serde_json::Value {
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

fn hit_region(
    name: &str,
    rect: (f32, f32, f32, f32),
    label: Option<&str>,
    action: &str,
) -> serde_json::Value {
    hit_region_fit(name, rect, label, action, None)
}

// A hit region with an explicit `fit` (reference-to-window mapping). `Some`
// keeps a region aligned with bottom-anchored furniture it covers.
// A runtime-filled overlay label: empty and hidden at build time (centered,
// native scale), shown and filled by the story system per page (the quick-row
// controls, choice options, and save slots). `fit` bottom-anchors the quick row
// to the window bottom like the dialog box it sits on.
fn hidden_label(
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

fn hit_region_fit(
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
fn button(
    name: &str,
    font: &str,
    text: &str,
    x: f32,
    y: f32,
    w: f32,
    action: &str,
) -> Vec<serde_json::Value> {
    let lbl = format!("{}_lbl", name);
    vec![
        label(
            &lbl,
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
        hit_region(
            &format!("{}_btn", name),
            (x, y, w, 40.0),
            Some(&lbl),
            action,
        ),
    ]
}

// A title-menu button: like `button`, but its hit region follows the label the
// story lays out at runtime (and goes inert while the label is empty), so the
// menu keeps only the applicable buttons contiguous with no dead click zones.
fn title_button(
    name: &str,
    font: &str,
    text: &str,
    y: f32,
    action: &str,
) -> Vec<serde_json::Value> {
    let win_w = UI_REFERENCE_SIZE[0];
    let x = win_w / 2.0 - 120.0;
    let lbl = format!("{}_lbl", name);
    let mut region = hit_region(
        &format!("{}_btn", name),
        (x, y, 240.0, 40.0),
        Some(&lbl),
        action,
    );
    region["args"]["follow_label"] = serde_json::json!(true);
    vec![
        label(
            &lbl,
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
