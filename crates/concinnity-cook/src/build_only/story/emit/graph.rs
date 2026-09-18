use concinnity_core::gfx::overlay::UI_REFERENCE_SIZE;

use super::media::MediaAssets;
use super::names::StoryNames;
use crate::build_only::story::helpers::wrap_text;
use crate::build_only::story::model::{Gate, ImageDims, Stage, Story, VarOp};

// Dialog paragraphs wrap at a fixed column because TextLabel only honors
// explicit newlines and font metrics are not available at this stage. The
// column is conservative for the dialog font size on the reference canvas.
const WRAP_COLUMNS: usize = 72;

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

// Compile the node graph. Jump and choice targets become node indices
// (validated against slugs during parse); media paths become the
// deduplicated asset names; speakers resolve to their display name and
// color; dialog text is pre-wrapped.
pub(super) fn compile_nodes(
    story: &Story,
    media: &mut MediaAssets,
    image_dims: ImageDims,
) -> Result<Vec<serde_json::Value>, String> {
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
            let music = page.music.as_ref().map(|p| media.clip(p));
            let sounds: Vec<String> = page.sounds.iter().map(|p| media.clip(p)).collect();
            pages_json.push(serde_json::json!({
                "speaker": speaker,
                "text": wrap_text(&page.text, WRAP_COLUMNS),
                "jump": page.jump.as_deref().map(&node_index),
                "music": music,
                "sounds": sounds,
                "stage": stage_entry(&page.stage, media, image_dims)?,
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
        let choice_music = node.choice_music.as_ref().map(|p| media.clip(p));
        let choice_sounds: Vec<String> = node.choice_sounds.iter().map(|p| media.clip(p)).collect();
        nodes_json.push(serde_json::json!({
            "slug": node.slug,
            "pages": pages_json,
            "choices": choices,
            "choice_stage": stage_entry(&node.choice_stage, media, image_dims)?,
            "choice_music": choice_music,
            "choice_sounds": choice_sounds,
            "choice_ops": ops_entries(&node.choice_ops),
            "choice_gates": gate_entries(&node.choice_gates, &node_index),
        }));
    }
    Ok(nodes_json)
}

// The compiled graph takes the import's own name: the one declaration the
// author wrote stays the one asset that carries the story. The scaffold
// block references the generated stage assets by name; the build resolves
// them to ids like every other cross-reference, so the runtime never needs
// the names.
pub(super) fn story_entry(
    names: &StoryNames,
    story: &Story,
    nodes: Vec<serde_json::Value>,
    text_speed: f32,
) -> serde_json::Value {
    serde_json::json!({
        "type": "Story",
        "args": {
            "$id": names.prefix,
            "title": story.title,
            "nodes": nodes,
            "text_speed": text_speed,
            "save_key": names.prefix,
            "scaffold": names.scaffold(),
        }
    })
}

fn stage_entry(
    stage: &Stage,
    media: &mut MediaAssets,
    image_dims: ImageDims,
) -> Result<serde_json::Value, String> {
    let (win_w, win_h) = (UI_REFERENCE_SIZE[0], UI_REFERENCE_SIZE[1]);
    let mut entry = serde_json::json!({});
    if let Some(path) = &stage.bg {
        entry["bg"] = serde_json::json!({
            "texture": media.image(path),
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
            "texture": media.image(path),
            "x": center_x - w / 2.0,
            "y": win_h - h,
            "width": w,
            "height": h,
        });
    }
    Ok(entry)
}

// The compiled variable operations for a page or choice menu.
fn ops_entries(ops: &[VarOp]) -> Vec<serde_json::Value> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_only::story::parse::parse_story;

    const SRC: &str = "---\ntitle: T\n---\n\n\
        # a\n\n[music](theme.ogg)\n\nhi\n\n[Go](#c)\n\n\
        # b\n\n[music](theme.ogg)\n\n```story\nif seen -> #a\n```\n\n\
        - [Back](#a)\n- [On](#c \"if seen\")\n\n\
        # c\n\n![left](p.png)\n\nbye\n";

    fn dims(_path: &str) -> Result<(u32, u32), String> {
        Ok((100, 200))
    }

    #[test]
    fn targets_resolve_to_node_indices() {
        let story = parse_story(SRC).unwrap();
        let mut media = MediaAssets::new("s");
        let nodes = compile_nodes(&story, &mut media, &dims).unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0]["pages"][1]["jump"], 2);
        assert_eq!(nodes[1]["choices"][0]["target"], 0);
        assert_eq!(nodes[1]["choices"][1]["target"], 2);
        assert_eq!(nodes[1]["choices"][1]["condition"]["name"], "seen");
        assert_eq!(nodes[1]["choice_gates"][0]["target"], 0);
        assert_eq!(nodes[1]["choice_gates"][0]["op"], "ne");
    }

    #[test]
    fn a_repeated_clip_path_compiles_to_one_name() {
        let story = parse_story(SRC).unwrap();
        let mut media = MediaAssets::new("s");
        let nodes = compile_nodes(&story, &mut media, &dims).unwrap();
        assert_eq!(nodes[0]["pages"][0]["music"], "s_clip0");
        assert_eq!(nodes[1]["choice_music"], "s_clip0");
        let clips = media
            .entries()
            .iter()
            .filter(|e| e["type"] == "AudioClip")
            .count();
        assert_eq!(clips, 1);
    }

    #[test]
    fn a_portrait_is_placed_at_its_native_size() {
        let story = parse_story(SRC).unwrap();
        let mut media = MediaAssets::new("s");
        let nodes = compile_nodes(&story, &mut media, &dims).unwrap();
        let left = &nodes[2]["pages"][0]["stage"]["left"];
        assert_eq!(left["texture"], "s_img0");
        assert_eq!(
            (left["width"].as_f64(), left["height"].as_f64()),
            (Some(100.0), Some(200.0))
        );
        assert_eq!(left["x"], PORTRAIT_LEFT_CENTER_X - 50.0);
    }
}
