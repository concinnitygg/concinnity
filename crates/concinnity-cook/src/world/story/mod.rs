// src/world/story.rs
// Build-time expansion: StoryImport -> Font / View / Sprite / TextLabel /
// HitRegion. A Markdown story file (frontmatter + headings + paragraphs +
// link lists) becomes a click-through, branching flow built entirely from
// existing UI assets: one View per page, a full-canvas HitRegion advancing to
// the next page, and choice buttons targeting other nodes. The whole graph is
// validated here, so a dangling jump or an undeclared speaker fails the build
// rather than the playthrough.

use std::collections::HashSet;

use super::expand::{asset_name, type_norm};
use crate::import::sanitize_name;

mod emit;
mod helpers;
mod image;
mod model;
mod parse;

#[cfg(test)]
mod tests;

use emit::emit_story;
use image::probe_image_dims;
use parse::parse_story;

// Replace every StoryImport asset with the UI asset entries its Markdown
// source expands to. Generated names are prefixed with the import's (unique)
// asset name, so they never collide with hand-authored assets; a collision is
// a hard error, as is any parse or graph-validation failure in the source.
pub(crate) fn expand_stories(assets: &mut Vec<serde_json::Value>) -> Result<(), String> {
    if !assets.iter().any(|v| type_norm(v) == "storyimport") {
        return Ok(());
    }

    let mut taken: HashSet<String> = assets
        .iter()
        .filter(|v| type_norm(v) != "storyimport")
        .map(asset_name)
        .filter(|n| !n.is_empty())
        .collect();

    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in assets.drain(..) {
        if type_norm(&value) != "storyimport" {
            result.push(value);
            continue;
        }

        let import_name = asset_name(&value);
        let args = value
            .get("args")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let source = args
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if source.is_empty() {
            return Err(format!("StoryImport '{}': missing `source`", import_name));
        }
        let title_screen = args
            .get("title_screen")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let text_speed = args
            .get("text_speed")
            .and_then(|v| v.as_f64())
            .unwrap_or(45.0) as f32;

        let content = std::fs::read_to_string(&source).map_err(|e| {
            format!(
                "StoryImport '{}': cannot read '{}': {}",
                import_name, source, e
            )
        })?;
        let story = parse_story(&content)
            .map_err(|e| format!("StoryImport '{}' ({}): {}", import_name, source, e))?;
        let entries = emit_story(
            &sanitize_name(&import_name),
            &story,
            title_screen,
            text_speed,
            &probe_image_dims,
        )
        .map_err(|e| format!("StoryImport '{}' ({}): {}", import_name, source, e))?;

        for entry in entries {
            let name = asset_name(&entry);
            if !name.is_empty() && !taken.insert(name.clone()) {
                return Err(format!(
                    "StoryImport '{}': generated asset name '{}' collides with an existing \
                     asset; rename the import or the conflicting asset",
                    import_name, name
                ));
            }
            result.push(entry);
        }
    }

    *assets = result;
    Ok(())
}
