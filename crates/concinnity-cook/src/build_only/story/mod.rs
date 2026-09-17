// Build-time expansion: StoryImport -> Story / Font / Screen / Sprite /
// TextLabel / HitRegion. A Markdown story file (frontmatter + headings +
// paragraphs + link lists) becomes a compiled node graph that plays inside one
// stage screen the story system fills page by page, beside generated title and
// ending screens. The whole graph is validated here, so a dangling jump or an
// undeclared speaker fails the build rather than the playthrough.

use std::collections::HashSet;

use super::expand::{asset_name, registered_type, schema_args};
use crate::authoring::registry::RegisteredType;
use crate::authoring::registry::build_only::StoryImport;
use crate::import::scene::sanitize_name;

mod emit;
mod frontmatter;
mod helpers;
mod image;
mod model;
mod parse;
mod parser;
pub(crate) mod schema;
mod script;

#[cfg(test)]
mod tests;

use emit::emit_story;
use image::probe_image_dims;
use parse::parse_story;

/// Validate Markdown story source without expanding it: the same parse +
/// graph-validation pass `expand_stories` runs, minus asset emission. Lets an
/// authoring front end (the editor's Story panel) reject a broken story before
/// writing it to disk.
pub fn validate_story_source(src: &str) -> Result<(), String> {
    parse_story(src).map(|_| ())
}

// Replace every StoryImport asset with the UI asset entries its Markdown
// source expands to. Generated names are prefixed with the import's (unique)
// asset name, so they never collide with hand-authored assets; a collision is
// a hard error, as is any parse or graph-validation failure in the source.
pub(crate) fn expand_stories(assets: &mut Vec<serde_json::Value>) -> Result<(), String> {
    if !assets
        .iter()
        .any(|v| registered_type(v) == Some(RegisteredType::StoryImport))
    {
        return Ok(());
    }

    let mut taken: HashSet<String> = assets
        .iter()
        .filter(|v| registered_type(v) != Some(RegisteredType::StoryImport))
        .map(asset_name)
        .filter(|n| !n.is_empty())
        .collect();

    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in assets.drain(..) {
        if registered_type(&value) != Some(RegisteredType::StoryImport) {
            result.push(value);
            continue;
        }

        let import_name = asset_name(&value);
        let StoryImport {
            source,
            title_screen,
            text_speed,
        } = schema_args(RegisteredType::StoryImport, &import_name, value.get("args"))?;
        if source.is_empty() {
            return Err(format!("StoryImport '{}': missing `source`", import_name));
        }

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
