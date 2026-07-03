// src/world/story.rs
// Build-time expansion: StoryImport -> Font / View / Sprite / TextLabel /
// HitRegion. A Markdown story file (frontmatter + headings + paragraphs +
// link lists) becomes a click-through, branching flow built entirely from
// existing UI assets: one View per page, a full-canvas HitRegion advancing to
// the next page, and choice buttons targeting other nodes. The whole graph is
// validated here, so a dangling jump or an undeclared speaker fails the build
// rather than the playthrough.

use std::collections::{BTreeMap, HashSet};
use std::ops::Range;

use pulldown_cmark::{Event, HeadingLevel, MetadataBlockKind, Options, Parser, Tag, TagEnd};

use super::expand::{asset_name, type_norm};
use crate::gfx::overlay::UI_REFERENCE_SIZE;
use crate::import::sanitize_name;

// Dialog paragraphs wrap at a fixed column because TextLabel only honors
// explicit newlines and font metrics are not available at this stage. The
// column is conservative for the dialog font size on the reference canvas.
const WRAP_COLUMNS: usize = 72;

const TITLE_FONT_PX: u32 = 56;
const MENU_FONT_PX: u32 = 28;
const DIALOG_FONT_PX: u32 = 22;

// A parsed story: frontmatter metadata plus the node graph, in document order.
#[derive(Debug, Default)]
pub(crate) struct Story {
    pub(crate) title: String,
    pub(crate) characters: BTreeMap<String, Character>,
    pub(crate) nodes: Vec<Node>,
}

#[derive(Debug, Clone)]
pub(crate) struct Character {
    pub(crate) name: String,
    pub(crate) color: [f32; 3],
}

// One `# heading` and everything under it. `choices`, when non-empty, is the
// node's final content: a menu of links out.
#[derive(Debug, Default)]
pub(crate) struct Node {
    pub(crate) slug: String,
    pub(crate) heading: String,
    pub(crate) pages: Vec<Page>,
    pub(crate) choices: Vec<Choice>,
}

// One click-through page. `jump` overrides the default advance (next page,
// then the node's choices or fall-through) with an explicit node target.
#[derive(Debug, Default)]
pub(crate) struct Page {
    pub(crate) speaker: Option<String>,
    pub(crate) text: String,
    pub(crate) jump: Option<String>,
}

#[derive(Debug)]
pub(crate) struct Choice {
    pub(crate) label: String,
    pub(crate) target: String,
}

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

        let content = std::fs::read_to_string(&source).map_err(|e| {
            format!(
                "StoryImport '{}': cannot read '{}': {}",
                import_name, source, e
            )
        })?;
        let story = parse_story(&content)
            .map_err(|e| format!("StoryImport '{}' ({}): {}", import_name, source, e))?;
        let entries = emit_story(&sanitize_name(&import_name), &story, title_screen)
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

// ---- Markdown parsing ----

// GitHub-style anchor slug for a heading: lowercase, alphanumerics kept,
// runs of spaces/hyphens/underscores collapsed to one hyphen, everything
// else dropped. `[link](#the-wood)` targets the heading `# The Wood`.
pub(crate) fn slug(text: &str) -> String {
    let mut out = String::new();
    for c in text.trim().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if (c == ' ' || c == '-' || c == '_') && !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_string()
}

// In-flight paragraph state: inline events accumulate here until the
// paragraph closes and is classified as narration, dialogue, or a jump.
#[derive(Default)]
struct ParaAcc {
    speaker: Option<String>,
    text: String,
    links: Vec<(String, String)>,
    has_plain_text: bool,
}

// Parse a story file into its node graph, rejecting anything outside the
// dialect. Errors carry a 1-based source line so the author can find the
// offending construct.
pub(crate) fn parse_story(src: &str) -> Result<Story, String> {
    let line_starts: Vec<usize> = std::iter::once(0)
        .chain(src.match_indices('\n').map(|(i, _)| i + 1))
        .collect();
    let line_of = |range: &Range<usize>| line_starts.partition_point(|&s| s <= range.start);
    let err = |range: &Range<usize>, msg: String| Err(format!("line {}: {}", line_of(range), msg));

    let mut story = Story::default();
    let mut cur_node: Option<Node> = None;
    let mut para: Option<ParaAcc> = None;
    let mut heading_text: Option<String> = None;
    let mut meta_text: Option<String> = None;
    let mut strong_text: Option<String> = None;
    let mut link: Option<(String, String)> = None; // (target, label so far)
    let mut in_list = false;
    let mut item_links: Vec<(String, String)> = Vec::new();
    let mut item_has_text = false;

    let options = Options::ENABLE_YAML_STYLE_METADATA_BLOCKS;
    for (event, range) in Parser::new_ext(src, options).into_offset_iter() {
        match event {
            Event::Start(Tag::MetadataBlock(MetadataBlockKind::YamlStyle)) => {
                meta_text = Some(String::new());
            }
            Event::End(TagEnd::MetadataBlock(_)) => {
                let text = meta_text.take().unwrap_or_default();
                parse_frontmatter(&text, &mut story)?;
            }

            Event::Start(Tag::Heading { level, .. }) => {
                if level != HeadingLevel::H1 {
                    return err(
                        &range,
                        format!(
                            "'{}' heading: only `#` headings (nodes) are supported",
                            "#".repeat(level as usize)
                        ),
                    );
                }
                heading_text = Some(String::new());
            }
            Event::End(TagEnd::Heading(_)) => {
                let text = heading_text.take().unwrap_or_default();
                let s = slug(&text);
                if s.is_empty() {
                    return err(
                        &range,
                        format!("heading '{}' produces an empty anchor", text),
                    );
                }
                if let Some(node) = cur_node.take() {
                    finish_node(node, &mut story, &range, &line_of)?;
                }
                cur_node = Some(Node {
                    slug: s,
                    heading: text,
                    ..Node::default()
                });
            }

            Event::Start(Tag::Paragraph) => {
                if meta_text.is_some() || in_list {
                    continue;
                }
                let Some(node) = cur_node.as_ref() else {
                    return err(&range, "content before the first `#` heading".to_string());
                };
                if !node.choices.is_empty() {
                    return err(
                        &range,
                        format!("node '{}': choices must be its last content", node.heading),
                    );
                }
                para = Some(ParaAcc::default());
            }
            Event::End(TagEnd::Paragraph) => {
                if in_list {
                    continue;
                }
                if let Some(acc) = para.take() {
                    let page = classify_paragraph(acc, &range, &line_of)?;
                    cur_node
                        .as_mut()
                        .expect("paragraph start checked the node")
                        .pages
                        .push(page);
                }
            }

            Event::Start(Tag::List(ordered)) => {
                if ordered.is_some() {
                    return err(
                        &range,
                        "choices must be a bullet list, not numbered".to_string(),
                    );
                }
                let Some(node) = cur_node.as_ref() else {
                    return err(&range, "content before the first `#` heading".to_string());
                };
                if !node.choices.is_empty() {
                    return err(
                        &range,
                        format!("node '{}': choices must be its last content", node.heading),
                    );
                }
                if in_list {
                    return err(&range, "nested lists are not supported".to_string());
                }
                in_list = true;
            }
            Event::End(TagEnd::List(_)) => {
                in_list = false;
            }
            Event::Start(Tag::Item) => {
                item_links.clear();
                item_has_text = false;
            }
            Event::End(TagEnd::Item) => {
                if item_links.len() != 1 || item_has_text {
                    return err(
                        &range,
                        "each choice must be exactly one link, e.g. `- [Go](#node)`".to_string(),
                    );
                }
                let (target, label) = item_links.pop().expect("length checked");
                cur_node
                    .as_mut()
                    .expect("list start checked the node")
                    .choices
                    .push(Choice { label, target });
            }

            Event::Start(Tag::Link { dest_url, .. }) => {
                let Some(target) = dest_url.strip_prefix('#') else {
                    return err(
                        &range,
                        format!("link '{}': only `#heading` targets are supported", dest_url),
                    );
                };
                link = Some((target.to_string(), String::new()));
            }
            Event::End(TagEnd::Link) => {
                let (target, label) = link.take().unwrap_or_default();
                if label.trim().is_empty() {
                    return err(&range, format!("link to '#{}' has no label text", target));
                }
                if in_list {
                    item_links.push((target, label.trim().to_string()));
                } else if let Some(acc) = para.as_mut() {
                    acc.links.push((label.trim().to_string(), target));
                } else {
                    return err(
                        &range,
                        "links may only appear in paragraphs or choice lists".to_string(),
                    );
                }
            }

            Event::Start(Tag::Strong) => {
                let misplaced = match para.as_ref() {
                    Some(acc) => {
                        acc.speaker.is_some() || acc.has_plain_text || !acc.links.is_empty()
                    }
                    None => true,
                };
                if misplaced {
                    return err(
                        &range,
                        "bold is reserved for speaker attribution at the start of a \
                         paragraph (`**id:** text`)"
                            .to_string(),
                    );
                }
                strong_text = Some(String::new());
            }
            Event::End(TagEnd::Strong) => {
                let text = strong_text.take().unwrap_or_default();
                let Some(id) = text.trim().strip_suffix(':') else {
                    return err(
                        &range,
                        format!(
                            "bold '{}' must be a speaker attribution ending in ':'",
                            text.trim()
                        ),
                    );
                };
                let id = id.trim().to_string();
                if !story.characters.contains_key(&id) {
                    return err(
                        &range,
                        format!(
                            "speaker '{}' is not declared in the frontmatter `characters`",
                            id
                        ),
                    );
                }
                para.as_mut()
                    .expect("strong start checked the paragraph")
                    .speaker = Some(id);
            }

            Event::Text(t) => {
                if let Some(meta) = meta_text.as_mut() {
                    meta.push_str(&t);
                } else if let Some(s) = strong_text.as_mut() {
                    s.push_str(&t);
                } else if let Some((_, label)) = link.as_mut() {
                    label.push_str(&t);
                } else if let Some(h) = heading_text.as_mut() {
                    h.push_str(&t);
                } else if in_list {
                    if !t.trim().is_empty() {
                        item_has_text = true;
                    }
                } else if let Some(acc) = para.as_mut() {
                    if !t.trim().is_empty() {
                        acc.has_plain_text = true;
                    }
                    acc.text.push_str(&t);
                }
            }
            Event::SoftBreak => {
                if let Some(acc) = para.as_mut() {
                    acc.text.push(' ');
                }
            }
            Event::HardBreak => {
                if let Some(acc) = para.as_mut() {
                    acc.text.push('\n');
                }
            }

            Event::Start(Tag::Image { .. }) => {
                return err(&range, "images are not supported yet".to_string());
            }
            Event::Start(Tag::CodeBlock(_)) => {
                return err(&range, "code blocks are not supported yet".to_string());
            }
            Event::Code(_) => {
                return err(&range, "inline code is not supported".to_string());
            }
            Event::Start(Tag::Emphasis) => {
                return err(&range, "emphasis (italics) is not supported".to_string());
            }
            Event::Start(Tag::BlockQuote(_)) => {
                return err(&range, "block quotes are not supported".to_string());
            }
            Event::Rule => {
                return err(
                    &range,
                    "thematic breaks (`---`) are not supported in the body".to_string(),
                );
            }
            Event::Html(_) | Event::InlineHtml(_) | Event::Start(Tag::HtmlBlock) => {
                return err(&range, "raw HTML is not supported".to_string());
            }
            Event::End(_) => {}
            other => {
                return err(
                    &range,
                    format!("unsupported Markdown construct: {:?}", other),
                );
            }
        }
    }

    if let Some(node) = cur_node.take() {
        let end = src.len()..src.len();
        finish_node(node, &mut story, &end, &line_of)?;
    }

    validate_story(&story)?;
    Ok(story)
}

fn finish_node(
    node: Node,
    story: &mut Story,
    range: &Range<usize>,
    line_of: &dyn Fn(&Range<usize>) -> usize,
) -> Result<(), String> {
    if node.pages.is_empty() && node.choices.is_empty() {
        return Err(format!(
            "line {}: node '{}' is empty; give it a paragraph or choices",
            line_of(range),
            node.heading
        ));
    }
    story.nodes.push(node);
    Ok(())
}

fn classify_paragraph(
    acc: ParaAcc,
    range: &Range<usize>,
    line_of: &dyn Fn(&Range<usize>) -> usize,
) -> Result<Page, String> {
    let line = line_of(range);
    match acc.links.len() {
        0 => {
            let text = acc.text.trim().to_string();
            if text.is_empty() {
                return Err(format!("line {}: empty paragraph", line));
            }
            Ok(Page {
                speaker: acc.speaker,
                text,
                jump: None,
            })
        }
        1 if !acc.has_plain_text && acc.speaker.is_none() => {
            let (label, target) = acc.links.into_iter().next().expect("length checked");
            Ok(Page {
                speaker: None,
                text: label,
                jump: Some(target),
            })
        }
        _ => Err(format!(
            "line {}: a link must stand alone in its paragraph (a jump) or sit in a \
             bullet list (choices)",
            line
        )),
    }
}

fn validate_story(story: &Story) -> Result<(), String> {
    if story.title.is_empty() {
        return Err("frontmatter must set `title`".to_string());
    }
    if story.nodes.is_empty() {
        return Err("story has no nodes; add a `# heading`".to_string());
    }

    let mut slugs = HashSet::new();
    for node in &story.nodes {
        if !slugs.insert(node.slug.as_str()) {
            return Err(format!(
                "duplicate node '{}' (anchor '#{}')",
                node.heading, node.slug
            ));
        }
    }

    fn targets(node: &Node) -> impl Iterator<Item = &str> {
        node.pages
            .iter()
            .filter_map(|p| p.jump.as_deref())
            .chain(node.choices.iter().map(|c| c.target.as_str()))
    }
    for node in &story.nodes {
        for target in targets(node) {
            if !slugs.contains(target) {
                return Err(format!(
                    "node '{}' links to '#{}', which matches no heading",
                    node.heading, target
                ));
            }
        }
    }
    Ok(())
}

// ---- Frontmatter ----

// The frontmatter is a deliberately strict YAML subset: a `title` line and a
// `characters:` block whose entries take three forms:
//
//   keeper: Innkeeper                                  (name only)
//   ayame: { name: Ayame, color: [1.0, 0.85, 0.8] }    (flow map)
//   ayame:                                             (block map)
//     name: Ayame
//     color: [1.0, 0.85, 0.8]
//
// Names may be quoted or plain; `color` values are JSON arrays.
fn parse_frontmatter(text: &str, story: &mut Story) -> Result<(), String> {
    let mut in_characters = false;
    // Indent of character-id lines, fixed by the first one seen.
    let mut id_indent: Option<usize> = None;
    // A block-form character collects its indented fields until a line at or
    // below the id indent closes it.
    let mut block: Option<BlockCharacter> = None;

    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim_end();
        if line.trim().is_empty() {
            continue;
        }
        let err = |msg: String| Err(format!("frontmatter line {}: {}", i + 1, msg));
        let indent = raw.len() - raw.trim_start().len();

        if indent == 0 {
            finish_block_character(&mut block, story)?;
            in_characters = false;
            id_indent = None;
            let Some((key, value)) = line.split_once(':') else {
                return err(format!("expected `key: value`, got '{}'", line));
            };
            match key.trim() {
                "title" => story.title = unquote(value.trim()).to_string(),
                "characters" => {
                    if !value.trim().is_empty() {
                        return err("`characters` takes an indented block".to_string());
                    }
                    in_characters = true;
                }
                other => {
                    return err(format!(
                        "unknown key '{}'; supported keys are `title` and `characters`",
                        other
                    ));
                }
            }
            continue;
        }

        if !in_characters {
            return err(format!("unexpected indented line '{}'", line.trim()));
        }

        // Deeper than the id indent: a field of the open block character.
        if indent > *id_indent.get_or_insert(indent) {
            let Some(b) = block.as_mut() else {
                return err(format!(
                    "unexpected indented line '{}'; character fields need an `id:` line \
                     above them",
                    line.trim()
                ));
            };
            let Some((key, val)) = line.trim().split_once(':') else {
                return err(format!("expected `key: value`, got '{}'", line.trim()));
            };
            let val = val.trim();
            match key.trim() {
                "name" => b.name = Some(parse_name_value(val).map_err(&err_str(i))?),
                "color" => b.color = parse_color_value(val).map_err(&err_str(i))?,
                other => return err(format!("unknown character key '{}'", other)),
            }
            continue;
        }

        finish_block_character(&mut block, story)?;

        let Some((id, value)) = line.trim().split_once(':') else {
            return err(format!("expected `id: ...`, got '{}'", line.trim()));
        };
        let id = unquote(id.trim()).to_string();
        if id.is_empty()
            || !id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return err(format!(
                "character id '{}' must be alphanumeric ('_' and '-' allowed)",
                id
            ));
        }
        let value = value.trim();
        if value.is_empty() {
            block = Some(BlockCharacter {
                id,
                line: i + 1,
                name: None,
                color: [1.0, 1.0, 1.0],
            });
        } else {
            let character = parse_character(value)
                .map_err(|e| format!("frontmatter line {}: character '{}': {}", i + 1, id, e))?;
            story.characters.insert(id, character);
        }
    }
    finish_block_character(&mut block, story)?;
    Ok(())
}

// A `id:` character whose fields arrive on the following indented lines.
struct BlockCharacter {
    id: String,
    line: usize,
    name: Option<String>,
    color: [f32; 3],
}

fn finish_block_character(
    block: &mut Option<BlockCharacter>,
    story: &mut Story,
) -> Result<(), String> {
    let Some(b) = block.take() else {
        return Ok(());
    };
    let Some(name) = b.name else {
        return Err(format!(
            "frontmatter line {}: character '{}': missing `name`",
            b.line, b.id
        ));
    };
    story.characters.insert(
        b.id,
        Character {
            name,
            color: b.color,
        },
    );
    Ok(())
}

fn err_str(i: usize) -> impl Fn(String) -> String {
    move |e| format!("frontmatter line {}: {}", i + 1, e)
}

fn parse_character(value: &str) -> Result<Character, String> {
    if !value.starts_with('{') {
        let name = unquote(value).to_string();
        if name.is_empty() {
            return Err("expected a display name or `{ name: ..., color: [...] }`".to_string());
        }
        return Ok(Character {
            name,
            color: [1.0, 1.0, 1.0],
        });
    }

    let inner = value
        .strip_prefix('{')
        .and_then(|v| v.strip_suffix('}'))
        .ok_or_else(|| "unterminated `{ ... }`".to_string())?;

    let mut name = None;
    let mut color = [1.0, 1.0, 1.0];
    for field in split_top_level(inner) {
        let field = field.trim();
        if field.is_empty() {
            continue;
        }
        let Some((key, val)) = field.split_once(':') else {
            return Err(format!("expected `key: value`, got '{}'", field));
        };
        let val = val.trim();
        match unquote(key.trim()) {
            "name" => name = Some(parse_name_value(val)?),
            "color" => color = parse_color_value(val)?,
            other => return Err(format!("unknown key '{}'", other)),
        }
    }
    let name = name.ok_or_else(|| "missing `name`".to_string())?;
    Ok(Character { name, color })
}

// A display name: quoted (JSON string) or plain text.
fn parse_name_value(val: &str) -> Result<String, String> {
    let name = if val.starts_with('"') {
        serde_json::from_str::<String>(val)
            .map_err(|_| format!("`name` has unbalanced quotes: {}", val))?
    } else {
        val.to_string()
    };
    if name.is_empty() {
        return Err("`name` must not be empty".to_string());
    }
    Ok(name)
}

fn parse_color_value(val: &str) -> Result<[f32; 3], String> {
    let parsed: Vec<f32> = serde_json::from_str(val)
        .map_err(|_| format!("`color` must be `[r, g, b]`, got '{}'", val))?;
    let [r, g, b] = parsed[..] else {
        return Err(format!(
            "`color` must have 3 components, got {}",
            parsed.len()
        ));
    };
    Ok([r, g, b])
}

// Split `a: 1, b: [2, 3]` on commas outside brackets.
fn split_top_level(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '[' | '{' => depth += 1,
            ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

fn unquote(s: &str) -> &str {
    let s = s.trim();
    s.strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .unwrap_or(s)
}

// ---- Emission ----

// Wrap text to the dialog column on word boundaries, preserving authored
// hard breaks.
pub(crate) fn wrap_text(text: &str, columns: usize) -> String {
    let mut lines: Vec<String> = Vec::new();
    for source_line in text.split('\n') {
        let mut cur = String::new();
        for word in source_line.split_whitespace() {
            if cur.is_empty() {
                cur = word.to_string();
            } else if cur.chars().count() + 1 + word.chars().count() <= columns {
                cur.push(' ');
                cur.push_str(word);
            } else {
                lines.push(std::mem::take(&mut cur));
                cur = word.to_string();
            }
        }
        lines.push(cur);
    }
    lines.join("\n")
}

// Emit the UI asset entries for one parsed story. `prefix` is the sanitized
// import name; every generated name starts with it.
pub(crate) fn emit_story(
    prefix: &str,
    story: &Story,
    title_screen: bool,
) -> Result<Vec<serde_json::Value>, String> {
    let (win_w, win_h) = (UI_REFERENCE_SIZE[0], UI_REFERENCE_SIZE[1]);
    let node_asset = |slug: &str| slug.replace('-', "_");

    // Node views live under a reserved `n` segment so a heading named
    // "title" or "ending" can never collide with the generated title and
    // ending views. First view of a node: its first page, or its choice menu
    // when it has no pages of its own.
    let first_view = |node: &Node| {
        if node.pages.is_empty() {
            format!("{}_n_{}_choice", prefix, node_asset(&node.slug))
        } else {
            format!("{}_n_{}_p0", prefix, node_asset(&node.slug))
        }
    };
    let first_view_of = |slug: &str| {
        story
            .nodes
            .iter()
            .find(|n| n.slug == slug)
            .map(&first_view)
            .expect("targets validated against node slugs")
    };

    let font_title = format!("{}_font_title", prefix);
    let font_menu = format!("{}_font_menu", prefix);
    let font_dialog = format!("{}_font_dialog", prefix);
    let title_view = format!("{}_title", prefix);
    let ending_view = format!("{}_ending", prefix);
    let entry_view = first_view(&story.nodes[0]);

    let mut out = vec![
        font(&font_title, TITLE_FONT_PX),
        font(&font_menu, MENU_FONT_PX),
        font(&font_dialog, DIALOG_FONT_PX),
    ];

    if title_screen {
        out.push(view(&title_view, true));
        out.push(sprite(
            &format!("{}_bg", title_view),
            0.0,
            0.0,
            win_w,
            win_h,
            [0.05, 0.06, 0.12, 1.0],
        ));
        out.push(label(
            &format!("{}_heading", title_view),
            &font_title,
            &story.title,
            LabelStyle {
                x: centered_x(&story.title, TITLE_FONT_PX, win_w),
                y: 180.0,
                color: [1.0, 0.92, 0.78],
                ..LabelStyle::default()
            },
        ));
        out.extend(button(
            &format!("{}_start", title_view),
            &font_menu,
            "Start",
            win_w / 2.0 - 120.0,
            430.0,
            240.0,
            &format!("view:show:{}", entry_view),
        ));
        out.extend(button(
            &format!("{}_quit", title_view),
            &font_menu,
            "Quit",
            win_w / 2.0 - 120.0,
            490.0,
            240.0,
            "quit",
        ));
    }

    let mut ending_used = false;
    let mut view_names: Vec<String> = Vec::new();
    if title_screen {
        view_names.push(title_view.clone());
    }

    for (ni, node) in story.nodes.iter().enumerate() {
        let node_name = node_asset(&node.slug);
        // Where the node goes when its last page advances: its own choices,
        // or the next node in document order, or the ending.
        let node_exit = if !node.choices.is_empty() {
            format!("{}_n_{}_choice", prefix, node_name)
        } else if let Some(next) = story.nodes.get(ni + 1) {
            first_view(next)
        } else {
            ending_used = true;
            ending_view.clone()
        };

        for (pi, page) in node.pages.iter().enumerate() {
            let page_view = format!("{}_n_{}_p{}", prefix, node_name, pi);
            view_names.push(page_view.clone());
            let initial = !title_screen && ni == 0 && pi == 0;
            out.push(view(&page_view, initial));
            out.push(sprite(
                &format!("{}_bg", page_view),
                0.0,
                0.0,
                win_w,
                win_h,
                [0.05, 0.06, 0.09, 1.0],
            ));
            if let Some(id) = &page.speaker {
                let character = &story.characters[id];
                out.push(label(
                    &format!("{}_name", page_view),
                    &font_menu,
                    &character.name,
                    LabelStyle {
                        x: 160.0,
                        y: 478.0,
                        color: character.color,
                        ..LabelStyle::default()
                    },
                ));
            }
            out.push(label(
                &format!("{}_text", page_view),
                &font_dialog,
                &wrap_text(&page.text, WRAP_COLUMNS),
                LabelStyle {
                    x: 160.0,
                    y: 530.0,
                    color: [1.0, 0.95, 0.85],
                    background: Some([0.0, 0.0, 0.0, 0.55]),
                },
            ));
            let target = match &page.jump {
                Some(slug) => first_view_of(slug),
                None if pi + 1 < node.pages.len() => {
                    format!("{}_n_{}_p{}", prefix, node_name, pi + 1)
                }
                None => node_exit.clone(),
            };
            out.push(hit_region(
                &format!("{}_next", page_view),
                0.0,
                0.0,
                win_w,
                win_h,
                None,
                &format!("view:show:{}", target),
            ));
        }

        if !node.choices.is_empty() {
            let choice_view = format!("{}_n_{}_choice", prefix, node_name);
            view_names.push(choice_view.clone());
            let initial = !title_screen && ni == 0 && node.pages.is_empty();
            out.push(view(&choice_view, initial));
            out.push(sprite(
                &format!("{}_bg", choice_view),
                0.0,
                0.0,
                win_w,
                win_h,
                [0.05, 0.06, 0.09, 1.0],
            ));
            out.push(sprite(
                &format!("{}_panel", choice_view),
                160.0,
                180.0,
                win_w - 320.0,
                360.0,
                [0.0, 0.0, 0.0, 0.55],
            ));
            let y0 = win_h / 2.0 - node.choices.len() as f32 * 30.0;
            for (ci, choice) in node.choices.iter().enumerate() {
                out.extend(button(
                    &format!("{}_opt{}", choice_view, ci),
                    &font_menu,
                    &choice.label,
                    280.0,
                    y0 + ci as f32 * 60.0,
                    win_w - 560.0,
                    &format!("view:show:{}", first_view_of(&choice.target)),
                ));
            }
        }
    }

    if ending_used {
        view_names.push(ending_view.clone());
        out.push(view(&ending_view, false));
        out.push(sprite(
            &format!("{}_bg", ending_view),
            0.0,
            0.0,
            win_w,
            win_h,
            [0.03, 0.03, 0.05, 1.0],
        ));
        out.push(label(
            &format!("{}_fin", ending_view),
            &font_title,
            "~ fin ~",
            LabelStyle {
                x: centered_x("~ fin ~", TITLE_FONT_PX, win_w),
                y: 260.0,
                color: [0.95, 0.88, 0.7],
                ..LabelStyle::default()
            },
        ));
        let (back_label, back_target) = if title_screen {
            ("Back to title", title_view.clone())
        } else {
            ("Restart", entry_view.clone())
        };
        out.extend(button(
            &format!("{}_back", ending_view),
            &font_menu,
            back_label,
            win_w / 2.0 - 160.0,
            490.0,
            320.0,
            &format!("view:show:{}", back_target),
        ));
    }

    // UI assets attach to a View by name prefix, so one generated view name
    // must never be a `_`-extension of another or the members of the longer
    // view would be ambiguous.
    let mut sorted = view_names.clone();
    sorted.sort();
    for pair in sorted.windows(2) {
        if pair[1].starts_with(&format!("{}_", pair[0])) {
            return Err(format!(
                "generated view '{}' is a name-prefix of '{}'; rename one of the headings",
                pair[0], pair[1]
            ));
        }
    }

    Ok(out)
}

fn font(name: &str, size_px: u32) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "type": "Font",
        "args": { "size_px": size_px }
    })
}

fn view(name: &str, initial: bool) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "type": "View",
        "args": { "initial": initial }
    })
}

fn sprite(name: &str, x: f32, y: f32, w: f32, h: f32, tint: [f32; 4]) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "type": "Sprite",
        "args": { "x": x, "y": y, "width": w, "height": h, "tint": tint }
    })
}

// Approximate text width for horizontal centering: this stage has no font
// metrics, so glyph advance is estimated from the font size. `centered` on
// TextLabel is not usable here; the renderer treats it as splash text and
// auto-scales it to fill the viewport.
fn est_text_width(text: &str, font_px: u32) -> f32 {
    text.chars().count() as f32 * 0.6 * font_px as f32
}

fn centered_x(text: &str, font_px: u32, span: f32) -> f32 {
    ((span - est_text_width(text, font_px)) / 2.0).max(0.0)
}

#[derive(Default)]
struct LabelStyle {
    x: f32,
    y: f32,
    color: [f32; 3],
    background: Option<[f32; 4]>,
}

fn label(name: &str, font: &str, content: &str, style: LabelStyle) -> serde_json::Value {
    let mut args = serde_json::json!({
        "font": font,
        "content": content,
        "x": style.x,
        "y": style.y,
        "color": style.color,
        "scale": 1.0,
    });
    if let Some(bg) = style.background {
        args["background"] = serde_json::json!(bg);
        args["padding"] = serde_json::json!(20.0);
    }
    serde_json::json!({ "name": name, "type": "TextLabel", "args": args })
}

fn hit_region(
    name: &str,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    label: Option<&str>,
    action: &str,
) -> serde_json::Value {
    let mut args = serde_json::json!({
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "action": action,
    });
    if let Some(l) = label {
        args["label"] = serde_json::json!(l);
        args["hover_color"] = serde_json::json!([1.0, 0.85, 0.3]);
        args["hover_scale"] = serde_json::json!(1.06);
    }
    serde_json::json!({ "name": name, "type": "HitRegion", "args": args })
}

// A clickable menu row: a TextLabel and the HitRegion that styles and fires
// it. The label is centered in the region by estimated text width; buttons
// always use the menu font.
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
                x: x + centered_x(text, MENU_FONT_PX, w).min(w - 20.0),
                y: y + 6.0,
                color: [0.92, 0.92, 0.92],
                ..LabelStyle::default()
            },
        ),
        hit_region(&format!("{}_btn", name), x, y, w, 40.0, Some(&lbl), action),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const CROSSROADS: &str = r#"---
title: The Crossroads
characters:
  ayame: { name: "Ayame", color: [1.0, 0.85, 0.8] }
  keeper: Innkeeper
---

# inn

You wake at a roadside inn. A note rests on the pillow.

**keeper:** Slept well? Someone left that for you.

# The Crossroads

The signpost points two ways.

- [Into the wood](#wood)
- [Toward the shore](#the-crossroads)

# wood

**ayame:** You came. I wasn't sure you would.

[The morning comes.](#ending)

# ending

You walk together toward the morning.
"#;

    fn find<'a>(entries: &'a [serde_json::Value], name: &str) -> &'a serde_json::Value {
        entries
            .iter()
            .find(|e| asset_name(e) == name)
            .unwrap_or_else(|| panic!("missing entry '{}'", name))
    }

    fn action(entry: &serde_json::Value) -> &str {
        entry["args"]["action"].as_str().unwrap_or("")
    }

    #[test]
    fn passes_through_without_imports() {
        let mut assets = vec![serde_json::json!({"name":"x","type":"Logger","args":{}})];
        expand_stories(&mut assets).unwrap();
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0]["type"], "Logger");
    }

    #[test]
    fn missing_source_is_an_error() {
        let mut assets = vec![serde_json::json!({
            "name": "story", "type": "StoryImport", "args": {}
        })];
        let err = expand_stories(&mut assets).unwrap_err();
        assert!(err.contains("missing `source`"));
    }

    #[test]
    fn parses_frontmatter_nodes_and_flow() {
        let story = parse_story(CROSSROADS).unwrap();
        assert_eq!(story.title, "The Crossroads");
        assert_eq!(story.characters["ayame"].name, "Ayame");
        assert_eq!(story.characters["ayame"].color, [1.0, 0.85, 0.8]);
        assert_eq!(story.characters["keeper"].name, "Innkeeper");
        assert_eq!(story.characters["keeper"].color, [1.0, 1.0, 1.0]);

        let slugs: Vec<&str> = story.nodes.iter().map(|n| n.slug.as_str()).collect();
        assert_eq!(slugs, ["inn", "the-crossroads", "wood", "ending"]);

        let inn = &story.nodes[0];
        assert_eq!(inn.pages.len(), 2);
        assert_eq!(inn.pages[1].speaker.as_deref(), Some("keeper"));

        let crossroads = &story.nodes[1];
        assert_eq!(crossroads.choices.len(), 2);
        assert_eq!(crossroads.choices[0].target, "wood");
        assert_eq!(crossroads.choices[1].target, "the-crossroads");

        let wood = &story.nodes[2];
        assert_eq!(wood.pages[1].jump.as_deref(), Some("ending"));
        assert_eq!(wood.pages[1].text, "The morning comes.");
    }

    #[test]
    fn emits_wired_views() {
        let story = parse_story(CROSSROADS).unwrap();
        let entries = emit_story("story", &story, true).unwrap();

        // Title screen starts the flow at the first node's first page.
        let title = find(&entries, "story_title");
        assert_eq!(title["args"]["initial"], true);
        let start = find(&entries, "story_title_start_btn");
        assert_eq!(action(start), "view:show:story_n_inn_p0");

        // Narration pages advance page -> page -> next node.
        assert_eq!(
            action(find(&entries, "story_n_inn_p0_next")),
            "view:show:story_n_inn_p1"
        );
        assert_eq!(
            action(find(&entries, "story_n_inn_p1_next")),
            "view:show:story_n_the_crossroads_p0"
        );

        // The speaker page carries a name plate in the character's color.
        let plate = find(&entries, "story_n_inn_p1_name");
        assert_eq!(plate["args"]["content"], "Innkeeper");

        // The choice node's page advances into its choice menu; each option
        // targets its node's first page.
        assert_eq!(
            action(find(&entries, "story_n_the_crossroads_p0_next")),
            "view:show:story_n_the_crossroads_choice"
        );
        assert_eq!(
            action(find(&entries, "story_n_the_crossroads_choice_opt0_btn")),
            "view:show:story_n_wood_p0"
        );
        assert_eq!(
            action(find(&entries, "story_n_the_crossroads_choice_opt1_btn")),
            "view:show:story_n_the_crossroads_p0"
        );

        // The jump page keeps its label text and targets the linked node.
        let jump_page = find(&entries, "story_n_wood_p1_text");
        assert_eq!(jump_page["args"]["content"], "The morning comes.");
        assert_eq!(
            action(find(&entries, "story_n_wood_p1_next")),
            "view:show:story_n_ending_p0"
        );

        // The last node falls through to the generated ending, which returns
        // to the title screen.
        assert_eq!(
            action(find(&entries, "story_n_ending_p0_next")),
            "view:show:story_ending"
        );
        assert_eq!(
            action(find(&entries, "story_ending_back_btn")),
            "view:show:story_title"
        );
    }

    #[test]
    fn no_title_screen_makes_first_page_initial() {
        let story = parse_story(CROSSROADS).unwrap();
        let entries = emit_story("story", &story, false).unwrap();
        assert!(!entries.iter().any(|e| asset_name(e) == "story_title"));
        assert_eq!(find(&entries, "story_n_inn_p0")["args"]["initial"], true);
        let back = find(&entries, "story_ending_back_btn");
        assert_eq!(action(back), "view:show:story_n_inn_p0");
        assert_eq!(
            find(&entries, "story_ending_back_lbl")["args"]["content"],
            "Restart"
        );
    }

    #[test]
    fn expands_from_file_and_replaces_the_import() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("story.md");
        std::fs::write(&path, CROSSROADS).unwrap();
        let mut assets = vec![serde_json::json!({
            "name": "story", "type": "StoryImport",
            "args": {"source": path.to_str().unwrap()}
        })];
        expand_stories(&mut assets).unwrap();
        assert!(!assets.iter().any(|v| type_norm(v) == "storyimport"));
        assert!(assets.iter().any(|v| type_norm(v) == "view"));
        assert!(assets.iter().any(|v| type_norm(v) == "hitregion"));
        assert!(assets.iter().any(|v| type_norm(v) == "font"));
    }

    #[test]
    fn generated_name_collision_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("story.md");
        std::fs::write(&path, CROSSROADS).unwrap();
        let mut assets = vec![
            serde_json::json!({
                "name": "story", "type": "StoryImport",
                "args": {"source": path.to_str().unwrap()}
            }),
            serde_json::json!({"name":"story_title","type":"View","args":{}}),
        ];
        let err = expand_stories(&mut assets).unwrap_err();
        assert!(err.contains("collides"));
    }

    #[test]
    fn block_style_characters_parse() {
        let src = "---\ntitle: T\ncharacters:\n  ayame:\n    name: Ayame\n    color: [1.0, 0.85, 0.8]\n  keeper: Innkeeper\n---\n\n# a\n\n**ayame:** hi\n\n**keeper:** bye\n";
        let story = parse_story(src).unwrap();
        assert_eq!(story.characters["ayame"].name, "Ayame");
        assert_eq!(story.characters["ayame"].color, [1.0, 0.85, 0.8]);
        assert_eq!(story.characters["keeper"].name, "Innkeeper");
        assert_eq!(story.characters["keeper"].color, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn unquoted_flow_map_name_parses() {
        let src = "---\ntitle: T\ncharacters:\n  a: { name: Ayame Doe, color: [1, 1, 1] }\n---\n\n# n\n\n**a:** hi\n";
        let story = parse_story(src).unwrap();
        assert_eq!(story.characters["a"].name, "Ayame Doe");
    }

    #[test]
    fn block_character_missing_name_is_an_error() {
        let src = "---\ntitle: T\ncharacters:\n  ayame:\n    color: [1, 1, 1]\n---\n\n# a\n\nhi\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("missing `name`"), "{err}");
        assert!(err.contains("ayame"), "{err}");
    }

    #[test]
    fn block_character_unknown_key_is_an_error() {
        let src = "---\ntitle: T\ncharacters:\n  ayame:\n    name: Ayame\n    voice: low\n---\n\n# a\n\nhi\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("unknown character key 'voice'"), "{err}");
    }

    #[test]
    fn fields_under_a_plain_character_are_an_error() {
        // `keeper: Innkeeper` is complete; a deeper line under it has no open
        // block to attach to.
        let src = "---\ntitle: T\ncharacters:\n  keeper: Innkeeper\n    color: [1, 1, 1]\n---\n\n# a\n\nhi\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("need an `id:` line"), "{err}");
    }

    #[test]
    fn dangling_link_target_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n[go](#nowhere)\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("#nowhere"), "{err}");
    }

    #[test]
    fn undeclared_speaker_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n**ghost:** boo\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("ghost"), "{err}");
        assert!(err.contains("line 7"), "{err}");
    }

    #[test]
    fn duplicate_heading_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\nhi\n\n# a\n\nbye\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("duplicate"), "{err}");
    }

    #[test]
    fn missing_title_is_an_error() {
        let src = "---\ncharacters:\n---\n\n# a\n\nhi\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("title"), "{err}");
    }

    #[test]
    fn content_before_first_heading_is_an_error() {
        let src = "---\ntitle: T\n---\n\nno node yet\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("before the first"), "{err}");
    }

    #[test]
    fn empty_node_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n# b\n\nhi\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn unsupported_constructs_are_errors() {
        let base = "---\ntitle: T\n---\n\n# a\n\n";
        for (body, needle) in [
            ("![img](x.png)\n", "images"),
            ("```\ncode\n```\n", "code blocks"),
            ("> quoted\n", "block quotes"),
            ("## sub\n", "headings"),
            ("*soft*\n", "emphasis"),
            ("1. [go](#a)\n", "bullet list"),
            ("see `code`\n", "inline code"),
        ] {
            let err = parse_story(&format!("{base}{body}")).unwrap_err();
            assert!(err.contains(needle), "{body:?} -> {err}");
        }
    }

    #[test]
    fn mixed_link_and_text_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\ngo [here](#a) now\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("stand alone"), "{err}");
    }

    #[test]
    fn content_after_choices_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n- [go](#a)\n\nafterthought\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("last content"), "{err}");
    }

    #[test]
    fn ambiguous_view_prefix_is_an_error() {
        // "wood p0" slugs to wood-p0, whose page-0 view name extends the
        // page-0 view of "wood".
        let src = "---\ntitle: T\n---\n\n# wood\n\nhi\n\n# wood p0\n\nbye\n";
        let story = parse_story(src).unwrap();
        let err = emit_story("s", &story, true).unwrap_err();
        assert!(err.contains("name-prefix"), "{err}");
    }

    #[test]
    fn slug_matches_github_style_anchors() {
        assert_eq!(slug("The Crossroads"), "the-crossroads");
        assert_eq!(slug("  Inn's  Door  "), "inns-door");
        assert_eq!(slug("wood"), "wood");
    }

    #[test]
    fn wrap_text_wraps_on_word_boundaries() {
        let wrapped = wrap_text("one two three four five", 9);
        assert_eq!(wrapped, "one two\nthree\nfour five");
        // Authored hard breaks survive.
        assert_eq!(wrap_text("a\nb", 80), "a\nb");
    }
}
