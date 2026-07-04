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
// node's final content: a menu of links out; the choice menu carries the
// stage dressing, music, and one-shot sounds current at the point the list
// appears.
#[derive(Debug, Default)]
pub(crate) struct Node {
    pub(crate) slug: String,
    pub(crate) heading: String,
    pub(crate) pages: Vec<Page>,
    pub(crate) choices: Vec<Choice>,
    pub(crate) choice_music: Option<String>,
    pub(crate) choice_sounds: Vec<String>,
    pub(crate) choice_stage: Stage,
}

// The visual dressing current at a page: the backdrop image and the
// character portraits standing on stage. A `![bg]` directive replaces
// the backdrop AND clears all portraits (a scene change); `![left]` /
// `![center]` / `![right]` swap one portrait and persist until the next
// scene change.
#[derive(Debug, Default, Clone)]
pub(crate) struct Stage {
    pub(crate) bg: Option<String>,
    pub(crate) left: Option<String>,
    pub(crate) center: Option<String>,
    pub(crate) right: Option<String>,
}

// One click-through page. `jump` overrides the default advance (next page,
// then the node's choices or fall-through) with an explicit node target.
// `music` is the audio-file path current at this page (from the most recent
// `[music]` directive in document order); `sounds` are the one-shots the
// directives directly above this page queued.
#[derive(Debug, Default)]
pub(crate) struct Page {
    pub(crate) speaker: Option<String>,
    pub(crate) text: String,
    pub(crate) jump: Option<String>,
    pub(crate) music: Option<String>,
    pub(crate) sounds: Vec<String>,
    pub(crate) stage: Stage,
}

#[derive(Debug)]
pub(crate) struct Choice {
    pub(crate) label: String,
    pub(crate) target: String,
}

// A media directive paragraph: a lone link whose label names the channel and
// whose target is an audio file, or an image whose alt names its stage role.
enum Directive {
    Music(String),
    Sound(String),
    Bg(String),
    Left(String),
    Center(String),
    Right(String),
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
    images: Vec<(String, String)>,
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
    let mut image: Option<(String, String)> = None; // (target, alt so far)
    let mut in_list = false;
    let mut item_links: Vec<(String, String)> = Vec::new();
    let mut item_has_text = false;
    // Media state in document order: the music current from the most recent
    // `[music]` directive, and one-shot `[sound]`s waiting for the next page
    // or choice list. A directive with nothing after it to attach to is dead
    // and rejected at end of parse.
    let mut current_music: Option<String> = None;
    let mut current_stage = Stage::default();
    let mut pending_sounds: Vec<String> = Vec::new();
    let mut unconsumed_directive: Option<usize> = None;

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
                    match classify_paragraph(acc, &range, &line_of)? {
                        ParaOut::Page(mut page) => {
                            page.music = current_music.clone();
                            page.stage = current_stage.clone();
                            page.sounds = std::mem::take(&mut pending_sounds);
                            unconsumed_directive = None;
                            cur_node
                                .as_mut()
                                .expect("paragraph start checked the node")
                                .pages
                                .push(page);
                        }
                        ParaOut::Directives(directives) => {
                            for directive in directives {
                                match directive {
                                    Directive::Music(path) => current_music = Some(path),
                                    Directive::Sound(path) => pending_sounds.push(path),
                                    // A backdrop change is a scene change:
                                    // the portraits leave with the old scene.
                                    Directive::Bg(path) => {
                                        current_stage = Stage {
                                            bg: Some(path),
                                            ..Stage::default()
                                        };
                                    }
                                    Directive::Left(path) => current_stage.left = Some(path),
                                    Directive::Center(path) => current_stage.center = Some(path),
                                    Directive::Right(path) => current_stage.right = Some(path),
                                }
                            }
                            unconsumed_directive = Some(line_of(&range));
                        }
                    }
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
                // The choice menu is shown like a page, so it carries the
                // current music and consumes any queued one-shots.
                let node = cur_node.as_mut().expect("list start checked the node");
                node.choice_music = current_music.clone();
                node.choice_stage = current_stage.clone();
                node.choice_sounds = std::mem::take(&mut pending_sounds);
                unconsumed_directive = None;
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
                if image.is_some() {
                    return err(
                        &range,
                        "links are not supported in image alt text".to_string(),
                    );
                }
                // Targets are classified when the enclosing construct closes:
                // `#heading` jumps/choices, audio-file media directives, or
                // image directives.
                link = Some((dest_url.to_string(), String::new()));
            }
            Event::End(TagEnd::Link) => {
                let (target, label) = link.take().unwrap_or_default();
                if label.trim().is_empty() {
                    return err(&range, format!("link to '{}' has no label text", target));
                }
                if in_list {
                    let Some(anchor) = target.strip_prefix('#') else {
                        return err(
                            &range,
                            format!("choice '{}' must link to a `#heading`", label.trim()),
                        );
                    };
                    item_links.push((anchor.to_string(), label.trim().to_string()));
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
                } else if let Some((_, alt)) = image.as_mut() {
                    alt.push_str(&t);
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

            Event::Start(Tag::Image { dest_url, .. }) => {
                if para.is_none() || in_list {
                    return err(
                        &range,
                        "images may only appear alone in their own paragraph".to_string(),
                    );
                }
                image = Some((dest_url.to_string(), String::new()));
            }
            Event::End(TagEnd::Image) => {
                let (target, alt) = image.take().unwrap_or_default();
                para.as_mut()
                    .expect("image start checked the paragraph")
                    .images
                    .push((alt.trim().to_string(), target));
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

    if let Some(line) = unconsumed_directive {
        return Err(format!(
            "line {}: a media directive needs a following paragraph or choice list to \
             attach to",
            line
        ));
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

// Reads an image file's pixel dimensions. Injected into emission (the real
// reader probes file headers) so tests run without image files on disk.
type ImageDims<'a> = &'a dyn Fn(&str) -> Result<(u32, u32), String>;

// What one paragraph contributes: a page of the story, or media directives
// that style the pages after it. Directives stack: a paragraph made only of
// `![bg]` / `[music]` / `[sound]` lines applies them all.
enum ParaOut {
    Page(Page),
    Directives(Vec<Directive>),
}

const AUDIO_EXTENSIONS: [&str; 4] = ["ogg", "wav", "mp3", "flac"];
const IMAGE_EXTENSIONS: [&str; 3] = ["png", "jpg", "jpeg"];

fn file_extension(target: &str) -> String {
    std::path::Path::new(target)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default()
}

fn classify_paragraph(
    acc: ParaAcc,
    range: &Range<usize>,
    line_of: &dyn Fn(&Range<usize>) -> usize,
) -> Result<ParaOut, String> {
    let line = line_of(range);

    // A jump: exactly one `#heading` link and nothing else.
    if acc.links.len() == 1
        && acc.images.is_empty()
        && !acc.has_plain_text
        && acc.speaker.is_none()
        && acc.links[0].1.starts_with('#')
    {
        let (label, target) = acc.links.into_iter().next().expect("length checked");
        return Ok(ParaOut::Page(Page {
            text: label,
            jump: Some(target[1..].to_string()),
            ..Page::default()
        }));
    }

    // A directives paragraph: images and file links only, no prose. They
    // often stack (a backdrop plus music at a scene top), so each line of
    // the paragraph is classified on its own.
    if !acc.images.is_empty() || acc.links.iter().any(|(_, t)| !t.starts_with('#')) {
        if acc.has_plain_text || acc.speaker.is_some() {
            return Err(format!(
                "line {}: media directives must stand alone in their own paragraph",
                line
            ));
        }
        let mut directives = Vec::new();
        for (alt, target) in acc.images {
            if !IMAGE_EXTENSIONS.contains(&file_extension(&target).as_str()) {
                return Err(format!(
                    "line {}: image '{}' must be a {} file",
                    line,
                    target,
                    IMAGE_EXTENSIONS.join("/")
                ));
            }
            match alt.as_str() {
                "bg" => directives.push(Directive::Bg(target)),
                "left" => directives.push(Directive::Left(target)),
                "center" => directives.push(Directive::Center(target)),
                "right" => directives.push(Directive::Right(target)),
                other => {
                    return Err(format!(
                        "line {}: image role '{}' is not supported; use `![bg]` for a \
                         backdrop or `![left]` / `![center]` / `![right]` for portraits",
                        line, other
                    ));
                }
            }
        }
        for (label, target) in acc.links {
            if target.starts_with('#') {
                return Err(format!(
                    "line {}: a jump link cannot share a paragraph with media directives",
                    line
                ));
            }
            if !AUDIO_EXTENSIONS.contains(&file_extension(&target).as_str()) {
                return Err(format!(
                    "line {}: link '{}' targets neither a `#heading` (a jump) nor an \
                     audio file ({})",
                    line,
                    target,
                    AUDIO_EXTENSIONS.join("/")
                ));
            }
            match label.as_str() {
                "music" => directives.push(Directive::Music(target)),
                "sound" => directives.push(Directive::Sound(target)),
                other => {
                    return Err(format!(
                        "line {}: audio link label must be `music` (looping) or `sound` \
                         (one-shot), got '{}'",
                        line, other
                    ));
                }
            }
        }
        return Ok(ParaOut::Directives(directives));
    }

    match acc.links.len() {
        0 => {
            let text = acc.text.trim().to_string();
            if text.is_empty() {
                return Err(format!("line {}: empty paragraph", line));
            }
            Ok(ParaOut::Page(Page {
                speaker: acc.speaker,
                text,
                ..Page::default()
            }))
        }
        _ => Err(format!(
            "line {}: a link must stand alone in its paragraph (a jump or media \
             directive) or sit in a bullet list (choices)",
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

// Emit the runtime assets for one parsed story: the compiled Story graph
// plus the stage scaffolding the story system drives at runtime. Instead of
// a View per page, the whole story plays inside one stage view whose labels
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
    let title_view = format!("{}_title", prefix);
    let stage_view = format!("{}_stage", prefix);
    let ending_view = format!("{}_ending", prefix);

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
            "story:start",
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

    // Audio files referenced by media directives, deduplicated by path in
    // first-use order; each becomes one AudioClip entry. Same for stage
    // images, which become Texture entries the stage sprites sample.
    let mut clips: Vec<(String, String)> = Vec::new();
    let mut images: Vec<(String, String)> = Vec::new();

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
            }));
        }
        let choices: Vec<serde_json::Value> = node
            .choices
            .iter()
            .map(|c| serde_json::json!({ "label": c.label, "target": node_index(&c.target) }))
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
        }));
    }
    // The compiled graph takes the import's own name: the one declaration the
    // author wrote stays the one asset that carries the story.
    out.push(serde_json::json!({
        "name": prefix,
        "type": "Story",
        "args": {
            "title": story.title,
            "nodes": nodes_json,
            "text_speed": text_speed,
        }
    }));

    // The stage: one view the story system drives. Sprites and labels are
    // placeholders here; the system fills text, swaps textures, and toggles
    // visibility page by page. Declaration order is draw order.
    out.push(view(&stage_view, !title_screen));
    out.push(stage_sprite(
        &format!("{}_bg", stage_view),
        [0.0, 0.0, win_w, win_h],
        [0.05, 0.06, 0.09, 1.0],
        true,
    ));
    for side in ["left", "center", "right"] {
        out.push(stage_sprite(
            &format!("{}_{}", stage_view, side),
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            false,
        ));
    }
    out.push(sprite(
        &format!("{}_box", stage_view),
        DIALOG_BOX.0,
        DIALOG_BOX.1,
        DIALOG_BOX.2,
        DIALOG_BOX.3,
        [0.0, 0.0, 0.0, 0.55],
    ));
    out.push(label(
        &format!("{}_name", stage_view),
        &font_menu,
        "",
        LabelStyle {
            x: 160.0,
            y: 478.0,
            color: [1.0, 1.0, 1.0],
            ..LabelStyle::default()
        },
    ));
    out.push(label(
        &format!("{}_text", stage_view),
        &font_dialog,
        "",
        LabelStyle {
            x: 160.0,
            y: 530.0,
            color: [1.0, 0.95, 0.85],
            ..LabelStyle::default()
        },
    ));
    out.push(hit_region(
        &format!("{}_advance", stage_view),
        0.0,
        0.0,
        win_w,
        win_h,
        None,
        "story:advance",
    ));
    out.push(serde_json::json!({
        "name": format!("{}_advance_key", prefix),
        "type": "KeyBinding",
        "args": { "key": "Space", "action": "story:advance" }
    }));

    // Choice furniture, sized for the widest menu in the story: a dim panel
    // and one button per option, hidden until the story system reaches a
    // choice. The buttons stay hit-active the whole time; the story system
    // ignores a choose action outside a menu (and an advance inside one), so
    // the overlap with the full-canvas advance region resolves by mode.
    let max_choices = story
        .nodes
        .iter()
        .map(|n| n.choices.len())
        .max()
        .unwrap_or(0);
    if max_choices > 0 {
        out.push(stage_sprite(
            &format!("{}_panel", stage_view),
            [160.0, 180.0, win_w - 320.0, 360.0],
            [0.0, 0.0, 0.0, 0.0],
            false,
        ));
        let y0 = win_h / 2.0 - max_choices as f32 * 30.0;
        for ci in 0..max_choices {
            let lbl = format!("{}_opt{}_lbl", stage_view, ci);
            let y = y0 + ci as f32 * 60.0;
            out.push(serde_json::json!({
                "name": lbl,
                "type": "TextLabel",
                "args": {
                    "font": font_menu,
                    "content": "",
                    "x": 280.0,
                    "y": y + 6.0,
                    "color": [0.92, 0.92, 0.92],
                    "scale": 1.0,
                    "visible": false,
                }
            }));
            out.push(serde_json::json!({
                "name": format!("{}_opt{}_btn", stage_view, ci),
                "type": "HitRegion",
                "args": {
                    "x": 280.0,
                    "y": y,
                    "width": win_w - 560.0,
                    "height": 40.0,
                    "label": lbl,
                    "hover_color": [1.0, 0.85, 0.3],
                    "hover_scale": 1.06,
                    "action": format!("story:choose:{}", ci),
                }
            }));
        }
    }

    // The ending screen, shown by the story system when the last node runs
    // out of pages.
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
    let (back_label, back_action) = if title_screen {
        ("Back to title", format!("view:show:{}", title_view))
    } else {
        ("Restart", "story:start".to_string())
    };
    out.extend(button(
        &format!("{}_back", ending_view),
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

    // UI assets attach to a View by name prefix, so one generated view name
    // must never be a `_`-extension of another or the members of the longer
    // view would be ambiguous.
    let mut view_names = vec![stage_view, ending_view];
    if title_screen {
        view_names.push(title_view);
    }
    view_names.sort();
    for pair in view_names.windows(2) {
        if pair[1].starts_with(&format!("{}_", pair[0])) {
            return Err(format!(
                "generated view '{}' is a name-prefix of '{}'",
                pair[0], pair[1]
            ));
        }
    }

    Ok(out)
}

// The fixed dialog box the stage's name plate and dialog text sit on.
const DIALOG_BOX: (f32, f32, f32, f32) = (140.0, 505.0, 1000.0, 190.0);

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

// A stage-owned sprite the story system mutates: cover fit (full-bleed stage
// imagery reaches the window edges without distorting) with an explicit
// initial visibility.
fn stage_sprite(name: &str, rect: [f32; 4], tint: [f32; 4], visible: bool) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "type": "Sprite",
        "args": {
            "x": rect[0], "y": rect[1], "width": rect[2], "height": rect[3],
            "tint": tint,
            "fit": "cover",
            "visible": visible,
        }
    })
}

// Read an image file's pixel dimensions from its header, without decoding
// the pixels.
fn probe_image_dims(path: &str) -> Result<(u32, u32), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("cannot read '{}': {}", path, e))?;
    if file_extension(path) == "png" {
        let decoder = png::Decoder::new(std::io::BufReader::new(file));
        let reader = decoder
            .read_info()
            .map_err(|e| format!("'{}': {}", path, e))?;
        let info = reader.info();
        Ok((info.width, info.height))
    } else {
        let mut decoder = jpeg_decoder::Decoder::new(std::io::BufReader::new(file));
        decoder
            .read_info()
            .map_err(|e| format!("'{}': {}", path, e))?;
        let info = decoder
            .info()
            .ok_or_else(|| format!("'{}': no image info", path))?;
        Ok((info.width as u32, info.height as u32))
    }
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

    // Fixed portrait-shaped dimensions so emission tests need no image files.
    fn stub_dims(_path: &str) -> Result<(u32, u32), String> {
        Ok((456, 700))
    }

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
    fn emits_the_compiled_graph_and_stage() {
        let story = parse_story(CROSSROADS).unwrap();
        let entries = emit_story("story", &story, true, 30.0, &stub_dims).unwrap();

        // The title screen is initial and starts the story system.
        let title = find(&entries, "story_title");
        assert_eq!(title["args"]["initial"], true);
        assert_eq!(
            action(find(&entries, "story_title_start_btn")),
            "story:start"
        );

        // The compiled graph takes the import's own name. Speakers resolve
        // to display name + color, jump and choice targets to node indices,
        // and the reveal speed rides along.
        let graph = &find(&entries, "story")["args"];
        assert_eq!(graph["title"], "The Crossroads");
        assert_eq!(graph["text_speed"], 30.0);
        let nodes = graph["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 4);
        let plate = &nodes[0]["pages"][1]["speaker"];
        assert_eq!(plate["name"], "Innkeeper");
        assert_eq!(nodes[1]["choices"][0]["label"], "Into the wood");
        assert_eq!(nodes[1]["choices"][0]["target"], 2);
        assert_eq!(nodes[1]["choices"][1]["target"], 1);
        assert_eq!(nodes[2]["pages"][1]["text"], "The morning comes.");
        assert_eq!(nodes[2]["pages"][1]["jump"], 3);

        // One stage view carries the whole story: backdrop, portrait slots,
        // dialog furniture, the advance region, and one button per option of
        // the widest choice menu (disabled until the story reaches one).
        assert_eq!(find(&entries, "story_stage")["args"]["initial"], false);
        assert_eq!(find(&entries, "story_stage_bg")["args"]["fit"], "cover");
        assert_eq!(
            find(&entries, "story_stage_center")["args"]["visible"],
            false
        );
        assert_eq!(
            action(find(&entries, "story_stage_advance")),
            "story:advance"
        );
        assert_eq!(
            find(&entries, "story_advance_key")["args"]["action"],
            "story:advance"
        );
        let opt0 = find(&entries, "story_stage_opt0_btn");
        assert_eq!(action(opt0), "story:choose:0");
        assert_eq!(
            find(&entries, "story_stage_opt0_lbl")["args"]["visible"],
            false
        );
        assert!(
            entries
                .iter()
                .any(|e| asset_name(e) == "story_stage_opt1_btn")
        );
        assert!(
            !entries
                .iter()
                .any(|e| asset_name(e) == "story_stage_opt2_btn")
        );

        // No per-page views or audio cues remain.
        assert!(!entries.iter().any(|e| asset_name(e).contains("_n_")));
        assert!(!entries.iter().any(|e| type_norm(e) == "audiocue"));

        // The ending returns to the title screen.
        assert_eq!(
            action(find(&entries, "story_ending_back_btn")),
            "view:show:story_title"
        );
    }

    #[test]
    fn no_title_screen_makes_the_stage_initial() {
        let story = parse_story(CROSSROADS).unwrap();
        let entries = emit_story("story", &story, false, 45.0, &stub_dims).unwrap();
        assert!(!entries.iter().any(|e| asset_name(e) == "story_title"));
        assert_eq!(find(&entries, "story_stage")["args"]["initial"], true);
        let back = find(&entries, "story_ending_back_btn");
        assert_eq!(action(back), "story:start");
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

    const SCORED: &str = r#"---
title: T
---

# inn

[music](assets/theme.ogg)

First page.

Second page.

[sound](assets/door.wav)

The door creaks.

# crossroads

[music](assets/tense.ogg)

- [Left](#inn)
- [Right](#crossroads)
"#;

    #[test]
    fn media_directives_parse_and_propagate() {
        let story = parse_story(SCORED).unwrap();
        let inn = &story.nodes[0];
        // Music persists from its directive across the following pages.
        assert_eq!(inn.pages[0].music.as_deref(), Some("assets/theme.ogg"));
        assert_eq!(inn.pages[1].music.as_deref(), Some("assets/theme.ogg"));
        // The one-shot attaches only to the page directly after it.
        assert!(inn.pages[1].sounds.is_empty());
        assert_eq!(inn.pages[2].sounds, ["assets/door.wav"]);
        // A pages-free choice node still carries the current music.
        let crossroads = &story.nodes[1];
        assert!(crossroads.pages.is_empty());
        assert_eq!(crossroads.choice_music.as_deref(), Some("assets/tense.ogg"));
    }

    #[test]
    fn media_directives_compile_to_deduped_clip_names() {
        let story = parse_story(SCORED).unwrap();
        let entries = emit_story("s", &story, true, 45.0, &stub_dims).unwrap();

        // Three distinct audio files -> three AudioClip entries; the theme is
        // deduplicated to one clip despite two pages sharing its music.
        let clips: Vec<&serde_json::Value> = entries
            .iter()
            .filter(|e| type_norm(e) == "audioclip")
            .collect();
        assert_eq!(clips.len(), 3);
        assert_eq!(clips[0]["args"]["source"], "assets/theme.ogg");

        // Pages carry the deduplicated clip names in the compiled graph.
        let nodes = &find(&entries, "s")["args"]["nodes"];
        assert_eq!(nodes[0]["pages"][0]["music"], clips[0]["name"]);
        assert_eq!(nodes[0]["pages"][1]["music"], clips[0]["name"]);

        // The one-shot lands on its page only.
        assert_eq!(nodes[0]["pages"][2]["sounds"][0], clips[1]["name"]);
        assert_eq!(nodes[0]["pages"][1]["sounds"].as_array().unwrap().len(), 0);

        // The choice menu carries the tense track.
        assert_eq!(nodes[1]["choice_music"], clips[2]["name"]);
    }

    #[test]
    fn bg_directive_parses_propagates_and_emits_textured_backdrops() {
        let src = "---\ntitle: T\n---\n\n# inn\n\n![bg](assets/inn.png)\n\nFirst.\n\nSecond.\n\n# out\n\n![bg](assets/street.png)\n\n- [Stay](#inn)\n";
        let story = parse_story(src).unwrap();
        // The backdrop persists across the pages after its directive.
        assert_eq!(
            story.nodes[0].pages[0].stage.bg.as_deref(),
            Some("assets/inn.png")
        );
        assert_eq!(
            story.nodes[0].pages[1].stage.bg.as_deref(),
            Some("assets/inn.png")
        );
        // A pages-free choice node carries the current backdrop.
        assert_eq!(
            story.nodes[1].choice_stage.bg.as_deref(),
            Some("assets/street.png")
        );

        let entries = emit_story("s", &story, true, 45.0, &stub_dims).unwrap();
        // Two distinct images -> two Texture entries.
        let textures: Vec<&serde_json::Value> = entries
            .iter()
            .filter(|e| type_norm(e) == "texture")
            .collect();
        assert_eq!(textures.len(), 2);
        assert_eq!(textures[0]["args"]["source"], "assets/inn.png");

        // Both pages of the node share the deduplicated texture in the
        // compiled graph; a full-canvas rectangle places the backdrop.
        let nodes = &find(&entries, "s")["args"]["nodes"];
        let bg = &nodes[0]["pages"][0]["stage"]["bg"];
        assert_eq!(bg["texture"], textures[0]["name"]);
        assert_eq!(bg["width"], 1280.0);
        assert_eq!(
            nodes[0]["pages"][1]["stage"]["bg"]["texture"],
            textures[0]["name"]
        );
        assert_eq!(
            nodes[1]["choice_stage"]["bg"]["texture"],
            textures[1]["name"]
        );

        // The title screen keeps its flat fill (no texture key at all).
        assert!(
            find(&entries, "s_title_bg")["args"]
                .get("texture")
                .is_none()
        );
    }

    #[test]
    fn stacked_directives_in_one_paragraph_all_apply() {
        // Adjacent directive lines form one Markdown paragraph; all apply.
        let src =
            "---\ntitle: T\n---\n\n# a\n\n![bg](x.png)\n[music](m.ogg)\n[sound](s.wav)\n\nhi\n";
        let story = parse_story(src).unwrap();
        let page = &story.nodes[0].pages[0];
        assert_eq!(page.stage.bg.as_deref(), Some("x.png"));
        assert_eq!(page.music.as_deref(), Some("m.ogg"));
        assert_eq!(page.sounds, ["s.wav"]);
    }

    #[test]
    fn jump_link_mixed_with_directives_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n![bg](x.png)\n[go](#a)\n\nhi\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("cannot share"), "{err}");
    }

    #[test]
    fn portraits_persist_and_bg_change_clears_them() {
        let src = "---\ntitle: T\n---\n\n# a\n\n![bg](room.png)\n![left](ana.png)\n\nOne.\n\n![right](ben.png)\n![center](cho.png)\n\nTwo.\n\n![bg](street.png)\n\nThree.\n";
        let story = parse_story(src).unwrap();
        let pages = &story.nodes[0].pages;
        // Page one: left portrait only.
        assert_eq!(pages[0].stage.left.as_deref(), Some("ana.png"));
        assert_eq!(pages[0].stage.center, None);
        assert_eq!(pages[0].stage.right, None);
        // Page two: the right and center portraits join; the left persists.
        assert_eq!(pages[1].stage.left.as_deref(), Some("ana.png"));
        assert_eq!(pages[1].stage.center.as_deref(), Some("cho.png"));
        assert_eq!(pages[1].stage.right.as_deref(), Some("ben.png"));
        // Page three: the scene change cleared every portrait.
        assert_eq!(pages[2].stage.bg.as_deref(), Some("street.png"));
        assert_eq!(pages[2].stage.left, None);
        assert_eq!(pages[2].stage.center, None);
        assert_eq!(pages[2].stage.right, None);
    }

    #[test]
    fn portraits_compile_at_native_size_and_bottom_anchor() {
        let src = "---\ntitle: T\n---\n\n# a\n\n![left](ana.png)\n\nhi\n";
        let story = parse_story(src).unwrap();
        // stub_dims reports 456x700: placed at native pixel size against the
        // 720 canvas, bottom-anchored; the cover-fit stage sprites put the
        // canvas bottom at the window bottom.
        let entries = emit_story("s", &story, true, 45.0, &stub_dims).unwrap();
        let p = &find(&entries, "s")["args"]["nodes"][0]["pages"][0]["stage"]["left"];
        assert_eq!(p["width"], 456.0);
        assert_eq!(p["height"], 700.0);
        assert_eq!(p["y"], 20.0);
        let x = p["x"].as_f64().unwrap() as f32;
        assert!((x - (320.0 - 456.0 / 2.0)).abs() < 1e-3);
        // The portrait image becomes a Texture entry like a backdrop.
        assert_eq!(p["texture"], find(&entries, "s_img0")["name"]);
    }

    #[test]
    fn oversized_portrait_scales_down_to_the_canvas_height() {
        let src = "---\ntitle: T\n---\n\n# a\n\n![center](big.png)\n\nhi\n";
        let story = parse_story(src).unwrap();
        let dims = |_: &str| Ok((900u32, 1440u32));
        let entries = emit_story("s", &story, true, 45.0, &dims).unwrap();
        let p = &find(&entries, "s")["args"]["nodes"][0]["pages"][0]["stage"]["center"];
        // Taller than the canvas: clamped to 720 with the width following.
        assert_eq!(p["height"], 720.0);
        assert_eq!(p["width"], 450.0);
        assert_eq!(p["y"], 0.0);
        // Centered on the canvas.
        let x = p["x"].as_f64().unwrap() as f32;
        assert!((x - (640.0 - 450.0 / 2.0)).abs() < 1e-3);
    }

    #[test]
    fn unknown_image_role_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n![portrait](x.png)\n\nhi\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("'portrait'"), "{err}");
        assert!(err.contains("![left]"), "{err}");
    }

    #[test]
    fn non_image_bg_target_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n![bg](x.gif)\n\nhi\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("png/jpg/jpeg"), "{err}");
    }

    #[test]
    fn image_mixed_with_text_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\nlook: ![bg](x.png)\n\nhi\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("stand alone"), "{err}");
    }

    #[test]
    fn trailing_bg_directive_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\nhi\n\n![bg](x.png)\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("needs a following"), "{err}");
    }

    #[test]
    fn trailing_media_directive_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\nhi\n\n[sound](x.wav)\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("needs a following"), "{err}");
    }

    #[test]
    fn bad_audio_label_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n[loop](x.ogg)\n\nhi\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("`music`"), "{err}");
        assert!(err.contains("loop"), "{err}");
    }

    #[test]
    fn non_audio_link_target_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n[music](x.txt)\n\nhi\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("neither"), "{err}");
    }

    #[test]
    fn choice_to_audio_file_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n- [music](x.ogg)\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("must link to a `#heading`"), "{err}");
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
    fn heading_names_cannot_collide_with_generated_views() {
        // Node names no longer mint views (the whole story plays inside one
        // stage view), so headings named after the scaffolding are fine.
        let src = "---\ntitle: T\n---\n\n# title\n\nhi\n\n# stage\n\nbye\n\n# ending\n\nfin\n";
        let story = parse_story(src).unwrap();
        let entries = emit_story("s", &story, true, 45.0, &stub_dims).unwrap();
        let nodes = &find(&entries, "s")["args"]["nodes"];
        assert_eq!(nodes.as_array().unwrap().len(), 3);
        // Exactly the three scaffolding views exist.
        let views: Vec<String> = entries
            .iter()
            .filter(|e| type_norm(e) == "view")
            .map(asset_name)
            .collect();
        assert_eq!(views, ["s_title", "s_stage", "s_ending"]);
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
