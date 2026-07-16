use std::collections::HashSet;
use std::ops::Range;

use pulldown_cmark::{Event, HeadingLevel, MetadataBlockKind, Options, Parser, Tag, TagEnd};

use super::helpers::{
    AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, file_extension, parse_flag, parse_int, slug,
    split_top_level, unquote,
};
use super::model::{
    BlockCharacter, Character, Choice, Condition, Directive, FlagOp, Gate, Node, Page, ParaAcc,
    ParaOut, ScriptLine, Stage, Story,
};

// One script line: `set <var>`, `clear <var>`, `set <var> = <int>`,
// `add <var> <int>`, or a conditional jump `if <condition> -> #anchor`.
fn parse_script_line(line: &str) -> Result<ScriptLine, String> {
    if let Some(rest) = line.strip_prefix("set ") {
        let rest = rest.trim();
        return Ok(ScriptLine::Op(match rest.split_once('=') {
            Some((name, value)) => FlagOp {
                name: parse_flag(name.trim())?,
                value: parse_int(value.trim())?,
                add: false,
            },
            None => FlagOp {
                name: parse_flag(rest)?,
                value: 1,
                add: false,
            },
        }));
    }
    if let Some(rest) = line.strip_prefix("clear ") {
        return Ok(ScriptLine::Op(FlagOp {
            name: parse_flag(rest.trim())?,
            value: 0,
            add: false,
        }));
    }
    if let Some(rest) = line.strip_prefix("add ") {
        let rest = rest.trim();
        let (name, amount) = rest
            .split_once(char::is_whitespace)
            .ok_or_else(|| format!("script line 'add {}' is missing the amount to add", rest))?;
        return Ok(ScriptLine::Op(FlagOp {
            name: parse_flag(name.trim())?,
            value: parse_int(amount.trim())?,
            add: true,
        }));
    }
    if let Some(rest) = line.strip_prefix("if ") {
        let (condition, target) = rest.split_once("->").ok_or_else(|| {
            format!(
                "script line '{}' is missing the `-> #anchor` jump target",
                line
            )
        })?;
        let target = target.trim();
        let Some(anchor) = target.strip_prefix('#') else {
            return Err(format!(
                "script jump target '{}' must be a `#heading` anchor",
                target
            ));
        };
        return Ok(ScriptLine::Gate(Gate {
            condition: parse_bare_condition(condition.trim())?,
            target: anchor.to_string(),
        }));
    }
    Err(format!(
        "script line '{}' is not `set <var> [= <int>]`, `clear <var>`, \
         `add <var> <int>`, or `if <condition> -> #anchor`",
        line
    ))
}

// The condition body after `if `: `<var>`, `not <var>`, or
// `<var> <op> <int>` with op one of == != < <= > >=. Compiled to a
// comparison against an integer (an unset variable reads as 0, so a bare
// flag test is `!= 0` and its negation `== 0`).
fn parse_bare_condition(condition: &str) -> Result<Condition, String> {
    if let Some(rest) = condition.strip_prefix("not ") {
        return Ok(Condition {
            name: parse_flag(rest.trim())?,
            op: "eq",
            value: 0,
        });
    }
    for (symbol, op) in [
        ("==", "eq"),
        ("!=", "ne"),
        ("<=", "le"),
        (">=", "ge"),
        ("<", "lt"),
        (">", "gt"),
    ] {
        if let Some((name, value)) = condition.split_once(symbol) {
            return Ok(Condition {
                name: parse_flag(name.trim())?,
                op,
                value: parse_int(value.trim())?,
            });
        }
    }
    Ok(Condition {
        name: parse_flag(condition)?,
        op: "ne",
        value: 0,
    })
}

// A choice condition from a link title: `if <condition>`.
fn parse_condition(title: &str) -> Result<Condition, String> {
    let Some(rest) = title.strip_prefix("if ") else {
        return Err(format!(
            "choice condition '{}' must be `if <var>`, `if not <var>`, or \
             `if <var> <op> <int>`",
            title
        ));
    };
    parse_bare_condition(rest.trim())
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
    let mut link: Option<(String, String, String)> = None; // (target, title, label so far)
    let mut image: Option<(String, String)> = None; // (target, alt so far)
    let mut in_list = false;
    let mut item_links: Vec<(String, String, Option<Condition>)> = Vec::new();
    let mut item_has_text = false;
    // A ```story script fence being accumulated; parsed into ops and gates
    // when it closes.
    let mut script_text: Option<String> = None;
    // Media state in document order: the music current from the most recent
    // `[music]` directive, and one-shot `[sound]`s waiting for the next page
    // or choice list. A directive with nothing after it to attach to is dead
    // and rejected at end of parse.
    let mut current_music: Option<String> = None;
    let mut current_stage = Stage::default();
    let mut pending_sounds: Vec<String> = Vec::new();
    // Script state waiting for the next page or choice menu, like the
    // one-shot sounds above.
    let mut pending_ops: Vec<FlagOp> = Vec::new();
    let mut pending_gates: Vec<Gate> = Vec::new();
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
                            page.ops = std::mem::take(&mut pending_ops);
                            page.gates = std::mem::take(&mut pending_gates);
                            unconsumed_directive = None;
                            cur_node
                                .as_mut()
                                .expect("paragraph start checked the node")
                                .pages
                                .push(*page);
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
                node.choice_ops = std::mem::take(&mut pending_ops);
                node.choice_gates = std::mem::take(&mut pending_gates);
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
                let (target, label, condition) = item_links.pop().expect("length checked");
                cur_node
                    .as_mut()
                    .expect("list start checked the node")
                    .choices
                    .push(Choice {
                        label,
                        target,
                        condition,
                    });
            }

            Event::Start(Tag::Link {
                dest_url, title, ..
            }) => {
                if image.is_some() {
                    return err(
                        &range,
                        "links are not supported in image alt text".to_string(),
                    );
                }
                // Targets are classified when the enclosing construct closes:
                // `#heading` jumps/choices, audio-file media directives, or
                // image directives.
                link = Some((dest_url.to_string(), title.to_string(), String::new()));
            }
            Event::End(TagEnd::Link) => {
                let (target, title, label) = link.take().unwrap_or_default();
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
                    // The optional link title gates the option:
                    // `- [Ask](#ask "if asked")`.
                    let condition = if title.trim().is_empty() {
                        None
                    } else {
                        match parse_condition(title.trim()) {
                            Ok(c) => Some(c),
                            Err(e) => return err(&range, e),
                        }
                    };
                    item_links.push((anchor.to_string(), label.trim().to_string(), condition));
                } else if let Some(acc) = para.as_mut() {
                    if !title.trim().is_empty() {
                        return err(
                            &range,
                            "link titles (conditions) are only supported on choices".to_string(),
                        );
                    }
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
                if let Some(script) = script_text.as_mut() {
                    script.push_str(&t);
                } else if let Some(meta) = meta_text.as_mut() {
                    meta.push_str(&t);
                } else if let Some(s) = strong_text.as_mut() {
                    s.push_str(&t);
                } else if let Some((_, alt)) = image.as_mut() {
                    alt.push_str(&t);
                } else if let Some((_, _, label)) = link.as_mut() {
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
            Event::Start(Tag::CodeBlock(kind)) => {
                let lang = match &kind {
                    pulldown_cmark::CodeBlockKind::Fenced(lang) => lang.to_string(),
                    pulldown_cmark::CodeBlockKind::Indented => {
                        return err(&range, "indented code blocks are not supported".to_string());
                    }
                };
                if lang != "story" {
                    return err(
                        &range,
                        format!(
                            "code fence '{}' is not supported; script blocks use ```story",
                            lang
                        ),
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
                script_text = Some(String::new());
            }
            Event::End(TagEnd::CodeBlock) => {
                let text = script_text.take().unwrap_or_default();
                for raw in text.lines() {
                    let trimmed = raw.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match parse_script_line(trimmed) {
                        Ok(ScriptLine::Op(op)) => pending_ops.push(op),
                        Ok(ScriptLine::Gate(gate)) => pending_gates.push(gate),
                        Err(e) => return err(&range, e),
                    }
                }
                unconsumed_directive = Some(line_of(&range));
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
        return Ok(ParaOut::Page(Box::new(Page {
            text: label,
            jump: Some(target[1..].to_string()),
            ..Page::default()
        })));
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
            Ok(ParaOut::Page(Box::new(Page {
                speaker: acc.speaker,
                text,
                ..Page::default()
            })))
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
            .chain(
                node.pages
                    .iter()
                    .flat_map(|p| p.gates.iter())
                    .chain(node.choice_gates.iter())
                    .map(|g| g.target.as_str()),
            )
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

// Frontmatter

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
                "background" => story.background = Some(unquote(value.trim()).to_string()),
                "characters" => {
                    if !value.trim().is_empty() {
                        return err("`characters` takes an indented block".to_string());
                    }
                    in_characters = true;
                }
                other => {
                    return err(format!(
                        "unknown key '{}'; supported keys are `title`, `background`, and \
                         `characters`",
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

#[cfg(test)]
mod tests {
    use super::*;

    // Wrap a frontmatter block in a minimal valid body. Frontmatter errors
    // surface before the body is validated, so a mis-set-up test still fails
    // loudly rather than passing on the wrong error.
    fn with_body(frontmatter: &str) -> String {
        format!("---\n{frontmatter}\n---\n\n# a\n\nhi\n")
    }

    // Script sub-parsers reached directly.

    #[test]
    fn script_if_line_needs_a_jump_target() {
        // `.err()` avoids requiring the Ok type (ScriptLine) to be Debug.
        let err = parse_script_line("if asked").err().unwrap();
        assert!(err.contains("jump target"), "{err}");
    }

    #[test]
    fn script_jump_target_must_be_an_anchor() {
        let err = parse_script_line("if asked -> b").err().unwrap();
        assert!(err.contains("anchor"), "{err}");
    }

    #[test]
    fn bare_condition_rejects_a_bad_flag() {
        assert!(
            parse_bare_condition("not BAD")
                .unwrap_err()
                .contains("not a variable name")
        );
        assert!(
            parse_bare_condition("BAD")
                .unwrap_err()
                .contains("not a variable name")
        );
    }

    #[test]
    fn choice_condition_requires_an_if_prefix() {
        let err = parse_condition("gold >= 3").unwrap_err();
        assert!(err.contains("choice condition"), "{err}");
        let ok = parse_condition("if gold >= 3").unwrap();
        assert_eq!((ok.name.as_str(), ok.op, ok.value), ("gold", "ge", 3));
    }

    // Frontmatter scalar-value parsers reached directly.

    #[test]
    fn name_value_parses_quoted_and_plain() {
        assert_eq!(parse_name_value("\"Ayame Doe\"").unwrap(), "Ayame Doe");
        assert_eq!(parse_name_value("Plain Name").unwrap(), "Plain Name");
    }

    #[test]
    fn name_value_rejects_unbalanced_quotes_and_empty() {
        assert!(
            parse_name_value("\"unterminated")
                .unwrap_err()
                .contains("unbalanced quotes")
        );
        assert!(
            parse_name_value("\"\"")
                .unwrap_err()
                .contains("must not be empty")
        );
    }

    #[test]
    fn color_value_parses_a_triple() {
        assert_eq!(
            parse_color_value("[1.0, 0.5, 0.25]").unwrap(),
            [1.0, 0.5, 0.25]
        );
    }

    #[test]
    fn color_value_rejects_non_arrays_and_wrong_lengths() {
        // A non-array JSON value trips the serde branch; a well-formed array
        // of the wrong length trips the component-count check.
        assert!(parse_color_value("nope").unwrap_err().contains("[r, g, b]"));
        assert!(
            parse_color_value("[1, 2]")
                .unwrap_err()
                .contains("3 components")
        );
    }

    #[test]
    fn inline_character_parses_and_reports_its_own_errors() {
        let ok = parse_character("{ name: \"Bo\", color: [1, 1, 1] }").unwrap();
        assert_eq!(ok.name, "Bo");
        let plain = parse_character("Bob").unwrap();
        assert_eq!((plain.name.as_str(), plain.color), ("Bob", [1.0, 1.0, 1.0]));

        assert!(
            parse_character("{ name: X")
                .unwrap_err()
                .contains("unterminated")
        );
        assert!(
            parse_character("{ name: X, bad }")
                .unwrap_err()
                .contains("key: value")
        );
        assert!(
            parse_character("{ name: X, voice: low }")
                .unwrap_err()
                .contains("unknown key")
        );
        assert!(
            parse_character("{ color: [1, 1, 1] }")
                .unwrap_err()
                .contains("missing `name`")
        );
        assert!(
            parse_character("\"\"")
                .unwrap_err()
                .contains("display name")
        );
    }

    // Frontmatter branches via full parses.

    #[test]
    fn frontmatter_unknown_key_is_an_error() {
        let err = parse_story(&with_body("title: T\nfoo: bar")).unwrap_err();
        assert!(err.contains("unknown key 'foo'"), "{err}");
    }

    #[test]
    fn characters_key_with_inline_value_is_an_error() {
        let err = parse_story(&with_body("title: T\ncharacters: stuff")).unwrap_err();
        assert!(err.contains("indented block"), "{err}");
    }

    #[test]
    fn top_level_line_without_a_colon_is_an_error() {
        let err = parse_story(&with_body("title: T\nnocolon")).unwrap_err();
        assert!(err.contains("key: value"), "{err}");
    }

    #[test]
    fn stray_indented_line_outside_characters_is_an_error() {
        let err = parse_story(&with_body("title: T\n  indented")).unwrap_err();
        assert!(err.contains("unexpected indented line"), "{err}");
    }

    #[test]
    fn character_field_without_a_colon_is_an_error() {
        let err = parse_story(&with_body("title: T\ncharacters:\n  a:\n    nocolon")).unwrap_err();
        assert!(err.contains("key: value"), "{err}");
    }

    #[test]
    fn character_id_line_without_a_colon_is_an_error() {
        let err = parse_story(&with_body("title: T\ncharacters:\n  nocolon")).unwrap_err();
        assert!(err.contains("id: ..."), "{err}");
    }

    #[test]
    fn invalid_character_id_is_an_error() {
        let err = parse_story(&with_body("title: T\ncharacters:\n  bad id: Name")).unwrap_err();
        assert!(err.contains("alphanumeric"), "{err}");
    }

    #[test]
    fn inline_character_missing_name_reports_its_id() {
        let err = parse_story(&with_body(
            "title: T\ncharacters:\n  a: { color: [1, 1, 1] }",
        ))
        .unwrap_err();
        assert!(err.contains("missing `name`"), "{err}");
        assert!(err.contains("character 'a'"), "{err}");
    }

    // Body branches via full parses.

    #[test]
    fn heading_with_no_word_characters_is_an_error() {
        let err = parse_story("---\ntitle: T\n---\n\n# !!!\n\nhi\n").unwrap_err();
        assert!(err.contains("empty anchor"), "{err}");
    }

    #[test]
    fn nested_lists_are_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n- [x](#a)\n  - [y](#a)\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("nested lists"), "{err}");
    }

    #[test]
    fn a_choice_must_be_exactly_one_link() {
        let src = "---\ntitle: T\n---\n\n# a\n\n- just words\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("exactly one link"), "{err}");
    }

    #[test]
    fn bold_away_from_the_paragraph_start_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\nhello **keeper:**\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("speaker attribution"), "{err}");
    }

    #[test]
    fn bold_without_a_trailing_colon_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n**keeper** hi\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("ending in ':'"), "{err}");
    }

    #[test]
    fn an_image_inside_a_list_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n- ![bg](x.png)\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("appear alone"), "{err}");
    }

    #[test]
    fn a_link_with_no_label_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n[](#a)\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("no label text"), "{err}");
    }

    #[test]
    fn a_link_in_a_heading_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# [x](#a)\n\nhi\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("only appear in paragraphs"), "{err}");
    }

    #[test]
    fn raw_html_is_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\n<div>hi</div>\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("raw HTML"), "{err}");
    }

    #[test]
    fn thematic_breaks_are_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\nhi\n\n***\n\nbye\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("thematic breaks"), "{err}");
    }

    #[test]
    fn indented_code_blocks_are_an_error() {
        let src = "---\ntitle: T\n---\n\n# a\n\nhi\n\n    indented code\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("indented code"), "{err}");
    }

    #[test]
    fn a_choice_link_title_needs_an_if_prefix() {
        let src = "---\ntitle: T\n---\n\n# a\n\n- [Go](#a \"gold >= 3\")\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("choice condition"), "{err}");
    }

    #[test]
    fn content_before_the_first_heading_in_a_list_is_an_error() {
        let src = "---\ntitle: T\n---\n\n- [x](#a)\n";
        let err = parse_story(src).unwrap_err();
        assert!(err.contains("before the first"), "{err}");
    }

    #[test]
    fn soft_and_hard_breaks_join_paragraph_text() {
        let soft = parse_story("---\ntitle: T\n---\n\n# a\n\nfirst\nsecond\n").unwrap();
        assert_eq!(soft.nodes[0].pages[0].text, "first second");
        let hard = parse_story("---\ntitle: T\n---\n\n# a\n\nfirst\\\nsecond\n").unwrap();
        assert_eq!(hard.nodes[0].pages[0].text, "first\nsecond");
    }
}
