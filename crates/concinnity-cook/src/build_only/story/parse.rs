use pulldown_cmark::{Options, Parser};
use std::collections::HashSet;

use super::helpers::{AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, file_extension};
use super::model::{Directive, Node, Page, ParaAcc, ParaOut, Story};
use super::parser::StoryParser;

// Parse a story file into its node graph, rejecting anything outside the
// dialect. Errors carry a 1-based source line so the author can find the
// offending construct.
pub(crate) fn parse_story(src: &str) -> Result<Story, String> {
    let mut parser = StoryParser::new(src);
    let options = Options::ENABLE_YAML_STYLE_METADATA_BLOCKS;
    for (event, range) in Parser::new_ext(src, options).into_offset_iter() {
        parser.on_event(event, range)?;
    }
    parser.finish()
}

pub(super) fn finish_node(node: Node, story: &mut Story, line: usize) -> Result<(), String> {
    if node.pages.is_empty() && node.choices.is_empty() {
        return Err(format!(
            "line {}: node '{}' is empty; give it a paragraph or choices",
            line, node.heading
        ));
    }
    story.nodes.push(node);
    Ok(())
}

pub(super) fn classify_paragraph(acc: ParaAcc, line: usize) -> Result<ParaOut, String> {
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

pub(super) fn validate_story(story: &Story) -> Result<(), String> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
