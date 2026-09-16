// The event-driven state behind `parse_story`: one handler per Markdown event
// family, each reading and writing the parser's own sub-states.

use pulldown_cmark::{Event, MetadataBlockKind, Tag, TagEnd};
use std::ops::Range;

use super::model::{Condition, Node, ParaAcc, Story};
use super::parse::{finish_node, validate_story};

mod block;
mod inline;
mod pending;

use pending::PendingMedia;

pub(super) struct StoryParser {
    story: Story,
    line_starts: Vec<usize>,
    src_len: usize,
    cur_node: Option<Node>,
    inline: InlineText,
    list: ListState,
    pending: PendingMedia,
}

// Text accumulators, each open between its construct's start and end events.
#[derive(Default)]
struct InlineText {
    para: Option<ParaAcc>,
    heading: Option<String>,
    meta: Option<String>,
    strong: Option<String>,
    // (target, title, label so far)
    link: Option<(String, String, String)>,
    // (target, alt so far)
    image: Option<(String, String)>,
    // A ```story script fence, parsed into ops and gates when it closes.
    script: Option<String>,
}

// The open choice list and the links of its current item.
#[derive(Default)]
struct ListState {
    active: bool,
    item_links: Vec<(String, String, Option<Condition>)>,
    item_has_text: bool,
}

impl StoryParser {
    pub(super) fn new(src: &str) -> Self {
        Self {
            story: Story::default(),
            line_starts: std::iter::once(0)
                .chain(src.match_indices('\n').map(|(i, _)| i + 1))
                .collect(),
            src_len: src.len(),
            cur_node: None,
            inline: InlineText::default(),
            list: ListState::default(),
            pending: PendingMedia::default(),
        }
    }

    fn line_of(&self, range: &Range<usize>) -> usize {
        self.line_starts.partition_point(|&s| s <= range.start)
    }

    fn err<T>(&self, range: &Range<usize>, msg: String) -> Result<T, String> {
        Err(format!("line {}: {}", self.line_of(range), msg))
    }

    // Block content needs a node to belong to, and nothing may follow a
    // node's choices.
    fn require_open_node(&self, range: &Range<usize>) -> Result<&Node, String> {
        let Some(node) = self.cur_node.as_ref() else {
            return self.err(range, "content before the first `#` heading".to_string());
        };
        if !node.choices.is_empty() {
            return self.err(
                range,
                format!("node '{}': choices must be its last content", node.heading),
            );
        }
        Ok(node)
    }

    pub(super) fn on_event(&mut self, event: Event<'_>, range: Range<usize>) -> Result<(), String> {
        match event {
            Event::Start(Tag::MetadataBlock(MetadataBlockKind::YamlStyle)) => {
                self.on_metadata_start();
                Ok(())
            }
            Event::End(TagEnd::MetadataBlock(_)) => self.on_metadata_end(),
            Event::Start(Tag::Heading { level, .. }) => self.on_heading_start(level, &range),
            Event::End(TagEnd::Heading(_)) => self.on_heading_end(&range),
            Event::Start(Tag::Paragraph) => self.on_paragraph_start(&range),
            Event::End(TagEnd::Paragraph) => self.on_paragraph_end(&range),
            Event::Start(Tag::List(ordered)) => self.on_list_start(ordered, &range),
            Event::End(TagEnd::List(_)) => {
                self.on_list_end();
                Ok(())
            }
            Event::Start(Tag::Item) => {
                self.on_item_start();
                Ok(())
            }
            Event::End(TagEnd::Item) => self.on_item_end(&range),
            Event::Start(Tag::Link {
                dest_url, title, ..
            }) => self.on_link_start(&dest_url, &title, &range),
            Event::End(TagEnd::Link) => self.on_link_end(&range),
            Event::Start(Tag::Strong) => self.on_strong_start(&range),
            Event::End(TagEnd::Strong) => self.on_strong_end(&range),
            Event::Text(text) => {
                self.on_text(&text);
                Ok(())
            }
            Event::SoftBreak => {
                self.on_break(' ');
                Ok(())
            }
            Event::HardBreak => {
                self.on_break('\n');
                Ok(())
            }
            Event::Start(Tag::Image { dest_url, .. }) => self.on_image_start(&dest_url, &range),
            Event::End(TagEnd::Image) => {
                self.on_image_end();
                Ok(())
            }
            Event::Start(Tag::CodeBlock(kind)) => self.on_code_block_start(&kind, &range),
            Event::End(TagEnd::CodeBlock) => self.on_code_block_end(&range),
            Event::End(_) => Ok(()),
            other => self.reject_unsupported(other, &range),
        }
    }

    fn reject_unsupported(&self, event: Event<'_>, range: &Range<usize>) -> Result<(), String> {
        let msg = match event {
            Event::Code(_) => "inline code is not supported".to_string(),
            Event::Start(Tag::Emphasis) => "emphasis (italics) is not supported".to_string(),
            Event::Start(Tag::BlockQuote(_)) => "block quotes are not supported".to_string(),
            Event::Rule => "thematic breaks (`---`) are not supported in the body".to_string(),
            Event::Html(_) | Event::InlineHtml(_) | Event::Start(Tag::HtmlBlock) => {
                "raw HTML is not supported".to_string()
            }
            other => format!("unsupported Markdown construct: {:?}", other),
        };
        self.err(range, msg)
    }

    pub(super) fn finish(mut self) -> Result<Story, String> {
        if let Some(line) = self.pending.unconsumed_directive {
            return Err(format!(
                "line {}: a media directive needs a following paragraph or choice list to \
                 attach to",
                line
            ));
        }
        if let Some(node) = self.cur_node.take() {
            let line = self.line_of(&(self.src_len..self.src_len));
            finish_node(node, &mut self.story, line)?;
        }

        validate_story(&self.story)?;
        Ok(self.story)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_only::story::model::Choice;

    const SRC: &str = "---\ntitle: T\n---\n\n# a\n";

    #[test]
    fn content_needs_an_open_node() {
        let parser = StoryParser::new(SRC);
        let err = parser.require_open_node(&(18..19)).unwrap_err();
        assert_eq!(err, "line 5: content before the first `#` heading");
    }

    #[test]
    fn content_after_choices_is_refused() {
        let mut parser = StoryParser::new(SRC);
        parser.cur_node = Some(Node {
            slug: "a".to_string(),
            heading: "a".to_string(),
            choices: vec![Choice {
                label: "Go".to_string(),
                target: "a".to_string(),
                condition: None,
            }],
            ..Node::default()
        });
        let err = parser.require_open_node(&(0..1)).unwrap_err();
        assert_eq!(err, "line 1: node 'a': choices must be its last content");
    }

    #[test]
    fn an_open_node_without_choices_accepts_content() {
        let mut parser = StoryParser::new(SRC);
        parser.cur_node = Some(Node {
            heading: "a".to_string(),
            ..Node::default()
        });
        assert_eq!(parser.require_open_node(&(0..1)).unwrap().heading, "a");
    }
}
