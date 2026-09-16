use pulldown_cmark::{CodeBlockKind, HeadingLevel};
use std::ops::Range;

use super::StoryParser;
use crate::build_only::story::frontmatter::parse_frontmatter;
use crate::build_only::story::helpers::slug;
use crate::build_only::story::model::{Choice, Node, ParaAcc, ParaOut};
use crate::build_only::story::parse::{classify_paragraph, finish_node};

impl StoryParser {
    pub(super) fn on_metadata_start(&mut self) {
        self.inline.meta = Some(String::new());
    }

    pub(super) fn on_metadata_end(&mut self) -> Result<(), String> {
        let text = self.inline.meta.take().unwrap_or_default();
        parse_frontmatter(&text, &mut self.story)
    }

    pub(super) fn on_heading_start(
        &mut self,
        level: HeadingLevel,
        range: &Range<usize>,
    ) -> Result<(), String> {
        if level != HeadingLevel::H1 {
            return self.err(
                range,
                format!(
                    "'{}' heading: only `#` headings (nodes) are supported",
                    "#".repeat(level as usize)
                ),
            );
        }
        self.inline.heading = Some(String::new());
        Ok(())
    }

    pub(super) fn on_heading_end(&mut self, range: &Range<usize>) -> Result<(), String> {
        let text = self.inline.heading.take().unwrap_or_default();
        let s = slug(&text);
        if s.is_empty() {
            return self.err(
                range,
                format!("heading '{}' produces an empty anchor", text),
            );
        }
        if let Some(node) = self.cur_node.take() {
            let line = self.line_of(range);
            finish_node(node, &mut self.story, line)?;
        }
        self.cur_node = Some(Node {
            slug: s,
            heading: text,
            ..Node::default()
        });
        Ok(())
    }

    pub(super) fn on_paragraph_start(&mut self, range: &Range<usize>) -> Result<(), String> {
        if self.inline.meta.is_some() || self.list.active {
            return Ok(());
        }
        self.require_open_node(range)?;
        self.inline.para = Some(ParaAcc::default());
        Ok(())
    }

    pub(super) fn on_paragraph_end(&mut self, range: &Range<usize>) -> Result<(), String> {
        if self.list.active {
            return Ok(());
        }
        let Some(acc) = self.inline.para.take() else {
            return Ok(());
        };
        match classify_paragraph(acc, self.line_of(range))? {
            ParaOut::Page(mut page) => {
                let attachment = self.pending.take_pending();
                page.music = attachment.music;
                page.stage = attachment.stage;
                page.sounds = attachment.sounds;
                page.ops = attachment.ops;
                page.gates = attachment.gates;
                self.cur_node
                    .as_mut()
                    .expect("paragraph start checked the node")
                    .pages
                    .push(*page);
            }
            ParaOut::Directives(directives) => {
                for directive in directives {
                    self.pending.apply_directive(directive);
                }
                self.pending.unconsumed_directive = Some(self.line_of(range));
            }
        }
        Ok(())
    }

    pub(super) fn on_list_start(
        &mut self,
        ordered: Option<u64>,
        range: &Range<usize>,
    ) -> Result<(), String> {
        if ordered.is_some() {
            return self.err(
                range,
                "choices must be a bullet list, not numbered".to_string(),
            );
        }
        self.require_open_node(range)?;
        if self.list.active {
            return self.err(range, "nested lists are not supported".to_string());
        }
        self.list.active = true;
        Ok(())
    }

    // The choice menu is shown like a page, so it carries the current music
    // and consumes any queued one-shots.
    pub(super) fn on_list_end(&mut self) {
        self.list.active = false;
        let attachment = self.pending.take_pending();
        let node = self.cur_node.as_mut().expect("list start checked the node");
        node.choice_music = attachment.music;
        node.choice_stage = attachment.stage;
        node.choice_sounds = attachment.sounds;
        node.choice_ops = attachment.ops;
        node.choice_gates = attachment.gates;
    }

    pub(super) fn on_item_start(&mut self) {
        self.list.item_links.clear();
        self.list.item_has_text = false;
    }

    pub(super) fn on_item_end(&mut self, range: &Range<usize>) -> Result<(), String> {
        if self.list.item_links.len() != 1 || self.list.item_has_text {
            return self.err(
                range,
                "each choice must be exactly one link, e.g. `- [Go](#node)`".to_string(),
            );
        }
        let (target, label, condition) = self.list.item_links.pop().expect("length checked");
        self.cur_node
            .as_mut()
            .expect("list start checked the node")
            .choices
            .push(Choice {
                label,
                target,
                condition,
            });
        Ok(())
    }

    pub(super) fn on_code_block_start(
        &mut self,
        kind: &CodeBlockKind<'_>,
        range: &Range<usize>,
    ) -> Result<(), String> {
        let lang = match kind {
            CodeBlockKind::Fenced(lang) => lang.to_string(),
            CodeBlockKind::Indented => {
                return self.err(range, "indented code blocks are not supported".to_string());
            }
        };
        if lang != "story" {
            return self.err(
                range,
                format!(
                    "code fence '{}' is not supported; script blocks use ```story",
                    lang
                ),
            );
        }
        self.require_open_node(range)?;
        self.inline.script = Some(String::new());
        Ok(())
    }

    pub(super) fn on_code_block_end(&mut self, range: &Range<usize>) -> Result<(), String> {
        let text = self.inline.script.take().unwrap_or_default();
        let line = self.line_of(range);
        self.pending.queue_script(&text, line)
    }
}
