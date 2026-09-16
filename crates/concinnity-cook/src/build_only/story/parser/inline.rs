use std::ops::Range;

use super::StoryParser;
use crate::build_only::story::script::parse_condition;

impl StoryParser {
    // Targets are classified when the enclosing construct closes: `#heading`
    // jumps/choices, audio-file media directives, or image directives.
    pub(super) fn on_link_start(
        &mut self,
        target: &str,
        title: &str,
        range: &Range<usize>,
    ) -> Result<(), String> {
        if self.inline.image.is_some() {
            return self.err(
                range,
                "links are not supported in image alt text".to_string(),
            );
        }
        self.inline.link = Some((target.to_string(), title.to_string(), String::new()));
        Ok(())
    }

    pub(super) fn on_link_end(&mut self, range: &Range<usize>) -> Result<(), String> {
        let (target, title, label) = self.inline.link.take().unwrap_or_default();
        if label.trim().is_empty() {
            return self.err(range, format!("link to '{}' has no label text", target));
        }
        if self.list.active {
            let Some(anchor) = target.strip_prefix('#') else {
                return self.err(
                    range,
                    format!("choice '{}' must link to a `#heading`", label.trim()),
                );
            };
            // The optional link title gates the option: `- [Ask](#ask "if asked")`.
            let condition = if title.trim().is_empty() {
                None
            } else {
                match parse_condition(title.trim()) {
                    Ok(c) => Some(c),
                    Err(e) => return self.err(range, e),
                }
            };
            self.list
                .item_links
                .push((anchor.to_string(), label.trim().to_string(), condition));
        } else if let Some(acc) = self.inline.para.as_mut() {
            if !title.trim().is_empty() {
                return self.err(
                    range,
                    "link titles (conditions) are only supported on choices".to_string(),
                );
            }
            acc.links.push((label.trim().to_string(), target));
        } else {
            return self.err(
                range,
                "links may only appear in paragraphs or choice lists".to_string(),
            );
        }
        Ok(())
    }

    pub(super) fn on_strong_start(&mut self, range: &Range<usize>) -> Result<(), String> {
        let misplaced = match self.inline.para.as_ref() {
            Some(acc) => acc.speaker.is_some() || acc.has_plain_text || !acc.links.is_empty(),
            None => true,
        };
        if misplaced {
            return self.err(
                range,
                "bold is reserved for speaker attribution at the start of a \
                 paragraph (`**id:** text`)"
                    .to_string(),
            );
        }
        self.inline.strong = Some(String::new());
        Ok(())
    }

    pub(super) fn on_strong_end(&mut self, range: &Range<usize>) -> Result<(), String> {
        let text = self.inline.strong.take().unwrap_or_default();
        let Some(id) = text.trim().strip_suffix(':') else {
            return self.err(
                range,
                format!(
                    "bold '{}' must be a speaker attribution ending in ':'",
                    text.trim()
                ),
            );
        };
        let id = id.trim().to_string();
        if !self.story.characters.contains_key(&id) {
            return self.err(
                range,
                format!(
                    "speaker '{}' is not declared in the frontmatter `characters`",
                    id
                ),
            );
        }
        self.inline
            .para
            .as_mut()
            .expect("strong start checked the paragraph")
            .speaker = Some(id);
        Ok(())
    }

    // Text goes to the innermost open accumulator.
    pub(super) fn on_text(&mut self, text: &str) {
        let inline = &mut self.inline;
        if let Some(script) = inline.script.as_mut() {
            script.push_str(text);
        } else if let Some(meta) = inline.meta.as_mut() {
            meta.push_str(text);
        } else if let Some(s) = inline.strong.as_mut() {
            s.push_str(text);
        } else if let Some((_, alt)) = inline.image.as_mut() {
            alt.push_str(text);
        } else if let Some((_, _, label)) = inline.link.as_mut() {
            label.push_str(text);
        } else if let Some(h) = inline.heading.as_mut() {
            h.push_str(text);
        } else if self.list.active {
            if !text.trim().is_empty() {
                self.list.item_has_text = true;
            }
        } else if let Some(acc) = inline.para.as_mut() {
            if !text.trim().is_empty() {
                acc.has_plain_text = true;
            }
            acc.text.push_str(text);
        }
    }

    // A soft break joins paragraph lines with a space, a hard break keeps
    // the newline.
    pub(super) fn on_break(&mut self, joiner: char) {
        if let Some(acc) = self.inline.para.as_mut() {
            acc.text.push(joiner);
        }
    }

    pub(super) fn on_image_start(
        &mut self,
        target: &str,
        range: &Range<usize>,
    ) -> Result<(), String> {
        if self.inline.para.is_none() || self.list.active {
            return self.err(
                range,
                "images may only appear alone in their own paragraph".to_string(),
            );
        }
        self.inline.image = Some((target.to_string(), String::new()));
        Ok(())
    }

    pub(super) fn on_image_end(&mut self) {
        let (target, alt) = self.inline.image.take().unwrap_or_default();
        self.inline
            .para
            .as_mut()
            .expect("image start checked the paragraph")
            .images
            .push((alt.trim().to_string(), target));
    }
}
