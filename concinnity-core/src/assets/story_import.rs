// src/assets/story_import.rs

use crate::ecs::{AssetOrigin, Component};

/// Imports a Markdown story file as a single declaration.
///
/// One `StoryImport` stands in for a whole branching, click-through story (a
/// visual-novel flow). The build parses the Markdown and expands the import
/// into the UI assets that play it: a [View](#view) per page with a backdrop
/// [Sprite](#sprite), [TextLabel](#textlabel)s for narration and speaker
/// names, and [HitRegion](#hitregion)s wiring page to page, so `world.jsonl`
/// stays a single readable line while the story lives in the Markdown file.
///
/// The `source` file is CommonMark Markdown opening with a YAML frontmatter
/// block:
///
/// - frontmatter declares the story `title` and its `characters`
/// - each `# heading` starts a node (a jump target)
/// - each paragraph is one click-through page of narration
/// - a paragraph opening `**id:**` attributes the line to a declared
///   character, shown as a name plate in that character's color
/// - a bullet list of links is a choice menu; each link targets a heading
///   (`[Into the wood](#the-wood)`)
/// - a paragraph that is a single link shows its label and jumps to its
///   target when clicked
/// - a node whose last page has no link falls through to the next heading
///   in document order; the final node ends the story
///
/// Any other Markdown construct (images, tables, code blocks, inline
/// emphasis, ...) is an error at build time, as are links to headings that
/// do not exist, undeclared speakers, and duplicate headings.
///
/// **Generated names** are prefixed with the import's own asset `name`
/// (`<name>_title`, `<name>_<node>_p0`, ...), so they never clash with
/// hand-authored assets.
///
/// Characters take a nested block, a one-line name, or a `{ ... }` flow map
/// (`ayame: { name: Ayame, color: [1.0, 0.85, 0.8] }`).
///
/// ```markdown
/// ---
/// title: The Crossroads
/// characters:
///   ayame:
///     name: Ayame
///     color: [1.0, 0.85, 0.8]
///   keeper: Innkeeper
/// ---
///
/// # inn
///
/// You wake at a roadside inn. A note rests on the pillow.
///
/// **ayame:** You came. I wasn't sure you would.
///
/// - [Into the wood](#wood)
/// - [Toward the shore](#shore)
/// ```
///
/// ```jsonl
/// {"name":"crossroads","type":"StoryImport","args":{"source":"assets/crossroads.md"}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StoryImport {
    /// Path to the Markdown story file, relative to the project root.
    pub source: String,
    /// Whether to generate a title screen (story title, Start and Quit
    /// buttons) as the initial view. When `false`, the story's first page is
    /// the initial view and the generated ending offers a Restart instead of
    /// Back to title.
    pub title_screen: bool,
}

impl Default for StoryImport {
    fn default() -> Self {
        Self {
            source: String::new(),
            title_screen: true,
        }
    }
}

impl Component for StoryImport {
    const NAME: &'static str = "StoryImport";
    const ORIGIN: AssetOrigin = AssetOrigin::BuildOnly;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}
