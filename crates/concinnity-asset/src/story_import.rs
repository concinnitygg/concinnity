// Markdown story-import schema.

use alloc::string::String;

/// Imports a Markdown story file as a single declaration.
///
/// One `StoryImport` stands in for a whole branching, click-through story (a
/// visual-novel flow). The build parses the Markdown and expands the import
/// into the UI assets that play it: a [Screen](#screen) per page with a backdrop
/// [Sprite](#sprite), [TextLabel](#textlabel)s for narration and speaker
/// names, and [HitRegion](#hitregion)s wiring page to page, so `world.jsonl`
/// stays a single readable line while the story lives in the Markdown file.
///
/// The `source` file is CommonMark Markdown opening with a YAML frontmatter
/// block:
///
/// - frontmatter declares the story `title`, its `characters`, and an
///   optional `background` (a full-bleed image drawn behind the title menu,
///   dimmed so the light menu text stays readable over a bright photo)
/// - each `# heading` starts a node (a jump target)
/// - each paragraph is one click-through page of narration
/// - a paragraph opening `**id:**` attributes the line to a declared
///   character, shown as a name plate in that character's color
/// - a bullet list of links is a choice menu; each link targets a heading
///   (`[Into the wood](#the-wood)`)
/// - a paragraph that is a single link shows its label and jumps to its
///   target when clicked
/// - a lone link to an audio file is a media directive: `music` loops from
///   the next page onward until replaced, `sound` plays once when the next
///   page shows (`[music](assets/theme.ogg)`, `[sound](assets/door.wav)`);
///   these expand to [AudioClip](#audioclip) + [AudioCue](#audiocue) entries
/// - `![bg](assets/inn.png)` sets the backdrop image from the next page
///   onward; `![left](ana.png)` / `![center](mid.png)` / `![right](ben.png)`
///   place a character portrait at that stage position, bottom-anchored at
///   the image's own pixel size (scaled down if taller than the canvas).
///   Portraits persist until replaced; a `![bg]` change is a scene change
///   and clears them all. Images expand to [Texture](#texture) entries drawn
///   by [Sprite](#sprite)s. Directives may stack on adjacent lines in one
///   paragraph
/// - a node whose last page has no link falls through to the next heading
///   in document order; the final node ends the story
/// - a ```` ```story ```` fenced block scripts state. All story state is
///   named integer variables starting at `0` each playthrough (a flag is a
///   variable holding `1`): `set <var>` assigns 1, `clear <var>` assigns 0,
///   `set <var> = <int>` assigns, and `add <var> <int>` adds (negatives
///   subtract). Operations run when the next page (or choice menu) shows.
///   `if <condition> -> #node` jumps there instead of showing it, where a
///   condition is `<var>` (not zero), `not <var>` (zero), or a comparison
///   `<var> <op> <int>` with `<op>` one of `==` `!=` `<` `<=` `>` `>=`
/// - a choice link's quoted title gates the option with the same condition
///   grammar: `- [Ask her](#ask "if asked")`, `- [Buy](#shop "if gold >= 3")`
///
/// The stage's dialog box carries a quick row of reader controls: Log (a
/// dialogue-history overlay), Auto (pages turn on their own once revealed),
/// Skip (instant reveal and rapid turns, stopping at menus), and Save
/// (numbered save slots). A pulsing marker shows when a fully revealed page
/// waits for a click.
///
/// Play position and variables auto-save page by page (under the project's
/// `.concinnity/data/`); the generated title screen's Continue resumes
/// them, and finishing the story clears the auto-save. Save writes one of
/// three numbered slots, resumed from the title screen's Load. The title
/// menu keeps only the buttons that apply, laid out contiguously: Continue
/// appears once an auto-save exists and Load once a slot does, so a fresh
/// project shows just Start and Quit with no gap.
///
/// Under the editor's debug run, saving the `source` file hot-reloads the
/// story: the graph re-compiles and swaps into the running game in place,
/// keeping the current position (matched by heading). New image or audio
/// files still need a restart.
///
/// Any other Markdown construct (tables, other code fences, inline
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
/// ```rust
/// # use concinnity_asset::StoryImport;
/// StoryImport {
///     source: "assets/crossroads.md".into(),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StoryImport {
    /// Path to the Markdown story file, relative to the project root.
    pub source: String,
    /// Whether to generate a title screen (story title, Start and Quit
    /// buttons) as the initial screen. When `false`, the story's first page is
    /// the initial screen and the generated ending offers a Restart instead of
    /// Back to title.
    pub title_screen: bool,
    /// Dialogue reveal speed in characters per second. `0` shows each page
    /// instantly.
    pub text_speed: f32,
}

impl Default for StoryImport {
    fn default() -> Self {
        Self {
            source: String::new(),
            title_screen: true,
            text_speed: 45.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_import_scaffolds_a_title_screen_at_the_story_default_speed() {
        let s = StoryImport::default();
        assert!(s.title_screen);
        // The same speed a Story defaults to, so the import adds no drift.
        assert_eq!(s.text_speed, 45.0);
        assert!(s.source.is_empty());
    }

    #[test]
    fn an_authored_import_parses_and_round_trips_through_postcard() {
        let s: StoryImport = serde_json::from_str(
            r#"{"source":"stories/ash.md","title_screen":false,"text_speed":80}"#,
        )
        .unwrap();
        assert_eq!(s.source, "stories/ash.md");
        assert!(!s.title_screen);

        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: StoryImport = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.source, "stories/ash.md");
        assert_eq!(back.text_speed, 80.0);
        assert!(!back.title_screen);
    }
}
