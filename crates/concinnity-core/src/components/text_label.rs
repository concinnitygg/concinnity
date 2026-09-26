// Screen-space UI text-label schema.

use crate::components::Screen;
use crate::components::SpriteFit;
use crate::components::Vocabulary;
use crate::ecs::FontHandle;
use crate::ecs::de_opt_font_handle;
use crate::ecs::{Ref, de_opt_ref};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Horizontal alignment of a [TextLabel](#textlabel) relative to its `x`.
///
/// `Center` and `Right` measure the rendered text with the real font metrics
/// each frame, so a label stays visually centered (or right-aligned) at any
/// scale without the author estimating glyph widths.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default, Vocabulary,
)]
#[serde(rename_all = "lowercase")]
pub enum TextAlign {
    /// `x` is the left edge of the text (the default).
    #[default]
    #[vocab("left")]
    Left,
    /// `x` is the horizontal center of the text.
    #[vocab("center")]
    Center,
    /// `x` is the right edge of the text.
    #[vocab("right")]
    Right,
}

/// Screen-space text drawn as a UI overlay on top of the 3D scene each frame.
///
/// Text is laid out using the referenced [Font](#font). A label naming none
/// draws with the engine's built-in face at 24px, which costs the build nothing
/// (the face ships inside the binary) and renders the same whether the world was
/// compiled or assembled in code. The `content` field can be updated every frame
/// (e.g. by an [FpsCounter](#fpscounter)).
///
/// A `\n` in `content` starts a new line. When `background` has an alpha > 0, a
/// box is filled behind the glyphs, extended outward by `padding` pixels,
/// useful for HUD chips.
///
/// Parts of the text can take colors of their own through `color_runs`: a
/// speaker's name in a line of dialogue, the key term in a hint, a warning
/// word in a status line. Every character no run covers draws in `color`.
///
/// ```rust
/// # use concinnity_core::components::TextLabel;
/// TextLabel {
///     content: "FPS: --".into(),
///     x: 10.0,
///     y: 10.0,
///     color: [1.0, 1.0, 1.0],
///     scale: 1.0,
///     ..Default::default()
/// };
/// ```
///
/// A line of dialogue with the speaker's name in gold:
///
/// ```rust
/// # use concinnity_core::components::{ColorRun, TextLabel};
/// TextLabel {
///     content: "Mara: The gate is open.".into(),
///     color_runs: vec![ColorRun {
///         start: 0,
///         length: 4,
///         color: [1.0, 0.8, 0.3],
///     }],
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct TextLabel {
    /// The [Font](#font) asset to use for rendering. Unset draws with the
    /// engine's built-in face at its native 24px.
    #[serde(deserialize_with = "de_opt_font_handle")]
    pub font: Option<FontHandle>,
    /// Text to display. Can be updated each frame.
    pub content: String,
    /// Horizontal position in pixels from the left edge of the window.
    pub x: f32,
    /// Vertical position in pixels from the top edge of the window.
    pub y: f32,
    /// Linear-space RGB text color, for every character no run in
    /// `color_runs` covers.
    pub color: [f32; 3],
    /// Spans of `content` drawn in colors of their own, in order along the
    /// text and never overlapping. Positions count characters, not bytes, and
    /// a `\n` counts as one. A run keeps its characters however the text wraps
    /// or aligns, since only color varies; the build rejects a run reaching
    /// past the end of `content` or starting before the previous one ends.
    pub color_runs: Vec<ColorRun>,
    /// Uniform scale applied on top of the font's `size_px` (24 for the
    /// built-in face). 1.0 = native size. Ignored when `centered` is set, which
    /// sizes the text to the viewport instead.
    pub scale: f32,
    /// When true, fit the label to the viewport and center it there each frame,
    /// so `x`, `y`, `align` and `scale` are all ignored.
    pub centered: bool,
    /// Horizontal alignment relative to `x` (measured with the real font
    /// metrics). Ignored when `centered` is set.
    pub align: TextAlign,
    /// How a screen-owned label maps from the reference canvas to the window when
    /// their aspect ratios differ (matches [Sprite](#sprite)'s `fit`). `Bottom`
    /// keeps a label flush with a bottom-anchored sprite it labels.
    pub fit: SpriteFit,
    /// RGBA fill of a box drawn behind the text. An alpha of 0 (the default)
    /// draws no box; any alpha > 0 draws the box at that opacity.
    pub background: [f32; 4],
    /// Pixels the background box extends past the text on every side. Only
    /// meaningful when `background` is visible.
    pub padding: f32,
    /// Width in the label's own pixels that text wraps within. `0` (the
    /// default) never wraps, so the text runs as far as it needs to. Any
    /// greater value breaks the content into lines at word boundaries, using
    /// the real font metrics, splitting a word only when it cannot fit a line
    /// on its own. Authored newlines are kept as breaks either way. Ignored
    /// when `centered` is set, since a centered label is sized to the viewport
    /// rather than to a container.
    pub wrap_width: f32,
    /// Most lines the label draws. `0` (the default) draws every line. When the
    /// text needs more than this, the last drawn line ends in an ellipsis, so
    /// text bounded by `wrap_width` is bounded in both directions and can never
    /// spill out of the box that holds it.
    pub max_lines: u32,
    /// When false, the label is hidden.
    pub visible: bool,
    /// [Screen](#screen) this label belongs to. `None` means the label is
    /// always visible.
    #[serde(default, deserialize_with = "de_opt_ref")]
    pub screen: Option<Ref<Screen>>,
}

impl Default for TextLabel {
    fn default() -> Self {
        Self {
            font: None,
            content: String::new(),
            x: 10.0,
            y: 10.0,
            color: [1.0, 1.0, 1.0],
            color_runs: Vec::new(),
            scale: 1.0,
            centered: false,
            align: TextAlign::Left,
            fit: SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            wrap_width: 0.0,
            max_lines: 0,
            visible: true,
            screen: None,
        }
    }
}

/// A span of a [TextLabel](#textlabel)'s `content` drawn in a color of its
/// own.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Default,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
)]
#[serde(default)]
pub struct ColorRun {
    /// The run's first character, counting from 0 at the start of `content`.
    pub start: u32,
    /// How many characters the run covers.
    pub length: u32,
    /// Linear-space RGB color of the covered characters.
    pub color: [f32; 3],
}

impl ColorRun {
    /// The character just past the run.
    pub fn end(&self) -> u64 {
        u64::from(self.start) + u64::from(self.length)
    }
}

/// Why a label's `color_runs` cannot color its `content`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorRunError {
    /// Run `run` ends at character `end`, past the `chars` the content holds.
    OutOfRange {
        /// Index of the run in `color_runs`.
        run: usize,
        /// The character just past the run.
        end: u64,
        /// Characters in `content`.
        chars: usize,
    },
    /// Run `run` starts before the run ahead of it ends.
    Overlap {
        /// Index of the run in `color_runs`.
        run: usize,
    },
}

impl fmt::Display for ColorRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::OutOfRange { run, end, chars } => write!(
                f,
                "color_runs[{run}] ends at character {end}, past the {chars} in content"
            ),
            Self::Overlap { run } => write!(
                f,
                "color_runs[{run}] starts before color_runs[{}] ends; runs must be in order and not overlap",
                run - 1
            ),
        }
    }
}

/// The first reason `runs` cannot color `content`, if any.
pub fn color_run_error(content: &str, runs: &[ColorRun]) -> Option<ColorRunError> {
    let chars = content.chars().count();
    let mut prev_end = 0u64;
    for (i, run) in runs.iter().enumerate() {
        if i > 0 && u64::from(run.start) < prev_end {
            return Some(ColorRunError::Overlap { run: i });
        }
        if run.end() > chars as u64 {
            return Some(ColorRunError::OutOfRange {
                run: i,
                end: run.end(),
                chars,
            });
        }
        prev_end = run.end();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::asset_id::AssetId;
    use alloc::string::ToString;

    #[test]
    fn a_blank_label_draws_white_left_aligned_text_with_no_background() {
        let l = TextLabel::default();
        assert!(l.content.is_empty());
        assert_eq!((l.x, l.y), (10.0, 10.0));
        assert_eq!(l.color, [1.0, 1.0, 1.0]);
        assert_eq!(l.scale, 1.0);
        assert!(!l.centered);
        assert_eq!(l.align, TextAlign::Left);
        assert_eq!(l.fit, SpriteFit::Fit);
        // A fully transparent background is what suppresses the chip box.
        assert_eq!(l.background, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(l.padding, 0.0);
        // Zero means unbounded: no wrapping and no line cap.
        assert_eq!(l.wrap_width, 0.0);
        assert_eq!(l.max_lines, 0);
        assert!(l.visible);
        assert!(l.font.is_none());
        assert!(l.screen.is_none());
        assert!(l.color_runs.is_empty());
        assert_eq!(TextAlign::default(), TextAlign::Left);
    }

    #[test]
    fn alignment_names_parse_in_lowercase() {
        let a = |s: &str| serde_json::from_str::<TextAlign>(s).unwrap();
        assert_eq!(a(r#""left""#), TextAlign::Left);
        assert_eq!(a(r#""center""#), TextAlign::Center);
        assert_eq!(a(r#""right""#), TextAlign::Right);
        assert_eq!(
            serde_json::to_string(&TextAlign::Center).unwrap(),
            r#""center""#
        );
    }

    #[test]
    fn a_wrapped_chip_parses_and_round_trips_through_postcard() {
        let l: TextLabel = crate::test_support::from_json(
            r#"{"font":"body","content":"Hello there","x":20,"y":40,"color":[1,0.9,0.5],
                "scale":1.25,"centered":true,"align":"right","fit":"cover",
                "background":[0,0,0,0.6],"padding":6,"wrap_width":320,"max_lines":3,
                "visible":false,"screen":"menu"}"#,
        );
        assert_eq!(l.font, Some(FontHandle(4)));
        assert_eq!(l.screen, Some(Ref::new(AssetId(4))));
        assert_eq!(l.align, TextAlign::Right);
        assert!(l.centered);
        assert!(!l.visible);

        let bytes = postcard::to_allocvec(&l).unwrap();
        let back: TextLabel = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.content, "Hello there");
        assert_eq!(back.color, [1.0, 0.9, 0.5]);
        assert_eq!(back.scale, 1.25);
        assert_eq!(back.fit, SpriteFit::Cover);
        assert_eq!(back.background, [0.0, 0.0, 0.0, 0.6]);
        assert_eq!(back.padding, 6.0);
        assert_eq!(back.wrap_width, 320.0);
        assert_eq!(back.max_lines, 3);
    }

    fn run(start: u32, length: u32) -> ColorRun {
        ColorRun {
            start,
            length,
            color: [1.0, 0.0, 0.0],
        }
    }

    #[test]
    fn color_runs_parse_and_round_trip_through_postcard() {
        let l: TextLabel = serde_json::from_str(
            r#"{"content":"Mara: hi","color_runs":[{"start":0,"length":4,"color":[1,0.8,0.3]}]}"#,
        )
        .unwrap();
        assert_eq!(
            l.color_runs,
            [ColorRun {
                start: 0,
                length: 4,
                color: [1.0, 0.8, 0.3]
            }]
        );
        let back: TextLabel = postcard::from_bytes(&postcard::to_allocvec(&l).unwrap()).unwrap();
        assert_eq!(back.color_runs, l.color_runs);
    }

    #[test]
    fn runs_in_order_within_the_content_are_accepted() {
        assert_eq!(color_run_error("hello world", &[]), None);
        assert_eq!(
            color_run_error("hello world", &[run(0, 5), run(5, 1), run(6, 5)]),
            None,
            "touching runs do not overlap, and a run may end at the last character"
        );
        assert_eq!(
            color_run_error("héllo", &[run(1, 4)]),
            None,
            "positions count characters, not bytes"
        );
        assert_eq!(
            color_run_error("a\nb", &[run(2, 1)]),
            None,
            "a newline is one character"
        );
    }

    #[test]
    fn a_run_past_the_content_is_out_of_range() {
        assert_eq!(
            color_run_error("hello", &[run(2, 4)]),
            Some(ColorRunError::OutOfRange {
                run: 0,
                end: 6,
                chars: 5
            })
        );
        assert_eq!(
            color_run_error("héllo", &[run(0, 6)]),
            Some(ColorRunError::OutOfRange {
                run: 0,
                end: 6,
                chars: 5
            }),
            "a two-byte character is still one"
        );
        assert!(
            color_run_error("x", &[run(u32::MAX, u32::MAX)]).is_some(),
            "the end does not wrap around"
        );
    }

    #[test]
    fn overlapping_or_backward_runs_are_rejected() {
        assert_eq!(
            color_run_error("hello world", &[run(0, 5), run(4, 2)]),
            Some(ColorRunError::Overlap { run: 1 })
        );
        assert_eq!(
            color_run_error("hello world", &[run(6, 2), run(0, 2)]),
            Some(ColorRunError::Overlap { run: 1 }),
            "runs are in order along the text"
        );
        let message = ColorRunError::Overlap { run: 1 }.to_string();
        assert!(message.contains("color_runs[1]") && message.contains("color_runs[0]"));
    }
}
