use std::sync::atomic::{AtomicUsize, Ordering};

use concinnity_core::components::ColorRun;
use concinnity_core::render::shader_programs::vocabulary::{ENTRIES, Kind};

use super::states::LineStates;
use super::window::window_runs;
use super::*;
use crate::editor::text_area::TextArea;

// `line`'s spans, as (text, token), starting from nothing open.
fn tokens(hl: &dyn Highlighter, line: &str) -> Vec<(String, Token)> {
    let mut spans = Vec::new();
    hl.line(line, LineState::default(), &mut spans);
    let chars: Vec<char> = line.chars().collect();
    spans
        .iter()
        .map(|s| (chars[s.start..s.start + s.len].iter().collect(), s.token))
        .collect()
}

fn token_of(hl: &dyn Highlighter, line: &str, text: &str) -> Option<Token> {
    tokens(hl, line)
        .into_iter()
        .find(|(t, _)| t == text)
        .map(|(_, token)| token)
}

// Each line's spans, carrying the state line to line.
fn lines(hl: &dyn Highlighter, text: &str) -> Vec<Vec<Span>> {
    let mut state = LineState::default();
    text.lines()
        .map(|line| {
            let mut spans = Vec::new();
            state = hl.line(line, state, &mut spans);
            spans
        })
        .collect()
}

#[test]
fn hlsl_colors_keywords_types_numbers_strings_and_comments() {
    let got = tokens(
        &HLSL,
        r#"if (x) return float4(0.5f, 0x1Fu, 2e-3, "a\"b"); // done"#,
    );
    let want = [
        ("if", Token::Keyword),
        ("return", Token::Keyword),
        ("float4", Token::Type),
        ("0.5f", Token::Number),
        ("0x1Fu", Token::Number),
        ("2e-3", Token::Number),
        (r#""a\"b""#, Token::String),
        ("// done", Token::Comment),
    ];
    let want: Vec<(String, Token)> = want.iter().map(|(t, k)| (t.to_string(), *k)).collect();
    assert_eq!(got, want);
    for ty in [
        "float",
        "uint2",
        "float4x4",
        "half3x2",
        "Texture2D",
        "SamplerState",
    ] {
        assert_eq!(token_of(&HLSL, ty, ty), Some(Token::Type), "{ty}");
    }
    assert_eq!(token_of(&HLSL, "float5", "float5"), None);
    assert_eq!(token_of(&HLSL, "x = .5;", ".5"), Some(Token::Number));
}

#[test]
fn hlsl_colors_a_directive() {
    let got = tokens(&HLSL, "  #define TINT 2");
    assert_eq!(got[0], ("#define".to_string(), Token::Preprocessor));
    assert_eq!(got[1], ("2".to_string(), Token::Number));
    assert_eq!(
        token_of(&HLSL, "a # b", "#"),
        None,
        "only at the start of a line"
    );
}

#[test]
fn hlsl_sets_apart_every_name_the_engine_provides() {
    for e in ENTRIES.iter().filter(|e| e.kind == Kind::Helper) {
        let line = format!("x = {}(a);", e.name);
        assert_eq!(
            token_of(&HLSL, &line, e.name),
            Some(Token::Engine),
            "{}",
            e.name
        );
    }
    for e in ENTRIES {
        let Some(owner) = e.kind.owner() else {
            continue;
        };
        let line = format!("y = {}.{};", owner, e.name);
        assert_eq!(
            token_of(&HLSL, &line, e.name),
            Some(Token::Engine),
            "{line}"
        );
        let block = matches!(e.kind, Kind::BlockField(_));
        assert_eq!(token_of(&HLSL, &line, owner).is_some(), block, "{line}");
    }
    let line = "float4 shade(VertexOut v, GpuObjectData od)";
    assert_eq!(token_of(&HLSL, line, "VertexOut"), Some(Token::Engine));
    assert_eq!(token_of(&HLSL, line, "GpuObjectData"), Some(Token::Engine));
}

#[test]
fn hlsl_colors_a_field_only_where_it_is_read_through_its_owner() {
    assert_eq!(token_of(&HLSL, "float elapsed = 1;", "elapsed"), None);
    assert_eq!(
        token_of(&HLSL, "x = LIGHTS.elapsed;", "elapsed"),
        None,
        "a block field belongs to its own block"
    );
    assert_eq!(
        token_of(&HLSL, "x = o.world_pos;", "world_pos"),
        Some(Token::Engine),
        "the hooks' structs may be held under any name"
    );
    let got = tokens(&HLSL, "x = v.color.b;");
    assert!(got.contains(&("color".to_string(), Token::Engine)));
    assert!(
        !got.iter().any(|(t, _)| t == "b"),
        "a swizzle is not a field"
    );
}

#[test]
fn hlsl_block_comments_run_across_lines() {
    let spans = lines(&HLSL, "a /* one\ntwo\nthree */ float b;\nfloat c;");
    let c = |start, len| Span {
        start,
        len,
        token: Token::Comment,
    };
    assert_eq!(spans[0], [c(2, 6)]);
    assert_eq!(spans[1], [c(0, 3)]);
    assert_eq!(spans[2][0], c(0, 8));
    assert_eq!(
        spans[2][1].token,
        Token::Type,
        "code resumes after the close"
    );
    assert_eq!(
        spans[3][0].token,
        Token::Type,
        "and the next line starts clear"
    );
    assert_eq!(
        lines(&HLSL, "/* a */ x /* b */")[0].len(),
        2,
        "two comments on one line"
    );
}

#[test]
fn markdown_colors_headings_emphasis_links_and_code() {
    assert_eq!(
        token_of(&MARKDOWN, "## Act one", "## Act one"),
        Some(Token::Heading)
    );
    assert_eq!(token_of(&MARKDOWN, "#hashtag", "#hashtag"), None);
    let got = tokens(
        &MARKDOWN,
        "A *soft* and **loud** word, `code`, [a link](next.md).",
    );
    let want = [
        ("*soft*", Token::Emphasis),
        ("**loud**", Token::Emphasis),
        ("`code`", Token::Code),
        ("[a link](next.md)", Token::Link),
    ];
    let want: Vec<(String, Token)> = want.iter().map(|(t, k)| (t.to_string(), *k)).collect();
    assert_eq!(got, want);
    assert!(tokens(&MARKDOWN, "a snake_case_name").is_empty());
    assert!(
        tokens(&MARKDOWN, "2 * 3 * 4").is_empty(),
        "spaced stars are not emphasis"
    );
    assert_eq!(token_of(&MARKDOWN, "a _b_ c", "_b_"), Some(Token::Emphasis));
    assert!(tokens(&MARKDOWN, "[not a link] here").is_empty());
}

#[test]
fn markdown_fences_run_across_lines() {
    let spans = lines(&MARKDOWN, "text\n```\n# not a heading\n```\n# heading");
    assert!(spans[0].is_empty());
    for line in &spans[1..4] {
        assert_eq!(line[0].token, Token::Code);
    }
    assert_eq!(spans[4][0].token, Token::Heading, "the fence closed");
}

// Counts the lines it is asked to scan.
#[derive(Debug, Default)]
struct Counting(AtomicUsize);

impl Highlighter for Counting {
    fn line(&self, line: &str, state: LineState, out: &mut Vec<Span>) -> LineState {
        self.0.fetch_add(1, Ordering::Relaxed);
        HLSL.line(line, state, out)
    }
}

impl Counting {
    fn take(&self) -> usize {
        self.0.swap(0, Ordering::Relaxed)
    }
}

#[test]
fn line_states_rescan_only_from_the_edited_line_down() {
    let mut text: Vec<String> = (0..100).map(|i| format!("float x{i};")).collect();
    let hl = Counting::default();
    let mut states = LineStates::default();
    states.start_of(99, &hl, |i| text[i].as_str());
    assert_eq!(hl.take(), 99, "the first ask scans every line above");
    states.start_of(99, &hl, |i| text[i].as_str());
    states.start_of(40, &hl, |i| text[i].as_str());
    assert_eq!(hl.take(), 0, "known states are not rescanned");

    text[60] = "/* open".to_string();
    states.edited(60);
    assert_eq!(states.known(), 61, "states above and at the edit stay");
    assert_eq!(
        states.start_of(30, &hl, |i| text[i].as_str()),
        LineState::default()
    );
    assert_eq!(hl.take(), 0);
    let inside = states.start_of(80, &hl, |i| text[i].as_str());
    assert_eq!(hl.take(), 20, "the rescan starts at the edited line");
    assert_ne!(
        inside,
        LineState::default(),
        "and sees the comment it opened"
    );
}

#[test]
fn learning_a_line_in_order_saves_its_scan() {
    let text = ["a", "b", "c"];
    let hl = Counting::default();
    let mut states = LineStates::default();
    let start = states.start_of(0, &hl, |i| text[i]);
    let after = hl.line(text[0], start, &mut Vec::new());
    states.learn(0, after);
    hl.take();
    states.start_of(1, &hl, |i| text[i]);
    assert_eq!(hl.take(), 0);
}

fn span(start: usize, len: usize) -> Span {
    Span {
        start,
        len,
        token: Token::Keyword,
    }
}

fn runs(line: &str, spans: &[Span], left: usize, width: usize) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    window_runs(line, spans, left, width, &mut out);
    out.iter().map(|r| (r.start, r.length)).collect()
}

#[test]
fn window_runs_clip_to_the_cells_a_row_shows() {
    let line = "return value;";
    let spans = [span(0, 6), span(7, 5)];
    assert_eq!(runs(line, &spans, 0, 40), [(0, 6), (7, 5)]);
    assert_eq!(
        runs(line, &spans, 3, 40),
        [(0, 3), (4, 5)],
        "scrolled right"
    );
    assert_eq!(
        runs(line, &spans, 0, 9),
        [(0, 6), (7, 2)],
        "cut at the right edge"
    );
    assert_eq!(runs(line, &spans, 7, 2), [(0, 2)]);
    assert!(runs(line, &spans, 20, 10).is_empty());
}

#[test]
fn window_runs_count_a_tab_as_the_cells_it_takes() {
    // The tab spans cells 0..4, so `if` sits in cells 4..6.
    let line = "\tif";
    assert_eq!(runs(line, &[span(1, 2)], 0, 20), [(4, 2)]);
    assert_eq!(
        runs(line, &[span(0, 1)], 2, 20),
        [(0, 2)],
        "a tab cut by the window"
    );
    let mut out = vec![ColorRun::default()];
    window_runs(line, &[], 0, 20, &mut out);
    assert_eq!(out.len(), 1, "appends, and nothing for no spans");
}

#[test]
fn a_text_area_rehighlights_after_an_edit_and_its_undo() {
    let mut area = TextArea::from_text("a\nb\nfloat c;").highlighted(&HLSL);
    let row = |area: &TextArea, line| {
        let mut out = Vec::new();
        area.line_runs(line, 0, 40, &mut out);
        out.iter()
            .map(|r| (r.start, r.length, r.color))
            .collect::<Vec<_>>()
    };
    let before = row(&area, 2);
    assert_eq!(before, [(0, 5, Token::Type.color())]);
    area.type_char('/');
    area.type_char('*');
    assert_eq!(
        row(&area, 2),
        [(0, 8, Token::Comment.color())],
        "a comment opened above swallows the line"
    );
    area.undo();
    assert_eq!(row(&area, 2), before);
    assert!(
        {
            let mut out = Vec::new();
            TextArea::from_text("float").line_runs(0, 0, 40, &mut out);
            out.is_empty()
        },
        "no highlighter, no runs"
    );
}
