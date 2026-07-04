use std::collections::BTreeMap;

// A parsed story: frontmatter metadata plus the node graph, in document order.
#[derive(Debug, Default)]
pub(crate) struct Story {
    pub(crate) title: String,
    pub(crate) characters: BTreeMap<String, Character>,
    pub(crate) nodes: Vec<Node>,
    // Optional title-screen backdrop image (frontmatter `background`), drawn
    // full-bleed behind the title menu.
    pub(crate) background: Option<String>,
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
    pub(crate) choice_ops: Vec<FlagOp>,
    pub(crate) choice_gates: Vec<Gate>,
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
    pub(crate) ops: Vec<FlagOp>,
    pub(crate) gates: Vec<Gate>,
}

#[derive(Debug)]
pub(crate) struct Choice {
    pub(crate) label: String,
    pub(crate) target: String,
    pub(crate) condition: Option<Condition>,
}

// A `set` / `clear` / `add` line from a ```story script block, run when the
// page (or choice menu) it precedes shows. All story state is named integer
// variables (a flag is a variable set to 1 / cleared to 0): `set x` assigns
// 1, `clear x` assigns 0, `set x = n` assigns n, `add x n` adds n.
#[derive(Debug, Clone)]
pub(crate) struct FlagOp {
    pub(crate) name: String,
    pub(crate) value: i32,
    pub(crate) add: bool,
}

// An `if ... -> #anchor` line from a ```story script block: a conditional
// jump evaluated before the page (or choice menu) it precedes shows.
#[derive(Debug, Clone)]
pub(crate) struct Gate {
    pub(crate) condition: Condition,
    pub(crate) target: String,
}

// A condition: `<var>` (not zero), `not <var>` (zero), or a comparison
// `<var> <op> <int>` with op one of == != < <= > >=. Used by gates and by
// choice-gating link titles.
#[derive(Debug, Clone)]
pub(crate) struct Condition {
    pub(crate) name: String,
    pub(crate) op: &'static str,
    pub(crate) value: i32,
}

// A media directive paragraph: a lone link whose label names the channel and
// whose target is an audio file, or an image whose alt names its stage role.
pub(super) enum Directive {
    Music(String),
    Sound(String),
    Bg(String),
    Left(String),
    Center(String),
    Right(String),
}

// One parsed line of a ```story script block.
pub(super) enum ScriptLine {
    Op(FlagOp),
    Gate(Gate),
}

// In-flight paragraph state: inline events accumulate here until the
// paragraph closes and is classified as narration, dialogue, or a jump.
#[derive(Default)]
pub(super) struct ParaAcc {
    pub(super) speaker: Option<String>,
    pub(super) text: String,
    pub(super) links: Vec<(String, String)>,
    pub(super) images: Vec<(String, String)>,
    pub(super) has_plain_text: bool,
}

// What one paragraph contributes: a page of the story, or media directives
// that style the pages after it. Directives stack: a paragraph made only of
// `![bg]` / `[music]` / `[sound]` lines applies them all.
pub(super) enum ParaOut {
    // Boxed: a Page is an order of magnitude larger than the other variant.
    Page(Box<Page>),
    Directives(Vec<Directive>),
}

// A `id:` character whose fields arrive on the following indented lines.
pub(super) struct BlockCharacter {
    pub(super) id: String,
    pub(super) line: usize,
    pub(super) name: Option<String>,
    pub(super) color: [f32; 3],
}

// Reads an image file's pixel dimensions. Injected into emission (the real
// reader probes file headers) so tests run without image files on disk.
pub(crate) type ImageDims<'a> = &'a dyn Fn(&str) -> Result<(u32, u32), String>;
