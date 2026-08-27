// The compiled form of a behavior body: every authored name resolved to a dense
// slot, so evaluating the body touches no strings.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::behavior::value::{Arith, Cmp, Val};
use crate::components::{Behavior, CueKind, StoryPlayback};
use crate::ecs::{AudioClipHandle, TracePath, asset_id::AssetId};

/// A slot-resolved expression.
#[derive(Debug)]
pub enum CExpr {
    /// A constant.
    Lit(Val),
    /// A world variable, by slot.
    Var(u16),
    /// One of the behavior's locals, by slot.
    Local(u16),
    /// A binding introduced by `let` or `for_each`, by slot.
    Bind(u16),
    /// The entity a name resolves to.
    Named(AssetId),
    /// The entity this run is scoped to.
    SelfEntity,
    /// Seconds this tick advances.
    Dt,
    /// Seconds of simulated time so far.
    Elapsed,
    /// An entity's world position.
    Position(Box<CExpr>),
    /// Distance between two entities.
    Distance(Box<CExpr>, Box<CExpr>),
    /// Whether an entity still exists.
    Alive(Box<CExpr>),
    /// The first entity a declared query selects, by slot.
    First(u16),
    /// How many entities a declared query selects, by slot.
    Count(u16),
    /// Two numbers combined.
    Arith(Arith, Box<CExpr>, Box<CExpr>),
    /// A vector scaled to unit length.
    Normalize(Box<CExpr>),
    /// Two values compared.
    Compare(Cmp, Box<CExpr>, Box<CExpr>),
    /// True when every operand is true.
    All(Vec<CExpr>),
    /// True when any operand is true.
    Any(Vec<CExpr>),
    /// Logical negation.
    Not(Box<CExpr>),
    /// An expression the checker should have rejected. Evaluates to nothing.
    Never,
}

/// A slot-resolved node with its compile-assigned identity: `id` is the node's
/// pre-order position across the whole body, indexing the program's `paths`
/// table so execution tracing can address the node the way the world checker's
/// faults do.
#[derive(Debug)]
pub struct CNode {
    /// Pre-order position across the body, and index into [`Program::paths`].
    pub id: u32,
    /// What the node does.
    pub op: COp,
}

/// A slot-resolved node operation.
#[derive(Debug)]
pub enum COp {
    /// Run one branch or the other.
    If {
        /// The tested condition.
        cond: CExpr,
        /// Branch taken when the condition holds.
        then: Vec<CNode>,
        /// Branch taken otherwise.
        otherwise: Vec<CNode>,
    },
    /// Run the body once per entity a declared query selects.
    ForEach {
        /// The query's slot.
        query: u16,
        /// Binding slot the iterated entity lands in.
        bind: u16,
        /// Nodes run per entity.
        body: Vec<CNode>,
    },
    /// Introduce a binding for the rest of the enclosing list.
    Let {
        /// Binding slot to fill.
        bind: u16,
        /// Value the binding takes.
        value: CExpr,
    },
    /// Write a world variable.
    SetVar {
        /// The variable's slot.
        slot: u16,
        /// Value to write.
        value: CExpr,
        /// Add to the current value rather than replacing it.
        add: bool,
    },
    /// Write one of the behavior's locals.
    SetLocal {
        /// The local's slot.
        slot: u16,
        /// Value to write.
        value: CExpr,
        /// Add to the current value rather than replacing it.
        add: bool,
    },
    /// Overwrite the named parts of an entity's transform.
    SetTransform {
        /// The entity to move.
        entity: CExpr,
        /// New position, when authored.
        position: Option<CExpr>,
        /// New rotation in degrees, when authored.
        rotation_deg: Option<CExpr>,
        /// New scale, when authored.
        scale: Option<CExpr>,
    },
    /// Request a copy of a template.
    Spawn {
        /// The template to copy.
        template: AssetId,
        /// Where the copy starts.
        position: [f32; 3],
        /// The copy's starting rotation, in degrees.
        rotation_deg: [f32; 3],
        /// The copy's starting scale.
        scale: [f32; 3],
        /// Seconds the copy lives, or zero for no limit.
        lifetime: f32,
        /// Binding slot the copy would land in, once it exists.
        bind: Option<u16>,
    },
    /// Request an entity's removal.
    Despawn(CExpr),
    /// Move an entity under another parent, or to the root.
    Reparent {
        /// The entity to move.
        child: CExpr,
        /// Its new parent, or `None` for the root.
        parent: Option<CExpr>,
    },
    /// Show or hide an entity.
    Visible(CExpr, bool),
    /// Play an audio cue.
    Sound {
        /// The clip to play.
        clip: AudioClipHandle,
        /// How the cue is voiced.
        kind: CueKind,
        /// Playback volume.
        volume: f32,
    },
    /// Request a scene change.
    Scene {
        /// The scene to load.
        scene: AssetId,
        /// The transition to play.
        transition: String,
    },
    /// Request a screen change.
    Screen(AssetId),
    /// Drive story playback.
    Story(StoryPlayback),
    /// Persist the world's behavior state.
    Save,
    /// A node the checker should have rejected. Does nothing.
    Never,
}

/// One compiled behavior.
#[derive(Debug)]
pub struct Program {
    /// The authored definition this was compiled from.
    pub def: Behavior,
    /// Components an entity must carry for this behavior to run against it.
    /// Empty runs the body once, world-scoped.
    pub scope: Vec<u8>,
    /// Starting value of each local, indexed by slot.
    pub local_inits: Vec<Val>,
    /// Component tags each declared query selects on, indexed by slot.
    pub queries: Vec<Vec<u8>>,
    /// The compiled body.
    pub body: Vec<CNode>,
    /// Each node's authored-tree path, indexed by [`CNode::id`].
    pub paths: Vec<TracePath>,
    /// How many binding slots a run of this body needs.
    pub bindings: usize,
}

impl Program {
    /// Whether the body runs per matching entity rather than once for the world.
    pub fn is_scoped(&self) -> bool {
        !self.scope.is_empty()
    }
}

/// The world's variables, in slot order: shared across behaviors, so slots are
/// assigned once across the whole set. A name the world's `Variables` asset
/// declares carries that declaration's type and starting value; any other name a
/// behavior mentions is an integer starting at zero.
#[derive(Debug, Default)]
pub struct VarTable {
    names: Vec<String>,
    inits: Vec<Val>,
}

impl VarTable {
    /// Declare a variable with its authored type and starting value. A repeated
    /// name keeps its first declaration; the world checker rejects duplicates.
    pub fn declare(&mut self, name: &str, init: Val) {
        if self.slot_of(name).is_some() {
            return;
        }
        self.names.push(name.to_string());
        self.inits.push(init);
    }

    // The slot for a name, assigning an undeclared integer if this is its first
    // mention.
    pub(crate) fn intern(&mut self, name: &str) -> u16 {
        if let Some(i) = self.slot_of(name) {
            return i;
        }
        self.declare(name, Val::Int(0));
        (self.names.len() - 1) as u16
    }

    /// The slot a name was assigned, if it has one.
    pub fn slot_of(&self, name: &str) -> Option<u16> {
        self.names.iter().position(|n| n == name).map(|i| i as u16)
    }

    /// Every declared name, in slot order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The starting values, in slot order.
    pub fn initial(&self) -> Vec<Val> {
        self.inits.clone()
    }
}
