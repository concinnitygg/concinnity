// The compiled form of a behavior body: every authored name resolved to a dense
// slot, so evaluating the body touches no strings.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::behavior::value::{Arith, Cmp, Val};
use crate::components::{Behavior, CueKind, SceneTransition, StoryPlayback};
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
    /// The entity of a declared query nearest a point.
    Nearest {
        /// The query's slot.
        query: u16,
        /// The point searched around.
        of: Box<CExpr>,
    },
    /// How many of a declared query's entities lie within a radius of a point.
    CountWithin {
        /// The query's slot.
        query: u16,
        /// The point searched around.
        of: Box<CExpr>,
        /// How far from it to search.
        radius: Box<CExpr>,
    },
    /// The nearest entity of a declared query a ray meets.
    Raycast {
        /// The query's slot.
        query: u16,
        /// Where the ray starts.
        from: Box<CExpr>,
        /// Which way it points.
        dir: Box<CExpr>,
        /// How far it reaches.
        distance: Box<CExpr>,
    },
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
    /// Run a block later, with the bindings live when this node was reached.
    After {
        /// Seconds of simulated time to wait.
        seconds: CExpr,
        /// Nodes run once the wait elapses.
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
        transition: SceneTransition,
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

    /// The block an [`COp::After`] node holds, by that node's id.
    ///
    /// A deferred run records the node it came from rather than a copy of the
    /// block, so nothing has to be kept alive beside the program; this is what
    /// resolves the one back to the other when the wait elapses. `None` when
    /// no node of that id defers anything, which is what a run scheduled
    /// against a body that has since been edited resolves to.
    pub fn deferred_block(&self, node: u32) -> Option<&[CNode]> {
        fn find(nodes: &[CNode], node: u32) -> Option<&[CNode]> {
            for n in nodes {
                let found = match &n.op {
                    COp::After { body, .. } if n.id == node => return Some(body),
                    COp::After { body, .. } => find(body, node),
                    COp::If {
                        then, otherwise, ..
                    } => find(then, node).or_else(|| find(otherwise, node)),
                    COp::ForEach { body, .. } => find(body, node),
                    _ => None,
                };
                if found.is_some() {
                    return found;
                }
            }
            None
        }
        find(&self.body, node)
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

    /// The starting value a name was declared with, if it has a slot.
    pub fn init_of(&self, name: &str) -> Option<Val> {
        self.slot_of(name)
            .and_then(|slot| self.inits.get(slot as usize))
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::{CExpr, CNode, COp, Program};
    use crate::behavior::Val;
    use alloc::vec;
    use alloc::vec::Vec;

    fn node(id: u32, op: COp) -> CNode {
        CNode { id, op }
    }

    fn after(id: u32, body: Vec<CNode>) -> CNode {
        node(
            id,
            COp::After {
                seconds: CExpr::Lit(Val::Float(1.0)),
                body,
            },
        )
    }

    fn program(body: Vec<CNode>) -> Program {
        Program {
            def: Default::default(),
            scope: Vec::new(),
            local_inits: Vec::new(),
            queries: Vec::new(),
            body,
            paths: Vec::new(),
            bindings: 0,
        }
    }

    // A deferred run records the node it came from, so this is what turns that
    // number back into the block to run.
    #[test]
    fn a_deferred_block_resolves_by_its_node_id() {
        let p = program(vec![node(0, COp::Save), after(1, vec![node(2, COp::Save)])]);
        let block = p.deferred_block(1).expect("node 1 defers a block");
        assert_eq!(block.len(), 1);
        assert_eq!(block[0].id, 2);
    }

    // The search descends branches and loop bodies, so an `after` nested
    // anywhere in the body is reachable.
    #[test]
    fn a_nested_deferred_block_is_found() {
        let p = program(vec![node(
            0,
            COp::If {
                cond: CExpr::Lit(Val::Bool(true)),
                then: vec![node(
                    1,
                    COp::ForEach {
                        query: 0,
                        bind: 0,
                        body: vec![after(2, vec![node(3, COp::Save)])],
                    },
                )],
                otherwise: Vec::new(),
            },
        )]);
        let block = p.deferred_block(2).expect("node 2 defers a block");
        assert_eq!(block[0].id, 3);
    }

    // An `after` inside an `after` block is reachable too: the search descends
    // a deferred block the way it descends any other list.
    #[test]
    fn an_after_inside_a_deferred_block_is_found() {
        let p = program(vec![after(0, vec![after(1, vec![node(2, COp::Save)])])]);
        assert_eq!(p.deferred_block(1).expect("the inner block")[0].id, 2);
    }

    // A node that defers nothing resolves to nothing, which is what a run
    // scheduled against a body that has since been edited lands on.
    #[test]
    fn a_node_that_defers_nothing_resolves_to_nothing() {
        let p = program(vec![node(0, COp::Save), after(1, Vec::new())]);
        assert!(p.deferred_block(0).is_none());
        assert!(p.deferred_block(9).is_none());
    }
}
