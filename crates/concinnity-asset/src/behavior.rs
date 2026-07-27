// Behavior schema: declarative logic over typed state, world reads, and a
// per-frame tick.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use crate::{AssetId, AudioClipHandle, StoryPlayback, de_opt_asset_ref, de_opt_audio_clip_handle};

/// A unit of world logic: an event source, the entities it runs against, its
/// state, the world it reads, and the nodes it runs.
///
/// A behavior with an empty `scope` runs once per firing, world-scoped. A
/// behavior with a `scope` runs once per matching entity, each with its own
/// copy of `locals`, and `"self"` resolves to that entity.
///
/// `locals` and `queries` are declared here and resolved to dense slots when
/// the world is built, so nothing is looked up by name while the world runs.
/// A query naming an unknown component is a build error.
///
/// `once` limits a behavior to a single firing; `cooldown` enforces a minimum
/// number of seconds between firings; `delay` postpones the nodes after the
/// firing decision (conditions are checked at fire time, not after the delay).
/// Timers, delays, and cooldowns freeze while a menu is open, like the rest of
/// the world clock.
///
/// ```jsonl
/// {"name":"greet","type":"Behavior","args":{"on":"start","do":[{"set":{"var":"visits","value":{"int":1},"add":true}}]}}
/// {"name":"drip","type":"Behavior","args":{"on":{"timer":{"interval":5.0,"repeat":true}},"do":[{"spawn":{"template":"drop","position":[0,3,0],"lifetime":4.0}}]}}
/// {"name":"chase","type":"Behavior","args":{"on":"tick","scope":["Prop"],"locals":[{"name":"speed","value":{"float":3.0}}],"queries":[{"name":"player","has":["Camera3D"]}],"do":[{"let":{"name":"target","value":{"first":"player"}}},{"if":{"cond":{"lt":[{"distance":["self",{"bind":"target"}]},{"float":20.0}]},"then":[]}}]}}
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Behavior {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The event that fires this behavior.
    pub on: BehaviorSource,
    /// Component names selecting the entities this behavior runs against. An
    /// empty list runs it once, world-scoped, with no `"self"`.
    pub scope: Vec<String>,
    /// Per-entity state. Each matching entity gets its own copy, reset to the
    /// declared value when the world starts. Locals are never persisted.
    pub locals: Vec<LocalDecl>,
    /// World reads, resolved once per tick into a stable-ordered entity list.
    pub queries: Vec<QueryDecl>,
    /// The nodes run, in order, each time the behavior fires.
    #[serde(rename = "do")]
    pub body: Vec<Node>,
    /// Fire at most once per run.
    pub once: bool,
    /// Seconds between the firing decision and the nodes running (`0` runs
    /// them immediately).
    pub delay: f32,
    /// Minimum seconds between firings (`0` allows every firing).
    pub cooldown: f32,
}

/// The event that fires a [Behavior](#behavior).
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BehaviorSource {
    /// Fires once when the world starts.
    #[default]
    Start,
    /// Fires every tick.
    Tick,
    /// Fires when `interval` seconds have elapsed; with `repeat`, every
    /// `interval` seconds.
    Timer {
        /// Seconds before the behavior fires.
        #[serde(default)]
        interval: f32,
        /// `true` fires every `interval` seconds; `false` fires once.
        #[serde(default)]
        repeat: bool,
    },
    /// Fires whenever the named world variable changes value.
    Variable(String),
    /// Fires when something enters the named [TriggerVolume](#triggervolume).
    Enter(#[serde(deserialize_with = "de_opt_asset_ref")] Option<AssetId>),
    /// Fires when something leaves the named [TriggerVolume](#triggervolume).
    Exit(#[serde(deserialize_with = "de_opt_asset_ref")] Option<AssetId>),
    /// Fires when the interact key is pressed on the named entity (a
    /// [Prop](#prop) declared `interactable`).
    Interact(#[serde(deserialize_with = "de_opt_asset_ref")] Option<AssetId>),
    /// Fires on an entity the tick after it spawns.
    Spawned,
}

/// A per-entity state slot declared by a [Behavior](#behavior). The declared
/// value fixes both the slot's type and its starting value.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LocalDecl {
    /// The name nodes read the slot by.
    pub name: String,
    /// The slot's type and starting value.
    pub value: Literal,
}

/// A world read declared by a [Behavior](#behavior), resolved once per tick
/// into the entities carrying every named component.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct QueryDecl {
    /// The name expressions read the result by.
    pub name: String,
    /// Component names an entity must all carry to match.
    pub has: Vec<String>,
}

/// A typed literal in a [Behavior](#behavior).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Literal {
    /// A boolean.
    Bool(bool),
    /// A signed integer.
    Int(i32),
    /// A 32-bit float.
    Float(f32),
    /// A 3-component vector.
    Vec3([f32; 3]),
}

impl Default for Literal {
    fn default() -> Self {
        Literal::Int(0)
    }
}

/// A value expression in a [Behavior](#behavior).
///
/// Expressions are pure: they read state, entities, and the clock, and never
/// change the world. Arithmetic works on both scalars and vectors, and mixing
/// the two is a build error.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Expr {
    /// A boolean literal.
    Bool(bool),
    /// An integer literal.
    Int(i32),
    /// A float literal.
    Float(f32),
    /// A vector literal.
    Vec3([f32; 3]),
    /// Reads a world variable.
    Var(String),
    /// Reads one of this behavior's per-entity locals.
    Local(String),
    /// Reads a name bound earlier by a `let` or `for_each` node.
    Bind(String),
    /// The entity this behavior instance runs for. Only valid with a `scope`.
    #[serde(rename = "self")]
    SelfEntity,
    /// Seconds elapsed since the previous tick.
    Dt,
    /// Seconds elapsed since the world started.
    Elapsed,
    /// World-space position of an entity.
    Position(Box<Expr>),
    /// Distance between two entities.
    Distance(Box<Expr>, Box<Expr>),
    /// The first entity of a declared query, or none when it is empty.
    First(String),
    /// How many entities a declared query matched.
    Count(String),
    /// Whether an entity is still alive.
    Alive(Box<Expr>),
    /// Sum of two values.
    Add(Box<Expr>, Box<Expr>),
    /// Difference of two values.
    Sub(Box<Expr>, Box<Expr>),
    /// Product of two values; a vector may be scaled by a scalar.
    Mul(Box<Expr>, Box<Expr>),
    /// Quotient of two values; dividing by zero yields zero.
    Div(Box<Expr>, Box<Expr>),
    /// A vector rescaled to unit length; a zero vector stays zero.
    Normalize(Box<Expr>),
    /// Equality test.
    Eq(Box<Expr>, Box<Expr>),
    /// Inequality test.
    Ne(Box<Expr>, Box<Expr>),
    /// Less-than test.
    Lt(Box<Expr>, Box<Expr>),
    /// Less-than-or-equal test.
    Le(Box<Expr>, Box<Expr>),
    /// Greater-than test.
    Gt(Box<Expr>, Box<Expr>),
    /// Greater-than-or-equal test.
    Ge(Box<Expr>, Box<Expr>),
    /// True when every operand is true.
    All(Vec<Expr>),
    /// True when any operand is true.
    Any(Vec<Expr>),
    /// Logical negation.
    Not(Box<Expr>),
}

impl Default for Expr {
    fn default() -> Self {
        Expr::Bool(false)
    }
}

/// One node run by a firing [Behavior](#behavior).
///
/// Nodes are the only way a behavior changes the world. There is no unbounded
/// loop and no recursion, so a behavior body always terminates.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Node {
    /// Runs `then` when `cond` holds, `else` otherwise.
    If {
        /// The tested expression.
        cond: Expr,
        /// Nodes run when `cond` holds.
        #[serde(default)]
        then: Vec<Node>,
        /// Nodes run when `cond` does not hold.
        #[serde(default, rename = "else")]
        otherwise: Vec<Node>,
    },
    /// Runs `do` once per entity a declared query matched, binding each to
    /// `bind`.
    ForEach {
        /// The declared query iterated.
        query: String,
        /// The name each entity is bound to.
        bind: String,
        /// Nodes run per entity.
        #[serde(default, rename = "do")]
        body: Vec<Node>,
    },
    /// Binds an expression to a name for the rest of the body.
    Let {
        /// The bound name.
        name: String,
        /// The bound expression.
        value: Expr,
    },
    /// Writes a world variable: assign `value`, or add it to the current value
    /// when `add` is `true`.
    Set {
        /// The variable written.
        var: String,
        /// The value assigned (or added).
        value: Expr,
        /// `false` assigns `value`; `true` adds it to the current value.
        #[serde(default)]
        add: bool,
    },
    /// Writes one of this behavior's per-entity locals.
    SetLocal {
        /// The local written.
        local: String,
        /// The value assigned (or added).
        value: Expr,
        /// `false` assigns `value`; `true` adds it to the current value.
        #[serde(default)]
        add: bool,
    },
    /// Writes an entity's transform. An omitted field is left unchanged.
    SetTransform {
        /// The entity moved.
        entity: Expr,
        /// New world-space position.
        #[serde(default)]
        position: Option<Expr>,
        /// New Euler rotation in degrees.
        #[serde(default)]
        rotation_deg: Option<Expr>,
        /// New scale.
        #[serde(default)]
        scale: Option<Expr>,
    },
    /// Creates a copy of an existing placement at a world position. Binding
    /// `bind` makes the copy addressable for the rest of the body.
    Spawn {
        /// The placement copied (e.g. a [Prop](#prop)).
        #[serde(default, deserialize_with = "de_opt_asset_ref")]
        template: Option<AssetId>,
        /// World-space position of the copy.
        #[serde(default)]
        position: [f32; 3],
        /// Euler rotation of the copy in degrees.
        #[serde(default)]
        rotation_deg: [f32; 3],
        /// Scale of the copy (`[1, 1, 1]` keeps the template's size).
        #[serde(default = "unit_scale")]
        scale: [f32; 3],
        /// Seconds the copy lives before auto-despawning (`0` lives forever).
        #[serde(default)]
        lifetime: f32,
        /// Name the new entity is bound to for the rest of the body.
        #[serde(default)]
        bind: Option<String>,
    },
    /// Removes an entity and its children from the world.
    Despawn {
        /// The entity removed.
        target: Expr,
    },
    /// Re-points an entity's parent edge.
    Reparent {
        /// The entity moved.
        child: Expr,
        /// The new parent, or unset to detach to a root.
        #[serde(default)]
        parent: Option<Expr>,
    },
    /// Makes a hidden entity (and its children) visible again.
    Show {
        /// The entity revealed.
        target: Expr,
    },
    /// Makes an entity (and its children) invisible without removing it. A
    /// hidden entity keeps simulating; `show` reverses this.
    Hide {
        /// The entity hidden.
        target: Expr,
    },
    /// Plays an [AudioClip](#audioclip) flat on the main mix (no 3D position).
    Sound {
        /// The clip played.
        #[serde(default, deserialize_with = "de_opt_audio_clip_handle")]
        clip: Option<AudioClipHandle>,
        /// Playback behavior: a looping `music` track or a one-shot `sound`.
        #[serde(default)]
        kind: crate::CueKind,
        /// Linear gain applied to the clip (`1.0` leaves it unchanged).
        #[serde(default = "unit_volume")]
        volume: f32,
    },
    /// Jumps the world to a named [Scene](#scene).
    Scene {
        /// The scene jumped to.
        #[serde(default, deserialize_with = "de_opt_asset_ref")]
        scene: Option<AssetId>,
        /// The transition: `"Cut"` or `"FadeBlack"`.
        #[serde(default = "default_transition")]
        transition: String,
    },
    /// Shows a [Screen](#screen), replacing the top of the screen stack.
    Screen {
        /// The screen shown.
        #[serde(default, deserialize_with = "de_opt_asset_ref")]
        screen: Option<AssetId>,
    },
    /// Controls the world's [Story](#story) playback.
    Story(StoryPlayback),
    /// Writes the world's logic state to disk: every world variable, plus
    /// which `once` behaviors have fired. Per-entity locals are not saved. The
    /// state is restored the next time the world starts, so a behavior gated
    /// on a saved variable carries across runs. Timers, delays, and cooldowns
    /// restart fresh; entities are not saved. A world with no `save` node
    /// never reads or writes state.
    Save,
}

impl Behavior {
    /// Whether any node plays an audio clip, so the runtime knows this
    /// behavior needs the audio system and its clip payloads cached.
    pub fn plays_sound(&self) -> bool {
        self.visit(&mut |n| matches!(n, Node::Sound { .. }))
    }

    /// Whether any node saves the world's logic state, so the runtime knows to
    /// restore persisted state when the world starts.
    pub fn saves_state(&self) -> bool {
        self.visit(&mut |n| matches!(n, Node::Save))
    }

    /// Whether the behavior runs against individual entities rather than the
    /// world as a whole.
    pub fn is_scoped(&self) -> bool {
        !self.scope.is_empty()
    }

    // True when any node in the body, at any nesting depth, satisfies `pred`.
    fn visit(&self, pred: &mut impl FnMut(&Node) -> bool) -> bool {
        fn walk(nodes: &[Node], pred: &mut impl FnMut(&Node) -> bool) -> bool {
            nodes.iter().any(|n| {
                pred(n)
                    || match n {
                        Node::If {
                            then, otherwise, ..
                        } => walk(then, pred) || walk(otherwise, pred),
                        Node::ForEach { body, .. } => walk(body, pred),
                        _ => false,
                    }
            })
        }
        walk(&self.body, pred)
    }
}

fn unit_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn unit_volume() -> f32 {
    1.0
}

fn default_transition() -> String {
    String::from("FadeBlack")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> Behavior {
        serde_json::from_str(json).expect("behavior parses")
    }

    #[test]
    fn defaults_are_world_scoped_and_empty() {
        let b = Behavior::default();
        assert!(!b.is_scoped());
        assert!(b.body.is_empty());
        assert_eq!(b.on, BehaviorSource::Start);
        assert!(!b.plays_sound());
        assert!(!b.saves_state());
    }

    #[test]
    fn tick_source_parses() {
        let b = parse(r#"{"on":"tick"}"#);
        assert_eq!(b.on, BehaviorSource::Tick);
    }

    #[test]
    fn spawned_source_parses() {
        let b = parse(r#"{"on":"spawned"}"#);
        assert_eq!(b.on, BehaviorSource::Spawned);
    }

    #[test]
    fn scope_and_locals_parse() {
        let b = parse(
            r#"{"on":"tick","scope":["Prop"],"locals":[{"name":"speed","value":{"float":3.0}}]}"#,
        );
        assert!(b.is_scoped());
        assert_eq!(b.scope, ["Prop"]);
        assert_eq!(b.locals.len(), 1);
        assert_eq!(b.locals[0].name, "speed");
        assert_eq!(b.locals[0].value, Literal::Float(3.0));
    }

    #[test]
    fn query_parses() {
        let b = parse(r#"{"queries":[{"name":"player","has":["Camera3D","Prop"]}]}"#);
        assert_eq!(b.queries.len(), 1);
        assert_eq!(b.queries[0].name, "player");
        assert_eq!(b.queries[0].has, ["Camera3D", "Prop"]);
    }

    #[test]
    fn comparison_expression_parses_as_pair() {
        let b = parse(
            r#"{"do":[{"if":{"cond":{"lt":[{"distance":["self",{"bind":"t"}]},{"float":20.0}]}}}]}"#,
        );
        let Node::If {
            cond,
            then,
            otherwise,
        } = &b.body[0]
        else {
            panic!("expected an if node");
        };
        assert!(then.is_empty());
        assert!(otherwise.is_empty());
        let Expr::Lt(lhs, rhs) = cond else {
            panic!("expected a less-than comparison");
        };
        assert_eq!(**rhs, Expr::Float(20.0));
        let Expr::Distance(a, b) = &**lhs else {
            panic!("expected a distance expression");
        };
        assert_eq!(**a, Expr::SelfEntity);
        assert_eq!(**b, Expr::Bind(String::from("t")));
    }

    #[test]
    fn spawn_binds_the_new_entity() {
        let b = parse(r#"{"do":[{"spawn":{"bind":"made"}}]}"#);
        let Node::Spawn { bind, scale, .. } = &b.body[0] else {
            panic!("expected a spawn node");
        };
        assert_eq!(bind.as_deref(), Some("made"));
        assert_eq!(*scale, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn nested_nodes_are_visited_for_sound_and_save() {
        let b = parse(
            r#"{"do":[{"for_each":{"query":"q","bind":"e","do":[{"if":{"cond":{"bool":true},"then":[{"sound":{}}],"else":[{"save":null}]}}]}}]}"#,
        );
        assert!(b.plays_sound());
        assert!(b.saves_state());
    }

    #[test]
    fn round_trips_through_json() {
        let b = parse(
            r#"{"on":{"timer":{"interval":5.0,"repeat":true}},"do":[{"set":{"var":"visits","value":{"int":1},"add":true}}]}"#,
        );
        let encoded = serde_json::to_string(&b).expect("behavior encodes");
        let again: Behavior = serde_json::from_str(&encoded).expect("behavior re-parses");
        assert_eq!(again.on, b.on);
        let Node::Set { var, value, add } = &again.body[0] else {
            panic!("expected a set node");
        };
        assert_eq!(var, "visits");
        assert_eq!(*value, Expr::Int(1));
        assert!(add);
    }

    #[test]
    fn round_trips_through_postcard() {
        let b = parse(r#"{"on":"tick","scope":["Prop"],"do":[{"despawn":{"target":"self"}}]}"#);
        let bytes = postcard::to_allocvec(&b).expect("behavior encodes");
        let again: Behavior = postcard::from_bytes(&bytes).expect("behavior decodes");
        assert_eq!(again.on, BehaviorSource::Tick);
        assert!(matches!(again.body[0], Node::Despawn { .. }));
    }
}
