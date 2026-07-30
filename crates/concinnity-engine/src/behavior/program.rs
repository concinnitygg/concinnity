// Compiled form of a Behavior: the authored asset with every name resolved to
// a dense slot, so evaluating a body touches no strings.
//
// Compilation runs once, in BehaviorSystem::init. The world crate's checker has
// already rejected unknown names and mistyped expressions, so compilation is
// total: anything it cannot resolve becomes `CExpr::Never`, which evaluates to
// nothing and skips its node rather than panicking.

use crate::assets::{Behavior, BehaviorSource, CueKind, Expr, Literal, Node, StoryPlayback};
use crate::ecs::{AudioClipHandle, Entity, TracePath, TraceStep, asset_id::AssetId};
use concinnity_core::ecs::ComponentTag;

// A value flowing through a behavior body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Val {
    Bool(bool),
    Int(i32),
    Float(f32),
    Vec3([f32; 3]),
    Entity(Entity),
}

impl Val {
    pub(super) fn as_bool(self) -> Option<bool> {
        match self {
            Val::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub(super) fn as_entity(self) -> Option<Entity> {
        match self {
            Val::Entity(e) => Some(e),
            _ => None,
        }
    }

    pub(super) fn as_vec3(self) -> Option<[f32; 3]> {
        match self {
            Val::Vec3(v) => Some(v),
            _ => None,
        }
    }

    // The scalar reading of a numeric value, for mixed vector/scalar forms.
    pub(super) fn as_f32(self) -> Option<f32> {
        match self {
            Val::Int(i) => Some(i as f32),
            Val::Float(f) => Some(f),
            _ => None,
        }
    }

    pub(super) fn from_literal(lit: &Literal) -> Val {
        match *lit {
            Literal::Bool(b) => Val::Bool(b),
            Literal::Int(i) => Val::Int(i),
            Literal::Float(f) => Val::Float(f),
            Literal::Vec3(v) => Val::Vec3(v),
        }
    }

    // The authored form, for persistence. Entities have no authored form and
    // are never persisted, so they save as their declared-away default.
    pub(super) fn to_literal(self) -> Literal {
        match self {
            Val::Bool(b) => Literal::Bool(b),
            Val::Int(i) => Literal::Int(i),
            Val::Float(f) => Literal::Float(f),
            Val::Vec3(v) => Literal::Vec3(v),
            Val::Entity(_) => Literal::Int(0),
        }
    }

    // The cross-boundary form execution tracing publishes (`Val` is private to
    // this system).
    pub(super) fn to_trace(self) -> crate::ecs::TraceVal {
        match self {
            Val::Bool(b) => crate::ecs::TraceVal::Bool(b),
            Val::Int(i) => crate::ecs::TraceVal::Int(i),
            Val::Float(f) => crate::ecs::TraceVal::Float(f),
            Val::Vec3(v) => crate::ecs::TraceVal::Vec3(v),
            Val::Entity(e) => crate::ecs::TraceVal::Entity(e.to_bits()),
        }
    }

    // Whether two values are the same shape, so a restored save can be
    // rejected when the world's declaration changed type under it.
    pub(super) fn same_type(self, other: Val) -> bool {
        core::mem::discriminant(&self) == core::mem::discriminant(&other)
    }
}

// How two numbers combine.
#[derive(Clone, Copy, Debug)]
pub(super) enum Arith {
    Add,
    Sub,
    Mul,
    Div,
}

// How two values compare.
#[derive(Clone, Copy, Debug)]
pub(super) enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

// A slot-resolved expression.
#[derive(Debug)]
pub(super) enum CExpr {
    Lit(Val),
    Var(u16),
    Local(u16),
    Bind(u16),
    Named(AssetId),
    SelfEntity,
    Dt,
    Elapsed,
    Position(Box<CExpr>),
    Distance(Box<CExpr>, Box<CExpr>),
    Alive(Box<CExpr>),
    First(u16),
    Count(u16),
    Arith(Arith, Box<CExpr>, Box<CExpr>),
    Normalize(Box<CExpr>),
    Compare(Cmp, Box<CExpr>, Box<CExpr>),
    All(Vec<CExpr>),
    Any(Vec<CExpr>),
    Not(Box<CExpr>),
    // An expression the checker should have rejected. Evaluates to nothing.
    Never,
}

// A slot-resolved node with its compile-assigned identity: `id` is the node's
// pre-order position across the whole body, indexing the program's `paths`
// table so execution tracing can address the node the way the world checker's
// faults do.
#[derive(Debug)]
pub(super) struct CNode {
    pub(super) id: u32,
    pub(super) op: COp,
}

// A slot-resolved node operation.
#[derive(Debug)]
pub(super) enum COp {
    If {
        cond: CExpr,
        then: Vec<CNode>,
        otherwise: Vec<CNode>,
    },
    ForEach {
        query: u16,
        bind: u16,
        body: Vec<CNode>,
    },
    Let {
        bind: u16,
        value: CExpr,
    },
    SetVar {
        slot: u16,
        value: CExpr,
        add: bool,
    },
    SetLocal {
        slot: u16,
        value: CExpr,
        add: bool,
    },
    SetTransform {
        entity: CExpr,
        position: Option<CExpr>,
        rotation_deg: Option<CExpr>,
        scale: Option<CExpr>,
    },
    Spawn {
        template: AssetId,
        position: [f32; 3],
        rotation_deg: [f32; 3],
        scale: [f32; 3],
        lifetime: f32,
        bind: Option<u16>,
    },
    Despawn(CExpr),
    Reparent {
        child: CExpr,
        parent: Option<CExpr>,
    },
    Visible(CExpr, bool),
    Sound {
        clip: AudioClipHandle,
        kind: CueKind,
        volume: f32,
    },
    Scene {
        scene: AssetId,
        transition: String,
    },
    Screen(AssetId),
    Story(StoryPlayback),
    Save,
    // A node the checker should have rejected. Does nothing.
    Never,
}

// One compiled behavior.
#[derive(Debug)]
pub(super) struct Program {
    pub(super) def: Behavior,
    // Components an entity must carry for this behavior to run against it.
    // Empty runs the body once, world-scoped.
    pub(super) scope: Vec<u8>,
    // Starting value of each local, indexed by slot.
    pub(super) local_inits: Vec<Val>,
    // Component tags each declared query selects on, indexed by slot.
    pub(super) queries: Vec<Vec<u8>>,
    pub(super) body: Vec<CNode>,
    // Each node's authored-tree path, indexed by `CNode::id`.
    pub(super) paths: Vec<TracePath>,
    // How many binding slots a run of this body needs.
    pub(super) bindings: usize,
}

impl Program {
    pub(super) fn is_scoped(&self) -> bool {
        !self.scope.is_empty()
    }
}

// The world's variables, in slot order: shared across behaviors, so slots are
// assigned once across the whole set. A name the world's `Variables` asset
// declares carries that declaration's type and starting value; any other name a
// behavior mentions is an integer starting at zero.
#[derive(Debug, Default)]
pub(super) struct VarTable {
    names: Vec<String>,
    inits: Vec<Val>,
}

impl VarTable {
    // Declare a variable with its authored type and starting value. A repeated
    // name keeps its first declaration; the world checker rejects duplicates.
    pub(super) fn declare(&mut self, name: &str, init: Val) {
        if self.slot_of(name).is_some() {
            return;
        }
        self.names.push(name.to_string());
        self.inits.push(init);
    }

    // The slot for a name, assigning an undeclared integer if this is its first
    // mention.
    fn intern(&mut self, name: &str) -> u16 {
        if let Some(i) = self.slot_of(name) {
            return i;
        }
        self.declare(name, Val::Int(0));
        (self.names.len() - 1) as u16
    }

    pub(super) fn slot_of(&self, name: &str) -> Option<u16> {
        self.names.iter().position(|n| n == name).map(|i| i as u16)
    }

    pub(super) fn names(&self) -> &[String] {
        &self.names
    }

    // The starting values, in slot order.
    pub(super) fn initial(&self) -> Vec<Val> {
        self.inits.clone()
    }
}

// Compile-time name scope, mirroring the world crate's checker.
struct Names<'a> {
    locals: &'a [String],
    queries: &'a [String],
    bindings: Vec<String>,
    // High-water mark of concurrently live bindings, which sizes the frame.
    peak: usize,
}

impl Names<'_> {
    fn bind(&mut self, name: &str) -> u16 {
        self.bindings.push(name.to_string());
        self.peak = self.peak.max(self.bindings.len());
        (self.bindings.len() - 1) as u16
    }

    fn binding(&self, name: &str) -> Option<u16> {
        self.bindings
            .iter()
            .rposition(|n| n == name)
            .map(|i| i as u16)
    }

    fn local(&self, name: &str) -> Option<u16> {
        self.locals.iter().position(|n| n == name).map(|i| i as u16)
    }

    fn query(&self, name: &str) -> Option<u16> {
        self.queries
            .iter()
            .position(|n| n == name)
            .map(|i| i as u16)
    }
}

pub(super) fn compile(def: Behavior, vars: &mut VarTable) -> Program {
    let scope: Vec<u8> = def
        .scope
        .iter()
        .filter_map(|c| ComponentTag::parse(c))
        .map(|t| t as u8)
        .collect();
    let local_names: Vec<String> = def.locals.iter().map(|l| l.name.clone()).collect();
    let local_inits: Vec<Val> = def
        .locals
        .iter()
        .map(|l| Val::from_literal(&l.value))
        .collect();
    let query_names: Vec<String> = def.queries.iter().map(|q| q.name.clone()).collect();
    let queries: Vec<Vec<u8>> = def
        .queries
        .iter()
        .map(|q| {
            q.has
                .iter()
                .filter_map(|c| ComponentTag::parse(c))
                .map(|t| t as u8)
                .collect()
        })
        .collect();

    // A variable source names a variable that must have a slot even if no node
    // ever writes it.
    if let BehaviorSource::Variable(name) = &def.on {
        vars.intern(name);
    }

    let mut names = Names {
        locals: &local_names,
        queries: &query_names,
        bindings: Vec::new(),
        peak: 0,
    };
    let mut paths = Vec::new();
    let body = compile_nodes(
        &def.body,
        &mut names,
        vars,
        &[TraceStep::Field("do")],
        &mut paths,
    );
    let bindings = names.peak;

    Program {
        def,
        scope,
        local_inits,
        queries,
        body,
        paths,
        bindings,
    }
}

// A branch list's path base: the parent node's path plus the verb and branch
// keys the authored JSON nests it under (matching the world checker's fault
// paths, so a traced node lands on the same row / card a fault would).
fn branch(path: &[TraceStep], verb: &'static str, list: &'static str) -> Vec<TraceStep> {
    let mut base = path.to_vec();
    base.push(TraceStep::Field(verb));
    base.push(TraceStep::Field(list));
    base
}

fn compile_nodes(
    nodes: &[Node],
    names: &mut Names<'_>,
    vars: &mut VarTable,
    base: &[TraceStep],
    paths: &mut Vec<TracePath>,
) -> Vec<CNode> {
    let depth = names.bindings.len();
    let out = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let mut path = base.to_vec();
            path.push(TraceStep::Index(i as u32));
            // Pre-order: the node claims its id (and path slot) before its
            // branches compile, so ids read top-down.
            let id = paths.len() as u32;
            paths.push(path.clone());
            let op = compile_node(n, names, vars, &path, paths);
            CNode { id, op }
        })
        .collect();
    names.bindings.truncate(depth);
    out
}

fn compile_node(
    node: &Node,
    names: &mut Names<'_>,
    vars: &mut VarTable,
    path: &[TraceStep],
    paths: &mut Vec<TracePath>,
) -> COp {
    match node {
        Node::If {
            cond,
            then,
            otherwise,
        } => COp::If {
            cond: compile_expr(cond, names, vars),
            then: compile_nodes(then, names, vars, &branch(path, "if", "then"), paths),
            otherwise: compile_nodes(otherwise, names, vars, &branch(path, "if", "else"), paths),
        },
        Node::ForEach { query, bind, body } => {
            let Some(query) = names.query(query) else {
                return COp::Never;
            };
            let depth = names.bindings.len();
            let bind = names.bind(bind);
            let body = compile_nodes(body, names, vars, &branch(path, "for_each", "do"), paths);
            names.bindings.truncate(depth);
            COp::ForEach { query, bind, body }
        }
        Node::Let { name, value } => {
            let value = compile_expr(value, names, vars);
            COp::Let {
                bind: names.bind(name),
                value,
            }
        }
        Node::Set { var, value, add } => COp::SetVar {
            slot: vars.intern(var),
            value: compile_expr(value, names, vars),
            add: *add,
        },
        Node::SetLocal { local, value, add } => match names.local(local) {
            Some(slot) => COp::SetLocal {
                slot,
                value: compile_expr(value, names, vars),
                add: *add,
            },
            None => COp::Never,
        },
        Node::SetTransform {
            entity,
            position,
            rotation_deg,
            scale,
        } => COp::SetTransform {
            entity: compile_expr(entity, names, vars),
            position: position.as_ref().map(|e| compile_expr(e, names, vars)),
            rotation_deg: rotation_deg.as_ref().map(|e| compile_expr(e, names, vars)),
            scale: scale.as_ref().map(|e| compile_expr(e, names, vars)),
        },
        Node::Spawn {
            template,
            position,
            rotation_deg,
            scale,
            lifetime,
            bind,
        } => match template {
            Some(template) => COp::Spawn {
                template: *template,
                position: *position,
                rotation_deg: *rotation_deg,
                // A zero scale would make the copy invisible; treat it as unit
                // scale, like the spawn request path.
                scale: if *scale == [0.0; 3] { [1.0; 3] } else { *scale },
                lifetime: *lifetime,
                bind: bind.as_ref().map(|b| names.bind(b)),
            },
            None => COp::Never,
        },
        Node::Despawn { target } => COp::Despawn(compile_expr(target, names, vars)),
        Node::Reparent { child, parent } => COp::Reparent {
            child: compile_expr(child, names, vars),
            parent: parent.as_ref().map(|e| compile_expr(e, names, vars)),
        },
        Node::Show { target } => COp::Visible(compile_expr(target, names, vars), true),
        Node::Hide { target } => COp::Visible(compile_expr(target, names, vars), false),
        Node::Sound { clip, kind, volume } => match clip {
            Some(clip) => COp::Sound {
                clip: *clip,
                kind: *kind,
                volume: *volume,
            },
            None => COp::Never,
        },
        Node::Scene { scene, transition } => match scene {
            Some(scene) => COp::Scene {
                scene: *scene,
                transition: transition.clone(),
            },
            None => COp::Never,
        },
        Node::Screen { screen } => match screen {
            Some(screen) => COp::Screen(*screen),
            None => COp::Never,
        },
        Node::Story(playback) => COp::Story(*playback),
        Node::Save => COp::Save,
    }
}

fn compile_expr(expr: &Expr, names: &mut Names<'_>, vars: &mut VarTable) -> CExpr {
    let binary = |a: &Expr, b: &Expr, names: &mut Names<'_>, vars: &mut VarTable| {
        (
            Box::new(compile_expr(a, names, vars)),
            Box::new(compile_expr(b, names, vars)),
        )
    };
    match expr {
        Expr::Bool(b) => CExpr::Lit(Val::Bool(*b)),
        Expr::Int(i) => CExpr::Lit(Val::Int(*i)),
        Expr::Float(f) => CExpr::Lit(Val::Float(*f)),
        Expr::Vec3(v) => CExpr::Lit(Val::Vec3(*v)),
        Expr::Var(name) => CExpr::Var(vars.intern(name)),
        Expr::Local(name) => names.local(name).map_or(CExpr::Never, CExpr::Local),
        Expr::Bind(name) => names.binding(name).map_or(CExpr::Never, CExpr::Bind),
        Expr::Named(id) => id.map_or(CExpr::Never, CExpr::Named),
        Expr::SelfEntity => CExpr::SelfEntity,
        Expr::Dt => CExpr::Dt,
        Expr::Elapsed => CExpr::Elapsed,
        Expr::Position(e) => CExpr::Position(Box::new(compile_expr(e, names, vars))),
        Expr::Alive(e) => CExpr::Alive(Box::new(compile_expr(e, names, vars))),
        Expr::Normalize(e) => CExpr::Normalize(Box::new(compile_expr(e, names, vars))),
        Expr::Not(e) => CExpr::Not(Box::new(compile_expr(e, names, vars))),
        Expr::Distance(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Distance(a, b)
        }
        Expr::First(q) => names.query(q).map_or(CExpr::Never, CExpr::First),
        Expr::Count(q) => names.query(q).map_or(CExpr::Never, CExpr::Count),
        Expr::Add(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Arith(Arith::Add, a, b)
        }
        Expr::Sub(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Arith(Arith::Sub, a, b)
        }
        Expr::Mul(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Arith(Arith::Mul, a, b)
        }
        Expr::Div(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Arith(Arith::Div, a, b)
        }
        Expr::Eq(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Compare(Cmp::Eq, a, b)
        }
        Expr::Ne(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Compare(Cmp::Ne, a, b)
        }
        Expr::Lt(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Compare(Cmp::Lt, a, b)
        }
        Expr::Le(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Compare(Cmp::Le, a, b)
        }
        Expr::Gt(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Compare(Cmp::Gt, a, b)
        }
        Expr::Ge(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Compare(Cmp::Ge, a, b)
        }
        Expr::All(items) => {
            CExpr::All(items.iter().map(|e| compile_expr(e, names, vars)).collect())
        }
        Expr::Any(items) => {
            CExpr::Any(items.iter().map(|e| compile_expr(e, names, vars)).collect())
        }
    }
}
