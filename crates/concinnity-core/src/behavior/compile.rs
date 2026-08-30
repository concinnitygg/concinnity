// Name resolution: the authored body walked once, every name replaced by the
// slot that holds its value at tick time.
//
// Compilation is total. The world crate's checker has already rejected unknown
// names and mistyped expressions, so anything unresolvable here becomes
// `CExpr::Never` or `COp::Never`, which evaluates to nothing and skips its node
// rather than panicking.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::behavior::program::{CExpr, CNode, COp, Program, VarTable};
use crate::behavior::value::{Arith, Cmp, Val};
use crate::components::{Behavior, BehaviorExpr, BehaviorNode, BehaviorSource};
use crate::ecs::{ComponentTag, TracePath, TraceStep};

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

// Resolve an authored component name to the tag that still holds its entities
// at tick time. A load-time pass drains some columns during `World::start`;
// `surviving_tag` maps those to the runtime component that replaces them
// (`Prop` -> `PropInstance`) and drops the ones nothing replaces, which the
// world checker rejects before a build gets here.
fn surviving_tag(name: &str) -> Option<u8> {
    ComponentTag::parse(name)
        .and_then(ComponentTag::surviving_tag)
        .map(|t| t as u8)
}

/// Compile one authored behavior against the world's shared variable table,
/// interning any variable name it mentions that no `Variables` asset declared.
pub fn compile(def: Behavior, vars: &mut VarTable) -> Program {
    let scope: Vec<u8> = def.scope.iter().filter_map(|c| surviving_tag(c)).collect();
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
        .map(|q| q.has.iter().filter_map(|c| surviving_tag(c)).collect())
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
    nodes: &[BehaviorNode],
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
    node: &BehaviorNode,
    names: &mut Names<'_>,
    vars: &mut VarTable,
    path: &[TraceStep],
    paths: &mut Vec<TracePath>,
) -> COp {
    match node {
        BehaviorNode::If {
            cond,
            then,
            otherwise,
        } => COp::If {
            cond: compile_expr(cond, names, vars),
            then: compile_nodes(then, names, vars, &branch(path, "if", "then"), paths),
            otherwise: compile_nodes(otherwise, names, vars, &branch(path, "if", "else"), paths),
        },
        BehaviorNode::ForEach { query, bind, body } => {
            let Some(query) = names.query(query) else {
                return COp::Never;
            };
            let depth = names.bindings.len();
            let bind = names.bind(bind);
            let body = compile_nodes(body, names, vars, &branch(path, "for_each", "do"), paths);
            names.bindings.truncate(depth);
            COp::ForEach { query, bind, body }
        }
        BehaviorNode::Let { name, value } => {
            let value = compile_expr(value, names, vars);
            COp::Let {
                bind: names.bind(name),
                value,
            }
        }
        BehaviorNode::Set { var, value, add } => COp::SetVar {
            slot: vars.intern(var),
            value: compile_expr(value, names, vars),
            add: *add,
        },
        BehaviorNode::SetLocal { local, value, add } => match names.local(local) {
            Some(slot) => COp::SetLocal {
                slot,
                value: compile_expr(value, names, vars),
                add: *add,
            },
            None => COp::Never,
        },
        BehaviorNode::SetTransform {
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
        BehaviorNode::Spawn {
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
        BehaviorNode::Despawn { target } => COp::Despawn(compile_expr(target, names, vars)),
        BehaviorNode::Reparent { child, parent } => COp::Reparent {
            child: compile_expr(child, names, vars),
            parent: parent.as_ref().map(|e| compile_expr(e, names, vars)),
        },
        BehaviorNode::Show { target } => COp::Visible(compile_expr(target, names, vars), true),
        BehaviorNode::Hide { target } => COp::Visible(compile_expr(target, names, vars), false),
        BehaviorNode::Sound { clip, kind, volume } => match clip {
            Some(clip) => COp::Sound {
                clip: *clip,
                kind: *kind,
                volume: *volume,
            },
            None => COp::Never,
        },
        BehaviorNode::Scene { scene, transition } => match scene {
            Some(scene) => COp::Scene {
                scene: *scene,
                transition: transition.clone(),
            },
            None => COp::Never,
        },
        BehaviorNode::Screen { screen } => match screen {
            Some(screen) => COp::Screen(*screen),
            None => COp::Never,
        },
        BehaviorNode::Story(playback) => COp::Story(*playback),
        BehaviorNode::Save => COp::Save,
    }
}

fn compile_expr(expr: &BehaviorExpr, names: &mut Names<'_>, vars: &mut VarTable) -> CExpr {
    let binary =
        |a: &BehaviorExpr, b: &BehaviorExpr, names: &mut Names<'_>, vars: &mut VarTable| {
            (
                Box::new(compile_expr(a, names, vars)),
                Box::new(compile_expr(b, names, vars)),
            )
        };
    match expr {
        BehaviorExpr::Bool(b) => CExpr::Lit(Val::Bool(*b)),
        BehaviorExpr::Int(i) => CExpr::Lit(Val::Int(*i)),
        BehaviorExpr::Float(f) => CExpr::Lit(Val::Float(*f)),
        BehaviorExpr::Vec3(v) => CExpr::Lit(Val::Vec3(*v)),
        BehaviorExpr::Var(name) => CExpr::Var(vars.intern(name)),
        BehaviorExpr::Local(name) => names.local(name).map_or(CExpr::Never, CExpr::Local),
        BehaviorExpr::Bind(name) => names.binding(name).map_or(CExpr::Never, CExpr::Bind),
        BehaviorExpr::Named(id) => id.map_or(CExpr::Never, CExpr::Named),
        BehaviorExpr::SelfEntity => CExpr::SelfEntity,
        BehaviorExpr::Dt => CExpr::Dt,
        BehaviorExpr::Elapsed => CExpr::Elapsed,
        BehaviorExpr::Position(e) => CExpr::Position(Box::new(compile_expr(e, names, vars))),
        BehaviorExpr::Alive(e) => CExpr::Alive(Box::new(compile_expr(e, names, vars))),
        BehaviorExpr::Normalize(e) => CExpr::Normalize(Box::new(compile_expr(e, names, vars))),
        BehaviorExpr::Not(e) => CExpr::Not(Box::new(compile_expr(e, names, vars))),
        BehaviorExpr::Distance(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Distance(a, b)
        }
        BehaviorExpr::First(q) => names.query(q).map_or(CExpr::Never, CExpr::First),
        BehaviorExpr::Count(q) => names.query(q).map_or(CExpr::Never, CExpr::Count),
        BehaviorExpr::Add(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Arith(Arith::Add, a, b)
        }
        BehaviorExpr::Sub(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Arith(Arith::Sub, a, b)
        }
        BehaviorExpr::Mul(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Arith(Arith::Mul, a, b)
        }
        BehaviorExpr::Div(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Arith(Arith::Div, a, b)
        }
        BehaviorExpr::Eq(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Compare(Cmp::Eq, a, b)
        }
        BehaviorExpr::Ne(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Compare(Cmp::Ne, a, b)
        }
        BehaviorExpr::Lt(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Compare(Cmp::Lt, a, b)
        }
        BehaviorExpr::Le(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Compare(Cmp::Le, a, b)
        }
        BehaviorExpr::Gt(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Compare(Cmp::Gt, a, b)
        }
        BehaviorExpr::Ge(a, b) => {
            let (a, b) = binary(a, b, names, vars);
            CExpr::Compare(Cmp::Ge, a, b)
        }
        BehaviorExpr::All(items) => {
            CExpr::All(items.iter().map(|e| compile_expr(e, names, vars)).collect())
        }
        BehaviorExpr::Any(items) => {
            CExpr::Any(items.iter().map(|e| compile_expr(e, names, vars)).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{BehaviorLiteral, BehaviorLocal, BehaviorQuery, CueKind};
    use crate::ecs::AudioClipHandle;
    use crate::ecs::asset_id::AssetId;
    use alloc::vec;

    // A behavior with one local, one query over props, and the given body, so
    // a test states only the nodes it is about.
    fn behavior(body: Vec<BehaviorNode>) -> Behavior {
        Behavior {
            locals: vec![BehaviorLocal {
                name: String::from("hp"),
                value: BehaviorLiteral::Int(3),
            }],
            queries: vec![BehaviorQuery {
                name: String::from("props"),
                has: vec![String::from("Prop")],
            }],
            body,
            ..Behavior::default()
        }
    }

    fn compiled(body: Vec<BehaviorNode>) -> Program {
        compile(behavior(body), &mut VarTable::default())
    }

    // The single compiled op a one-node body produces.
    fn op(node: BehaviorNode) -> COp {
        let mut body = compiled(vec![node]).body;
        assert_eq!(body.len(), 1, "expected exactly one compiled node");
        body.remove(0).op
    }

    fn expr(e: BehaviorExpr) -> CExpr {
        match op(BehaviorNode::Despawn { target: e }) {
            COp::Despawn(e) => e,
            other => panic!("expected a despawn, got {other:?}"),
        }
    }

    fn int(i: i32) -> Box<BehaviorExpr> {
        Box::new(BehaviorExpr::Int(i))
    }

    // Compilation is total: the world checker has already rejected these, so a
    // name that does not resolve becomes a node that does nothing rather than
    // a panic.
    #[test]
    fn an_unresolvable_name_compiles_to_a_node_that_does_nothing() {
        assert!(matches!(
            op(BehaviorNode::ForEach {
                query: String::from("undeclared"),
                bind: String::from("it"),
                body: Vec::new(),
            }),
            COp::Never
        ));
        assert!(matches!(
            op(BehaviorNode::SetLocal {
                local: String::from("undeclared"),
                value: BehaviorExpr::Int(1),
                add: false,
            }),
            COp::Never
        ));
        assert!(matches!(
            op(BehaviorNode::Spawn {
                template: None,
                position: [0.0; 3],
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
                lifetime: 0.0,
                bind: None,
            }),
            COp::Never
        ));
        assert!(matches!(
            op(BehaviorNode::Sound {
                clip: None,
                kind: CueKind::Sound,
                volume: 1.0,
            }),
            COp::Never
        ));
        assert!(matches!(
            op(BehaviorNode::Scene {
                scene: None,
                transition: String::from("Cut"),
            }),
            COp::Never
        ));
        assert!(matches!(
            op(BehaviorNode::Screen { screen: None }),
            COp::Never
        ));
    }

    #[test]
    fn a_resolvable_name_compiles_to_the_node_it_denotes() {
        assert!(matches!(
            op(BehaviorNode::Sound {
                clip: Some(AudioClipHandle(2)),
                kind: CueKind::Music,
                volume: 0.5,
            }),
            COp::Sound {
                clip: AudioClipHandle(2),
                kind: CueKind::Music,
                volume: 0.5,
            }
        ));
        assert!(matches!(
            op(BehaviorNode::Scene {
                scene: Some(AssetId(4)),
                transition: String::from("FadeBlack"),
            }),
            COp::Scene {
                scene: AssetId(4),
                ..
            }
        ));
        assert!(matches!(
            op(BehaviorNode::Screen {
                screen: Some(AssetId(5)),
            }),
            COp::Screen(AssetId(5))
        ));
    }

    // A zero scale would make the copy invisible, so it reads as the
    // template's own size instead.
    #[test]
    fn a_zero_scaled_spawn_compiles_to_unit_scale() {
        let spawn = |scale| {
            op(BehaviorNode::Spawn {
                template: Some(AssetId(1)),
                position: [1.0, 2.0, 3.0],
                rotation_deg: [0.0; 3],
                scale,
                lifetime: 2.0,
                bind: Some(String::from("copy")),
            })
        };
        let COp::Spawn { scale, bind, .. } = spawn([0.0; 3]) else {
            panic!("expected a spawn");
        };
        assert_eq!(scale, [1.0; 3]);
        assert_eq!(bind, Some(0));

        let COp::Spawn { scale, .. } = spawn([2.0; 3]) else {
            panic!("expected a spawn");
        };
        assert_eq!(scale, [2.0; 3]);
    }

    #[test]
    fn show_and_hide_compile_to_the_same_node_with_opposite_visibility() {
        let target = || BehaviorExpr::Named(Some(AssetId(1)));
        assert!(matches!(
            op(BehaviorNode::Show { target: target() }),
            COp::Visible(_, true)
        ));
        assert!(matches!(
            op(BehaviorNode::Hide { target: target() }),
            COp::Visible(_, false)
        ));
    }

    #[test]
    fn a_reparent_compiles_both_its_parent_forms() {
        assert!(matches!(
            op(BehaviorNode::Reparent {
                child: BehaviorExpr::SelfEntity,
                parent: Some(BehaviorExpr::Named(Some(AssetId(1)))),
            }),
            COp::Reparent {
                parent: Some(_),
                ..
            }
        ));
        assert!(matches!(
            op(BehaviorNode::Reparent {
                child: BehaviorExpr::SelfEntity,
                parent: None,
            }),
            COp::Reparent { parent: None, .. }
        ));
    }

    #[test]
    fn a_local_and_the_tick_readings_compile_to_their_slots() {
        assert!(matches!(
            expr(BehaviorExpr::Local(String::from("hp"))),
            CExpr::Local(0)
        ));
        assert!(matches!(
            expr(BehaviorExpr::Local(String::from("undeclared"))),
            CExpr::Never
        ));
        assert!(matches!(expr(BehaviorExpr::Dt), CExpr::Dt));
        assert!(matches!(expr(BehaviorExpr::Elapsed), CExpr::Elapsed));
    }

    #[test]
    fn the_unary_expressions_compile_to_their_operators() {
        assert!(matches!(
            expr(BehaviorExpr::Normalize(int(1))),
            CExpr::Normalize(_)
        ));
        assert!(matches!(expr(BehaviorExpr::Not(int(1))), CExpr::Not(_)));
    }

    #[test]
    fn every_arithmetic_operator_compiles_to_its_own_kind() {
        let arith = |e| match expr(e) {
            CExpr::Arith(op, _, _) => op,
            other => panic!("expected arithmetic, got {other:?}"),
        };
        assert!(matches!(
            arith(BehaviorExpr::Add(int(1), int(2))),
            Arith::Add
        ));
        assert!(matches!(
            arith(BehaviorExpr::Sub(int(1), int(2))),
            Arith::Sub
        ));
        assert!(matches!(
            arith(BehaviorExpr::Mul(int(1), int(2))),
            Arith::Mul
        ));
        assert!(matches!(
            arith(BehaviorExpr::Div(int(1), int(2))),
            Arith::Div
        ));
    }

    #[test]
    fn every_comparison_operator_compiles_to_its_own_kind() {
        let cmp = |e| match expr(e) {
            CExpr::Compare(op, _, _) => op,
            other => panic!("expected a comparison, got {other:?}"),
        };
        assert!(matches!(cmp(BehaviorExpr::Eq(int(1), int(2))), Cmp::Eq));
        assert!(matches!(cmp(BehaviorExpr::Ne(int(1), int(2))), Cmp::Ne));
        assert!(matches!(cmp(BehaviorExpr::Lt(int(1), int(2))), Cmp::Lt));
        assert!(matches!(cmp(BehaviorExpr::Le(int(1), int(2))), Cmp::Le));
        assert!(matches!(cmp(BehaviorExpr::Gt(int(1), int(2))), Cmp::Gt));
        assert!(matches!(cmp(BehaviorExpr::Ge(int(1), int(2))), Cmp::Ge));
    }

    #[test]
    fn the_variadic_expressions_compile_each_operand() {
        let items = || vec![BehaviorExpr::Bool(true), BehaviorExpr::Dt];
        let CExpr::All(all) = expr(BehaviorExpr::All(items())) else {
            panic!("expected an all");
        };
        assert_eq!(all.len(), 2);
        let CExpr::Any(any) = expr(BehaviorExpr::Any(items())) else {
            panic!("expected an any");
        };
        assert_eq!(any.len(), 2);
    }
}
