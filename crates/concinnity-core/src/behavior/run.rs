// Evaluating a compiled body: expressions read the world through a per-tick
// view, nodes append to an effect buffer the caller drains onto the runtime's
// request queues.
//
// Nothing here mutates the world. An expression that cannot produce a value
// (an empty query's `first`, an unresolved name, a despawned entity) yields
// None, and the node holding it is skipped rather than guessing.

use alloc::string::String;
use alloc::vec::Vec;

use crate::behavior::program::{CExpr, CNode, COp};
use crate::behavior::value::{Arith, Cmp, Val};
use crate::components::{PlayCue, StoryPlayback, Transform};
use crate::ecs::{Entity, asset_id::AssetId};
use crate::math::sqrt;

/// What a behavior may read this tick.
pub struct View<'a> {
    /// Seconds this tick advances.
    pub dt: f32,
    /// Seconds of simulated time so far.
    pub elapsed: f32,
    /// The world's variables, in slot order.
    pub vars: &'a [Val],
    /// This instance's locals, in slot order.
    pub locals: &'a [Val],
    /// Exactly as wide as the body's binding high-water mark, so a slot is
    /// always in range and the frame never grows mid-run.
    pub bindings: &'a mut [Option<Val>],
    /// The entities each declared query selected this tick, in slot order.
    pub queries: &'a [Vec<Entity>],
    /// Resolves a name to the entity carrying it.
    pub by_name: &'a dyn Fn(AssetId) -> Option<Entity>,
    /// Reads an entity's transform.
    pub transforms: &'a dyn Fn(Entity) -> Option<Transform>,
    /// Whether an entity still exists.
    pub alive: &'a dyn Fn(Entity) -> bool,
    /// The entity this run is scoped to, if any.
    pub self_entity: Option<Entity>,
    /// Node ids executed this run, recorded only while tracing is requested
    /// (`None` costs one branch per node).
    pub trace: &'a mut Option<Vec<u32>>,
}

/// One world change a behavior asked for, in body order.
#[derive(Debug, Clone)]
pub enum Effect {
    /// Write a world variable.
    SetVar {
        /// The variable's slot.
        slot: u16,
        /// The value produced.
        value: Val,
        /// Add to the current value rather than replacing it.
        add: bool,
    },
    /// Write one of the instance's locals.
    SetLocal {
        /// The local's slot.
        slot: u16,
        /// The value produced.
        value: Val,
        /// Add to the current value rather than replacing it.
        add: bool,
    },
    /// Replace an entity's transform.
    SetTransform {
        /// The entity to move.
        entity: Entity,
        /// Its new transform.
        transform: Transform,
    },
    /// Copy a template into the world.
    Spawn(SpawnEffect),
    /// Remove an entity.
    Despawn(Entity),
    /// Move an entity under another parent, or to the root.
    Reparent {
        /// The entity to move.
        child: Entity,
        /// Its new parent, or `None` for the root.
        parent: Option<Entity>,
    },
    /// Show or hide an entity.
    Visible(Entity, bool),
    /// Play an audio cue.
    Sound(PlayCue),
    /// Load a scene.
    Scene {
        /// The scene to load.
        scene: AssetId,
        /// The transition to play.
        transition: String,
    },
    /// Show a screen.
    Screen(AssetId),
    /// Drive story playback.
    Story(StoryPlayback),
    /// Persist the world's behavior state.
    Save,
}

/// A requested copy of a template.
#[derive(Debug, Clone)]
pub struct SpawnEffect {
    /// The template to copy.
    pub template: AssetId,
    /// Where the copy starts.
    pub transform: Transform,
    /// Seconds the copy lives, or `None` for no limit.
    pub lifetime: Option<f32>,
}

fn eval(expr: &CExpr, view: &View<'_>) -> Option<Val> {
    match expr {
        CExpr::Lit(v) => Some(*v),
        CExpr::Var(slot) => view.vars.get(*slot as usize).copied(),
        CExpr::Local(slot) => view.locals.get(*slot as usize).copied(),
        CExpr::Bind(slot) => view.bindings.get(*slot as usize).copied().flatten(),
        CExpr::Named(id) => (view.by_name)(*id).map(Val::Entity),
        CExpr::SelfEntity => view.self_entity.map(Val::Entity),
        CExpr::Dt => Some(Val::Float(view.dt)),
        CExpr::Elapsed => Some(Val::Float(view.elapsed)),
        CExpr::Position(e) => {
            let entity = eval(e, view)?.as_entity()?;
            Some(Val::Vec3((view.transforms)(entity)?.position))
        }
        CExpr::Alive(e) => {
            // An expression that yields no entity at all is not alive.
            let alive = eval(e, view)
                .and_then(Val::as_entity)
                .is_some_and(view.alive);
            Some(Val::Bool(alive))
        }
        CExpr::Distance(a, b) => {
            let a = (view.transforms)(eval(a, view)?.as_entity()?)?.position;
            let b = (view.transforms)(eval(b, view)?.as_entity()?)?.position;
            let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
            Some(Val::Float(sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])))
        }
        CExpr::First(slot) => view
            .queries
            .get(*slot as usize)?
            .first()
            .copied()
            .map(Val::Entity),
        CExpr::Count(slot) => Some(Val::Int(view.queries.get(*slot as usize)?.len() as i32)),
        CExpr::Normalize(e) => {
            let v = eval(e, view)?.as_vec3()?;
            let len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            Some(Val::Vec3(if len > f32::EPSILON {
                [v[0] / len, v[1] / len, v[2] / len]
            } else {
                [0.0; 3]
            }))
        }
        CExpr::Arith(op, a, b) => arith(*op, eval(a, view)?, eval(b, view)?),
        CExpr::Compare(op, a, b) => compare(*op, eval(a, view)?, eval(b, view)?),
        CExpr::Not(e) => Some(Val::Bool(!eval(e, view)?.as_bool()?)),
        CExpr::All(items) => {
            for item in items {
                if !eval(item, view)?.as_bool()? {
                    return Some(Val::Bool(false));
                }
            }
            Some(Val::Bool(true))
        }
        CExpr::Any(items) => {
            for item in items {
                if eval(item, view)?.as_bool()? {
                    return Some(Val::Bool(true));
                }
            }
            Some(Val::Bool(false))
        }
        CExpr::Never => None,
    }
}

fn arith(op: Arith, a: Val, b: Val) -> Option<Val> {
    let scalar = |x: f32, y: f32| match op {
        Arith::Add => x + y,
        Arith::Sub => x - y,
        Arith::Mul => x * y,
        // Division by zero yields zero rather than an infinity that would
        // silently poison every downstream transform.
        Arith::Div => {
            if y.abs() > f32::EPSILON {
                x / y
            } else {
                0.0
            }
        }
    };
    match (a, b) {
        (Val::Int(x), Val::Int(y)) => Some(Val::Int(scalar(x as f32, y as f32) as i32)),
        (Val::Vec3(x), Val::Vec3(y)) => Some(Val::Vec3([
            scalar(x[0], y[0]),
            scalar(x[1], y[1]),
            scalar(x[2], y[2]),
        ])),
        (Val::Vec3(v), other) => {
            let s = other.as_f32()?;
            Some(Val::Vec3([
                scalar(v[0], s),
                scalar(v[1], s),
                scalar(v[2], s),
            ]))
        }
        (other, Val::Vec3(v)) => {
            let s = other.as_f32()?;
            Some(Val::Vec3([
                scalar(s, v[0]),
                scalar(s, v[1]),
                scalar(s, v[2]),
            ]))
        }
        (x, y) => Some(Val::Float(scalar(x.as_f32()?, y.as_f32()?))),
    }
}

fn compare(op: Cmp, a: Val, b: Val) -> Option<Val> {
    let equal = match (a, b) {
        (Val::Entity(x), Val::Entity(y)) => x == y,
        (Val::Bool(x), Val::Bool(y)) => x == y,
        (x, y) => x.as_f32()? == y.as_f32()?,
    };
    Some(Val::Bool(match op {
        Cmp::Eq => equal,
        Cmp::Ne => !equal,
        Cmp::Lt => a.as_f32()? < b.as_f32()?,
        Cmp::Le => a.as_f32()? <= b.as_f32()?,
        Cmp::Gt => a.as_f32()? > b.as_f32()?,
        Cmp::Ge => a.as_f32()? >= b.as_f32()?,
    }))
}

/// Run a compiled body against `view`, appending what it asked for to `out`.
pub fn exec(nodes: &[CNode], view: &mut View<'_>, out: &mut Vec<Effect>) {
    for node in nodes {
        exec_node(node, view, out);
    }
}

fn exec_node(node: &CNode, view: &mut View<'_>, out: &mut Vec<Effect>) {
    if let Some(t) = view.trace.as_mut() {
        t.push(node.id);
    }
    match &node.op {
        COp::If {
            cond,
            then,
            otherwise,
        } => {
            let Some(Val::Bool(pass)) = eval(cond, view) else {
                return;
            };
            exec(if pass { then } else { otherwise }, view, out);
        }
        COp::ForEach { query, bind, body } => {
            let Some(entities) = view.queries.get(*query as usize) else {
                return;
            };
            // Cloned so the body may read other queries (and mutate bindings)
            // while this one is iterated.
            for entity in entities.clone() {
                set_binding(view, *bind, Some(Val::Entity(entity)));
                exec(body, view, out);
            }
        }
        COp::Let { bind, value } => {
            let value = eval(value, view);
            set_binding(view, *bind, value);
        }
        COp::SetVar { slot, value, add } => {
            let Some(value) = eval(value, view) else {
                return;
            };
            out.push(Effect::SetVar {
                slot: *slot,
                value,
                add: *add,
            });
        }
        COp::SetLocal { slot, value, add } => {
            let Some(value) = eval(value, view) else {
                return;
            };
            out.push(Effect::SetLocal {
                slot: *slot,
                value,
                add: *add,
            });
        }
        COp::SetTransform {
            entity,
            position,
            rotation_deg,
            scale,
        } => {
            let Some(entity) = eval(entity, view).and_then(Val::as_entity) else {
                return;
            };
            let Some(mut transform) = (view.transforms)(entity) else {
                return;
            };
            let field = |expr: &Option<CExpr>, into: &mut [f32; 3]| {
                if let Some(expr) = expr
                    && let Some(v) = eval(expr, view).and_then(Val::as_vec3)
                {
                    *into = v;
                }
            };
            field(position, &mut transform.position);
            field(rotation_deg, &mut transform.rotation_deg);
            field(scale, &mut transform.scale);
            out.push(Effect::SetTransform { entity, transform });
        }
        COp::Spawn {
            template,
            position,
            rotation_deg,
            scale,
            lifetime,
            bind,
        } => {
            out.push(Effect::Spawn(SpawnEffect {
                template: *template,
                transform: Transform {
                    position: *position,
                    rotation_deg: *rotation_deg,
                    scale: *scale,
                },
                lifetime: (*lifetime > 0.0).then_some(*lifetime),
            }));
            // The entity does not exist until SpawnSystem applies the request,
            // so the binding holds nothing this tick and reads skip their node.
            if let Some(bind) = bind {
                set_binding(view, *bind, None);
            }
        }
        COp::Despawn(target) => {
            if let Some(entity) = eval(target, view).and_then(Val::as_entity) {
                out.push(Effect::Despawn(entity));
            }
        }
        COp::Reparent { child, parent } => {
            let Some(child) = eval(child, view).and_then(Val::as_entity) else {
                return;
            };
            // A named-but-unresolvable parent skips, so a stale reference never
            // silently detaches the child to a root.
            let parent = match parent {
                Some(expr) => match eval(expr, view).and_then(Val::as_entity) {
                    Some(entity) => Some(entity),
                    None => return,
                },
                None => None,
            };
            out.push(Effect::Reparent { child, parent });
        }
        COp::Visible(target, visible) => {
            if let Some(entity) = eval(target, view).and_then(Val::as_entity) {
                out.push(Effect::Visible(entity, *visible));
            }
        }
        COp::Sound { clip, kind, volume } => out.push(Effect::Sound(PlayCue {
            clip: *clip,
            kind: *kind,
            volume: *volume,
            priority: 0,
        })),
        COp::Scene { scene, transition } => out.push(Effect::Scene {
            scene: *scene,
            transition: transition.clone(),
        }),
        COp::Screen(screen) => out.push(Effect::Screen(*screen)),
        COp::Story(playback) => out.push(Effect::Story(*playback)),
        COp::Save => out.push(Effect::Save),
        COp::Never => {}
    }
}

fn set_binding(view: &mut View<'_>, slot: u16, value: Option<Val>) {
    if let Some(slot) = view.bindings.get_mut(slot as usize) {
        *slot = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::CueKind;
    use crate::ecs::AudioClipHandle;
    use alloc::boxed::Box;
    use alloc::vec;
    use core::num::NonZeroU32;

    fn entity(index: u32) -> Entity {
        Entity::new(index, NonZeroU32::MIN)
    }

    fn moved(position: [f32; 3]) -> Transform {
        Transform {
            position,
            ..Transform::default()
        }
    }

    // What one evaluated run reads, so a test states only the parts it is
    // about. Every list defaults empty: an entity absent from `entities` is
    // dead, one absent from `transforms` has none, and a name absent from
    // `names` resolves to nothing.
    #[derive(Default)]
    struct Run {
        dt: f32,
        elapsed: f32,
        vars: Vec<Val>,
        locals: Vec<Val>,
        queries: Vec<Vec<Entity>>,
        entities: Vec<Entity>,
        transforms: Vec<(Entity, Transform)>,
        names: Vec<(AssetId, Entity)>,
        self_entity: Option<Entity>,
        bindings: usize,
    }

    impl Run {
        fn with_view<R>(&self, body: impl FnOnce(&mut View<'_>) -> R) -> R {
            let mut bindings = vec![None; self.bindings];
            let mut trace = None;
            let mut view = View {
                dt: self.dt,
                elapsed: self.elapsed,
                vars: &self.vars,
                locals: &self.locals,
                bindings: &mut bindings,
                queries: &self.queries,
                by_name: &|id| self.names.iter().find(|(n, _)| *n == id).map(|(_, e)| *e),
                transforms: &|e| {
                    self.transforms
                        .iter()
                        .find(|(t, _)| *t == e)
                        .map(|(_, t)| *t)
                },
                alive: &|e| self.entities.contains(&e),
                self_entity: self.self_entity,
                trace: &mut trace,
            };
            body(&mut view)
        }

        fn eval(&self, expr: &CExpr) -> Option<Val> {
            self.with_view(|view| eval(expr, view))
        }

        fn exec(&self, nodes: &[CNode]) -> Vec<Effect> {
            let mut out = Vec::new();
            self.with_view(|view| exec(nodes, view, &mut out));
            out
        }
    }

    fn node(op: COp) -> CNode {
        CNode { id: 0, op }
    }

    fn lit(v: Val) -> Box<CExpr> {
        Box::new(CExpr::Lit(v))
    }

    #[test]
    fn a_local_reads_its_slot_and_an_out_of_range_one_yields_nothing() {
        let run = Run {
            locals: vec![Val::Int(4), Val::Float(1.5)],
            ..Run::default()
        };
        assert_eq!(run.eval(&CExpr::Local(1)), Some(Val::Float(1.5)));
        assert_eq!(run.eval(&CExpr::Local(9)), None);
    }

    #[test]
    fn dt_and_elapsed_read_the_tick() {
        let run = Run {
            dt: 0.25,
            elapsed: 12.0,
            ..Run::default()
        };
        assert_eq!(run.eval(&CExpr::Dt), Some(Val::Float(0.25)));
        assert_eq!(run.eval(&CExpr::Elapsed), Some(Val::Float(12.0)));
    }

    #[test]
    fn normalize_scales_to_unit_length() {
        let run = Run::default();
        let v = run
            .eval(&CExpr::Normalize(lit(Val::Vec3([0.0, 3.0, 4.0]))))
            .expect("a vector normalizes");
        assert_eq!(v, Val::Vec3([0.0, 0.6, 0.8]));
    }

    // A zero-length vector has no direction, so it normalizes to zero rather
    // than to the infinities a division by its length would produce.
    #[test]
    fn normalize_of_a_zero_vector_is_zero() {
        let run = Run::default();
        assert_eq!(
            run.eval(&CExpr::Normalize(lit(Val::Vec3([0.0; 3])))),
            Some(Val::Vec3([0.0; 3]))
        );
    }

    #[test]
    fn normalize_of_a_non_vector_yields_nothing() {
        let run = Run::default();
        assert_eq!(run.eval(&CExpr::Normalize(lit(Val::Int(3)))), None);
    }

    #[test]
    fn not_inverts_a_bool_and_rejects_anything_else() {
        let run = Run::default();
        assert_eq!(
            run.eval(&CExpr::Not(lit(Val::Bool(false)))),
            Some(Val::Bool(true))
        );
        assert_eq!(run.eval(&CExpr::Not(lit(Val::Int(1)))), None);
    }

    #[test]
    fn all_holds_only_when_every_operand_does() {
        let run = Run::default();
        let all = |items: Vec<CExpr>| run.eval(&CExpr::All(items));
        assert_eq!(all(Vec::new()), Some(Val::Bool(true)));
        assert_eq!(
            all(vec![
                CExpr::Lit(Val::Bool(true)),
                CExpr::Lit(Val::Bool(true))
            ]),
            Some(Val::Bool(true))
        );
        assert_eq!(
            all(vec![
                CExpr::Lit(Val::Bool(true)),
                CExpr::Lit(Val::Bool(false))
            ]),
            Some(Val::Bool(false))
        );
        assert_eq!(all(vec![CExpr::Lit(Val::Int(1))]), None);
    }

    #[test]
    fn any_holds_as_soon_as_one_operand_does() {
        let run = Run::default();
        let any = |items: Vec<CExpr>| run.eval(&CExpr::Any(items));
        assert_eq!(any(Vec::new()), Some(Val::Bool(false)));
        assert_eq!(
            any(vec![
                CExpr::Lit(Val::Bool(false)),
                CExpr::Lit(Val::Bool(true))
            ]),
            Some(Val::Bool(true))
        );
        assert_eq!(
            any(vec![
                CExpr::Lit(Val::Bool(false)),
                CExpr::Lit(Val::Bool(false))
            ]),
            Some(Val::Bool(false))
        );
        assert_eq!(any(vec![CExpr::Lit(Val::Int(1))]), None);
    }

    #[test]
    fn a_never_expression_yields_nothing() {
        assert_eq!(Run::default().eval(&CExpr::Never), None);
    }

    #[test]
    fn arithmetic_on_two_ints_stays_an_int() {
        let run = Run::default();
        let op = |op| run.eval(&CExpr::Arith(op, lit(Val::Int(7)), lit(Val::Int(2))));
        assert_eq!(op(Arith::Add), Some(Val::Int(9)));
        assert_eq!(op(Arith::Sub), Some(Val::Int(5)));
        assert_eq!(op(Arith::Mul), Some(Val::Int(14)));
        assert_eq!(op(Arith::Div), Some(Val::Int(3)));
    }

    // A quotient by zero yields zero rather than the infinity that would
    // silently poison every transform downstream of it.
    #[test]
    fn division_by_zero_yields_zero() {
        let run = Run::default();
        assert_eq!(
            run.eval(&CExpr::Arith(
                Arith::Div,
                lit(Val::Int(7)),
                lit(Val::Int(0))
            )),
            Some(Val::Int(0))
        );
        assert_eq!(
            run.eval(&CExpr::Arith(
                Arith::Div,
                lit(Val::Vec3([1.0, 2.0, 3.0])),
                lit(Val::Float(0.0)),
            )),
            Some(Val::Vec3([0.0; 3]))
        );
    }

    #[test]
    fn arithmetic_on_two_vectors_is_component_wise() {
        let run = Run::default();
        assert_eq!(
            run.eval(&CExpr::Arith(
                Arith::Add,
                lit(Val::Vec3([1.0, 2.0, 3.0])),
                lit(Val::Vec3([10.0, 20.0, 30.0])),
            )),
            Some(Val::Vec3([11.0, 22.0, 33.0]))
        );
    }

    #[test]
    fn a_vector_and_a_scalar_combine_component_wise_either_way_round() {
        let run = Run::default();
        assert_eq!(
            run.eval(&CExpr::Arith(
                Arith::Mul,
                lit(Val::Vec3([1.0, 2.0, 3.0])),
                lit(Val::Float(2.0)),
            )),
            Some(Val::Vec3([2.0, 4.0, 6.0]))
        );
        // Order matters for the non-commutative operators: the scalar is the
        // left operand of every component.
        assert_eq!(
            run.eval(&CExpr::Arith(
                Arith::Sub,
                lit(Val::Int(10)),
                lit(Val::Vec3([1.0, 2.0, 3.0])),
            )),
            Some(Val::Vec3([9.0, 8.0, 7.0]))
        );
    }

    #[test]
    fn a_vector_against_a_non_numeric_yields_nothing() {
        let run = Run::default();
        assert_eq!(
            run.eval(&CExpr::Arith(
                Arith::Add,
                lit(Val::Vec3([1.0; 3])),
                lit(Val::Bool(true)),
            )),
            None
        );
        assert_eq!(
            run.eval(&CExpr::Arith(
                Arith::Add,
                lit(Val::Bool(true)),
                lit(Val::Vec3([1.0; 3])),
            )),
            None
        );
    }

    #[test]
    fn a_mixed_int_and_float_widens_to_a_float() {
        let run = Run::default();
        assert_eq!(
            run.eval(&CExpr::Arith(
                Arith::Add,
                lit(Val::Int(1)),
                lit(Val::Float(0.5)),
            )),
            Some(Val::Float(1.5))
        );
        assert_eq!(
            run.eval(&CExpr::Arith(
                Arith::Add,
                lit(Val::Bool(true)),
                lit(Val::Int(1)),
            )),
            None
        );
    }

    #[test]
    fn entities_and_bools_compare_by_identity_rather_than_as_numbers() {
        let run = Run::default();
        let cmp = |op, a, b| run.eval(&CExpr::Compare(op, lit(a), lit(b)));
        let (a, b) = (Val::Entity(entity(1)), Val::Entity(entity(2)));
        assert_eq!(cmp(Cmp::Eq, a, a), Some(Val::Bool(true)));
        assert_eq!(cmp(Cmp::Eq, a, b), Some(Val::Bool(false)));
        assert_eq!(cmp(Cmp::Ne, a, b), Some(Val::Bool(true)));
        assert_eq!(
            cmp(Cmp::Eq, Val::Bool(true), Val::Bool(true)),
            Some(Val::Bool(true))
        );
        // Neither side reads as a number, so an ordering comparison of them
        // has no answer.
        assert_eq!(cmp(Cmp::Lt, a, b), None);
    }

    #[test]
    fn numbers_compare_in_every_ordering() {
        let run = Run::default();
        let cmp = |op| {
            run.eval(&CExpr::Compare(op, lit(Val::Int(1)), lit(Val::Float(2.0))))
                .and_then(Val::as_bool)
        };
        assert_eq!(cmp(Cmp::Eq), Some(false));
        assert_eq!(cmp(Cmp::Ne), Some(true));
        assert_eq!(cmp(Cmp::Lt), Some(true));
        assert_eq!(cmp(Cmp::Le), Some(true));
        assert_eq!(cmp(Cmp::Gt), Some(false));
        assert_eq!(cmp(Cmp::Ge), Some(false));
    }

    #[test]
    fn an_unevaluable_comparison_yields_nothing() {
        let run = Run::default();
        assert_eq!(
            run.eval(&CExpr::Compare(
                Cmp::Eq,
                lit(Val::Bool(true)),
                lit(Val::Int(1)),
            )),
            None
        );
    }

    #[test]
    fn a_condition_that_is_not_a_bool_runs_neither_branch() {
        let run = Run::default();
        let effects = run.exec(&[node(COp::If {
            cond: CExpr::Lit(Val::Int(1)),
            then: vec![node(COp::Save)],
            otherwise: vec![node(COp::Save)],
        })]);
        assert!(effects.is_empty(), "{effects:?}");
    }

    #[test]
    fn a_for_each_over_an_undeclared_query_runs_nothing() {
        let run = Run {
            bindings: 1,
            ..Run::default()
        };
        let effects = run.exec(&[node(COp::ForEach {
            query: 3,
            bind: 0,
            body: vec![node(COp::Save)],
        })]);
        assert!(effects.is_empty(), "{effects:?}");
    }

    #[test]
    fn setting_a_local_from_an_unevaluable_value_asks_for_nothing() {
        let run = Run::default();
        let effects = run.exec(&[node(COp::SetLocal {
            slot: 0,
            value: CExpr::Never,
            add: false,
        })]);
        assert!(effects.is_empty(), "{effects:?}");
    }

    #[test]
    fn a_transform_write_skips_an_entity_it_cannot_resolve() {
        let ghost = entity(7);
        // Resolves to no entity at all.
        let run = Run::default();
        let effects = run.exec(&[node(COp::SetTransform {
            entity: CExpr::Never,
            position: None,
            rotation_deg: None,
            scale: None,
        })]);
        assert!(effects.is_empty(), "{effects:?}");

        // Resolves to an entity that carries no transform to overwrite.
        let effects = run.exec(&[node(COp::SetTransform {
            entity: CExpr::Lit(Val::Entity(ghost)),
            position: None,
            rotation_deg: None,
            scale: None,
        })]);
        assert!(effects.is_empty(), "{effects:?}");
    }

    #[test]
    fn a_transform_write_keeps_the_fields_it_was_not_given() {
        let e = entity(1);
        let run = Run {
            transforms: vec![(e, moved([1.0, 2.0, 3.0]))],
            ..Run::default()
        };
        let effects = run.exec(&[node(COp::SetTransform {
            entity: CExpr::Lit(Val::Entity(e)),
            position: Some(CExpr::Lit(Val::Vec3([9.0; 3]))),
            // Neither an absent field nor one that evaluates to a non-vector
            // overwrites what the entity already carries.
            rotation_deg: None,
            scale: Some(CExpr::Lit(Val::Int(2))),
        })]);
        let [Effect::SetTransform { entity, transform }] = effects.as_slice() else {
            panic!("expected one transform write, got {effects:?}");
        };
        assert_eq!(*entity, e);
        assert_eq!(transform.position, [9.0; 3]);
        assert_eq!(transform.scale, Transform::default().scale);
    }

    #[test]
    fn a_reparent_skips_a_child_it_cannot_resolve() {
        let run = Run::default();
        let effects = run.exec(&[node(COp::Reparent {
            child: CExpr::Never,
            parent: None,
        })]);
        assert!(effects.is_empty(), "{effects:?}");
    }

    // A stale parent reference skips the whole node rather than detaching the
    // child to the root, which is what a `None` parent means.
    #[test]
    fn a_reparent_onto_an_unresolvable_parent_skips_rather_than_detaching() {
        let run = Run::default();
        let effects = run.exec(&[node(COp::Reparent {
            child: CExpr::Lit(Val::Entity(entity(1))),
            parent: Some(CExpr::Never),
        })]);
        assert!(effects.is_empty(), "{effects:?}");
    }

    #[test]
    fn a_reparent_moves_the_child_under_a_parent_or_to_the_root() {
        let (child, parent) = (entity(1), entity(2));
        let run = Run::default();
        let effects = run.exec(&[
            node(COp::Reparent {
                child: CExpr::Lit(Val::Entity(child)),
                parent: Some(CExpr::Lit(Val::Entity(parent))),
            }),
            node(COp::Reparent {
                child: CExpr::Lit(Val::Entity(child)),
                parent: None,
            }),
        ]);
        let [
            Effect::Reparent {
                child: a,
                parent: Some(p),
            },
            Effect::Reparent {
                child: b,
                parent: None,
            },
        ] = effects.as_slice()
        else {
            panic!("expected two reparents, got {effects:?}");
        };
        assert_eq!((*a, *p, *b), (child, parent, child));
    }

    #[test]
    fn the_request_only_nodes_each_push_what_they_name() {
        let run = Run::default();
        let effects = run.exec(&[
            node(COp::Sound {
                clip: AudioClipHandle(3),
                kind: CueKind::Music,
                volume: 0.5,
            }),
            node(COp::Scene {
                scene: AssetId(1),
                transition: String::from("fade"),
            }),
            node(COp::Screen(AssetId(2))),
            // A node the checker should have rejected asks for nothing rather
            // than guessing at what it meant.
            node(COp::Never),
        ]);
        let [
            Effect::Sound(cue),
            Effect::Scene { scene, transition },
            Effect::Screen(screen),
        ] = effects.as_slice()
        else {
            panic!("expected three requests, got {effects:?}");
        };
        assert_eq!(
            (cue.clip, cue.kind, cue.volume),
            (AudioClipHandle(3), CueKind::Music, 0.5)
        );
        assert_eq!(*scene, AssetId(1));
        assert_eq!(transition, "fade");
        assert_eq!(*screen, AssetId(2));
    }
}
