// Evaluating a compiled body: expressions read the world through a per-tick
// view, nodes append to an effect buffer the system drains onto the runtime's
// request queues.
//
// Nothing here mutates the world. An expression that cannot produce a value
// (an empty query's `first`, an unresolved name, a despawned entity) yields
// None, and the node holding it is skipped rather than guessing.

use crate::assets::Transform;
use crate::ecs::{Entity, asset_id::AssetId};

use super::program::{Arith, CExpr, CNode, COp, Cmp, Val};

// What a behavior may read this tick.
pub(super) struct View<'a> {
    pub(super) dt: f32,
    pub(super) elapsed: f32,
    pub(super) vars: &'a [Val],
    pub(super) locals: &'a [Val],
    // Exactly as wide as the body's binding high-water mark, so a slot is
    // always in range and the frame never grows mid-run.
    pub(super) bindings: &'a mut [Option<Val>],
    pub(super) queries: &'a [Vec<Entity>],
    pub(super) by_name: &'a dyn Fn(AssetId) -> Option<Entity>,
    pub(super) transforms: &'a dyn Fn(Entity) -> Option<Transform>,
    pub(super) alive: &'a dyn Fn(Entity) -> bool,
    pub(super) self_entity: Option<Entity>,
    // BehaviorNode ids executed this run, recorded only while tracing is requested
    // (`None` costs one branch per node).
    pub(super) trace: &'a mut Option<Vec<u32>>,
}

// One world change a behavior asked for, in body order.
#[derive(Debug, Clone)]
pub(super) enum Effect {
    SetVar {
        slot: u16,
        value: Val,
        add: bool,
    },
    SetLocal {
        slot: u16,
        value: Val,
        add: bool,
    },
    SetTransform {
        entity: Entity,
        transform: Transform,
    },
    Spawn(SpawnEffect),
    Despawn(Entity),
    Reparent {
        child: Entity,
        parent: Option<Entity>,
    },
    Visible(Entity, bool),
    Sound(crate::assets::PlayCue),
    Scene {
        scene: AssetId,
        transition: String,
    },
    Screen(AssetId),
    Story(crate::assets::StoryPlayback),
    Save,
}

#[derive(Debug, Clone)]
pub(super) struct SpawnEffect {
    pub(super) template: AssetId,
    pub(super) transform: Transform,
    pub(super) lifetime: Option<f32>,
}

pub(super) fn eval(expr: &CExpr, view: &View<'_>) -> Option<Val> {
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
            Some(Val::Float((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()))
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
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
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

pub(super) fn exec(nodes: &[CNode], view: &mut View<'_>, out: &mut Vec<Effect>) {
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
        COp::Sound { clip, kind, volume } => out.push(Effect::Sound(crate::assets::PlayCue {
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
