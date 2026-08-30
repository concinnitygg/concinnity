// The write phase of a tick: land one body's effects on the world.
//
// Nothing here reads the world back, so the runs apply in job order and each
// sees exactly what the one before it left.

use super::BehaviorSystem;
use super::instance::Instance;
use crate::behavior::{Effect, Val};
use crate::components::{
    DespawnRequest, PlayCue, ReparentRequest, SceneCommand, ScreenCommand, SpawnRequest,
    StoryCommand, StoryPlayback, Transform, VisibilityRequest,
};
use crate::ecs::{Entity, PipelineContext};

impl BehaviorSystem {
    // Land one body's effects. Returns whether a `save` was requested.
    pub(super) fn apply(
        &mut self,
        ctx: &mut PipelineContext,
        i: usize,
        entity: Option<Entity>,
        effects: impl Iterator<Item = Effect>,
    ) -> bool {
        let mut save_requested = false;
        for effect in effects {
            match effect {
                Effect::SetVar { slot, value, add } => {
                    if let Some(current) = self.vars.get_mut(slot as usize) {
                        *current = if add {
                            add_vals(*current, value)
                        } else {
                            value
                        };
                    }
                }
                Effect::SetLocal { slot, value, add } => {
                    let Some(instance) = self.instance_mut(i, entity) else {
                        continue;
                    };
                    let Some(current) = instance.locals.get_mut(slot as usize) else {
                        continue;
                    };
                    *current = if add {
                        add_vals(*current, value)
                    } else {
                        value
                    };
                }
                Effect::SetTransform { entity, transform } => {
                    if let Some(current) = ctx.get_mut::<Transform>(entity) {
                        *current = transform;
                    }
                }
                Effect::Spawn(spawn) => {
                    ctx.events_mut::<SpawnRequest>().send(SpawnRequest {
                        template: spawn.template,
                        name: None,
                        transform: spawn.transform,
                        lifetime_secs: spawn.lifetime,
                    });
                }
                Effect::Despawn(entity) => {
                    ctx.events_mut::<DespawnRequest>().send(DespawnRequest {
                        target: entity.into(),
                    });
                }
                Effect::Reparent { child, parent } => {
                    ctx.events_mut::<ReparentRequest>().send(ReparentRequest {
                        child: child.into(),
                        parent: parent.map(Into::into),
                    });
                }
                Effect::Visible(entity, visible) => {
                    ctx.events_mut::<VisibilityRequest>()
                        .send(VisibilityRequest {
                            target: entity.into(),
                            visible,
                        });
                }
                Effect::Sound(cue) => {
                    ctx.events_mut::<PlayCue>().send(cue);
                }
                Effect::Scene { scene, transition } => {
                    ctx.events_mut::<SceneCommand>()
                        .send(SceneCommand { scene, transition });
                }
                Effect::Screen(screen) => {
                    ctx.events_mut::<ScreenCommand>()
                        .send(ScreenCommand::Show(screen));
                }
                Effect::Story(playback) => {
                    let command = match playback {
                        StoryPlayback::Start => StoryCommand::Start,
                        StoryPlayback::Continue => StoryCommand::Continue,
                    };
                    ctx.events_mut::<StoryCommand>().send(command);
                }
                Effect::Save => save_requested = true,
            }
        }
        save_requested
    }

    fn instance_mut(&mut self, i: usize, entity: Option<Entity>) -> Option<&mut Instance> {
        self.instances[i]
            .iter_mut()
            .find(|inst| inst.entity == entity)
    }
}

// Adding to a local keeps its declared type.
fn add_vals(current: Val, delta: Val) -> Val {
    match (current, delta) {
        (Val::Int(a), b) => Val::Int(a.saturating_add(b.as_f32().unwrap_or(0.0) as i32)),
        (Val::Float(a), b) => Val::Float(a + b.as_f32().unwrap_or(0.0)),
        (Val::Vec3(a), Val::Vec3(b)) => Val::Vec3([a[0] + b[0], a[1] + b[1], a[2] + b[2]]),
        // Booleans and entities have no addition; assignment wins.
        (_, b) => b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::SpawnEffect;
    use crate::behavior::system::test_world::{TestWorld, world_with};
    use crate::components::{
        Behavior, BehaviorLiteral, BehaviorLocal, BehaviorSource, CueKind, PropInstance,
    };
    use crate::ecs::{AudioClipHandle, EventCursor, System, asset_id::AssetId};
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    // One prop-scoped behavior carrying a single integer local, over a world
    // holding one prop, ticked once so the system holds its programs and the
    // instance they run against: instances are created by the tick's resync,
    // not by init, and only a scoped one carries locals.
    fn started(local: BehaviorLiteral) -> (BehaviorSystem, TestWorld, Entity) {
        let mut world = world_with(vec![Behavior {
            on: BehaviorSource::Start,
            scope: vec![String::from("Prop")],
            locals: vec![BehaviorLocal {
                name: String::from("hp"),
                value: local,
            }],
            ..Behavior::default()
        }]);
        let prop = world.components.push_typed(PropInstance);
        let mut sys = BehaviorSystem::new();
        sys.init(&mut world.ctx());
        sys.tick(&mut world.ctx(), 0.016, 0.016);
        (sys, world, prop)
    }

    fn local(sys: &BehaviorSystem) -> Val {
        sys.instances[0][0].locals[0]
    }

    fn apply(
        sys: &mut BehaviorSystem,
        world: &mut TestWorld,
        entity: Entity,
        effects: Vec<Effect>,
    ) -> bool {
        sys.apply(&mut world.ctx(), 0, Some(entity), effects.into_iter())
    }

    fn sent<E: 'static>(world: &mut TestWorld) -> usize {
        let mut cursor = EventCursor::default();
        world
            .ctx()
            .events::<E>()
            .map(|e| e.read(&mut cursor).count())
            .unwrap_or(0)
    }

    #[test]
    fn a_local_is_assigned_or_added_to() {
        let (mut sys, mut world, prop) = started(BehaviorLiteral::Int(3));
        let write = |value, add| Effect::SetLocal {
            slot: 0,
            value,
            add,
        };

        apply(&mut sys, &mut world, prop, vec![write(Val::Int(4), false)]);
        assert_eq!(local(&sys), Val::Int(4));

        apply(&mut sys, &mut world, prop, vec![write(Val::Int(2), true)]);
        assert_eq!(local(&sys), Val::Int(6));
    }

    // A write addressed to an instance or a slot that does not exist is
    // dropped rather than landing on a neighbour.
    #[test]
    fn a_write_to_an_absent_instance_or_slot_lands_nowhere() {
        let (mut sys, mut world, prop) = started(BehaviorLiteral::Int(3));

        // No instance is scoped to the world, so this addresses none of them.
        sys.apply(
            &mut world.ctx(),
            0,
            None,
            [Effect::SetLocal {
                slot: 0,
                value: Val::Int(9),
                add: false,
            }]
            .into_iter(),
        );
        assert_eq!(local(&sys), Val::Int(3));

        apply(
            &mut sys,
            &mut world,
            prop,
            vec![Effect::SetLocal {
                slot: 7,
                value: Val::Int(9),
                add: false,
            }],
        );
        assert_eq!(local(&sys), Val::Int(3));
    }

    #[test]
    fn a_write_to_an_undeclared_variable_slot_lands_nowhere() {
        let (mut sys, mut world, prop) = started(BehaviorLiteral::Int(0));
        let before = sys.vars.clone();
        apply(
            &mut sys,
            &mut world,
            prop,
            vec![Effect::SetVar {
                slot: 99,
                value: Val::Int(1),
                add: false,
            }],
        );
        assert_eq!(sys.vars, before);
    }

    #[test]
    fn each_request_effect_sends_its_own_event() {
        let (mut sys, mut world, prop) = started(BehaviorLiteral::Int(0));
        let entity = world.components.push_typed(PropInstance);
        let saved = apply(
            &mut sys,
            &mut world,
            prop,
            vec![
                Effect::Spawn(SpawnEffect {
                    template: AssetId(1),
                    transform: Transform::default(),
                    lifetime: Some(2.0),
                }),
                Effect::Despawn(entity),
                Effect::Reparent {
                    child: entity,
                    parent: Some(entity),
                },
                Effect::Reparent {
                    child: entity,
                    parent: None,
                },
                Effect::Visible(entity, false),
                Effect::Sound(PlayCue {
                    clip: AudioClipHandle(1),
                    kind: CueKind::Sound,
                    volume: 1.0,
                    priority: 0,
                }),
                Effect::Scene {
                    scene: AssetId(2),
                    transition: String::from("Cut"),
                },
                Effect::Screen(AssetId(3)),
                Effect::Story(StoryPlayback::Start),
                Effect::Story(StoryPlayback::Continue),
                Effect::Save,
            ],
        );
        assert!(saved, "a save effect has to be reported to the caller");
        assert_eq!(sent::<SpawnRequest>(&mut world), 1);
        assert_eq!(sent::<DespawnRequest>(&mut world), 1);
        assert_eq!(sent::<ReparentRequest>(&mut world), 2);
        assert_eq!(sent::<VisibilityRequest>(&mut world), 1);
        assert_eq!(sent::<PlayCue>(&mut world), 1);
        assert_eq!(sent::<SceneCommand>(&mut world), 1);
        assert_eq!(sent::<ScreenCommand>(&mut world), 1);
        assert_eq!(sent::<StoryCommand>(&mut world), 2);
    }

    #[test]
    fn a_transform_effect_writes_the_entity_that_carries_one() {
        let (mut sys, mut world, prop) = started(BehaviorLiteral::Int(0));
        let entity = world.components.push_typed(PropInstance);
        world.components.insert_typed(entity, Transform::default());
        let moved = Transform {
            position: [1.0, 2.0, 3.0],
            ..Transform::default()
        };
        apply(
            &mut sys,
            &mut world,
            prop,
            vec![Effect::SetTransform {
                entity,
                transform: moved,
            }],
        );
        assert_eq!(
            world
                .components
                .get::<Transform>(entity)
                .map(|t| t.position),
            Some([1.0, 2.0, 3.0])
        );

        // An entity carrying no transform is left alone rather than gaining one.
        let bare = world.components.push_typed(PropInstance);
        apply(
            &mut sys,
            &mut world,
            prop,
            vec![Effect::SetTransform {
                entity: bare,
                transform: moved,
            }],
        );
        assert!(world.components.get::<Transform>(bare).is_none());
    }

    // Adding keeps the target's declared type: the delta is read through it
    // rather than widening the slot.
    #[test]
    fn addition_keeps_the_declared_type() {
        assert_eq!(add_vals(Val::Int(1), Val::Float(2.7)), Val::Int(3));
        assert_eq!(add_vals(Val::Float(1.5), Val::Int(2)), Val::Float(3.5));
        assert_eq!(
            add_vals(Val::Vec3([1.0, 2.0, 3.0]), Val::Vec3([1.0; 3])),
            Val::Vec3([2.0, 3.0, 4.0])
        );
    }

    // A delta that has no numeric reading contributes nothing rather than
    // corrupting the slot.
    #[test]
    fn a_non_numeric_delta_adds_nothing() {
        assert_eq!(add_vals(Val::Int(5), Val::Bool(true)), Val::Int(5));
        assert_eq!(add_vals(Val::Float(1.5), Val::Bool(true)), Val::Float(1.5));
    }

    // Booleans and entities have no addition at all, so the delta simply
    // replaces them.
    #[test]
    fn a_type_without_addition_takes_the_delta_whole() {
        assert_eq!(add_vals(Val::Bool(false), Val::Bool(true)), Val::Bool(true));
        assert_eq!(
            add_vals(Val::Vec3([1.0; 3]), Val::Int(2)),
            Val::Int(2),
            "a vector with a scalar delta has no component-wise reading"
        );
    }
}
