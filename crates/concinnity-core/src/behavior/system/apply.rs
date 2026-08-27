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
