// Action dispatch: each firing rule's actions map onto the runtime's existing
// request queues (spawn/despawn/reparent churn, audio, scene, screen, story)
// or write the shared variable store.

use super::vars::Variables;
use crate::assets::{
    DespawnRequest, PlayCue, ReactionAction, ReparentRequest, SceneCommand, ScreenCommand,
    SpawnRequest, StoryCommand, StoryPlayback, Transform, VisibilityRequest,
};
use crate::ecs::PipelineContext;

pub(super) fn execute(ctx: &mut PipelineContext, actions: &[ReactionAction]) {
    for action in actions {
        match action {
            ReactionAction::Set { name, value, add } => {
                if let Some(vars) = ctx.resource_mut::<Variables>() {
                    vars.apply(name, *value, *add);
                }
            }
            ReactionAction::Spawn {
                template,
                position,
                rotation_deg,
                scale,
                lifetime,
            } => {
                let Some(template) = *template else { continue };
                // A zero scale would make the copy invisible; treat it as
                // unit scale, like the debug spawn path.
                let scale = if *scale == [0.0; 3] { [1.0; 3] } else { *scale };
                ctx.events_mut::<SpawnRequest>().send(SpawnRequest {
                    template,
                    name: None,
                    transform: Transform {
                        position: *position,
                        rotation_deg: *rotation_deg,
                        scale,
                    },
                    lifetime_secs: (*lifetime > 0.0).then_some(*lifetime),
                });
            }
            ReactionAction::Despawn { target } => {
                let Some(target) = *target else { continue };
                ctx.events_mut::<DespawnRequest>()
                    .send(DespawnRequest { name: target });
            }
            ReactionAction::Reparent { child, parent } => {
                let Some(child) = *child else { continue };
                ctx.events_mut::<ReparentRequest>().send(ReparentRequest {
                    child,
                    parent: *parent,
                });
            }
            ReactionAction::Sound { clip, kind, volume } => {
                let Some(clip) = *clip else { continue };
                ctx.events_mut::<PlayCue>().send(PlayCue {
                    clip,
                    kind: *kind,
                    volume: *volume,
                });
            }
            ReactionAction::Scene { scene, transition } => {
                let Some(scene) = *scene else { continue };
                ctx.events_mut::<SceneCommand>().send(SceneCommand {
                    scene,
                    transition: transition.clone(),
                });
            }
            ReactionAction::Screen { screen } => {
                let Some(screen) = *screen else { continue };
                ctx.events_mut::<ScreenCommand>()
                    .send(ScreenCommand::Show(screen));
            }
            ReactionAction::Story(playback) => {
                let command = match playback {
                    StoryPlayback::Start => StoryCommand::Start,
                    StoryPlayback::Continue => StoryCommand::Continue,
                };
                ctx.events_mut::<StoryCommand>().send(command);
            }
            ReactionAction::Show { target } | ReactionAction::Hide { target } => {
                let Some(name) = *target else { continue };
                let visible = matches!(action, ReactionAction::Show { .. });
                ctx.events_mut::<VisibilityRequest>()
                    .send(VisibilityRequest { name, visible });
            }
        }
    }
}
