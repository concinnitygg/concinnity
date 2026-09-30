// GraphicsSystem scene-flow wiring and per-frame scene visibility application.

use concinnity_core::components::{Hidden, RenderHandle, Scene, SceneMember};
use concinnity_core::ecs::Entity;
use concinnity_core::ecs::PipelineContext;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::render::scene_flow;

use super::*;

// The per-entity scene-visibility snapshot plus the scratch its refresh needs,
// both reused across frames: a fade refreshes this every frame it runs, so
// nothing here may allocate in steady state.
#[derive(Default)]
pub(crate) struct SceneVisibilityScratch {
    pub(crate) visibility: scene_flow::SceneVisibility,
    scene_of: std::collections::HashMap<Entity, AssetId>,
}

// Rebuild the (draw-slots, scene) visibility pairs from the per-entity
// components: every entity with a RenderHandle contributes its GPU draw slots,
// tagged with the SceneMember scene it belongs to (None = always visible),
// consumed by the scene_flow visibility functions.
pub(crate) fn refresh_visibility_snapshot(
    ctx: &PipelineContext,
    scratch: &mut SceneVisibilityScratch,
) {
    scratch.scene_of.clear();
    scratch.scene_of.extend(
        ctx.join2::<SceneMember, RenderHandle>()
            .map(|(entity, member, _)| (entity, member.0)),
    );
    scratch.visibility.clear();
    for (entity, handle) in ctx.query_with_entity::<RenderHandle>() {
        scratch
            .visibility
            .begin_prop(scratch.scene_of.get(&entity).copied());
        // A Hidden entity contributes no slots: its draws were switched off
        // by a hide request, and a scene switch must not relight them.
        if ctx.get::<Hidden>(entity).is_none() {
            for &slot in handle.draws.iter() {
                scratch.visibility.push_draw(slot);
            }
        }
    }
}

impl GraphicsSystem {
    // Drain the world's Scene assets into the flow state. The first declared
    // Scene is active at world start; its props are shown and every other
    // scene's props are hidden.
    pub(super) fn setup_scene_flow(&mut self, ctx: &mut PipelineContext) {
        let scenes: Vec<AssetId> = ctx
            .drain_with_ids::<Scene>()
            .into_iter()
            .filter_map(|(id, _)| id)
            .collect();
        if scenes.is_empty() {
            return;
        }
        let active_scene = scenes[0];
        self.apply_scene_visibility(ctx, active_scene);
        self.scene_flow = Some(scene_flow::SceneFlow {
            scenes,
            current: active_scene,
            fade: scene_flow::FadePhase::None,
        });
    }

    pub(super) fn apply_scene_visibility(&mut self, ctx: &PipelineContext, active_scene: AssetId) {
        // Snapshot visibility from the per-entity components before borrowing the
        // backend, so the ctx borrow is released by the time set_scene_visibility
        // runs.
        refresh_visibility_snapshot(ctx, &mut self.scene_visibility);
        if let Some(backend) = self.backend.as_deref_mut() {
            scene_flow::set_scene_visibility(
                &self.scene_visibility.visibility,
                active_scene,
                backend,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::ecs::World;

    // Collect the snapshot's pairs for assertions.
    fn snapshot_pairs(ctx: &PipelineContext) -> (Vec<Vec<DrawIndex>>, Vec<Option<AssetId>>) {
        let mut scratch = SceneVisibilityScratch::default();
        refresh_visibility_snapshot(ctx, &mut scratch);
        let mut draws = Vec::new();
        let mut scenes = Vec::new();
        for (d, s) in scratch.visibility.props() {
            draws.push(d.to_vec());
            scenes.push(s);
        }
        (draws, scenes)
    }

    // The snapshot pairs each entity's draw slots with its scene; scene-less
    // entities are always visible.
    #[test]
    fn snapshot_pairs_each_entity_draws_with_its_scene() {
        let mut world = World::new();
        let mut ctx = world.context();

        // Entity in scene 7 with two draw slots.
        let a = ctx.components.spawn();
        ctx.insert(
            a,
            RenderHandle {
                draws: [DrawIndex(10), DrawIndex(11)].into(),
            },
        );
        ctx.insert(a, SceneMember(AssetId(7)));
        // Entity with no scene (always visible), one slot.
        let b = ctx.components.spawn();
        ctx.insert(
            b,
            RenderHandle {
                draws: [DrawIndex(20)].into(),
            },
        );
        // Entity in scene 8, one slot.
        let c = ctx.components.spawn();
        ctx.insert(
            c,
            RenderHandle {
                draws: [DrawIndex(30)].into(),
            },
        );
        ctx.insert(c, SceneMember(AssetId(8)));

        let (draws, scenes) = snapshot_pairs(&ctx);

        // Pairs follow RenderHandle column order (a, b, c).
        assert_eq!(
            draws,
            vec![
                vec![DrawIndex(10), DrawIndex(11)],
                vec![DrawIndex(20)],
                vec![DrawIndex(30)]
            ]
        );
        assert_eq!(scenes, vec![Some(AssetId(7)), None, Some(AssetId(8))]);
    }

    // A Hidden entity contributes an empty slot list, so a scene switch never
    // relights slots a hide request turned off.
    #[test]
    fn snapshot_blanks_hidden_entities_draws() {
        let mut world = World::new();
        let mut ctx = world.context();

        let a = ctx.components.spawn();
        ctx.insert(
            a,
            RenderHandle {
                draws: [DrawIndex(10)].into(),
            },
        );
        let b = ctx.components.spawn();
        ctx.insert(
            b,
            RenderHandle {
                draws: [DrawIndex(20)].into(),
            },
        );
        ctx.insert(b, Hidden);

        let (draws, scenes) = snapshot_pairs(&ctx);
        assert_eq!(draws, vec![vec![DrawIndex(10)], vec![]]);
        assert_eq!(scenes, vec![None, None]);
    }

    // An entity carrying SceneMember but no RenderHandle contributes no draws
    // (it is not in the render set), so it never appears in the snapshot.
    #[test]
    fn snapshot_skips_scene_members_without_a_render_handle() {
        let mut world = World::new();
        let mut ctx = world.context();

        let only_scene = ctx.components.spawn();
        ctx.insert(only_scene, SceneMember(AssetId(7)));
        let rendered = ctx.components.spawn();
        ctx.insert(
            rendered,
            RenderHandle {
                draws: [DrawIndex(5)].into(),
            },
        );

        let (draws, scenes) = snapshot_pairs(&ctx);
        assert_eq!(draws, vec![vec![DrawIndex(5)]]);
        assert_eq!(scenes, vec![None]);
    }
}
