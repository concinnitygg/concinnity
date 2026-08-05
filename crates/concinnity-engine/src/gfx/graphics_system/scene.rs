// GraphicsSystem scene-flow wiring and per-frame scene visibility application.

use crate::assets::{RenderHandle, Scene, SceneMember};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::gfx::scene_flow;

use super::*;

// Build the (draw-slots, scene) visibility pairs from the per-entity
// components: every entity with a RenderHandle contributes its GPU draw slots,
// tagged with the SceneMember scene it belongs to (None = always visible),
// consumed by the scene_flow visibility functions. The two returned vectors are
// index-aligned: pair i is one entity's draws and its scene.
pub(crate) fn decomposed_visibility_snapshot(
    ctx: &PipelineContext,
) -> (Vec<Vec<usize>>, Vec<Option<AssetId>>) {
    let scene_of: std::collections::HashMap<crate::ecs::Entity, AssetId> = ctx
        .join2::<SceneMember, RenderHandle>()
        .map(|(entity, member, _)| (entity, member.0))
        .collect();
    let mut draws = Vec::new();
    let mut scenes = Vec::new();
    for (entity, handle) in ctx.query_with_entity::<RenderHandle>() {
        // A Hidden entity contributes no slots: its draws were switched off
        // by a hide request, and a scene switch must not relight them.
        if ctx.get::<crate::assets::Hidden>(entity).is_some() {
            draws.push(Vec::new());
        } else {
            draws.push(handle.draws.iter().map(|&slot| slot as usize).collect());
        }
        scenes.push(scene_of.get(&entity).copied());
    }
    (draws, scenes)
}

impl GraphicsSystem {
    // Drain the world's Scene assets into the flow state. The first declared
    // Scene is active at world start; its props are shown and every other
    // scene's props are hidden.
    pub(super) fn setup_scene_flow(&mut self, ctx: &mut PipelineContext) {
        let scenes: Vec<AssetId> = ctx
            .drain::<Scene>()
            .into_iter()
            .map(|s| s.asset_id)
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
        let (draws, scenes) = decomposed_visibility_snapshot(ctx);
        if let Some(backend) = self.backend.as_deref_mut() {
            scene_flow::set_scene_visibility(&draws, &scenes, active_scene, backend);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::BlobData;
    use crate::ecs::{ComponentStorage, Resources};
    use crate::gfx::profile::FrameProfile;

    // The snapshot pairs each entity's draw slots with its scene; scene-less
    // entities are always visible.
    #[test]
    fn snapshot_pairs_each_entity_draws_with_its_scene() {
        let mut components = ComponentStorage::default();
        let mut blob = BlobData::empty();
        let mut profile = FrameProfile::default();
        let mut resources = Resources::new();
        let scratch = crate::ecs::Arena::with_capacity(64 * 1024);
        let mut ctx = PipelineContext {
            components: &mut components,
            blob: &mut blob,
            profile: &mut profile,
            resources: &mut resources,
            frame: crate::ecs::FrameContext::new(&scratch),
        };

        // Entity in scene 7 with two draw slots.
        let a = ctx.components.spawn();
        ctx.insert(
            a,
            RenderHandle {
                draws: [10, 11].into(),
            },
        );
        ctx.insert(a, SceneMember(AssetId(7)));
        // Entity with no scene (always visible), one slot.
        let b = ctx.components.spawn();
        ctx.insert(b, RenderHandle { draws: [20].into() });
        // Entity in scene 8, one slot.
        let c = ctx.components.spawn();
        ctx.insert(c, RenderHandle { draws: [30].into() });
        ctx.insert(c, SceneMember(AssetId(8)));

        let (draws, scenes) = decomposed_visibility_snapshot(&ctx);

        // Pairs follow RenderHandle column order (a, b, c).
        assert_eq!(draws, vec![vec![10usize, 11], vec![20], vec![30]]);
        assert_eq!(scenes, vec![Some(AssetId(7)), None, Some(AssetId(8))]);
    }

    // A Hidden entity contributes an empty slot list, so a scene switch never
    // relights slots a hide request turned off.
    #[test]
    fn snapshot_blanks_hidden_entities_draws() {
        let mut components = ComponentStorage::default();
        let mut blob = BlobData::empty();
        let mut profile = FrameProfile::default();
        let mut resources = Resources::new();
        let scratch = crate::ecs::Arena::with_capacity(64 * 1024);
        let mut ctx = PipelineContext {
            components: &mut components,
            blob: &mut blob,
            profile: &mut profile,
            resources: &mut resources,
            frame: crate::ecs::FrameContext::new(&scratch),
        };

        let a = ctx.components.spawn();
        ctx.insert(a, RenderHandle { draws: [10].into() });
        let b = ctx.components.spawn();
        ctx.insert(b, RenderHandle { draws: [20].into() });
        ctx.insert(b, crate::assets::Hidden);

        let (draws, scenes) = decomposed_visibility_snapshot(&ctx);
        assert_eq!(draws, vec![vec![10usize], vec![]]);
        assert_eq!(scenes, vec![None, None]);
    }

    // An entity carrying SceneMember but no RenderHandle contributes no draws
    // (it is not in the render set), so it never appears in the snapshot.
    #[test]
    fn snapshot_skips_scene_members_without_a_render_handle() {
        let mut components = ComponentStorage::default();
        let mut blob = BlobData::empty();
        let mut profile = FrameProfile::default();
        let mut resources = Resources::new();
        let scratch = crate::ecs::Arena::with_capacity(64 * 1024);
        let mut ctx = PipelineContext {
            components: &mut components,
            blob: &mut blob,
            profile: &mut profile,
            resources: &mut resources,
            frame: crate::ecs::FrameContext::new(&scratch),
        };

        let only_scene = ctx.components.spawn();
        ctx.insert(only_scene, SceneMember(AssetId(7)));
        let rendered = ctx.components.spawn();
        ctx.insert(rendered, RenderHandle { draws: [5].into() });

        let (draws, scenes) = decomposed_visibility_snapshot(&ctx);
        assert_eq!(draws, vec![vec![5usize]]);
        assert_eq!(scenes, vec![None]);
    }
}
