// GraphicsSystem scene-flow wiring and per-frame scene visibility application.

use concinnity_core::components::Scene;
use concinnity_core::ecs::PipelineContext;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::render::scene_flow;

use super::*;

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
        scene_flow::refresh_visibility_snapshot(ctx, &mut self.scene_visibility);
        if let Some(backend) = self.backend.as_deref_mut() {
            scene_flow::set_scene_visibility(
                &self.scene_visibility.visibility,
                active_scene,
                backend,
            );
        }
    }
}
