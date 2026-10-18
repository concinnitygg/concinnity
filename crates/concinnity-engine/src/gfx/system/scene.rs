// GraphicsSystem scene-flow wiring and per-frame scene visibility application.

use concinnity_core::components::Scene;
use concinnity_core::ecs::PipelineContext;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::render::scene_flow;

use super::*;

impl GraphicsSystem {
    // Drain the world's Scene assets into the flow state. The start scene's
    // props are shown and every other scene's props are hidden.
    pub(super) fn setup_scene_flow(&mut self, ctx: &mut PipelineContext) {
        let flow = scene_flow::SceneFlow::starting(ctx);
        ctx.drain::<Scene>();
        let Some(flow) = flow else {
            return;
        };
        self.apply_scene_visibility(ctx, flow.current);
        self.scene_flow = Some(flow);
    }

    pub(super) fn apply_scene_visibility(&mut self, ctx: &PipelineContext, active_scene: AssetId) {
        // Snapshot visibility from the per-entity components before borrowing the
        // backend, so the ctx borrow is released by the time set_scene_visibility
        // runs.
        let scratch = self.extractor.scene_visibility_mut();
        scene_flow::refresh_visibility_snapshot(ctx, scratch);
        if let Some(backend) = self.backend.as_deref_mut() {
            scene_flow::set_scene_visibility(&scratch.visibility, active_scene, backend);
        }
    }
}
