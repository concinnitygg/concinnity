//! The per-frame params blocks `MtlContext::draw_frame` hoists ahead of its
//! render-graph dispatch, so one `GraphFrameParams` carries the union.

use concinnity_core::gfx::render_types;
use concinnity_core::render::error;
use concinnity_core::render::lights;
use concinnity_core::render::post::rt_reflections::RtParamsInputs;
use concinnity_core::render::render_graph;
use concinnity_core::render::uniforms::metal::VelocityUniforms;
use concinnity_core::render::volumetric_fog::FogSettings;
use concinnity_core::transform::mat4_inverse;
use concinnity_core::transform::mat4_mul;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLTexture;

use crate::metal::context::MtlContext;

// The camera and resolution inputs every params block derives from.
pub(super) struct PassUniformArgs {
    pub(super) fov_y_radians: f32,
    pub(super) aspect: f32,
    pub(super) near: f32,
    pub(super) far: f32,
    pub(super) cam_pos: [f32; 3],
    pub(super) sky_rot: [[f32; 4]; 3],
    pub(super) proj: [[f32; 4]; 4],
    pub(super) vp: [[f32; 4]; 4],
    pub(super) render_w: u32,
    pub(super) render_h: u32,
}

// One params block per pass that takes one, plus the gates the graph inputs
// and `GraphFrameParams` read alongside them.
pub(super) struct PassUniforms {
    pub(super) ssao_params: Option<render_types::SsaoParams>,
    pub(super) ssr_params: Option<render_types::SsrParams>,
    pub(super) ssgi_params: Option<render_types::SsgiParams>,
    pub(super) rt_reflection_params: Option<render_types::RtParams>,
    pub(super) fog_settings: Option<FogSettings>,
    pub(super) fog_params: Option<render_types::FogParams>,
    pub(super) fog_froxel_params: Option<render_types::FogFroxelParams>,
    pub(super) clustered: bool,
    pub(super) cluster_params: render_types::ClusterParams,
    pub(super) velocity_active: bool,
    pub(super) vel_uniforms: Option<VelocityUniforms>,
    pub(super) scene_input: Retained<ProtocolObject<dyn MTLTexture>>,
    pub(super) scene_color: Retained<ProtocolObject<dyn MTLTexture>>,
    pub(super) transparent_active: bool,
}

impl MtlContext {
    pub(super) fn frame_pass_uniforms(
        &mut self,
        args: PassUniformArgs,
    ) -> error::RenderResult<PassUniforms> {
        let PassUniformArgs {
            fov_y_radians,
            aspect,
            near,
            far,
            cam_pos,
            sky_rot,
            proj,
            vp,
            render_w,
            render_h,
        } = args;
        // Per-frame pass uniforms hoisted upfront.
        // Every pass that needs a struct of per-frame params builds its
        // uniforms here so a single GraphFrameParams can carry
        // the union into `execute_graph`.
        let ssao_params = self
            .ssao
            .settings
            .map(|settings| settings.params(fov_y_radians, aspect));
        let ssr_params = self.ssr.settings.map(|settings| {
            let v = self.view.matrix;
            let inv_view_rot = [
                [v[0][0], v[1][0], v[2][0], 0.0],
                [v[0][1], v[1][1], v[2][1], 0.0],
                [v[0][2], v[1][2], v[2][2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let prefilter_mip_count = self.scene.env_map.prefilter_mip_count as f32;
            settings.params(
                fov_y_radians,
                aspect,
                inv_view_rot,
                cam_pos,
                prefilter_mip_count,
                sky_rot,
            )
        });
        let ssgi_params = self
            .ssgi
            .settings
            .map(|settings| settings.params(fov_y_radians, aspect));
        // RT-reflection params: built only when the acceleration structure is
        // live (so they stay in lockstep with `rt_reflections_enabled`). Carries
        // the camera-to-world transform + sun the kernel shades hits with, like
        // SSR's params plus the world-space camera + sun.
        let rt_reflection_params =
            self.rt
                .settings
                .filter(|_| self.rt.accel.is_some())
                .map(|settings| {
                    let v = self.view.matrix;
                    let inv_view_rot = [
                        [v[0][0], v[1][0], v[2][0], 0.0],
                        [v[0][1], v[1][1], v[2][1], 0.0],
                        [v[0][2], v[1][2], v[2][2], 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ];
                    let prefilter_mip_count = self.scene.env_map.prefilter_mip_count as f32;
                    let sun = &self.light_uniforms.directional[0];
                    let sun_color = [
                        sun.color[0] * sun.intensity,
                        sun.color[1] * sun.intensity,
                        sun.color[2] * sun.intensity,
                    ];
                    settings.params(RtParamsInputs {
                        fov_y_radians,
                        aspect,
                        inv_view_rot,
                        cam_pos,
                        sun_dir: sun.direction,
                        sun_color,
                        prefilter_mip_count,
                        sky_rot,
                    })
                });
        // The live settings, dropped when the medium cannot affect the frame (a
        // zero density integrates to a transparent black over the whole volume).
        // One source for the two param blocks and the `frame_graph_inputs` gate, so
        // `GraphFrameParams`'s "Some only when the Fog pass is in the graph"
        // contract holds.
        let fog_settings = self.fog.settings.filter(|s| s.contributes());
        let fog_params = fog_settings.map(|fog| {
            // Sun = the first directional light; falls back to the
            // LightUniforms::DEFAULT direction if the world declared none.
            let sun = &self.light_uniforms.directional[0];
            let sun_color = [
                sun.color[0] * sun.intensity,
                sun.color[1] * sun.intensity,
                sun.color[2] * sun.intensity,
            ];
            // Fog renders into hdr_resolve, which is render-resolution
            // when the upscaler is on. The fog shader uses the viewport
            // to reconstruct world position from screen UV, so it must
            // match the actual render target's pixel grid.
            let viewport = [render_w as f32, render_h as f32];
            // Reconstruct the froxel volume with the UN-jittered view-projection.
            // Fog is volumetric, so its screen-space contribution does not follow
            // the surface motion vectors TAA reprojects by. Feeding it the jittered
            // inv_vp shifts the whole volume sub-pixel every frame; on a large
            // smooth low-contrast surface, where the fog is the dominant
            // high-frequency signal, TAA cannot reconcile that per-frame shift with
            // the jitter-free history, so the fog flickers (a moving moire). The
            // un-jittered inv_vp keeps the volume stable frame to frame; its offset
            // versus the jittered depth buffer is far below the coarse froxel grid.
            let fog_inv_vp = mat4_inverse(mat4_mul(proj, self.view.matrix));
            fog.params(fog_inv_vp, cam_pos, sun.direction, sun_color, viewport)
        });
        // FogFroxel volume extras: view matrix + volume dimensions + near/far
        // so the compute kernel can place each froxel in world-space and the
        // fragment shader can map a scene depth into the volume's Z axis.
        let fog_froxel_params = fog_settings.map(|fog| render_types::FogFroxelParams {
            view: self.view.matrix,
            froxel_dims: [
                render_graph::FOG_FROXEL_X,
                render_graph::FOG_FROXEL_Y,
                render_graph::FOG_FROXEL_Z,
            ],
            _pad_align: 0,
            z_near: near.max(1e-3),
            z_far: fog.max_distance,
            _pad: [0.0; 2],
        });
        // Clustered light-binning params (main camera). The compute pass reads
        // these to build each cluster's world-space AABB (un-jittered inverse VP
        // + camera forward, matching the fog froxel convention) and the forward
        // pass reads the grid dims / depth range / screen size to place a
        // fragment. `use_clusters` is set only when the world has local lights
        // (the pipeline is built iff so) and at least one is still live;
        // otherwise the forward pass brute-forces an empty list and the LightCull
        // graph node is omitted, so a list the skipped pass did not write is never
        // read. Stored on self so the shared main-pass bind can push it; a local
        // copy feeds the LightCull arm.
        let clustered = lights::clustered_lighting_active(
            self.light_cull.pipeline.is_some(),
            self.light_uniforms.num_local_lights,
        );
        let cluster_inv_vp = mat4_inverse(mat4_mul(proj, self.view.matrix));
        self.cluster_params = render_types::ClusterParams {
            inv_view_proj: cluster_inv_vp,
            cam_pos,
            z_near: near.max(1e-3),
            view_forward: [
                -self.view.matrix[0][2],
                -self.view.matrix[1][2],
                -self.view.matrix[2][2],
            ],
            z_far: far,
            grid_x: render_types::CLUSTER_GRID_X,
            grid_y: render_types::CLUSTER_GRID_Y,
            grid_z: render_types::CLUSTER_GRID_Z,
            num_lights: self.light_uniforms.num_local_lights.max(0) as u32,
            screen_w: render_w as f32,
            screen_h: render_h as f32,
            use_clusters: u32::from(clustered),
            _pad: 0,
        };
        let cluster_params = self.cluster_params;
        // Velocity (motion vectors in the G-buffer pre-pass) is needed whenever
        // temporal reconstruction runs: that's TAA or the MetalFX upscaler.
        let velocity_active = self.taa.enabled || self.upscale.scaler.is_some();
        let vel_uniforms = if velocity_active {
            Some(VelocityUniforms {
                jittered_vp: vp,
                cur_vp: mat4_mul(proj, self.view.matrix),
                prev_vp: self.prev_view_proj,
            })
        } else {
            None
        };
        // `scene_input` is the engine-owned texture the post-decoration stack
        // treats as the pre-TAA scene: `ssr_targets.output` when a reflection
        // path is live, else the raw `hdr_resolve`.
        //
        // `output` is the *composited* scene, not the reflection. Both the SSR
        // and the RT resolve write radiance into `ssr_targets.reflection`, then
        // call the shared `encode_reflection_composite`, which blends that over
        // `hdr_resolve` into `output`. Worth stating precisely: the DirectX
        // equivalent split the two apart and left its upscaler reading the
        // radiance buffer as if it were the scene.
        //
        // `scene_color` is what Bloom + Composite read:
        //   - the upscaler's output (drawable-res) when MetalFX is on,
        //   - the TAA resolve target when TAA is on,
        //   - otherwise just the pre-TAA scene (no temporal stage).
        let scene_input = if self.ssr.settings.is_some() || self.rt.accel.is_some() {
            self.ssr
                .targets
                .as_ref()
                .ok_or_else(|| {
                    error::RenderError::Other("reflections enabled but SSR targets missing".into())
                })?
                .output
                .clone()
        } else {
            self.targets.hdr.hdr_resolve.clone()
        };
        let scene_color = if let Some(u) = &self.upscale.scaler {
            u.output.clone()
        } else if let Some(out) = self.taa.output() {
            out.clone()
        } else {
            scene_input.clone()
        };

        // The transparent pass runs when any translucent producer is live.
        // Drives both the graph-input gate (whether the slot is inserted) and the
        // `scene_pre_taa` supply in `GraphFrameParams` (the pass reads + writes
        // it). With
        // SSR off `scene_input` aliases `hdr_resolve`, which is the correct
        // RMW target: the transparent encoder blits a scene copy first, so the
        // self-read for refraction is safe.
        let transparent_active = (self.water.pipeline.is_some()
            && self.water.surfaces.iter().any(|s| s.visible))
            || (self.glass.pipeline.is_some() && self.glass.panels.iter().any(|p| p.visible))
            || self.mesh_glass_visible();
        Ok(PassUniforms {
            ssao_params,
            ssr_params,
            ssgi_params,
            rt_reflection_params,
            fog_settings,
            fog_params,
            fog_froxel_params,
            clustered,
            cluster_params,
            velocity_active,
            vel_uniforms,
            scene_input,
            scene_color,
            transparent_active,
        })
    }
}
