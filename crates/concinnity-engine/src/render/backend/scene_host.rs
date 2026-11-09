//! The scene a backend draws from, lent to the defaulted operations that only
//! edit it.
//!
//! Most of what the engine pushes into a backend each frame (camera, model
//! matrices, skinned poses, slot visibility) and most of what the streamer
//! does (placing a mesh or chunk, cloning a draw) is bookkeeping on the CPU
//! copy of the scene plus, at most, some bytes written into the shared
//! geometry buffers. [`SceneState`] holds that bookkeeping once for every
//! backend, so those operations are defaulted over this trait instead of being
//! implemented three times. A backend answers with its scene and with a
//! [`GeometryWriter`] for its buffers; everything else stays its own.

use crate::render::backend::error::RenderResult;
use crate::render::scene_state::{GeometryWriter, SceneState};

/// An edit of the scene that may write geometry bytes through the writer.
pub type GeometryEdit<'a> =
    &'a mut dyn FnMut(&mut SceneState, &mut dyn GeometryWriter) -> RenderResult<()>;

/// Lends a backend's [`SceneState`] to the scene-only defaults of
/// [`RenderBackend`](super::RenderBackend) and its operation families.
///
/// A backend with no scene of its own, such as a test double, answers `None`
/// throughout; the defaults then do nothing, or report
/// [`RenderError::Unsupported`](crate::render::backend::error::RenderError::Unsupported)
/// where the operation is fallible.
pub trait SceneHost {
    /// The scene, for the defaulted queries.
    fn scene(&self) -> Option<&SceneState>;
    /// The scene, for the defaulted edits that write no geometry.
    fn scene_mut(&mut self) -> Option<&mut SceneState>;
    /// Run `edit` over the scene with the writer that lands its bytes in this
    /// backend's shared geometry buffers, returning its result.
    fn edit_geometry(&mut self, edit: GeometryEdit<'_>) -> Option<RenderResult<()>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::mesh_payload::{SkinnedVertex, Vertex};
    use crate::gfx::render_types::{
        DrawIndex, MATERIAL_PARAM_COUNT, MaterialUniforms, SkinnedDrawObject, SkinnedIndex,
    };
    use crate::input::snapshot::InputSnapshot;
    use crate::render::backend::{
        BackendProbe, CullReadback, DecalPass, DeviceCapabilities, DrawStreaming, EmitterPass,
        FrameCapture, FrameParams, GeometryBuffers, GeometryRebuild, LightTables, PlanetReshape,
        RenderBackend, SdfVolumeShaders, SkinnedDraws, SkinnedFrame, TextureUploads, WindowControl,
        WorldReload, WorldShaders,
    };
    use crate::render::config::apply::SettingsTarget;
    use crate::render::scene_state::{DrawList, GeometryBuffer};
    use crate::render::slots::draw_slot::SlotAlloc;
    use crate::test_support::draw_object;
    use alloc::vec;
    use alloc::vec::Vec;

    const MOVED: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0, 1.0],
    ];

    // Records the buffers a write reached.
    #[derive(Default)]
    struct Writes(Vec<GeometryBuffer>);

    impl GeometryWriter for Writes {
        fn write(&mut self, buffer: GeometryBuffer, _: usize, _: &[u8]) -> RenderResult<()> {
            self.0.push(buffer);
            Ok(())
        }
    }

    // A backend whose only state is a scene and the writes lent to it.
    struct Hosted {
        scene: SceneState,
        writes: Writes,
        // Decal slot bindings, `(id, Some(texture))` bound and `(id, None)`
        // released.
        decal_bindings: Vec<(usize, Option<usize>)>,
        // Emitter pools, `(id, true)` created and `(id, false)` destroyed;
        // the next create fails when `fail_emitter` is set.
        emitter_pools: Vec<(usize, bool)>,
        fail_emitter: bool,
        // What the shared geometry buffers were asked, in order.
        geometry: Vec<&'static str>,
        replaced: Option<(usize, usize)>,
    }

    impl Hosted {
        // Two build-time slots expecting 3 vertices and 3 indices, and one
        // skinned slot.
        fn new() -> Self {
            let objects = (0..2)
                .map(|_| {
                    let mut obj = draw_object();
                    obj.vertex_count = 3;
                    obj.index_count = 3;
                    obj
                })
                .collect();
            let mut scene = SceneState::new(
                DrawList::unreserved(objects, 0),
                [0.0; 4],
                crate::render::light::state::LightState::new(
                    crate::gfx::render_types::LightUniforms::DEFAULT,
                    &[],
                    1,
                ),
                crate::render::material_params::MaterialParamTable::new(Vec::new(), 1),
                crate::render::environment::volumetric_fog::SceneFog::default(),
                crate::render::decal::DecalSet::new(0, 1),
            );
            scene
                .skinned
                .draw_objects
                .push(crate::test_support::skinned_draw_object());
            scene.skinned.joint_matrices.push(vec![MOVED]);
            scene.skinned.morph_weights.push(vec![0.0]);
            Self {
                scene,
                writes: Writes::default(),
                decal_bindings: Vec::new(),
                emitter_pools: Vec::new(),
                fail_emitter: false,
                geometry: Vec::new(),
                replaced: None,
            }
        }
    }

    impl SceneHost for Hosted {
        fn scene(&self) -> Option<&SceneState> {
            Some(&self.scene)
        }
        fn scene_mut(&mut self) -> Option<&mut SceneState> {
            Some(&mut self.scene)
        }
        fn edit_geometry(&mut self, edit: GeometryEdit<'_>) -> Option<RenderResult<()>> {
            Some(edit(&mut self.scene, &mut self.writes))
        }
    }

    impl RenderBackend for Hosted {
        fn wait_idle(&self) {}
        fn draw_frame(&mut self, _: FrameParams<'_>) -> RenderResult<()> {
            Ok(())
        }
    }

    impl WindowControl for Hosted {
        fn window_closed(&mut self) -> bool {
            false
        }
        fn request_cursor_capture(&mut self) {}
        fn take_input(&mut self) -> InputSnapshot {
            InputSnapshot::default()
        }
    }

    impl SkinnedDraws for Hosted {
        fn upload_skinned(
            &mut self,
            _: &[SkinnedVertex],
            _: &[u32],
            _: Vec<SkinnedDrawObject>,
        ) -> RenderResult<()> {
            Ok(())
        }
    }

    impl TextureUploads for Hosted {
        fn evict_texture_slot(
            &mut self,
            _: crate::gfx::render_types::TextureSlot,
        ) -> RenderResult<()> {
            Ok(())
        }
        fn update_texture_slot(
            &mut self,
            _: crate::gfx::render_types::TextureSlot,
            _: &crate::bake::texture::TextureImage,
        ) -> RenderResult<()> {
            Ok(())
        }
    }

    impl DrawStreaming for Hosted {
        fn setup_chunk_streaming(&mut self, _: usize, _: usize) -> RenderResult<()> {
            Ok(())
        }
    }

    impl BackendProbe for Hosted {
        fn capabilities(&self) -> DeviceCapabilities {
            DeviceCapabilities::ALL
        }
    }
    impl CullReadback for Hosted {}
    impl FrameCapture for Hosted {}
    impl GeometryBuffers for Hosted {
        fn read_static_geometry(&mut self) -> RenderResult<(Vec<Vertex>, Vec<u32>)> {
            self.geometry.push("read");
            Ok((Vec::new(), Vec::new()))
        }
        fn replace_static_geometry(
            &mut self,
            vertices: &[Vertex],
            indices: &[u32],
        ) -> RenderResult<()> {
            self.geometry.push("replace");
            self.replaced = Some((vertices.len(), indices.len()));
            Ok(())
        }
        fn static_geometry_moved(&mut self) {
            self.geometry.push("moved");
        }
    }
    impl GeometryRebuild for Hosted {}
    impl PlanetReshape for Hosted {}
    impl SettingsTarget for Hosted {}
    impl DecalPass for Hosted {
        fn ensure_decal_pass(&mut self) -> RenderResult<()> {
            Ok(())
        }
        fn bind_decal_texture(&mut self, id: usize, texture_slot: usize) {
            self.decal_bindings.push((id, Some(texture_slot)));
        }
        fn release_decal_texture(&mut self, id: usize) {
            self.decal_bindings.push((id, None));
        }
    }
    impl EmitterPass for Hosted {
        fn ensure_emitter_pass(&mut self) -> RenderResult<()> {
            Ok(())
        }
        fn create_emitter(
            &mut self,
            id: usize,
            _: &crate::render::particles::ParticleEmitterRecord,
        ) -> RenderResult<()> {
            if self.fail_emitter {
                return Err(crate::render::backend::error::RenderError::Other(
                    "no pool".into(),
                ));
            }
            self.emitter_pools.push((id, true));
            Ok(())
        }
        fn destroy_emitter(&mut self, id: usize) {
            self.emitter_pools.push((id, false));
        }
    }
    impl LightTables for Hosted {}
    impl SdfVolumeShaders for Hosted {}
    impl WorldReload for Hosted {}
    impl WorldShaders for Hosted {}

    fn vertices() -> Vec<Vertex> {
        vec![crate::test_support::vertex(); 3]
    }

    #[test]
    fn the_frame_pushes_land_in_the_scene() {
        let mut b = Hosted::new();
        b.update_view(MOVED);
        b.update_models(&[(DrawIndex(1), MOVED)]);
        b.retire_draw_object(DrawIndex(0));
        assert_eq!(b.scene.view.matrix, MOVED);
        assert_eq!(b.scene.draw.objects[1].model, MOVED);
        assert!(!b.scene.draw.objects[0].resident);
    }

    #[test]
    fn the_skinned_pushes_land_in_the_slots() {
        let mut b = Hosted::new();
        let mut poses = crate::render::frame::snapshot::SpanBuffer::default();
        poses.push(SkinnedIndex(0), &[MOVED, MOVED]);
        let mut morphs = crate::render::frame::snapshot::SpanBuffer::default();
        morphs.push(SkinnedIndex(0), &[0.5]);
        b.update_skinned(SkinnedFrame {
            poses: &poses,
            morphs: &morphs,
            models: &[(SkinnedIndex(0), MOVED)],
        });
        b.reveal_skinned_instance(SkinnedIndex(0), MOVED);
        assert_eq!(b.scene.skinned.morph_weights[0], vec![0.5]);
        assert!(b.scene.skinned.draw_objects[0].visible);
        b.retire_skinned_draw_object(SkinnedIndex(0));
        assert!(!b.scene.skinned.draw_objects[0].visible);
        assert!(b.update_skinned_skeleton(SkinnedIndex(0), 4).is_ok());
        assert_eq!(b.scene.skinned.joint_matrices[0].len(), 4);
    }

    #[test]
    fn streaming_writes_through_the_lent_writer() {
        let mut b = Hosted::new();
        b.seed_mesh_streaming(0, 3 * size_of::<Vertex>() as u64, 0, 12)
            .expect("seed");
        b.upload_mesh(DrawIndex(0), &vertices(), &[0, 1, 2], 0)
            .expect("room for the mesh");
        assert_eq!(
            b.writes.0,
            vec![GeometryBuffer::Vertex, GeometryBuffer::Index]
        );
        assert!(b.scene.draw.objects[0].resident);
        b.evict_mesh(DrawIndex(0), 0).expect("slot 0 exists");
        assert!(!b.scene.draw.objects[0].resident);
        b.update_mesh_geometry(DrawIndex(1), &vertices(), &[0, 1, 2], &[])
            .expect("counts match");
        assert_eq!(b.writes.0.len(), 4);
    }

    #[test]
    fn chunks_and_clones_land_at_their_slots() {
        let mut b = Hosted::new();
        b.scene
            .placement
            .chunk_vtx
            .free(0, 3 * size_of::<Vertex>() as u64, 0);
        b.scene.placement.chunk_idx.free(0, 12, 0);
        let verts = vertices();
        let chunk = crate::render::backend::ChunkMesh {
            verts: &verts,
            idxs: &[0, 1, 2],
            model: MOVED,
            texture_slot: 0,
            normal_map_slot: 0,
            material: MaterialUniforms::DEFAULT,
            frame: 0,
            planet_tile: None,
            bounds: None,
        };
        b.add_chunk_mesh(chunk, SlotAlloc::Append(DrawIndex(2)))
            .expect("room for the chunk");
        b.set_chunk_model(DrawIndex(2), crate::transform::IDENTITY)
            .expect("slot 2 exists");
        b.remove_chunk_mesh(DrawIndex(2), 0).expect("slot 2 exists");
        b.clone_static_draw_object(DrawIndex(0), MOVED, SlotAlloc::Append(DrawIndex(3)))
            .expect("source exists");
        assert_eq!(b.scene.draw.objects.len(), 4);
        assert_eq!(b.scene.draw.objects[3].model, MOVED);
    }

    #[test]
    fn the_live_edits_rewrite_the_slot() {
        let mut b = Hosted::new();
        b.set_draw_material(DrawIndex(0), MaterialUniforms::DEFAULT, 5, 6);
        b.set_draw_cull_distance(DrawIndex(0), 9.0);
        assert_eq!(b.scene.draw.objects[0].texture_slot, 5);
        assert_eq!(b.scene.draw.objects[0].cull_distance, 9.0);
        assert_eq!(b.draw_geometry_size(DrawIndex(0)), Some((3, 3)));
        assert_eq!(b.draw_lod_index_counts(DrawIndex(0)), Some(vec![]));
    }

    // A material's new params land in the scene's table, the one every
    // backend copy is written from; the zero row stays zero.
    #[test]
    fn material_params_rewrite_the_scenes_row() {
        let mut b = Hosted::new();
        b.scene.material_params = crate::render::material_params::MaterialParamTable::new(
            crate::render::material_params::rows([[0.0; MATERIAL_PARAM_COUNT]]),
            2,
        );
        let params = [1.0; MATERIAL_PARAM_COUNT];
        b.set_material_params(1, params);
        b.set_material_params(0, params);
        assert_eq!(b.scene.material_params.rows()[1].values, params);
        assert_eq!(
            b.scene.material_params.rows()[0].values,
            [0.0; MATERIAL_PARAM_COUNT]
        );
        assert!(b.scene.material_params.take_upload(0).is_some());
    }

    // A runtime decal lands in the scene's set and the backend binds its
    // albedo; removing it frees the slot and releases the binding.
    #[test]
    fn a_runtime_decal_lives_in_the_scenes_set() {
        let mut b = Hosted::new();
        b.scene.decals = crate::render::decal::DecalSet::new(1, 1);
        let record = crate::render::decal::DecalRecord {
            model: MOVED,
            inv_model: MOVED,
            texture_slot: 4,
            tint: [1.0; 4],
        };
        let id = b.add_decal(record).unwrap();
        assert!(!b.scene.decals.is_empty());
        let full = b.add_decal(record).unwrap_err();
        assert!(alloc::format!("{full}").contains("(1 decals)"), "{full}");
        b.remove_decal(id).unwrap();
        assert!(b.scene.decals.is_empty());
        assert!(b.remove_decal(id).is_err());
        assert_eq!(b.decal_bindings, [(id, Some(4)), (id, None)]);
    }

    // A runtime emitter lands in the scene's particle system and the backend
    // allocates its pool; a pool that cannot be allocated leaves no emitter.
    #[test]
    fn a_runtime_emitter_lives_in_the_scenes_particle_system() {
        let mut b = Hosted::new();
        let record = crate::render::particles::ParticleEmitterRecord::new(
            &crate::components::ParticleEmitter::default(),
            0,
        );
        let id = b.add_emitter(record).unwrap();
        assert!(!b.scene.particles.is_empty());
        b.fail_emitter = true;
        assert!(b.add_emitter(record).is_err());
        b.fail_emitter = false;
        b.remove_emitter(id).unwrap();
        assert!(b.scene.particles.is_empty());
        assert!(b.remove_emitter(id).is_err());
        assert_eq!(b.emitter_pools, [(id, true), (id, false)]);
    }

    // A size-changing rebuild reads the backend's buffers back, repacks them,
    // swaps the new ones in, then lays the draws out over them before the
    // backend refreshes what was built over the old ones.
    #[test]
    fn a_static_rebuild_repacks_through_the_backends_buffers() {
        let mut b = Hosted::new();
        let change = |draw: u32, vertex_count: usize| crate::render::backend::DrawGeometryUpdate {
            draw_idx: DrawIndex(draw),
            vertices: vec![crate::test_support::vertex(); vertex_count],
            indices: vec![0, 1, 2],
            lod_alternates: Vec::new(),
        };
        b.rebuild_static_geometry(vec![change(0, 4), change(1, 3)])
            .unwrap();
        assert_eq!(b.geometry, ["read", "replace", "moved"]);
        assert_eq!(b.replaced, Some((7, 6)));
        assert_eq!(b.scene.draw.objects[0].vertex_count, 4);
        assert_eq!(b.scene.draw.objects[1].vertex_count, 3);
    }
}
