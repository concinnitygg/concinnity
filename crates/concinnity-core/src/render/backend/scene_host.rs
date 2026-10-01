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

use crate::render::error::RenderResult;
use crate::render::scene_state::{GeometryWriter, SceneState};

/// An edit of the scene that may write geometry bytes through the writer.
pub type GeometryEdit<'a> =
    &'a mut dyn FnMut(&mut SceneState, &mut dyn GeometryWriter) -> RenderResult<()>;

/// Lends a backend's [`SceneState`] to the scene-only defaults of
/// [`RenderBackend`](super::RenderBackend) and its operation families.
///
/// A backend with no scene of its own, such as a test double, answers `None`
/// throughout; the defaults then do nothing, or report
/// [`RenderError::Unsupported`](crate::render::error::RenderError::Unsupported)
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
    use crate::gfx::render_types::{DrawIndex, MaterialUniforms, SkinnedDrawObject, SkinnedIndex};
    use crate::input::snapshot::InputSnapshot;
    use crate::render::backend::{
        BackendProbe, DeviceCapabilities, DrawStreaming, FrameParams, LiveEdit, RenderBackend,
        RenderTuning, SceneEffects, SkinnedDraws, WindowControl,
    };
    use crate::render::draw_slot::SlotAlloc;
    use crate::render::scene_flow::SceneControl;
    use crate::render::scene_state::{DrawList, GeometryBuffer};
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
            let mut scene = SceneState::new(DrawList::unreserved(objects, 0), [0.0; 4]);
            scene
                .skinned
                .draw_objects
                .push(crate::test_support::skinned_draw_object());
            scene.skinned.joint_matrices.push(vec![MOVED]);
            scene.skinned.morph_weights.push(vec![0.0]);
            Self {
                scene,
                writes: Writes::default(),
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

    impl SceneControl for Hosted {
        fn update_visibility(&mut self, _: DrawIndex, _: bool) {}
        fn set_fade(&mut self, _: f32) {}
    }

    impl RenderBackend for Hosted {
        fn window_closed(&mut self) -> bool {
            false
        }
        fn request_cursor_capture(&mut self) {}
        fn take_input(&mut self) -> InputSnapshot {
            InputSnapshot::default()
        }
        fn wait_idle(&self) {}
        fn draw_frame(&mut self, _: FrameParams<'_>) -> RenderResult<()> {
            Ok(())
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

    impl DrawStreaming for Hosted {
        fn evict_texture_slot(&mut self, _: usize) -> RenderResult<()> {
            Ok(())
        }
        fn update_texture_slot(
            &mut self,
            _: usize,
            _: &crate::bake::texture::TextureImage,
        ) -> RenderResult<()> {
            Ok(())
        }
        fn setup_chunk_streaming(&mut self, _: usize, _: usize) -> RenderResult<()> {
            Ok(())
        }
    }

    impl BackendProbe for Hosted {
        fn capabilities(&self) -> DeviceCapabilities {
            DeviceCapabilities::ALL
        }
    }
    impl LiveEdit for Hosted {}
    impl RenderTuning for Hosted {}
    impl SceneEffects for Hosted {}
    impl WindowControl for Hosted {}

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
        b.update_skinned_pose(SkinnedIndex(0), &[MOVED, MOVED]);
        b.update_morph_weights(SkinnedIndex(0), &[0.5]);
        b.update_skinned_models(&[(SkinnedIndex(0), MOVED)]);
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
        b.seed_mesh_streaming(0, 3 * core::mem::size_of::<Vertex>() as u64, 0, 12);
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
            .free(0, 3 * core::mem::size_of::<Vertex>() as u64, 0);
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
}
