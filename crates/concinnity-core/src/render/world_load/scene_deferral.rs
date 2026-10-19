//! The content a world's load leaves to streaming because a scene other than
//! the start scene exclusively owns it: texture slots, mesh sources and shader
//! buckets whose payloads are not read at init. The streaming pools bring them
//! in when their scene pins, so with streaming off, or no scenes, or a blob
//! without baked scene groups, everything decodes at init.

use alloc::vec::Vec;
use hashbrown::HashSet;

use crate::ecs::asset_id::AssetId;
use crate::ecs::{BlobMeshBounds, BlobSceneGroups, PipelineContext, ResourceKind};
use crate::render::draw_list::DeferredMeshSources;
use crate::render::scene_flow::start_scene;

/// Texture slots (below `slot_count`) whose payloads init skips.
pub fn deferred_texture_slots(
    ctx: &PipelineContext,
    streaming: bool,
    slot_count: usize,
) -> HashSet<usize> {
    let mut deferred = HashSet::new();
    if !streaming {
        return deferred;
    }
    let Some(start) = start_scene(ctx) else {
        return deferred;
    };
    let Some(groups) = ctx.resource::<BlobSceneGroups>() else {
        return deferred;
    };
    for group in &groups.0 {
        if group.scene == start {
            continue;
        }
        for &(kind, handle) in &group.resources {
            if kind == ResourceKind::Texture && (handle as usize) < slot_count {
                deferred.insert(handle as usize);
            }
        }
    }
    deferred
}

/// Mesh sources whose payload decode init defers, with the baked bounds and
/// counts their draw records are built from. Sources without a baked bounds
/// record fall out (they decode eagerly).
pub fn deferred_mesh_sources(ctx: &PipelineContext, streaming: bool) -> DeferredMeshSources {
    let mut out = DeferredMeshSources::default();
    if !streaming {
        return out;
    }
    let Some(start) = start_scene(ctx) else {
        return out;
    };
    let Some(groups) = ctx.resource::<BlobSceneGroups>() else {
        return out;
    };
    let Some(baked) = ctx.resource::<BlobMeshBounds>() else {
        return out;
    };
    for record in &baked.0 {
        out.bounds.insert(record.handle, (record.min, record.max));
        out.counts
            .insert(record.handle, (record.vertex_count, record.index_count));
    }
    for group in &groups.0 {
        if group.scene == start {
            continue;
        }
        for &(kind, handle) in &group.resources {
            if kind == ResourceKind::Mesh {
                out.by_handle.insert(handle);
            }
        }
        for &def in &group.defs {
            out.by_def.insert(def);
        }
    }
    out
}

/// Shader buckets whose payload decode and pipeline build init skips, each
/// paired with the scene that exclusively owns it, in bucket order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShaderDeferral {
    scenes: Vec<(u32, AssetId)>,
}

impl ShaderDeferral {
    /// The buckets a streamed world defers: those a non-start scene owns.
    /// `shader_ids` is the world's Shaders in bucket order. Bucket 0, the world
    /// default program, is never deferred.
    pub fn plan(ctx: &PipelineContext, streaming: bool, shader_ids: &[Option<AssetId>]) -> Self {
        let mut scenes = Vec::new();
        if !streaming {
            return Self { scenes };
        }
        let Some(start) = start_scene(ctx) else {
            return Self { scenes };
        };
        let Some(groups) = ctx.resource::<BlobSceneGroups>() else {
            return Self { scenes };
        };
        for group in &groups.0 {
            if group.scene == start {
                continue;
            }
            for &def in &group.defs {
                if let Some(bucket) = shader_ids.iter().position(|&id| id == Some(def))
                    && bucket > 0
                {
                    scenes.push((bucket as u32, group.scene));
                }
            }
        }
        scenes.sort_unstable();
        Self { scenes }
    }

    /// Whether init skips `bucket`.
    pub fn is_deferred(&self, bucket: u32) -> bool {
        self.scenes.iter().any(|&(b, _)| b == bucket)
    }

    /// A deferred bucket whose source cannot be kept for later still has to
    /// render: init builds it, and no scene claims it any more.
    pub fn build_at_init(&mut self, bucket: u32) {
        self.scenes.retain(|&(b, _)| b != bucket);
    }

    /// Each deferred bucket with the scene that owns it.
    pub fn scenes(&self) -> &[(u32, AssetId)] {
        &self.scenes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Scene;
    use crate::ecs::{MeshBoundsRecord, SceneGroup, World};
    use alloc::vec;

    // The declared Scenes of a deferral test, in declaration order, so the
    // first is the start scene the deferral spares.
    fn residency_world(scenes: &[AssetId]) -> World {
        let mut world = World::new();
        for &id in scenes {
            world.push_identified(id, Scene { camera_shot: None });
        }
        world
    }

    fn group(
        scene: AssetId,
        resources: Vec<(ResourceKind, u32)>,
        defs: Vec<AssetId>,
    ) -> SceneGroup {
        SceneGroup {
            scene,
            resources,
            defs,
        }
    }

    fn bounds_record(handle: u32, vertex_count: u32, index_count: u32) -> MeshBoundsRecord {
        MeshBoundsRecord {
            handle,
            min: [-1.0; 3],
            max: [1.0; 3],
            vertex_count,
            index_count,
        }
    }

    const TEXTURE_KIND: ResourceKind = ResourceKind::Texture;
    const MESH_KIND: ResourceKind = ResourceKind::Mesh;
    const START: AssetId = AssetId(10);
    const LATER: AssetId = AssetId(11);

    // The start scene renders on the first frame, so its slots decode eagerly.
    #[test]
    fn texture_deferral_spares_the_start_scene() {
        let mut world = residency_world(&[START, LATER]);
        world.insert_resource(BlobSceneGroups(vec![
            group(START, vec![(TEXTURE_KIND, 0)], Vec::new()),
            group(
                LATER,
                vec![(TEXTURE_KIND, 1), (TEXTURE_KIND, 2)],
                Vec::new(),
            ),
        ]));
        let deferred = deferred_texture_slots(&world.context(), true, 8);
        assert_eq!(deferred, HashSet::from([1, 2]));
    }

    // An entry that is not a Texture, or whose handle is past the streamed slot
    // count, is not a texture slot this path can defer.
    #[test]
    fn texture_deferral_ignores_other_kinds_and_out_of_range_handles() {
        let mut world = residency_world(&[START, LATER]);
        world.insert_resource(BlobSceneGroups(vec![group(
            LATER,
            vec![(TEXTURE_KIND, 1), (MESH_KIND, 2), (TEXTURE_KIND, 9)],
            Vec::new(),
        )]));
        let deferred = deferred_texture_slots(&world.context(), true, 3);
        assert_eq!(deferred, HashSet::from([1]));
    }

    #[test]
    fn texture_deferral_is_empty_without_streaming_scenes_or_groups() {
        let groups = vec![group(LATER, vec![(TEXTURE_KIND, 1)], Vec::new())];

        let mut unstreamed = residency_world(&[START, LATER]);
        unstreamed.insert_resource(BlobSceneGroups(groups.clone()));
        assert!(deferred_texture_slots(&unstreamed.context(), false, 8).is_empty());

        let mut sceneless = residency_world(&[]);
        sceneless.insert_resource(BlobSceneGroups(groups));
        assert!(deferred_texture_slots(&sceneless.context(), true, 8).is_empty());

        let mut ungrouped = residency_world(&[START, LATER]);
        assert!(deferred_texture_slots(&ungrouped.context(), true, 8).is_empty());
    }

    // Mesh deferral covers both mesh-source forms (resource-stream handles and
    // payload-carrying defs), while every baked record is kept: they are looked
    // up by handle for whichever sources end up deferred.
    #[test]
    fn mesh_deferral_marks_later_scene_sources_and_keeps_every_baked_record() {
        let start_def = AssetId(20);
        let later_def = AssetId(21);
        let mut world = residency_world(&[START, LATER]);
        world.insert_resource(BlobSceneGroups(vec![
            group(START, vec![(MESH_KIND, 0)], vec![start_def]),
            group(
                LATER,
                vec![(MESH_KIND, 1), (TEXTURE_KIND, 5)],
                vec![later_def],
            ),
        ]));
        world.insert_resource(BlobMeshBounds(vec![
            bounds_record(0, 24, 36),
            bounds_record(1, 8, 12),
        ]));

        let sources = deferred_mesh_sources(&world.context(), true);
        assert_eq!(sources.by_handle, HashSet::from([1]));
        assert_eq!(sources.by_def, HashSet::from([later_def]));
        assert_eq!(sources.counts.get(&0), Some(&(24, 36)));
        assert_eq!(sources.counts.get(&1), Some(&(8, 12)));
        assert_eq!(sources.bounds.get(&1), Some(&([-1.0; 3], [1.0; 3])));
    }

    // A source with no baked bounds record cannot be deferred: init has nothing
    // to size its draw record from.
    #[test]
    fn mesh_deferral_is_empty_without_streaming_or_baked_bounds() {
        let groups = vec![group(LATER, vec![(MESH_KIND, 1)], Vec::new())];

        let mut unstreamed = residency_world(&[START, LATER]);
        unstreamed.insert_resource(BlobSceneGroups(groups.clone()));
        unstreamed.insert_resource(BlobMeshBounds(vec![bounds_record(1, 8, 12)]));
        assert!(
            deferred_mesh_sources(&unstreamed.context(), false)
                .by_handle
                .is_empty()
        );

        let mut unbaked = residency_world(&[START, LATER]);
        unbaked.insert_resource(BlobSceneGroups(groups));
        let sources = deferred_mesh_sources(&unbaked.context(), true);
        assert!(sources.by_handle.is_empty());
        assert!(sources.counts.is_empty());
    }

    // A Shader owned by a later scene is deferred with that scene. Bucket 0 is
    // the world default program: the world cannot render without it.
    #[test]
    fn shader_deferral_names_the_owning_scene_and_spares_bucket_zero() {
        const DEFAULT_SHADER: AssetId = AssetId(20);
        const SCENE_SHADER: AssetId = AssetId(21);
        const OTHER_DEF: AssetId = AssetId(22);
        let shaders = [Some(DEFAULT_SHADER), Some(SCENE_SHADER)];

        let mut world = residency_world(&[START, LATER]);
        world.insert_resource(BlobSceneGroups(vec![
            group(START, Vec::new(), vec![DEFAULT_SHADER]),
            group(LATER, Vec::new(), vec![SCENE_SHADER, OTHER_DEF]),
        ]));
        assert_eq!(
            ShaderDeferral::plan(&world.context(), true, &shaders).scenes(),
            [(1, LATER)]
        );

        let mut owned_default = residency_world(&[START, LATER]);
        owned_default.insert_resource(BlobSceneGroups(vec![group(
            LATER,
            Vec::new(),
            vec![DEFAULT_SHADER],
        )]));
        assert!(
            ShaderDeferral::plan(&owned_default.context(), true, &shaders)
                .scenes()
                .is_empty()
        );
    }

    // A bucket that cannot stay deferred is built at init and leaves its scene.
    #[test]
    fn a_bucket_built_at_init_is_no_longer_deferred() {
        let mut deferral = ShaderDeferral {
            scenes: vec![(1, LATER), (2, LATER)],
        };
        assert!(deferral.is_deferred(1));
        deferral.build_at_init(1);
        assert!(!deferral.is_deferred(1));
        assert_eq!(deferral.scenes(), [(2, LATER)]);
    }

    #[test]
    fn shader_deferral_is_empty_without_streaming_scenes_or_groups() {
        let shaders = [Some(AssetId(20)), Some(AssetId(21))];
        let groups = vec![group(LATER, Vec::new(), vec![AssetId(21)])];

        let mut unstreamed = residency_world(&[START, LATER]);
        unstreamed.insert_resource(BlobSceneGroups(groups.clone()));
        assert!(
            ShaderDeferral::plan(&unstreamed.context(), false, &shaders)
                .scenes()
                .is_empty()
        );

        let mut sceneless = residency_world(&[]);
        sceneless.insert_resource(BlobSceneGroups(groups));
        assert!(
            ShaderDeferral::plan(&sceneless.context(), true, &shaders)
                .scenes()
                .is_empty()
        );

        let mut ungrouped = residency_world(&[START, LATER]);
        assert!(
            ShaderDeferral::plan(&ungrouped.context(), true, &shaders)
                .scenes()
                .is_empty()
        );
    }
}
