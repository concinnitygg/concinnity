use super::*;
use concinnity_engine::animation::skinned_mesh_map::SkinnedMeshNameIndex;
use concinnity_engine::components::{CharacterCapsule, SkinnedMesh};
use concinnity_engine::ecs::{ResourceKind, ResourceRecord};
use concinnity_engine::render::frame::queues::ActiveRenderQueues;

// Morph targets upload right after the skinned geometry, and a failed morph
// upload fails init the way a failed geometry upload does.
#[test]
fn morph_target_upload_failure_fails_init() {
    const MORPHED: AssetId = AssetId(846);
    let delta = mesh_payload::MorphDelta {
        position: [0.0, 1.0, 0.0],
        normal: [0.0; 3],
    };
    let morphs =
        mesh_payload::PayloadMorphs::from_dense(vec!["raise".to_string()], 3, &[delta; 3]).unwrap();
    let payload = skinned_payload_with_morphs(3, &[], &morphs);

    for fail in [
        None,
        Some(error::RenderError::Other(
            "morph buffer rejected".to_string(),
        )),
    ] {
        let failing = fail.is_some();
        let (state, hooks) = recording_hooks();
        lock(&state).fail_morph_upload = fail;
        let mut b = scene_builder();
        push_skinned_payload(&mut b, MORPHED, SkinnedMesh::default(), &payload);
        let mut world = b.build();
        let gs = init_graphics(&mut world, hooks);

        assert_eq!(gs.failed, failing);
        let s = lock(&state);
        assert!(s.saw(&Call::UploadSkinnedMorphs), "{:?}", s.calls);
    }
}

// A SkinnedMesh's geometry is decoded, merged into the shared skinned buffers,
// and uploaded; each mesh publishes a SkeletonPose for AnimationSystem to drive,
// registers its name so a runtime spawn can resolve the template, and a mesh
// with a capsule additionally gets a CharacterRig for PhysicsSystem.
#[test]
fn skinned_mesh_world_uploads_geometry_and_publishes_poses() {
    const RIGGED: AssetId = AssetId(840);
    const PLAIN: AssetId = AssetId(841);

    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    push_skinned_mesh(
        &mut b,
        RIGGED,
        SkinnedMesh {
            material: Some(MaterialHandle::new(0)),
            position: [5.0, 0.0, 0.0],
            capsule: Some(CharacterCapsule {
                half_height: 0.9,
                radius: 0.3,
            }),
            ..Default::default()
        },
        4,
    );
    push_skinned_mesh(&mut b, PLAIN, SkinnedMesh::default(), 3);
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    {
        let s = lock(&state);
        assert!(
            s.saw(&Call::UploadSkinned {
                // The two meshes' vertices merge into one shared buffer.
                vertices: 7,
                draws: 2,
            }),
            "both meshes' geometry rides one upload: {:?}",
            s.calls
        );
        // Skinned draws are counted at construction so the GPU-cull buffers are
        // sized for the merged total, even though the geometry uploads after.
        assert_eq!(s.init.as_ref().unwrap().n_skinned, 2);
    }

    let ctx = world.context();
    // One SkeletonPose per mesh, each keyed to its handle and template draw.
    let mut poses: Vec<(u32, SkinnedIndex)> = ctx
        .query::<SkeletonPose>()
        .map(|p| (p.mesh_id.0, p.skinned_index))
        .collect();
    poses.sort();
    assert_eq!(poses, vec![(0, SkinnedIndex(0)), (1, SkinnedIndex(1))]);
    // Only the capsule-carrying mesh gets a rig.
    let rigs: Vec<u32> = ctx.query::<CharacterRig>().map(|r| r.target.0).collect();
    assert_eq!(rigs, vec![0], "only the capsule mesh gets a character rig");
    // The name index the debug animation commands address a mesh by.
    let names = ctx
        .resource::<SkinnedMeshNameIndex>()
        .expect("name index published before AnimationSystem inits");
    assert_eq!(names.0.get(&RIGGED).map(|h| h.0), Some(0));
    assert_eq!(names.0.get(&PLAIN).map(|h| h.0), Some(1));
    // Each template is registered under its mesh name, so a runtime SpawnRequest
    // resolves it the same way a static placement resolves.
    assert!(ctx.entity_of(RIGGED).is_some());
    assert!(ctx.entity_of(PLAIN).is_some());
}

// `max_instances` pre-reserves hidden bind-pose copies of a skinned mesh, each
// with its OWN vertex region in the shared buffer: the GPU skin fold writes the
// deformed buffer keyed by global vertex index, so two live instances sharing a
// region would clobber each other's pose. A runtime spawn reveals one of these
// without growing any GPU buffer.
#[test]
fn skinned_instance_reserves_get_their_own_vertex_regions() {
    const HERO: AssetId = AssetId(842);
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    push_skinned_mesh(
        &mut b,
        HERO,
        SkinnedMesh {
            max_instances: 3,
            ..Default::default()
        },
        4,
    );
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let s = lock(&state);
    assert!(
        s.saw(&Call::UploadSkinned {
            // The template plus three copies, each carrying its own 4 vertices.
            vertices: 16,
            draws: 4,
        }),
        "each reserved copy owns a distinct vertex region: {:?}",
        s.calls
    );
    assert_eq!(
        world
            .resource::<ActiveRenderQueues>()
            .and_then(|slot| slot.0.as_ref())
            .expect("graphics init publishes the recording surfaces")
            .slots
            .skinned_free(),
        3,
        "the engine's instance pool is seeded with the reserved copies"
    );
    assert_eq!(s.init.as_ref().unwrap().n_skinned, 4);
}

// A SkinnedMesh referencing a material that is not in the table fails the world
// build: a silent fallback would render the character with the wrong surface.
#[test]
fn a_skinned_mesh_with_an_unknown_material_fails_init() {
    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    push_skinned_mesh(
        &mut b,
        AssetId(843),
        SkinnedMesh {
            material: Some(MaterialHandle::new(99)),
            ..Default::default()
        },
        3,
    );
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(gs.failed);
    assert!(!backend_parked(&world));
}

// A SkinnedMesh whose geometry payload is malformed, and one with no compiled
// payload at all, each fail the world build rather than dropping the character.
#[test]
fn a_skinned_mesh_without_usable_geometry_fails_init() {
    let record = |b: &mut WorldBuilder, payload: Option<PayloadLocator>| {
        let data = postcard::to_allocvec(&(844u32, SkinnedMesh::default())).unwrap();
        b.skinned_records.push(ResourceRecord {
            resource_kind: ResourceKind::SkinnedMesh,
            handle: 0,
            payload,
            data_bytes: data,
        });
    };

    // Malformed payload bytes (no "SKMV" magic).
    let mut b = scene_builder();
    let loc = b.payload(b"not-a-skinned-mesh");
    record(&mut b, Some(loc));
    let (_state, hooks) = recording_hooks();
    let mut world = b.build();
    assert!(init_graphics(&mut world, hooks).failed, "malformed payload");

    // No payload at all (the build never compiled the geometry).
    let mut b = scene_builder();
    record(&mut b, None);
    let (_state, hooks) = recording_hooks();
    let mut world = b.build();
    assert!(init_graphics(&mut world, hooks).failed, "no payload");

    // Baked data that does not decode as a SkinnedMesh record.
    let mut b = scene_builder();
    b.skinned_records.push(ResourceRecord {
        resource_kind: ResourceKind::SkinnedMesh,
        handle: 0,
        payload: None,
        data_bytes: vec![0xFF; 3],
    });
    let (_state, hooks) = recording_hooks();
    let mut world = b.build();
    assert!(init_graphics(&mut world, hooks).failed, "undecodable data");
}

// The per-frame skinned push: a pose reaches the GPU when its matrices were
// rewritten (flagged by whoever wrote them), so an unanimated mesh uploads
// its seeded bind pose once and nothing after. A rewritten pose uploads
// again, unless a menu is open (animation is frozen behind it).
#[test]
fn skinned_poses_upload_when_flagged_and_freeze_behind_a_menu() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    push_skinned_mesh(&mut b, AssetId(845), SkinnedMesh::default(), 3);
    let mut world = b.build();
    let mut gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let flag_pose = |world: &mut World| {
        let mut ctx = world.context();
        let pose = ctx.query_mut::<SkeletonPose>().next().unwrap();
        pose.updated = true;
    };

    step(&mut gs, &mut world);
    assert!(
        lock(&state).saw(&Call::UpdateSkinnedPose(SkinnedIndex(0))),
        "the seeded bind pose is pushed on the first frame"
    );

    // No animation rewrote the pose: the next frame pushes nothing.
    lock(&state).calls.clear();
    step(&mut gs, &mut world);
    assert!(
        !lock(&state).saw(&Call::UpdateSkinnedPose(SkinnedIndex(0))),
        "an untouched pose is not re-uploaded"
    );

    // A rewritten pose (flagged the way AnimationSystem does) is pushed once.
    flag_pose(&mut world);
    lock(&state).calls.clear();
    step(&mut gs, &mut world);
    assert!(
        lock(&state).saw(&Call::UpdateSkinnedPose(SkinnedIndex(0))),
        "a flagged pose is uploaded"
    );

    // Behind a real menu (the overlay reports it active) even a flagged pose
    // waits: the skinned draw is skipped behind it anyway.
    flag_pose(&mut world);
    world.insert_resource(ScreenStack {
        pauses_world: true,
        ..Default::default()
    });
    lock(&state).calls.clear();
    step(&mut gs, &mut world);
    assert!(
        !lock(&state).saw(&Call::UpdateSkinnedPose(SkinnedIndex(0))),
        "pose uploads are skipped while a menu is open"
    );

    // The editor's freeze keeps the world drawn and reseeds poses live, so
    // its override lets the flagged pose through.
    world.insert_resource(MenuOverride(Some(true)));
    lock(&state).calls.clear();
    step(&mut gs, &mut world);
    assert!(
        lock(&state).saw(&Call::UpdateSkinnedPose(SkinnedIndex(0))),
        "the editor override uploads a flagged pose"
    );
}

// A skinned mesh's LOD alternates share its vertex region: the runtime skinned
// index buffer is u16, so every alternate's mesh-relative indices are rebased
// onto the same vertex base as LOD0. Each reserved instance copy gets its own
// rebased set against its own region.
#[test]
fn skinned_lod_alternates_rebase_onto_their_slot_vertex_region() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // LOD0 is 3 indices; one alternate adds 3 more, all over the same 4 vertices.
    let locator = b.payload(&skinned_payload(4, &[(10.0, vec![0, 1, 2])]));
    let data = postcard::to_allocvec(&(
        852u32,
        SkinnedMesh {
            max_instances: 1,
            ..Default::default()
        },
    ))
    .unwrap();
    b.skinned_records.push(ResourceRecord {
        resource_kind: ResourceKind::SkinnedMesh,
        handle: 0,
        payload: Some(locator),
        data_bytes: data,
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let s = lock(&state);
    assert!(
        s.saw(&Call::UploadSkinned {
            // Template + one reserved copy, each with its own 4 vertices; the LOD
            // alternate reuses its slot's region rather than duplicating it.
            vertices: 8,
            draws: 2,
        }),
        "an LOD alternate shares its slot's vertex region: {:?}",
        s.calls
    );
    assert_eq!(
        world
            .resource::<ActiveRenderQueues>()
            .and_then(|slot| slot.0.as_ref())
            .expect("graphics init publishes the recording surfaces")
            .slots
            .skinned_free(),
        1,
        "the engine's instance pool is seeded with the reserved copy"
    );
}
