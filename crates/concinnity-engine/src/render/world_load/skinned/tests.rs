use super::*;
use crate::components::SkeletonPose;
use crate::ecs::World;
use alloc::vec;

fn vertex(pos: [f32; 3]) -> SkinnedVertex {
    SkinnedVertex {
        pos,
        normal: [0.0, 1.0, 0.0],
        tangent: [1.0, 0.0, 0.0],
        color: [1.0; 3],
        uv: [0.0; 2],
        joints: [0; 4],
        weights: [1.0, 0.0, 0.0, 0.0],
    }
}

fn geometry(handle: u32, max_instances: u32, material: Option<MaterialHandle>) -> SkinnedGeometry {
    SkinnedGeometry {
        handle: SkinnedMeshHandle::new(handle),
        name_id: AssetId(100 + handle),
        mesh: SkinnedMesh {
            material,
            max_instances,
            ..Default::default()
        },
        vertices: vec![
            vertex([-1.0, 0.0, 0.0]),
            vertex([2.0, 3.0, 0.0]),
            vertex([0.0, 0.0, -4.0]),
        ],
        indices: vec![0, 1, 2],
        joint_defs: vec![SkeletonJoint::default()],
        morphs: PayloadMorphs::default(),
        lod_alternates: Vec::new(),
    }
}

// Each mesh is followed by its hidden copies, each in its own vertex region
// with indices rebased onto it, and every copy is paired with its template.
#[test]
fn copies_follow_their_template_in_their_own_regions() {
    let meshes = [geometry(0, 2, None), geometry(1, 0, None)];
    let out = assemble_skinned(&meshes, &HashMap::new()).expect("assembles");
    assert_eq!(out.draw_objects.len(), 4);
    assert_eq!(out.vertices.len(), 12);
    let visible: Vec<bool> = out.draw_objects.iter().map(|d| d.visible).collect();
    assert_eq!(visible, [true, false, false, true]);
    let bases: Vec<u32> = out.draw_objects.iter().map(|d| d.vertex_base).collect();
    assert_eq!(bases, [0, 3, 6, 9]);
    assert_eq!(&out.indices[3..6], &[3, 4, 5]);
    assert_eq!(
        out.pool_reservations,
        [
            (SkinnedIndex::from_usize(0), SkinnedIndex::from_usize(1)),
            (SkinnedIndex::from_usize(0), SkinnedIndex::from_usize(2)),
        ]
    );
    let templates: Vec<SkinnedIndex> = out.templates.iter().map(|t| t.template_index).collect();
    assert_eq!(
        templates,
        [SkinnedIndex::from_usize(0), SkinnedIndex::from_usize(3)]
    );
    assert_eq!(
        out.templates[0].local_bounds,
        ([-1.0, 0.0, -4.0], [2.0, 3.0, 0.0])
    );
    assert_eq!(out.draw_objects[1].local_bb_max, [2.0, 3.0, 0.0]);
}

#[test]
fn a_mesh_naming_an_unknown_material_fails_the_load() {
    let meshes = [geometry(0, 0, Some(MaterialHandle::new(7)))];
    let err = assemble_skinned(&meshes, &HashMap::new())
        .err()
        .expect("fails");
    assert!(err.contains("unknown material 7"), "{err}");
}

#[test]
fn an_empty_mesh_has_zero_bounds() {
    assert_eq!(bind_pose_bounds(&[]), ([0.0; 3], [0.0; 3]));
}

// Each template becomes an identified posed entity; one with a capsule gets a
// rig, and a published pick index makes it pickable.
#[test]
fn templates_spawn_posed_and_rigged_and_pickable() {
    let mut rigged = geometry(0, 0, None);
    rigged.mesh.capsule = Some(CharacterCapsule {
        half_height: 0.9,
        radius: 0.01,
    });
    let out = assemble_skinned(&[rigged, geometry(1, 0, None)], &HashMap::new()).unwrap();
    let mut world = World::new();
    world.insert_resource(PickIndex::default());
    let mut picks = Vec::new();
    spawn_skinned_templates(&mut world.context(), out.templates, &mut picks);
    let ctx = world.context();
    assert_eq!(ctx.query::<SkeletonPose>().count(), 2);
    let rig = ctx.query::<CharacterRig>().next().expect("one rig");
    assert_eq!(ctx.query::<CharacterRig>().count(), 1);
    assert_eq!(rig.radius, 0.05, "a thin capsule is clamped");
    assert_eq!(picks.len(), 2);
    assert!(ctx.entity_of(AssetId(101)).is_some());
}

// Under a pick index a template is placed by its authored Transform, so the
// extracted index follows a move of it and the move reaches the skinned model
// push; without one the template stays a bare pose.
#[test]
fn a_pickable_template_follows_its_transform_into_the_index_and_the_model() {
    use crate::components::Transform;
    use crate::render::frame::extract::FrameExtractor;
    use crate::render::frame::snapshot::RenderSnapshot;

    let mut placed = geometry(0, 0, None);
    placed.mesh.position = [5.0, 0.0, 0.0];
    let out = assemble_skinned(&[placed], &HashMap::new()).unwrap();
    let template = out.templates[0].template_index;
    let mut world = World::new();
    world.insert_resource(PickIndex::default());
    let mut fx = FrameExtractor::default();
    spawn_skinned_templates(
        &mut world.context(),
        out.templates,
        fx.pick_candidates_mut(),
    );
    assert_eq!(fx.pick_candidates().len(), 1);
    let mut extract = |world: &mut World| {
        let mut snap = RenderSnapshot::default();
        fx.extract(&mut world.context(), &mut snap, 0);
        let bb_min = world.resource::<PickIndex>().unwrap().entries[0].bb_min[0];
        (snap, bb_min)
    };
    let (_, bb_min) = extract(&mut world);
    assert!(bb_min >= 5.0, "bounds follow the authored position");

    let entity = world.context().entity_of(AssetId(100)).unwrap();
    world
        .context()
        .get_mut::<Transform>(entity)
        .unwrap()
        .position = [7.0, 0.0, 0.0];
    let (snap, bb_min) = extract(&mut world);
    assert!(snap.skinned_models.iter().any(|(idx, _)| *idx == template));
    assert!(bb_min >= 7.0, "the index follows the move");

    let out = assemble_skinned(&[geometry(0, 0, None)], &HashMap::new()).unwrap();
    let mut world = World::new();
    let mut picks = Vec::new();
    spawn_skinned_templates(&mut world.context(), out.templates, &mut picks);
    assert!(picks.is_empty());
    assert_eq!(world.context().query::<Transform>().count(), 0);
}
