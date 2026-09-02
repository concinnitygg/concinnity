// The parity oracle between the two ways a world gets its computed data.
//
// A payload baked by the `bake` functions and handed to the world's data-entry
// methods must be the same bytes, at the same handle, as the cook pipeline
// compiles for the same typed declaration. The two run different code -- the
// cook goes through an authored JSON form, its expansion passes, and the
// payload cache; the bake functions work from the typed values directly -- so
// nothing but a test holds them to each other.
//
// The reference declarations are the cube example's, with the image-based
// lighting at a fraction of the example's cube-face size so the test is cheap;
// the convolution, the payload format, and the code path either side reaches
// are the same at any size.

use crate::components::{Camera3D, ProceduralMesh, Prop};
use crate::{World, bake, cook};

const PREFILTER_FACE: u32 = 32;
const IRRADIANCE_FACE: u32 = 8;
const PREFILTER_SAMPLES: u32 = 16;

fn cube_mesh() -> ProceduralMesh {
    ProceduralMesh {
        generator: "box".to_string(),
        half_extents: Some([0.7, 0.7, 0.7]),
        ..Default::default()
    }
}

fn cube_material() -> bake::Material {
    bake::Material {
        roughness: 0.20,
        metallic: 0.30,
        tint: [0.88, 0.86, 0.84],
        ..Default::default()
    }
}

fn camera_args() -> bake::Camera3D {
    bake::Camera3D {
        fov_y_degrees: 40.0,
        near: 0.05,
        far: 100.0,
        position: [0.0, 2.0, 8.0],
        yaw: 0.0,
        pitch: -0.245,
        controller: None,
    }
}

fn sky() -> bake::EnvironmentMap {
    bake::EnvironmentMap {
        generator: "sky".to_string(),
        prefilter_face_size: PREFILTER_FACE,
        irradiance_face_size: IRRADIANCE_FACE,
        prefilter_samples: PREFILTER_SAMPLES,
        ..Default::default()
    }
}

// The cube's declarations compiled by the cook: the reference the raw path is
// held to.
fn cooked_world() -> World {
    let mut spec = cook::world();
    spec.add("camera", camera_args())
        .add("sky", sky())
        .add("cube_mesh", cube_mesh())
        .add("cube_material", cube_material())
        .add("cube", Prop::default())
        .reference("mesh", "cube_mesh")
        .reference("material", "cube_material");
    spec.compile().expect("the declared world cooks")
}

// The same declarations baked in place and handed over raw.
fn raw_world() -> World {
    let mut world = World::new();
    world.add_component(bake::camera(camera_args()));
    let ibl = bake::environment_map(&sky()).expect("the sky bakes");
    world.add_environment_map(ibl);
    let mesh = cube_mesh();
    let payload = bake::procedural_mesh(&mesh).expect("the box bakes");
    let mesh = world.add_mesh(mesh, payload);
    let material = world.add_material(cube_material());
    world.add_component(Prop {
        mesh: Some(mesh),
        material: Some(material),
        ..Default::default()
    });
    world
}

// The compiled geometry behind a world's one ProceduralMesh, wherever that
// world keeps it: a cooked mesh's payload rides its locator, a raw one's sits
// in the runtime payload store.
fn mesh_payload(world: &mut World) -> Vec<u8> {
    let mesh = world
        .inner()
        .query::<ProceduralMesh>()
        .next()
        .expect("the cube's mesh")
        .clone();
    match mesh.locator {
        Some(locator) => world
            .inner_mut()
            .context()
            .blob
            .read(&locator)
            .expect("the payload reads back")
            .to_vec(),
        None => world
            .inner()
            .resource::<concinnity_core::resource::RuntimeMeshPayloads>()
            .expect("the runtime payload store")
            .get(mesh.asset_id)
            .expect("the mesh's payload")
            .to_vec(),
    }
}

// The compiled image-based lighting behind a world's environment map.
fn environment_payload(world: &mut World) -> Vec<u8> {
    let entry = world
        .inner()
        .resource::<concinnity_core::resource::EnvironmentMapTable>()
        .expect("the environment map table")
        .0
        .first()
        .expect("the sky is at handle 0")
        .clone();
    if let Some(baked) = entry.baked_bytes() {
        return baked.to_vec();
    }
    match entry.payload {
        Some(locator) => world
            .inner_mut()
            .context()
            .blob
            .read(&locator)
            .expect("the payload reads back")
            .to_vec(),
        None => panic!("an environment map entry carries its payload"),
    }
}

// The generated geometry is the same bytes either way: the typed value carries
// every generator argument, so the authored path's own defaults never come
// into it.
#[test]
fn the_generated_mesh_payload_is_byte_identical() {
    assert_eq!(
        mesh_payload(&mut raw_world()),
        mesh_payload(&mut cooked_world())
    );
}

// The image-based lighting is the same bytes either way: one convolution
// serves both, and the row schedule it is spread over cannot change a texel.
#[test]
fn the_environment_map_payload_is_byte_identical() {
    assert_eq!(
        environment_payload(&mut raw_world()),
        environment_payload(&mut cooked_world())
    );
}

// A material's bytes are its clamped parameters; `add_material` runs the same
// registered validator the authored path does.
#[test]
fn the_material_data_bytes_are_byte_identical() {
    let bytes = |world: &World| {
        world
            .inner()
            .resource::<concinnity_core::resource::MaterialTable>()
            .expect("the material table")
            .data_bytes(0)
            .expect("the cube's material is at handle 0")
            .to_vec()
    };
    assert_eq!(bytes(&raw_world()), bytes(&cooked_world()));
}

// The handle a data-entry method returns is the same index the cook resolves
// the equivalent named reference to, so a Prop reads the same table slot
// whichever path produced its world.
#[test]
fn the_resolved_handles_match() {
    let handles = |world: &World| {
        let prop = world.inner().query::<Prop>().next().expect("the cube prop");
        (
            prop.mesh.map(|h| h.index()),
            prop.material.map(|h| h.index()),
        )
    };
    assert_eq!(handles(&raw_world()), handles(&cooked_world()));
    assert_eq!(handles(&raw_world()), (Some(0), Some(0)));
}

// The baked camera is the compiled camera: same view, same lens.
#[test]
fn the_baked_camera_matches_the_cooked_one() {
    let cooked_world = cooked_world();
    let cooked = cooked_world
        .inner()
        .query::<Camera3D>()
        .next()
        .expect("the cooked camera");
    let baked = bake::camera(camera_args());
    assert_eq!(baked.view_matrix, cooked.view_matrix);
    assert_eq!(baked.fov_y_degrees, cooked.fov_y_degrees);
    assert_eq!(baked.position, cooked.position);
}

// Both worlds start: the raw one's data is not just byte-equal but runnable.
#[test]
fn both_worlds_start() {
    for world in [raw_world(), cooked_world()] {
        crate::test_support::assert_starts_headless(crate::App::from_world(world));
    }
}
