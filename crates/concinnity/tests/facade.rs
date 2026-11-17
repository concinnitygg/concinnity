//! The facade end to end: a world assembled through `concinnity`'s own paths
//! starts, and a blob the `cook` module writes loads back into a runnable app.

use concinnity::World;
use concinnity::components::TextLabel;

// A world holding anything renderable resolves to a windowed run, so windows
// are banned first: a regression that takes the windowed loop panics naming the
// backend instead of blocking on an event loop the harness cannot end.
#[cfg(feature = "std")]
fn assert_starts_headless(app: concinnity::App) {
    concinnity_testing::forbid_windows();
    let mut driver = app.into_headless().into_driver();
    driver.start().expect("the world starts");
}

fn hello() -> TextLabel {
    TextLabel {
        content: "Hello, world!".into(),
        ..Default::default()
    }
}

#[test]
fn a_world_holds_the_components_added_to_it() {
    let mut world = World::new();
    world.add_component(hello());

    assert_eq!(world.component_count(), 1);
    assert_eq!(
        world.query::<TextLabel>().next().unwrap().content,
        "Hello, world!"
    );
}

// The headless loop runs the world it was handed until nothing is left to
// step, on whichever tier is built.
#[test]
fn a_headless_app_runs_to_completion() {
    let mut world = World::new();
    world.add_component(hello());

    concinnity::App::from_world(world)
        .into_headless()
        .run()
        .expect("the run ends");
}

// A world with nothing to draw takes the headless loop by default.
#[cfg(feature = "std")]
#[test]
fn a_world_with_nothing_to_draw_starts_headless() {
    use concinnity::components::PhysicsConfig;

    let mut world = World::new();
    world.add_component(PhysicsConfig::default());

    concinnity_testing::forbid_windows();
    let mut driver = concinnity::App::from_world(world).into_driver();
    driver.start().expect("the world starts");
}

#[cfg(feature = "std")]
#[test]
fn a_world_asking_for_headless_starts_headless() {
    use concinnity::components::AppConfig;

    let mut world = World::new();
    world.add_component(hello());
    world.add_component(AppConfig {
        headless: true,
        ..Default::default()
    });

    concinnity_testing::forbid_windows();
    let mut driver = concinnity::App::from_world(world).into_driver();
    driver.start().expect("the world starts");
}

// The doc example's whole path, minus the window: bake, hand over, run.
#[cfg(feature = "std")]
#[test]
fn baked_data_reaches_a_world_that_starts() {
    use concinnity::bake;
    use concinnity::components::{ProceduralMesh, Prop};

    let mesh = ProceduralMesh {
        generator: "box".into(),
        half_extents: Some([0.7, 0.7, 0.7]),
        ..Default::default()
    };
    let mut world = World::new();
    let mesh = world
        .add_mesh(bake::procedural_mesh(mesh).expect("the box bakes"))
        .expect("the first mint");
    let stone = world.add_material(bake::Material {
        roughness: 0.2,
        ..Default::default()
    });
    world.add_component(Prop {
        mesh: Some(mesh),
        material: Some(stone),
        ..Default::default()
    });
    let sky = bake::environment_map(&bake::EnvironmentMap {
        generator: "sky".into(),
        prefilter_face_size: 16,
        irradiance_face_size: 8,
        prefilter_samples: 4,
        ..Default::default()
    })
    .expect("the sky bakes");
    world.add_environment_map(sky);
    world.add_font(bake::font(&bake::Font::default()).expect("the built-in face bakes"));

    assert_starts_headless(concinnity::App::from_world(world));
}

// What `write_blob` puts on disk is what `from_blob` reads back, with no
// state-tree anchor in between.
#[cfg(feature = "cook")]
#[test]
fn a_written_blob_loads_back_into_a_runnable_app() {
    use concinnity::components::DirectionalLight;

    let tree = concinnity_testing::TempTree::new();
    let primary = tree.join("data/0");

    concinnity::cook::world()
        .add(
            "sun",
            &DirectionalLight {
                intensity: 3.5,
                ..Default::default()
            },
        )
        .write_blob(&primary)
        .expect("the world is written");

    let app = concinnity::App::from_blob(&primary).expect("the written blob loads");
    assert_starts_headless(app);
}

// A path with no blob behind it is an error naming the file, not a panic and
// not an empty world that fails later.
#[cfg(feature = "std")]
#[test]
fn a_missing_blob_reports_the_path_it_could_not_read() {
    let tree = concinnity_testing::TempTree::new();
    let missing = tree.join("concinnity-no-such-blob/0");
    let err = concinnity::App::from_blob(&missing).expect_err("nothing to load");
    assert!(
        matches!(err, concinnity::StartupError::MissingData { .. }),
        "{err:?}"
    );
    assert!(err.to_string().contains("concinnity-no-such-blob"), "{err}");
}
