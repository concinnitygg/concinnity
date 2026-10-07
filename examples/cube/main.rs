//! A spinning black cube over a still dark pool, the editor's clapper-board
//! mark on each face and its edges picked out in light, under a green sun that
//! rises behind the viewpoint, crosses the frame, and sets into the water. The
//! sky behind it is a generated starfield, and it turns as a whole, so the
//! stars, the sun's light, its body and the image-based lighting all travel
//! together. The renderer's optional work -- ray-traced reflections
//! (screen-space where the GPU has no ray tracing), screen-space global
//! illumination, ambient occlusion, temporal anti-aliasing -- is not requested
//! here: it is what the engine defaults to, clamped per GPU by the `Auto`
//! graphics quality preset. What this world does author is the look of a
//! nearly black scene: its exposure, bloom threshold, and ambient fill.
//!
//! Nothing is read from or written to disk, and no authoring step runs: the
//! world is assembled here, component by component, and what is not a
//! component -- the cube's geometry, the mark and frame laid over it, the
//! lighting it reflects -- is baked in place with the [`bake`] functions and
//! handed over raw. Everything that happens after [`App::run`] is the engine
//! reacting to that data: the water grid is tessellated, the spin behavior is
//! compiled, the world gains the defaults it declares nothing of its own for,
//! and the starfield is drawn behind everything as the background.
//!
//! The camera has no controller, so the viewpoint stays on the cube.
//!
//! Run it with `cargo run --release --example cube`.

mod logo;

use concinnity::components::{
    Behavior, BehaviorExpr, BehaviorNode, BehaviorSource, DirectionalLight, GraphicsConfig,
    PostProcessConfig, ProceduralMesh, Prop, SkyRotation, WaterSurface, WaterWave, Window,
};
use concinnity::{App, AssetId, MaterialHandle, MeshHandle, Ref, World, bake};

// Degrees the cube turns per second.
const SPIN_DEGREES_PER_SECOND: f32 = 36.0;
// Pitch and roll the cube is held at, so three of its faces stay in view.
const CUBE_TILT_DEGREES: [f32; 3] = [24.0, 0.0, 16.0];
// Height the cube floats at, clear of the corner it swings through and of the
// crest the two waves below reach when they meet.
const CUBE_HEIGHT: f32 = 1.3;
// Half the cube's width.
const CUBE_HALF_EXTENT: f32 = 0.7;
// How much of a face's width the mark spans, and how wide the band along
// each edge is.
const MARK_SPAN: f32 = 0.62;
const EDGE_WIDTH: f32 = 0.02;
// How far the mark and the frame sit off the cube's surface, so they draw
// over it rather than fighting it for depth.
const SURFACE_LIFT: f32 = 0.004;
// The sun's direction, how far out it hangs, and its size. The body sits on
// the light's own direction, so the two agree wherever the sky has carried
// them. It hangs far enough out that the pool can reach almost to the far
// plane and still end where the sun sets; the radius keeps the disc the same
// size on screen as it would be at a twentieth of the distance.
const SUN_DIRECTION: [f32; 3] = [0.35, 0.55, 1.0];
const SUN_DISTANCE: f32 = 480.0;
const SUN_RADIUS: f32 = 12.0;
// The celestial sphere: its pole, and how fast it turns. About a minute to
// bring the sun from behind the camera, overhead, and down into the pool.
const SKY_AXIS: [f32; 3] = [1.0, 0.0, 0.0];
const SKY_DEGREES_PER_SECOND: f32 = 3.0;
// The name the sun hangs off, so it orbits with the sky rather than sitting
// still while its light moves.
const SKY_PIVOT: AssetId = AssetId(1);
// The cube's three layers, named so the spin below can reach them and nothing
// else -- the sun is a prop too, and turning it would pull it off its orbit.
const CUBE_LAYERS: [AssetId; 3] = [AssetId(2), AssetId(3), AssetId(4)];
// Half the pool's width, reaching exactly as far as the point the sun sets
// at. A body that sets beyond the far edge loses its reflection off that edge
// before it reaches the edge itself; one that sets short of the edge crosses
// the edge line on the way down. Setting on the edge, the disc meets its own
// reflection on the line where the water and the sky meet. The edge is far
// enough out that, from a camera two units up, it sits within a quarter of a
// degree of eye level: the stars meet their reflections on that line too.
const WATER_HALF_EXTENT: f32 = 459.0;
// The fixed viewpoint: on the cube, tipped down just far enough that the
// horizon sits a little above the frame's center, with sky over it.
const CAMERA_POSITION: [f32; 3] = [0.0, 2.0, 9.0];
const CAMERA_PITCH: f32 = -0.09;
const CAMERA_FOV_Y_DEGREES: f32 = 40.0;

fn main() {
    let world = cube_world().expect("the cube world bakes");
    App::from_world(world).run().expect("the app runs");
}

fn cube_world() -> Result<World, concinnity::Error> {
    let mut world = World::new();

    world.add_component(Window {
        title: "Cube".to_string(),
        width: 1280,
        height: 720,
        resizable: true,
        ..Default::default()
    });
    world.add_component(GraphicsConfig {
        clear_color: [0.0, 0.0, 0.0, 1.0],
        vsync: true,
        ..Default::default()
    });
    // No controller: the viewpoint is fixed on the cube, rendering from the
    // view this bakes.
    world.add_component(bake::camera(bake::Camera3D {
        fov_y_degrees: CAMERA_FOV_Y_DEGREES,
        near: 0.05,
        view_distance: None,
        position: CAMERA_POSITION,
        yaw: 0.0,
        pitch: CAMERA_PITCH,
        controller: None,
    }));
    // Grading for a frame that is deliberately almost all black: the ambient
    // fill stays low over a sky that is nearly black to begin with, so the
    // sun alone carries the cube, and bloom catches only the edge frame, the
    // sun's disc and the brightest stars. Auto-exposure stays off (its
    // default) for the same reason -- an average-brightness meter would lift
    // this frame to gray. Which effects run is not decided here; the quality
    // preset picks those from the GPU.
    world.add_component(PostProcessConfig {
        ambient_intensity: 0.2,
        bloom_intensity: 0.45,
        bloom_threshold: 1.2,
        vignette_strength: 0.3,
        exposure_ev: -0.1,
        ..Default::default()
    });

    // A generated starfield, convolved here into the two cubemaps the shaders
    // light with. The map is also drawn as the background, so the same bake is
    // both the stars on screen and the scene's cold fill. The face
    // size is what the stars are drawn at on screen, so it is four times the
    // one this used to bake at; the sample count, which only affects the
    // blurred reflection mips, comes down to pay for it.
    let ibl = bake::environment_map(&bake::EnvironmentMap {
        generator: "stars".to_string(),
        prefilter_face_size: 1024,
        irradiance_face_size: 32,
        prefilter_samples: 128,
        ..Default::default()
    })?;
    world.add_environment_map(ibl);

    // The turning sky. It carries the sun's light and its body together, so
    // the two never disagree about where the sun is: the light's direction is
    // rotated each frame and the body orbits the pivot as its parent.
    world.add_identified(
        SKY_PIVOT,
        SkyRotation {
            axis: SKY_AXIS,
            degrees_per_second: SKY_DEGREES_PER_SECOND,
            angle_deg: 0.0,
        },
    );

    // The sun: one green light, starting high behind the camera and rising
    // over it as the sky turns. The one directional light in the world, which
    // also keeps the engine's neutral fallback light out. Bright enough that
    // its specular lobe reads as a glitter path on the water and as a hard
    // highlight on whichever cube face is turned to it.
    world.add_component(DirectionalLight {
        direction: SUN_DIRECTION,
        color: [0.52, 0.95, 0.70],
        intensity: 10.0,
    });

    // The sun's body. A directional light has none, so the sun is an emissive
    // sphere hung on the light's own direction, parented to the sky so the two
    // travel together. It starts behind the camera and out of frame, rises over
    // it, and sets into the pool, where the water mirrors it once it is in
    // front. It carries no light of its own; the emissive is well over the
    // bloom threshold, so the disc blooms rather than reading as a flat circle.
    let sun = ProceduralMesh {
        generator: "sphere".to_string(),
        radius: Some(SUN_RADIUS),
        segments: Some(48),
        rings: Some(24),
        ..Default::default()
    };
    let sun_mesh = world.add_mesh(bake::procedural_mesh(sun)?)?;
    let sun_material = world.add_material(bake::Material {
        roughness: 1.0,
        metallic: 0.0,
        tint: [0.0, 0.0, 0.0],
        emissive_factor: [0.45, 4.60, 1.30],
        ..Default::default()
    });
    world.add_component(Prop {
        mesh: Some(sun_mesh),
        material: Some(sun_material),
        position: sun_position(),
        parent: Some(Ref::new(SKY_PIVOT)),
        ..Default::default()
    });

    // Near-black water, almost flat. The renderer mirrors the scene across the
    // surface's plane and renders it again into it, so the cube reflects in the
    // pool; the wave normals then bend that reflection, which is the only thing
    // that gives the stillness away. Both waves are long, shallow and slow (a
    // crest crosses its own wavelength in about fifteen seconds), and the foam
    // a shoreline would carry is turned off. A low `fresnel_power` keeps the
    // reflection strong head-on instead of only at grazing angles. The
    // roughness does double duty: it widens the sun's specular lobe into the
    // glitter path running back toward the camera, and it scales how far the
    // wave normals push the mirror's screen lookup. The grid
    // itself is the engine's to build: the backends tessellate it at start
    // from nothing but this component. The grid's 16-bit indices cap it at
    // 255 cells a side, so over a pool this wide a cell spans a couple of
    // waves; that is fine, since the wave normal the shading and the mirror
    // use is evaluated per fragment, and the displacement itself is too
    // shallow to see at that scale.
    world.add_component(WaterSurface {
        center: [0.0, 0.0, 0.0],
        extent: [WATER_HALF_EXTENT, WATER_HALF_EXTENT],
        subdivisions: 240,
        waves: vec![
            WaterWave {
                amplitude: 0.050,
                wavelength: 3.4,
                speed: 0.22,
                direction: [1.0, 0.30],
                steepness: 0.05,
            },
            WaterWave {
                amplitude: 0.035,
                wavelength: 2.1,
                speed: 0.16,
                direction: [-0.35, 1.0],
                steepness: 0.05,
            },
        ],
        deep_color: [0.006, 0.010, 0.018],
        shallow_color: [0.02, 0.05, 0.07],
        depth_falloff_meters: 1.5,
        foam_width_meters: 0.0,
        foam_intensity: 0.0,
        fresnel_power: 0.5,
        roughness: 0.10,
        refraction_strength: 0.02,
        visible: true,
    });

    // The cube is three props at one transform: the black body from the box
    // generator, and over it the mark and the edge frame as raw geometry, each
    // with the material that sets it apart from the body.
    let body = ProceduralMesh {
        generator: "box".to_string(),
        half_extents: Some([CUBE_HALF_EXTENT; 3]),
        ..Default::default()
    };
    let body_mesh = world.add_mesh(bake::procedural_mesh(body)?)?;
    // Near-black, but glossy enough that the sun shows as a broad sheen on
    // whichever face turns toward it.
    let body_material = world.add_material(bake::Material {
        roughness: 0.22,
        metallic: 0.40,
        tint: [0.03, 0.03, 0.035],
        ..Default::default()
    });
    place_on_cube(&mut world, CUBE_LAYERS[0], body_mesh, body_material);

    let mark = logo::mark_on_box(CUBE_HALF_EXTENT, MARK_SPAN, SURFACE_LIFT);
    let mark_mesh = world.add_mesh(bake::mesh(&mark)?)?;
    let mark_material = world.add_material(bake::Material {
        roughness: 0.55,
        metallic: 0.0,
        tint: [0.93, 0.96, 0.94],
        emissive_factor: [0.30, 0.34, 0.31],
        ..Default::default()
    });
    place_on_cube(&mut world, CUBE_LAYERS[1], mark_mesh, mark_material);

    let frame = logo::edge_frame(CUBE_HALF_EXTENT, EDGE_WIDTH, SURFACE_LIFT);
    let frame_mesh = world.add_mesh(bake::mesh(&frame)?)?;
    let frame_material = world.add_material(bake::Material {
        roughness: 0.6,
        metallic: 0.0,
        tint: [0.45, 0.85, 0.62],
        emissive_factor: [0.55, 1.35, 0.85],
        ..Default::default()
    });
    place_on_cube(&mut world, CUBE_LAYERS[2], frame_mesh, frame_material);

    world.add_component(spin_behavior());

    Ok(world)
}

// The sun's body sits on the light's direction, `SUN_DISTANCE` out. Local to
// the sky pivot, which is at the origin, so the sky's rotation carries it.
fn sun_position() -> [f32; 3] {
    let d = SUN_DIRECTION;
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    std::array::from_fn(|k| d[k] / len * SUN_DISTANCE)
}

// One layer of the cube, at the cube's transform.
fn place_on_cube(world: &mut World, id: AssetId, mesh: MeshHandle, material: MaterialHandle) {
    world.add_identified(
        id,
        Prop {
            mesh: Some(mesh),
            material: Some(material),
            position: [0.0, CUBE_HEIGHT, 0.0],
            rotation_deg: CUBE_TILT_DEGREES,
            ..Default::default()
        },
    );
}

// The spin, written to each of the cube's three layers by name. It runs once a
// tick, world-scoped: a behavior scoped to `Prop` would reach every prop in the
// world, and the sun is a prop too.
fn spin_behavior() -> Behavior {
    Behavior {
        on: BehaviorSource::Tick,
        body: CUBE_LAYERS
            .iter()
            .map(|id| BehaviorNode::SetTransform {
                entity: BehaviorExpr::Named(Some(Ref::new(*id))),
                position: None,
                rotation_deg: Some(spin_rotation()),
                scale: None,
            })
            .collect(),
        ..Default::default()
    }
}

// `CUBE_TILT_DEGREES + [0, 1, 0] * elapsed * SPIN_DEGREES_PER_SECOND`: a fixed
// tilt whose yaw is driven by the world clock.
fn spin_rotation() -> BehaviorExpr {
    let yaw = BehaviorExpr::Mul(
        Box::new(BehaviorExpr::Elapsed),
        Box::new(BehaviorExpr::Float(SPIN_DEGREES_PER_SECOND)),
    );
    BehaviorExpr::Add(
        Box::new(BehaviorExpr::Vec3(CUBE_TILT_DEGREES)),
        Box::new(BehaviorExpr::Mul(
            Box::new(BehaviorExpr::Vec3([0.0, 1.0, 0.0])),
            Box::new(yaw),
        )),
    )
}
