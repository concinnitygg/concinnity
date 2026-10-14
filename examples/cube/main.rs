//! A spinning black cube over a dark pool ruffled by a light breeze, the editor's clapper-board
//! mark on each face and its edges picked out in light, under a green sun that
//! rises behind the viewpoint, crosses the frame, and sets into the water. The
//! sky is airless: black between sharp stars, a Milky Way and a few faint
//! galaxies, with the sun's disc among them. It turns as a whole, so the
//! stars, the sun's light and disc, and the reflections and fill the sky casts
//! all travel together. The renderer's optional work -- ray-traced reflections
//! (screen-space where the GPU has no ray tracing), screen-space global
//! illumination, ambient occlusion, temporal anti-aliasing -- is not requested
//! here: it is what the engine defaults to, clamped per GPU by the `Auto`
//! graphics quality preset. What this world does author is the look of a
//! nearly black scene: its exposure, bloom threshold and ambient fill, and
//! stars bright enough to hold their own beside the sun.
//!
//! Nothing is read from or written to disk, and no authoring step runs: the
//! world is assembled here, component by component, and what is not a
//! component -- the cube's geometry and the mark and frame laid over it -- is
//! baked in place with the [`bake`] functions and handed over raw. Everything
//! that happens after [`App::run`] is the engine reacting to that data: the
//! water grid is tessellated, the spin behavior is compiled, the world gains
//! the defaults it declares nothing of its own for, and the sky is drawn behind
//! everything and captured into the cube the scene reflects and is lit by.
//!
//! The camera has no controller, so the viewpoint stays on the cube.
//!
//! Run it with `cargo run --release --example cube`.

mod logo;

use concinnity::components::{
    Atmosphere, Behavior, BehaviorExpr, BehaviorNode, BehaviorSource, DirectionalLight,
    GraphicsConfig, NightSky, PostProcessConfig, ProceduralMesh, Prop, SkyRotation, WaterSurface,
    Wind, Window,
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
// The sun's direction, the width of its disc, and the disc's brightness as a
// multiple of the sunlight: enough to bloom into a green halo.
const SUN_DIRECTION: [f32; 3] = [0.35, 0.55, 1.0];
const SUN_DIAMETER_DEGREES: f32 = 2.9;
const SUN_DISC_INTENSITY: f32 = 0.5;
// The celestial sphere: its pole, and how fast it turns. About a minute to
// bring the sun from behind the camera, overhead, and down into the pool.
const SKY_AXIS: [f32; 3] = [1.0, 0.0, 0.0];
const SKY_DEGREES_PER_SECOND: f32 = 3.0;
// The stars, the Milky Way and the galaxies, as multiples of the real sky's.
// Far past physical: real stars vanish beside a sun, these keep their place
// in the frame.
const STAR_BRIGHTNESS: f32 = 3.0e6;
const MILKY_WAY_BRIGHTNESS: f32 = 5.0e5;
const GALAXIES: u32 = 6;
// The cube's three layers, named so the spin below can reach them.
const CUBE_LAYERS: [AssetId; 3] = [AssetId(1), AssetId(2), AssetId(3)];
// Half the pool's width. The sun and the stars are at infinity, so each meets
// its reflection at eye level; this far out, from a camera two units up, the
// pool's edge sits a twentieth of a degree under that line, less than a
// pixel, so the water runs right up to where they meet.
const WATER_HALF_EXTENT: f32 = 2400.0;
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
    // this frame to gray. With a fixed exposure the scene is written at that
    // exposure, so the grading here is the grading on screen. Which effects
    // run is not decided here; the quality preset picks those from the GPU.
    world.add_component(PostProcessConfig {
        ambient_intensity: 0.2,
        bloom_intensity: 0.45,
        bloom_threshold: 1.2,
        vignette_strength: 0.3,
        exposure_ev: -0.1,
        ..Default::default()
    });

    // The sky beyond the air, drawn per pixel, so the stars stay single sharp
    // points. No moon: the green sun is the one body in this sky.
    world.add_component(NightSky {
        star_brightness: STAR_BRIGHTNESS,
        milky_way_brightness: MILKY_WAY_BRIGHTNESS,
        galaxies: GALAXIES,
        moon: false,
        ..Default::default()
    });
    // No air: nothing scatters, so the sky stays black around the sun. Declared
    // only for the sun's disc, which is the light's own color, wider than the
    // real sun's and far dimmer.
    world.add_component(Atmosphere {
        rayleigh_scattering: [0.0; 3],
        mie_scattering: [0.0; 3],
        mie_absorption: [0.0; 3],
        ozone_absorption: [0.0; 3],
        ground_albedo: [0.0; 3],
        sun_angular_diameter_deg: SUN_DIAMETER_DEGREES,
        sun_disc_intensity: SUN_DISC_INTENSITY,
        ..Default::default()
    });

    // The turning sky. It carries the stars, the sun's light and its disc
    // together, so none of them disagrees about where the sun is.
    world.add_component(SkyRotation {
        axis: SKY_AXIS,
        degrees_per_second: SKY_DEGREES_PER_SECOND,
        angle_deg: 0.0,
    });

    // The sun: one green light, starting high behind the camera and rising
    // over it as the sky turns. The one directional light in the world, which
    // also keeps the engine's neutral fallback light out. Bright enough that
    // its specular lobe reads as a glitter path on the water and as a hard
    // highlight on whichever cube face is turned to it. Its disc sets into the
    // pool and meets its reflection at the horizon.
    world.add_component(DirectionalLight {
        direction: SUN_DIRECTION,
        color: [0.52, 0.95, 0.70],
        intensity: 10.0,
    });

    // A light, gusty breeze: the only thing moving the water. Its gusts roll
    // across the pool as darker, rougher patches, with glassier water between.
    world.add_component(Wind {
        direction: [1.0, 0.30],
        strength: 2.0,
        gustiness: 0.6,
        gust_scale: 30.0,
    });

    // Near-black water. The renderer mirrors the scene across the surface's
    // plane and renders it again into it, so the cube reflects in the pool;
    // the breeze's waves and ripples then bend that reflection. The wind
    // raises them over the pool's own width (`fetch_meters`), so they stay
    // small, and every one is finer than the grid: they shade the surface
    // rather than move it, evaluated per fragment, which is what lets a pool
    // this wide keep a grid of 255 cells a side (the cap its 16-bit indices
    // set). The foam a shoreline would carry is turned off. A low
    // `fresnel_power` keeps the reflection strong head-on instead of only at
    // grazing angles. The roughness is the surface's floor: the ripples too
    // fine for a pixel widen it further, which is what spreads the sun's
    // specular lobe into the glitter path running back toward the camera. The
    // body colors are reflectances lit by the sun, so the water darkens as the
    // sun sinks and is black once it has set; they lean blue against the
    // green light. The grid itself is the engine's to build: the backends
    // tessellate it at start from nothing but this component.
    world.add_component(WaterSurface {
        center: [0.0, 0.0, 0.0],
        extent: [WATER_HALF_EXTENT, WATER_HALF_EXTENT],
        subdivisions: 240,
        fetch_meters: 2.0 * WATER_HALF_EXTENT,
        wind_response: 1.0,
        waves: Vec::new(),
        deep_color: [0.004, 0.005, 0.016],
        shallow_color: [0.012, 0.017, 0.032],
        depth_falloff_meters: 1.5,
        foam_width_meters: 0.0,
        foam_intensity: 0.0,
        whitecaps: 0.0,
        fresnel_power: 0.5,
        roughness: 0.05,
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

// The spin, written to each of the cube's three layers by name, once a tick.
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
                cut: false,
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
