//! A spinning cube over a still dark pool. The renderer's optional work --
//! ray-traced reflections (screen-space where the GPU has no ray tracing),
//! screen-space global illumination, ambient occlusion, temporal anti-aliasing
//! -- is not requested here: it is what the engine defaults to, clamped per GPU
//! by the `Auto` graphics quality preset. What this world does author is the
//! look of a nearly black scene: its exposure, bloom threshold, and ambient
//! fill.
//!
//! Nothing is read from or written to disk, and no authoring step runs: the
//! world is assembled here, component by component, and what is not a
//! component -- the cube's geometry, the lighting it reflects -- is baked in
//! place with the [`bake`] functions and handed over raw. Everything that
//! happens after [`App::run`] is the engine reacting to that data: the water
//! grid is tessellated, the spin behavior is compiled, and the world is
//! completed with the defaults it does not opt out of.
//!
//! The camera has no controller, so the viewpoint stays on the cube.
//!
//! Run it with `cargo run --release --example cube`.

use concinnity::components::{
    Behavior, BehaviorExpr, BehaviorNode, BehaviorSource, DirectionalLight, EngineDefaults,
    GraphicsConfig, PointLight, PostProcessConfig, ProceduralMesh, Prop, WaterSurface, WaterWave,
    Window,
};
use concinnity::{App, World, bake};

// Degrees the cube turns per second.
const SPIN_DEGREES_PER_SECOND: f32 = 36.0;
// Pitch and roll the cube is held at, so three of its faces stay in view.
const CUBE_TILT_DEGREES: [f32; 3] = [24.0, 0.0, 16.0];
// Height the cube floats at, clear of the corner it swings through and of the
// crest the two waves below reach when they meet.
const CUBE_HEIGHT: f32 = 1.3;

fn main() {
    let world = cube_world().expect("the cube world bakes");
    App::from_world(world).run().expect("the app runs");
}

fn cube_world() -> Result<World, String> {
    let mut world = World::new();

    world.add_component(Window {
        title: "Cube".to_string(),
        width: 1280,
        height: 720,
        resizable: true,
        ..Default::default()
    });
    world.add_component(GraphicsConfig {
        clear_color: [0.004, 0.005, 0.008, 1.0],
        vsync: true,
        ..Default::default()
    });
    // No controller: the viewpoint is fixed on the cube, rendering from the
    // view this bakes.
    world.add_component(bake::camera(bake::Camera3D {
        fov_y_degrees: 40.0,
        near: 0.05,
        far: 100.0,
        position: [0.0, 2.0, 8.0],
        yaw: 0.0,
        pitch: -0.245,
        controller: None,
    }));
    // Grading for a frame that is deliberately almost all black: the ambient
    // fill is pulled well down so the two point lights carry the cube, and bloom
    // only catches their highlights. Auto-exposure stays off (its default) for
    // the same reason -- an average-brightness meter would lift this frame to
    // grey. Which effects run is not decided here; the quality preset picks
    // those from the GPU.
    world.add_component(PostProcessConfig {
        ambient_intensity: 0.2,
        bloom_intensity: 0.45,
        bloom_threshold: 1.2,
        vignette_strength: 0.3,
        exposure_ev: -0.1,
        ..Default::default()
    });

    // A generated sky, convolved here into the two cubemaps the shaders light
    // with, and used for that image-based lighting alone: the sky mesh the
    // world would otherwise gain at start would both light up the background
    // and put a second prop in front of the spin behavior below, so
    // EngineDefaults opts out of it.
    let ibl = bake::environment_map(&bake::EnvironmentMap {
        generator: "sky".to_string(),
        prefilter_face_size: 256,
        irradiance_face_size: 32,
        ..Default::default()
    })?;
    world.add_environment_map(ibl);
    world.add_component(EngineDefaults {
        sky: false,
        ..Default::default()
    });

    // Key light, warm, from overhead.
    world.add_component(DirectionalLight {
        direction: [0.10, 0.92, -0.25],
        color: [1.0, 0.96, 0.90],
        intensity: 2.2,
    });
    // Cool and warm from either side, so each face changes colour as it turns.
    world.add_component(PointLight {
        position: [-3.2, 1.5, 2.2],
        color: [0.35, 0.60, 1.0],
        intensity: 11.0,
        range: 12.0,
    });
    world.add_component(PointLight {
        position: [3.0, 1.2, 1.4],
        color: [1.0, 0.50, 0.22],
        intensity: 13.0,
        range: 12.0,
    });

    // Near-black water, almost flat. The renderer mirrors the scene across the
    // surface's plane and renders it again into it, so the cube reflects in the
    // pool; the wave normals then bend that reflection, which is the only thing
    // that gives the stillness away. Both waves are long, shallow and slow (a
    // crest crosses its own wavelength in about fifteen seconds), and the foam
    // a shoreline would carry is turned off. A low `fresnel_power` keeps the
    // reflection strong head-on instead of only at grazing angles. The grid
    // itself is the engine's to build: the backends tessellate it at start
    // from nothing but this component.
    world.add_component(WaterSurface {
        centre: [0.0, 0.0, 0.0],
        extent: [14.0, 14.0],
        subdivisions: 160,
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
        deep_colour: [0.006, 0.010, 0.018],
        shallow_colour: [0.02, 0.05, 0.07],
        depth_falloff_metres: 1.5,
        foam_width_metres: 0.0,
        foam_intensity: 0.0,
        fresnel_power: 0.5,
        roughness: 0.03,
        refraction_strength: 0.02,
        visible: true,
        ..Default::default()
    });

    // The cube itself: its geometry baked here from the box generator, its
    // material clamped on the way in, and a prop holding the handles the world
    // returned for both.
    let mesh = ProceduralMesh {
        generator: "box".to_string(),
        half_extents: Some([0.7, 0.7, 0.7]),
        ..Default::default()
    };
    let payload = bake::procedural_mesh(&mesh)?;
    let cube_mesh = world.add_mesh(mesh, payload);
    let cube_material = world.add_material(bake::Material {
        roughness: 0.20,
        metallic: 0.30,
        tint: [0.88, 0.86, 0.84],
        ..Default::default()
    });
    world.add_component(Prop {
        mesh: Some(cube_mesh),
        material: Some(cube_material),
        position: [0.0, CUBE_HEIGHT, 0.0],
        rotation_deg: CUBE_TILT_DEGREES,
        ..Default::default()
    });

    // The cube is the only prop in the world, so scoping the spin to `Prop`
    // and writing `self` saves naming what it turns.
    world.add_component(Behavior {
        on: BehaviorSource::Tick,
        scope: vec!["Prop".to_string()],
        body: vec![BehaviorNode::SetTransform {
            entity: BehaviorExpr::SelfEntity,
            position: None,
            rotation_deg: Some(spin_rotation()),
            scale: None,
        }],
        ..Default::default()
    });

    Ok(world)
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

#[cfg(test)]
mod tests {
    use super::*;

    // The bakes are checked here, not by the compiler: an unknown generator or
    // an unbakeable declaration surfaces as the error string it returns.
    #[test]
    fn the_cube_world_bakes() {
        cube_world().expect("the cube world bakes");
    }
}
