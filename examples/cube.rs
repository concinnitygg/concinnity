//! A spinning cube over a still dark pool, with the renderer's optional work
//! turned on: ray-traced reflections (screen-space where the GPU has no ray
//! tracing), screen-space global illumination, ambient occlusion, temporal
//! anti-aliasing, bloom, and image-based lighting.
//!
//! Nothing is read from or written to disk: the world is declared here and
//! compiled in memory, its geometry comes from the built-in `box` generator,
//! and no asset names a source file.
//!
//! The camera has no controller, so the viewpoint stays on the cube.
//!
//! Run it with `cargo run --release --example cube --features cook`.

use concinnity::assets::{
    AaMode, Behavior, BehaviorExpr, BehaviorNode, BehaviorSource, Camera3DArgs, DirectionalLight,
    EngineDefaults, EnvironmentMap, GraphicsConfig, IndirectLighting, Material, PointLight,
    PostProcessConfig, ProceduralMesh, Prop, ReflectionBlurResolution, ShadowUpdate,
    SsgiResolution, WaterSurface, WaterWave, Window,
};
use concinnity::cook::{self, WorldBuilder};
use concinnity::{App, World};

// Degrees the cube turns per second.
const SPIN_DEGREES_PER_SECOND: f32 = 36.0;
// Pitch and roll the cube is held at, so three of its faces stay in view.
const CUBE_TILT_DEGREES: [f32; 3] = [24.0, 0.0, 16.0];
// Height the cube floats at, clear of the corner it swings through and of the
// crest the two waves below reach when they meet.
const CUBE_HEIGHT: f32 = 1.3;

fn main() {
    let world = cube_world().expect("the cube world compiles");
    App::from_world(world).run().expect("the app runs");
}

fn cube_world() -> std::io::Result<World> {
    let mut spec = cook::world();
    declare(&mut spec);
    spec.compile()
}

fn declare(spec: &mut WorldBuilder) {
    spec.add(
        "window",
        Window {
            title: "Cube".to_string(),
            width: 1280,
            height: 720,
            resizable: true,
            ..Default::default()
        },
    )
    .add(
        "gfx",
        GraphicsConfig {
            clear_color: [0.004, 0.005, 0.008, 1.0],
            vsync: true,
            shadow_map_size: 4096,
            shadow_update: ShadowUpdate::EveryFrame,
            anisotropy: 16,
            ..Default::default()
        },
    )
    // No controller: the viewpoint is fixed on the cube.
    .add(
        "camera",
        Camera3DArgs {
            fov_y_degrees: 40.0,
            near: 0.05,
            far: 100.0,
            position: [0.0, 2.0, 8.0],
            yaw: 0.0,
            pitch: -0.245,
            controller: None,
        },
    )
    // Auto-exposure is the one effect left off: this frame is deliberately
    // almost all black, which is what an average-brightness meter lifts to grey.
    .add(
        "post",
        PostProcessConfig {
            aa_mode: AaMode::Taa,
            ssao: true,
            ssr: true,
            ssr_intensity: 1.0,
            ray_traced_reflections: true,
            reflection_blur_resolution: ReflectionBlurResolution::Full,
            indirect_lighting: IndirectLighting::Ssgi,
            ssgi_resolution: SsgiResolution::Full,
            ssgi_rays: 16,
            ssgi_steps: 24,
            ambient_intensity: 0.2,
            bloom_intensity: 0.45,
            bloom_threshold: 1.2,
            vignette_strength: 0.3,
            exposure_ev: -0.1,
            occlusion_two_pass: true,
            ..Default::default()
        },
    )
    // A generated sky, used for its image-based lighting alone: the injected
    // sky mesh would both light up the background and put a second mesh in
    // front of the spin behavior below.
    .add(
        "sky",
        EnvironmentMap {
            generator: "sky".to_string(),
            prefilter_face_size: 256,
            irradiance_face_size: 32,
            ..Default::default()
        },
    )
    .add(
        "defaults",
        EngineDefaults {
            sky: false,
            ..Default::default()
        },
    )
    // Key light, warm, from overhead.
    .add(
        "key",
        DirectionalLight {
            direction: [0.10, 0.92, -0.25],
            color: [1.0, 0.96, 0.90],
            intensity: 2.2,
        },
    )
    // Cool and warm from either side, so each face changes colour as it turns.
    .add(
        "cool",
        PointLight {
            position: [-3.2, 1.5, 2.2],
            color: [0.35, 0.60, 1.0],
            intensity: 11.0,
            range: 12.0,
        },
    )
    .add(
        "warm",
        PointLight {
            position: [3.0, 1.2, 1.4],
            color: [1.0, 0.50, 0.22],
            intensity: 13.0,
            range: 12.0,
        },
    )
    // Near-black water, almost flat. The renderer mirrors the scene across the
    // surface's plane and renders it again into it, so the cube reflects in the
    // pool; the wave normals then bend that reflection, which is the only thing
    // that gives the stillness away. Both waves are long, shallow and slow (a
    // crest crosses its own wavelength in about fifteen seconds), and the foam
    // a shoreline would carry is turned off. A low `fresnel_power` keeps the
    // reflection strong head-on instead of only at grazing angles.
    .add(
        "water",
        WaterSurface {
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
        },
    )
    .add(
        "cube_mesh",
        ProceduralMesh {
            generator: "box".to_string(),
            half_extents: Some([0.7, 0.7, 0.7]),
            ..Default::default()
        },
    )
    .add(
        "cube_material",
        Material {
            roughness: 0.20,
            metallic: 0.30,
            tint: [0.88, 0.86, 0.84],
            ..Default::default()
        },
    )
    .add(
        "cube",
        Prop {
            position: [0.0, CUBE_HEIGHT, 0.0],
            rotation_deg: CUBE_TILT_DEGREES,
            ..Default::default()
        },
    )
    .reference("mesh", "cube_mesh")
    .reference("material", "cube_material")
    // The cube is the only prop in the world, so scoping the spin to `Prop`
    // and writing `self` saves naming what it turns.
    .add(
        "spin",
        Behavior {
            on: BehaviorSource::Tick,
            scope: vec!["Prop".to_string()],
            body: vec![BehaviorNode::SetTransform {
                entity: BehaviorExpr::SelfEntity,
                position: None,
                rotation_deg: Some(spin_rotation()),
                scale: None,
            }],
            ..Default::default()
        },
    );
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

    // The declarations are checked by the cook, not by the compiler: an
    // unresolved reference or a mistyped behavior expression surfaces here.
    #[test]
    fn the_declared_world_compiles() {
        cube_world().expect("the cube world compiles");
    }

    // `spin` turns every prop in the world, which reads as "the cube" only
    // while the cube is the only one. The skybox prop the build would
    // otherwise inject is the other candidate, and `EngineDefaults` is what
    // keeps it out.
    #[test]
    fn the_cube_is_the_worlds_only_prop() {
        let mut spec = cook::world();
        declare(&mut spec);
        let props: Vec<&str> = spec
            .declared()
            .filter(|(_, ty)| *ty == "Prop")
            .map(|(name, _)| name)
            .collect();
        assert_eq!(props, ["cube"]);
        assert!(spec.declared().any(|(_, ty)| ty == "EngineDefaults"));
    }
}
