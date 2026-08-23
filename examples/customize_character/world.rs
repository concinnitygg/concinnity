// examples/customize_character/world.rs
//
// The customizable-humanoid world, declared as typed asset structs rather
// than a world.jsonl file. A field that points at another asset holds a
// resolved handle, so its name is given with `reference` beside the value.

use concinnity::assets::{
    Animation, Camera3DArgs, CameraController, CharacterCapsule, CharacterModel, CharacterShape,
    DirectionalLight, GraphicsConfig, JointProportion, Material, ProceduralMesh, Prop, ShapeSlider,
};
use concinnity::cook::WorldBuilder;

use crate::BODY_GLB;

// The bundled schema every conforming body is validated against, and whose
// regions the synthesized sliders below come from.
const SCHEMA: &str = "builtin:humanoid";

pub(crate) fn declare(world: &mut WorldBuilder) {
    world
        .add(
            "gfx",
            GraphicsConfig {
                clear_color: [0.3, 0.4, 0.6, 1.0],
                fps_cap: 60,
                ..Default::default()
            },
        )
        .add(
            "cam",
            Camera3DArgs {
                position: [0.0, 1.1, 3.2],
                fov_y_degrees: 50.0,
                yaw: 0.0,
                pitch: -0.05,
                controller: Some(CameraController {
                    free_fly: true,
                    move_speed: 3.0,
                    mouse_sensitivity: 0.0015,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .add(
            "sun",
            DirectionalLight {
                color: [1.0, 1.0, 1.0],
                direction: [-0.4, 0.8, 0.4],
                intensity: 2.5,
            },
        )
        .add(
            "mat_floor",
            Material {
                roughness: 0.8,
                tint: [0.5, 0.5, 0.5],
                ..Default::default()
            },
        )
        .add(
            "floor_mesh",
            ProceduralMesh {
                generator: "plane".into(),
                half_width: 10.0,
                half_depth: 10.0,
                ..Default::default()
            },
        )
        .add("floor", Prop::default())
        .reference("mesh", "floor_mesh")
        .reference("material", "mat_floor")
        .add(
            "mat_skin",
            Material {
                roughness: 0.55,
                tint: [0.85, 0.62, 0.5],
                ..Default::default()
            },
        )
        // The shaped body: three levels of detail decimated from the one
        // source, switching at 6 m and 12 m.
        .add(
            "body",
            CharacterModel {
                schema: SCHEMA.into(),
                source: BODY_GLB.into(),
                lod_levels: 3,
                lod_distances: vec![6.0, 12.0],
                position: [0.0, 0.03, 0.0],
                capsule: Some(CharacterCapsule {
                    half_height: 0.88,
                    radius: 0.3,
                }),
                ..Default::default()
            },
        )
        .reference("material", "mat_skin")
        // Fifteen sliders, most of them synthesized from the mesh by the
        // schema rather than sculpted in Blender.
        .add(
            "body_shape",
            CharacterShape {
                sliders: sliders(&[
                    ("face", 1.0),
                    ("weight", 0.3),
                    ("muscle", 0.5),
                    ("shoulders", 0.5),
                    ("biceps", 0.8),
                    ("deltoid", 0.6),
                    ("pectoral", 0.5),
                    ("thigh_girth", 0.5),
                    ("calf_muscle", 0.7),
                    ("waist_girth", -0.4),
                    ("neck_girth", 0.3),
                    ("brow_ridge", 0.8),
                    ("cheekbone_l", 0.6),
                    ("cheekbone_r", 0.6),
                    ("nose", 0.5),
                ]),
                proportions: vec![
                    scaled("spine", 1.04),
                    lengthened("thigh_l", 0.03),
                    lengthened("thigh_r", 0.03),
                ],
                ..Default::default()
            },
        )
        .reference("target", "body")
        .add(
            "body_idle",
            Animation {
                source: BODY_GLB.into(),
                animation_name: "idle".into(),
                looping: true,
                ..Default::default()
            },
        )
        .reference("target", "body");
}

fn sliders(values: &[(&str, f32)]) -> Vec<ShapeSlider> {
    values
        .iter()
        .map(|(name, value)| ShapeSlider {
            name: (*name).into(),
            value: *value,
        })
        .collect()
}

fn scaled(joint: &str, scale: f32) -> JointProportion {
    JointProportion {
        joint: joint.into(),
        scale,
        ..Default::default()
    }
}

fn lengthened(joint: &str, length: f32) -> JointProportion {
    JointProportion {
        joint: joint.into(),
        length,
        ..Default::default()
    }
}
