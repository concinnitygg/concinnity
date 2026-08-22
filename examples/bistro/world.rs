// examples/bistro/world.rs
//
// The Bistro showcase world, declared as typed asset structs rather than a
// world.jsonl file, so the example authors its content the same way the
// framework's own documentation does.
//
// Two conventions this content depends on: the first declared Scene is the one
// active at world start (so `menu` must come before `bistro`), and an asset
// joins a scene by name prefix, which is what puts `bistro_exterior` in the
// `bistro` scene while the unprefixed assets stay visible in both.

use concinnity::assets::{
    AaMode, Camera3DArgs, CameraController, DirectionalLight, EnvironmentMap, GraphicsConfig,
    IndirectLighting, MainMenu, MainMenuItem, PostProcessConfig, ReflectionProbe, Scene,
    SceneImport, StreamingConfig,
};
use concinnity::cook::WorldBuilder;

use crate::{FBX_REL, HDR_REL};

pub(crate) fn declare(world: &mut WorldBuilder) {
    world
        .add("menu", Scene::default())
        .add("bistro", Scene::default())
        .add(
            "stream",
            StreamingConfig {
                texture_cap: 512,
                ..Default::default()
            },
        )
        .add(
            "scene_graphics",
            GraphicsConfig {
                clear_color: [0.05, 0.06, 0.09, 1.0],
                shadow_map_size: 2048,
                ..Default::default()
            },
        )
        .add(
            "scene_camera",
            Camera3DArgs {
                position: [-17.94, 5.04, 1.29],
                fov_y_degrees: 60.0,
                yaw: -1.446,
                pitch: -0.187,
                far: 900.0,
                controller: Some(CameraController {
                    free_fly: true,
                    move_speed: 8.0,
                    mouse_sensitivity: 0.0015,
                    player_radius: 0.3,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .add(
            "sun",
            DirectionalLight {
                color: [1.0, 0.95, 0.85],
                direction: [-0.4, 0.78, 0.5],
                intensity: 3.0,
            },
        )
        .add(
            "env_sky",
            EnvironmentMap {
                source: HDR_REL.to_string(),
                ..Default::default()
            },
        )
        .add(
            "post_all",
            PostProcessConfig {
                ssr: true,
                ray_traced_reflections: true,
                ssr_intensity: 0.9,
                ssr_max_distance: 120.0,
                ssao: true,
                ssao_radius: 0.6,
                ssao_intensity: 1.3,
                indirect_lighting: IndirectLighting::Ibl,
                aa_mode: AaMode::Taa,
                bloom_intensity: 0.35,
                bloom_threshold: 1.5,
                occlusion_two_pass: true,
                auto_exposure: false,
                exposure_ev: -1.0,
                lut_strength: 0.9,
                ..Default::default()
            },
        )
        .add(
            "bistro_exterior",
            SceneImport {
                source: FBX_REL.to_string(),
                texture_max_size: 512,
                ..Default::default()
            },
        )
        .add(
            "lobby",
            ReflectionProbe {
                position: [-3.34, 1.7, 0.82],
                half_extents: [5.0, 4.0, 5.0],
            },
        )
        .add(
            "main_menu",
            MainMenu {
                title: "Bistro_v5_2".to_string(),
                initial: true,
                items: vec![
                    item("Start", "scene:bistro"),
                    item("Settings", "settings"),
                    item("Quit", "quit"),
                ],
                ..Default::default()
            },
        );
}

fn item(label: &str, action: &str) -> MainMenuItem {
    MainMenuItem {
        label: label.to_string(),
        action: action.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The scene conventions the content relies on, neither of which the type
    // system can catch: `menu` is declared first so it is the active scene at
    // startup, and the imported geometry carries the `bistro_` prefix that
    // files it under the `bistro` scene.
    #[test]
    fn scene_order_and_prefixes_hold() {
        let mut spec = concinnity::cook::world();
        declare(&mut spec);
        let declared: Vec<_> = spec.declared().collect();

        let scenes: Vec<&str> = declared
            .iter()
            .filter(|(_, ty)| *ty == "Scene")
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(scenes, ["menu", "bistro"], "menu must start active");

        let (import, _) = declared
            .iter()
            .find(|(_, ty)| *ty == "SceneImport")
            .expect("the exterior import");
        assert!(
            import.starts_with("bistro_"),
            "the import must file under the bistro scene, got {import}"
        );
    }
}
