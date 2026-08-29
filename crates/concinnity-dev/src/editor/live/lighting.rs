// src/editor/live/lighting.rs
//
// The lighting live path. Unlike a Sprite or a TextLabel, none of these assets
// is data the running world re-reads each frame: the sun is packed into the
// renderer's light uniforms at load and the fog / shadow / post-process assets
// are consumed there outright, so writing their columns would apply nothing.
// The engine's `gfx::lighting_preview` seam is what actually reaches the
// renderer; this module decides whether an edit can go through it and bakes the
// component it takes.
//
// What a rebuild would show is the standard: a knob the renderer sizes a GPU
// resource from, and fog on a world that started without it, are declined here
// so the caller rebuilds rather than showing a change that does not stick.

use crate::ecs::World;
use crate::gfx::lighting_preview;
use concinnity_core::ecs::ComponentAsset;
use concinnity_world::registry::RegisteredType;
use serde_json::{Map, Value};

use super::{Apply, RenderConfig, component};

/// Plan the backend push for an edited lighting asset, or `None` when the type
/// is not one of them, the running world has no renderer, or a changed field is
/// one the renderer only reads at load.
pub(super) fn plan(
    world: &World,
    ct: RegisteredType,
    name: &str,
    args: &Map<String, Value>,
    keys: &[String],
) -> Option<Apply> {
    if !is_expressible(ct, keys) || !lighting_preview::is_available(world) {
        return None;
    }
    let id = crate::ecs::asset_id::intern(name);
    match component::bake(ct, id, args).ok()? {
        ComponentAsset::DirectionalLight(light) => {
            let entity = world
                .resource::<concinnity_core::ecs::EntityByName>()?
                .get(id)?;
            Some(Apply::Sun { entity, light })
        }
        ComponentAsset::VolumetricFog(fog) => {
            // The fog pass has a pipeline only where the world declared enabled
            // fog at load, so turning it on live is not expressible.
            if fog.enabled && !lighting_preview::fog_pass_built(world) {
                return None;
            }
            Some(Apply::RenderConfig(RenderConfig::Fog(fog)))
        }
        ComponentAsset::GraphicsConfig(config) => {
            Some(Apply::RenderConfig(RenderConfig::Graphics(config)))
        }
        ComponentAsset::PostProcessConfig(config) => {
            Some(Apply::RenderConfig(RenderConfig::Post(Box::new(config))))
        }
        _ => None,
    }
}

// Whether this is a lighting type at all, and whether every field the edit
// changed is one the renderer can be handed. The sun and the fog are replaced
// wholesale, so every field of theirs qualifies; the other two carry knobs that
// size a GPU resource at load beside the ones that do not.
fn is_expressible(ct: RegisteredType, keys: &[String]) -> bool {
    match ct {
        RegisteredType::DirectionalLight | RegisteredType::VolumetricFog => true,
        RegisteredType::GraphicsConfig => {
            live_keys(keys, lighting_preview::LIVE_GRAPHICS_CONFIG_FIELDS)
        }
        RegisteredType::PostProcessConfig => {
            live_keys(keys, lighting_preview::LIVE_POST_PROCESS_FIELDS)
        }
        _ => false,
    }
}

/// Perform a planned lighting push.
pub(super) fn commit(world: &mut World, config: RenderConfig) {
    match config {
        RenderConfig::Fog(fog) => {
            lighting_preview::apply_fog(world, Some(&fog));
        }
        RenderConfig::Graphics(config) => {
            lighting_preview::apply_graphics_config(world, &config);
        }
        RenderConfig::Post(config) => {
            lighting_preview::apply_post_process_config(world, &config);
        }
    }
}

/// Replace one directional light and re-push the whole set. The renderer packs
/// the lights in column order, which is the order this reads them back in, so
/// the edited light lands in the slot it already occupied.
pub(super) fn commit_sun(
    world: &mut World,
    entity: crate::ecs::Entity,
    light: crate::components::DirectionalLight,
) {
    world.replace_component(entity, ComponentAsset::DirectionalLight(light));
    let lights: Vec<crate::components::DirectionalLight> = world
        .query::<crate::components::DirectionalLight>()
        .cloned()
        .collect();
    lighting_preview::apply_directional_lights(world, &lights);
}

// Whether every changed key is one the renderer can be handed live.
fn live_keys(keys: &[String], live: &[&str]) -> bool {
    keys.iter().all(|k| live.contains(&k.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn keys(names: &[&str]) -> Vec<String> {
        names.iter().map(|k| k.to_string()).collect()
    }

    // The sun and the fog are replaced wholesale, so no field of theirs can
    // decline; a type this module does not own never claims the edit.
    #[test]
    fn the_wholesale_types_take_any_field() {
        assert!(is_expressible(
            RegisteredType::DirectionalLight,
            &keys(&["direction", "color", "intensity"]),
        ));
        assert!(is_expressible(
            RegisteredType::VolumetricFog,
            &keys(&["enabled", "density"]),
        ));
        assert!(!is_expressible(
            RegisteredType::PointLight,
            &keys(&["intensity"])
        ));
        assert!(!is_expressible(RegisteredType::Sprite, &keys(&["width"])));
    }

    // A shadow knob the per-frame cascade math reads goes live; the map
    // resolution sizes the shadow array at load, so it rebuilds.
    #[test]
    fn a_graphics_config_edit_declines_the_load_time_knobs() {
        assert!(is_expressible(
            RegisteredType::GraphicsConfig,
            &keys(&["shadow_distance", "shadow_cascades"]),
        ));
        assert!(!is_expressible(
            RegisteredType::GraphicsConfig,
            &keys(&["shadow_map_size"]),
        ));
        assert!(
            !is_expressible(
                RegisteredType::GraphicsConfig,
                &keys(&["shadow_distance", "frames_in_flight"]),
            ),
            "one knob the renderer cannot take declines the whole change"
        );
    }

    // The look-tuning scalars go live; the feature toggles beside them gate
    // passes built at load.
    #[test]
    fn a_post_process_edit_declines_the_toggles() {
        assert!(is_expressible(
            RegisteredType::PostProcessConfig,
            &keys(&["ambient_intensity", "exposure_ev", "ssgi_intensity"]),
        ));
        assert!(!is_expressible(
            RegisteredType::PostProcessConfig,
            &keys(&["ssao"]),
        ));
        assert!(!is_expressible(
            RegisteredType::PostProcessConfig,
            &keys(&["aa_mode"]),
        ));
    }

    // A world with no renderer has nothing to push to, so every lighting edit
    // falls back to the rebuild rather than being swallowed.
    #[test]
    fn every_lighting_type_declines_without_a_renderer() {
        let world = World::new();
        assert!(!lighting_preview::is_available(&world));
        for (ty, args, changed) in [
            ("DirectionalLight", json!({ "intensity": 2.0 }), "intensity"),
            ("VolumetricFog", json!({ "density": 0.1 }), "density"),
            (
                "GraphicsConfig",
                json!({ "shadow_distance": 90 }),
                "shadow_distance",
            ),
            (
                "PostProcessConfig",
                json!({ "ambient_intensity": 0.5 }),
                "ambient_intensity",
            ),
        ] {
            let ct = RegisteredType::parse(ty).expect("a registered type");
            assert!(
                plan(
                    &world,
                    ct,
                    "lighting",
                    args.as_object().unwrap(),
                    &keys(&[changed]),
                )
                .is_none(),
                "{ty} declines with no renderer"
            );
        }
    }

    // The sun's component still lives in the world, so the ECS half of the
    // commit lands even where there is no renderer to push to.
    #[test]
    fn committing_a_sun_writes_the_component() {
        use crate::components::DirectionalLight;
        let mut world = World::new();
        let entity = world.push(DirectionalLight::default());
        let edited = DirectionalLight {
            direction: [0.0, -1.0, 0.0],
            color: [0.2, 0.4, 0.6],
            intensity: 5.0,
        };
        commit_sun(&mut world, entity, edited.clone());
        let held = world
            .query::<DirectionalLight>()
            .next()
            .expect("the light column still holds it");
        assert_eq!(held.direction, edited.direction);
        assert_eq!(held.color, edited.color);
        assert_eq!(held.intensity, edited.intensity);
    }

    #[test]
    fn live_keys_needs_every_key() {
        let live = ["shadow_distance", "shadow_cascades"];
        assert!(live_keys(&[], &live), "no change is trivially live");
        assert!(live_keys(&keys(&["shadow_cascades"]), &live));
        assert!(!live_keys(&keys(&["shadow_cascades", "vsync"]), &live));
    }
}
