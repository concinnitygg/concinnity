//! The lighting live path. Unlike a Sprite or a TextLabel, none of these assets
//! is data the running world re-reads each frame: the sun is packed into the
//! renderer's light uniforms at load and the fog / shadow / post-process assets
//! are consumed there outright, so writing their columns would apply nothing.
//! Core's `live_preview::lighting` module is what actually reaches the
//! renderer; this module decides whether an edit can go through it and bakes the
//! component it takes.
//!
//! What a rebuild would show is the standard: a knob the renderer sizes a GPU
//! resource from, and fog on a world that started without it, are declined here
//! so the caller rebuilds rather than showing a change that does not stick.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_core::components::DirectionalLight;
use concinnity_core::ecs::{ComponentAsset, Entity, World};
use concinnity_core::render::live_preview::lighting as lighting_preview;
use concinnity_host::thread::asset_id;
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
    match component::bake(ct, args).ok()? {
        ComponentAsset::DirectionalLight(light) => {
            let entity = world.entity_of(asset_id::lookup(name)?)?;
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
        ComponentAsset::GraphicsConfig(graphics) => Some(Apply::RenderConfig(
            RenderConfig::Graphics(Box::new(graphics)),
        )),
        _ => None,
    }
}

// Whether this is a lighting type at all, and whether every field the edit
// changed is one the renderer can be handed. The sun and the fog are replaced
// wholesale, so every field of theirs qualifies; `GraphicsConfig` carries
// knobs that size a GPU resource at load beside the ones that do not, each
// changed key naming its `section.field`.
fn is_expressible(ct: RegisteredType, keys: &[String]) -> bool {
    match ct {
        RegisteredType::DirectionalLight | RegisteredType::VolumetricFog => true,
        RegisteredType::GraphicsConfig => keys.iter().all(|key| {
            key.split_once('.').is_some_and(|(section, field)| {
                lighting_preview::live_settings_fields(section)
                    .is_some_and(|live| live.contains(&field))
            })
        }),
        _ => false,
    }
}

/// Perform a planned lighting push.
pub(super) fn commit(world: &mut World, config: RenderConfig) {
    match config {
        RenderConfig::Fog(fog) => {
            lighting_preview::apply_fog(world, Some(&fog));
        }
        RenderConfig::Graphics(graphics) => {
            lighting_preview::apply_graphics(world, &graphics);
        }
    }
}

/// Replace one directional light and re-push the whole set. The renderer packs
/// the lights in column order, which is the order this reads them back in, so
/// the edited light lands in the slot it already occupied.
pub(super) fn commit_sun(world: &mut World, entity: Entity, light: DirectionalLight) {
    world.replace_component(entity, ComponentAsset::DirectionalLight(light));
    let lights: Vec<DirectionalLight> = world.query::<DirectionalLight>().cloned().collect();
    lighting_preview::apply_directional_lights(world, &lights);
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
    fn a_shadows_edit_declines_the_load_time_knobs() {
        let gfx = RegisteredType::GraphicsConfig;
        assert!(is_expressible(
            gfx,
            &keys(&["shadows.distance", "shadows.cascades"])
        ));
        assert!(!is_expressible(gfx, &keys(&["shadows.map_size"])));
        assert!(
            !is_expressible(gfx, &keys(&["shadows.distance", "shadows.map_size"])),
            "one knob the renderer cannot take declines the whole change"
        );
        assert!(!is_expressible(
            gfx,
            &keys(&["frame_pacing.frames_in_flight"])
        ));
        assert!(!is_expressible(gfx, &keys(&["shadows"])));
    }

    // The look-tuning scalars go live; the feature toggles beside them gate
    // passes built at load.
    #[test]
    fn a_look_edit_declines_the_toggles() {
        let gfx = RegisteredType::GraphicsConfig;
        assert!(is_expressible(
            gfx,
            &keys(&[
                "indirect_lighting.ambient_intensity",
                "indirect_lighting.ssgi_intensity"
            ]),
        ));
        assert!(is_expressible(gfx, &keys(&["exposure.ev"])));
        assert!(!is_expressible(gfx, &keys(&["ambient_occlusion.enabled"])));
        assert!(!is_expressible(gfx, &keys(&["anti_aliasing.mode"])));
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
                json!({ "shadows": { "distance": 90 } }),
                "shadows.distance",
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
        use concinnity_core::components::DirectionalLight;
        let mut world = World::new();
        let entity = world.push(DirectionalLight::default());
        let edited = DirectionalLight {
            direction: [0.0, -1.0, 0.0],
            color: [0.2, 0.4, 0.6],
            intensity: 5.0,
        };
        commit_sun(&mut world, entity, edited);
        let held = world
            .query::<DirectionalLight>()
            .next()
            .expect("the light column still holds it");
        assert_eq!(held.direction, edited.direction);
        assert_eq!(held.color, edited.color);
        assert_eq!(held.intensity, edited.intensity);
    }
}
