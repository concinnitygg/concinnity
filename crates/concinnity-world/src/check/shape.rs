// src/check/shape.rs
//
// World-shape rules: structural constraints over the expanded world that no
// single asset's args can express. Stage-B validation (`check_world_with`)
// runs every rule here after expansion, so the checks see the world exactly as
// it will be baked: a rule that a build-time pass fills (the renderable
// contract, filled by companion injection) holds for any world that triggers
// the fill, and a violation always means the declared assets themselves
// conflict. Each rule documents its filler; a rule with no filler is a pure
// authoring constraint.
//
// The rules are driven by the registry's structural metadata (the `singleton`
// flag), so the expansion passes, the editor, and these assertions share one
// source of truth. Reference RESOLUTION is not a shape rule: every registry
// `refs:` field is resolved generically by the cross-reference validator
// (`validate_registry_refs`); the rules here judge relationships between
// assets that already resolve.

use crate::registry::RegisteredType;
use crate::world::WorldJsonlAsset;
use std::collections::HashSet;

fn norm(t: &str) -> String {
    t.to_lowercase().replace('_', "")
}

// A non-empty string arg, i.e. an explicit authored reference. Non-string and
// empty values are left to the per-asset arg checks.
fn str_arg<'a>(asset: &'a WorldJsonlAsset, field: &str) -> Option<&'a str> {
    asset
        .args
        .get(field)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
}

// Names of every asset whose normalized type matches.
fn names_of_type<'a>(assets: &'a [WorldJsonlAsset], type_norm: &str) -> HashSet<&'a str> {
    assets
        .iter()
        .filter(|a| norm(&a.asset_type) == type_norm)
        .map(|a| a.name.as_str())
        .collect()
}

// The screen an element belongs to: its explicit `screen` field first, else
// the longest `<screen>_` name prefix. Mirrors the build's membership
// resolution (`resolve_scene_refs`), which runs after these checks.
fn owning_screen<'a>(
    element: &'a WorldJsonlAsset,
    screens: &'a HashSet<&'a str>,
) -> Option<&'a str> {
    if let Some(explicit) = str_arg(element, "screen") {
        return Some(explicit);
    }
    screens
        .iter()
        .filter(|sn| element.name.starts_with(&format!("{sn}_")))
        .max_by_key(|sn| sn.len())
        .copied()
}

// Run every world-shape rule, collecting all violations.
pub(crate) fn check_shape(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    check_singletons(assets, errors);
    check_initial_screens(assets, errors);
    check_focus_ownership(assets, errors);
    check_renderable_contract(assets, errors);
    check_shader_budget(assets, errors);
    check_material_shader_consumers(assets, errors);
    super::physics::check_layers(assets, errors);
}

// At most one instance of every `singleton`-flagged type. No filler: companion
// injection only adds a missing singleton (it skips when the type is already
// present), so a violation always means two declared or generated instances.
fn check_singletons(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    for ty in RegisteredType::all().iter().filter(|t| t.singleton()) {
        let type_norm = norm(ty.as_str());
        let names: Vec<&str> = assets
            .iter()
            .filter(|a| norm(&a.asset_type) == type_norm)
            .map(|a| a.name.as_str())
            .collect();
        if names.len() > 1 {
            errors.push(format!(
                "{} is a world singleton but {} are declared ({}); keep one",
                ty.as_str(),
                names.len(),
                names.join(", ")
            ));
        }
    }
}

// At most one Screen seeds the open stack. No filler. The runtime opens the
// first `initial` screen it encounters, so a second one would silently lose to
// declaration order.
fn check_initial_screens(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    let initial: Vec<&str> = assets
        .iter()
        .filter(|a| norm(&a.asset_type) == "screen")
        .filter(|a| a.args.get("initial").and_then(|v| v.as_bool()) == Some(true))
        .map(|a| a.name.as_str())
        .collect();
    if initial.len() > 1 {
        errors.push(format!(
            "{} Screens are marked initial ({}); only one screen can seed the open stack",
            initial.len(),
            initial.join(", ")
        ));
    }
}

// A Screen's `focus` must reference a TextInput on that same screen: an input
// on another screen is not even visible while this one is up, so focusing it
// would send keystrokes off-screen. An unowned (global) input is allowed. No
// filler. Existence of the focus target is covered by the generic registry-ref
// resolution in the cross-reference validator; this rule only judges
// ownership, so a dangling focus reports once.
fn check_focus_ownership(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    let screens = names_of_type(assets, "screen");
    for screen in assets.iter().filter(|a| norm(&a.asset_type) == "screen") {
        let Some(focus) = str_arg(screen, "focus") else {
            continue;
        };
        let Some(input) = assets
            .iter()
            .find(|a| norm(&a.asset_type) == "textinput" && a.name == focus)
        else {
            continue;
        };
        let owner = owning_screen(input, &screens);
        if let Some(owner) = owner
            && owner != screen.name
        {
            errors.push(format!(
                "Screen '{}': focus '{}' belongs to screen '{}'; a screen can only focus its own TextInput",
                screen.name, focus, owner
            ));
        }
    }
}

// A rendering world (one with a GraphicsConfig) needs a Window.
// Filled by companion injection: any `renders`-flagged type pulls in the
// GraphicsConfig marker, which pulls in a Window. This fires only when the
// world declares an incomplete render stack of its own. A Shader is not
// required: a world that declares none renders through the engine's own
// main-pass program.
fn check_renderable_contract(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    let has_graphics = assets
        .iter()
        .any(|a| norm(&a.asset_type) == "graphicsconfig");
    if !has_graphics {
        return;
    }
    let has_window = assets.iter().any(|a| norm(&a.asset_type) == "window");
    if !has_window {
        errors.push(
            "world renders (has a GraphicsConfig) but has no Window; declare one \
             or remove the GraphicsConfig"
                .to_string(),
        );
    }
}

// The renderer gives every Shader its own pipeline and its own indirect
// command buffer per draw pass, sized at init from a fixed bucket count. No
// filler: the count is exactly what the world declares.
fn check_shader_budget(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    let shaders = names_of_type(assets, "shader");
    let max = concinnity_core::gfx::render_types::MAX_SHADER_BUCKETS;
    if shaders.len() > max {
        let mut names: Vec<&str> = shaders.into_iter().collect();
        names.sort_unstable();
        errors.push(format!(
            "world declares {} Shaders but at most {max} are supported ({}); \
             share one Shader between more materials",
            names.len(),
            names.join(", ")
        ));
    }
}

// A Material naming a Shader may only be used where the renderer can honour it.
// Instanced, skinned, and voxel-chunk draws render under the world's default
// Shader, so a custom Shader on their material would silently shade them as if
// it were not there. No filler: this is a pure authoring constraint.
fn check_material_shader_consumers(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    // (consumer type, what renders it) for the draw paths with no bucket.
    const UNSUPPORTED: &[(&str, &str)] = &[
        ("instancedprop", "instanced draws"),
        ("skinnedmesh", "skinned draws"),
        ("voxelworld", "voxel chunk draws"),
    ];
    let shaded: HashSet<&str> = assets
        .iter()
        .filter(|a| norm(&a.asset_type) == "material" && str_arg(a, "shader").is_some())
        .map(|a| a.name.as_str())
        .collect();
    if shaded.is_empty() {
        return;
    }
    for (consumer_type, draws) in UNSUPPORTED {
        for consumer in assets
            .iter()
            .filter(|a| norm(&a.asset_type) == *consumer_type)
        {
            let Some(material) = str_arg(consumer, "material") else {
                continue;
            };
            if !shaded.contains(material) {
                continue;
            }
            errors.push(format!(
                "{} '{}' uses material '{}', which names a Shader, but {} always render with the \
                 world's default Shader; drop the Shader from that material or give '{}' a \
                 material without one",
                consumer.asset_type, consumer.name, material, draws, consumer.name
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str, asset_type: &str, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            name: name.to_string(),
            asset_type: asset_type.to_string(),
            args,
        }
    }

    fn errors_for(assets: &[WorldJsonlAsset]) -> Vec<String> {
        let mut errors = Vec::new();
        check_shape(assets, &mut errors);
        errors
    }

    fn render_stack() -> Vec<WorldJsonlAsset> {
        vec![
            asset("gfx", "GraphicsConfig", serde_json::json!({})),
            asset("win", "Window", serde_json::json!({})),
            asset(
                "scene_shader",
                "Shader",
                serde_json::json!({
                    "vertex": {"sources": {"metal": "x.metal"}},
                    "fragment": {"sources": {"metal": "x.metal"}}
                }),
            ),
        ]
    }

    #[test]
    fn a_second_singleton_instance_is_an_error() {
        let mut assets = render_stack();
        assets.push(asset("win2", "Window", serde_json::json!({})));
        let errs = errors_for(&assets);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains("Window is a world singleton"));
        assert!(errs[0].contains("win") && errs[0].contains("win2"));
    }

    #[test]
    fn one_of_each_singleton_passes() {
        let mut assets = render_stack();
        assets.push(asset("app", "AppConfig", serde_json::json!({})));
        assets.push(asset("phys", "PhysicsConfig", serde_json::json!({})));
        assert!(errors_for(&assets).is_empty());
    }

    #[test]
    fn two_initial_screens_are_an_error() {
        let mut assets = render_stack();
        assets.push(asset(
            "menu",
            "Screen",
            serde_json::json!({"initial": true}),
        ));
        assets.push(asset("hud", "Screen", serde_json::json!({"initial": true})));
        let errs = errors_for(&assets);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains("menu") && errs[0].contains("hud"));
    }

    #[test]
    fn one_initial_screen_or_none_passes() {
        let mut assets = render_stack();
        assets.push(asset(
            "menu",
            "Screen",
            serde_json::json!({"initial": true}),
        ));
        assets.push(asset("hud", "Screen", serde_json::json!({})));
        assert!(errors_for(&assets).is_empty());
    }

    #[test]
    fn focus_on_another_screens_input_is_an_error() {
        let mut assets = render_stack();
        assets.push(asset(
            "pause",
            "Screen",
            serde_json::json!({"focus": "menu_search"}),
        ));
        assets.push(asset("menu", "Screen", serde_json::json!({})));
        // Owned by `menu` via the name prefix.
        assets.push(asset("menu_search", "TextInput", serde_json::json!({})));
        let errs = errors_for(&assets);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains("Screen 'pause'"));
        assert!(errs[0].contains("belongs to screen 'menu'"));
    }

    #[test]
    fn focus_on_own_or_global_input_passes() {
        let mut assets = render_stack();
        assets.push(asset(
            "menu",
            "Screen",
            serde_json::json!({"focus": "menu_search"}),
        ));
        assets.push(asset("menu_search", "TextInput", serde_json::json!({})));
        // A global (unowned) input may be focused from any screen.
        assets.push(asset(
            "pause",
            "Screen",
            serde_json::json!({"focus": "console_line"}),
        ));
        assets.push(asset("console_line", "TextInput", serde_json::json!({})));
        assert!(errors_for(&assets).is_empty());
    }

    #[test]
    fn explicit_screen_field_overrides_the_prefix_for_ownership() {
        let mut assets = render_stack();
        assets.push(asset("menu", "Screen", serde_json::json!({})));
        assets.push(asset(
            "pause",
            "Screen",
            serde_json::json!({"focus": "menu_search"}),
        ));
        // Named under `menu_` but explicitly owned by `pause`: the explicit
        // field wins, exactly as the build's membership resolution decides.
        assets.push(asset(
            "menu_search",
            "TextInput",
            serde_json::json!({"screen": "pause"}),
        ));
        assert!(errors_for(&assets).is_empty());
    }

    // A dangling focus is a resolution failure, reported once by the generic
    // registry-ref pass in the cross-reference validator; the ownership rule
    // skips it rather than piling on a second error.
    #[test]
    fn ownership_skips_a_dangling_focus() {
        let mut assets = render_stack();
        assets.push(asset(
            "menu",
            "Screen",
            serde_json::json!({"focus": "ghost"}),
        ));
        assert!(errors_for(&assets).is_empty());
    }

    #[test]
    fn graphics_config_without_a_window_reports_it() {
        let assets = vec![asset("gfx", "GraphicsConfig", serde_json::json!({}))];
        let errs = errors_for(&assets);
        assert!(errs.iter().any(|e| e.contains("no Window")), "{errs:?}");
    }

    #[test]
    fn graphics_config_without_a_shader_is_fine() {
        // A world that declares no Shader renders with the engine's own program.
        let assets = vec![
            asset("gfx", "GraphicsConfig", serde_json::json!({})),
            asset("win", "Window", serde_json::json!({})),
        ];
        assert!(errors_for(&assets).is_empty());
    }

    #[test]
    fn a_complete_render_stack_passes() {
        assert!(errors_for(&render_stack()).is_empty());
    }

    #[test]
    fn a_non_rendering_world_needs_no_render_stack() {
        let assets = vec![asset("clip", "AudioClip", serde_json::json!({}))];
        assert!(errors_for(&assets).is_empty());
    }

    fn shader(name: &str) -> WorldJsonlAsset {
        asset(
            name,
            "Shader",
            serde_json::json!({
                "vertex": {"sources": {"metal": "x.metal"}},
                "fragment": {"sources": {"metal": "x.metal"}}
            }),
        )
    }

    #[test]
    fn more_shaders_than_buckets_is_an_error() {
        let max = concinnity_core::gfx::render_types::MAX_SHADER_BUCKETS;
        let mut assets = render_stack();
        for i in 0..max {
            assets.push(shader(&format!("extra_{i}")));
        }
        let errs = errors_for(&assets);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains(&format!("at most {max}")), "{errs:?}");

        // Exactly the cap passes.
        assets.pop();
        assert!(errors_for(&assets).is_empty());
    }

    // The draw paths that carry no shader bucket must not silently drop a
    // material's Shader: authoring one is an error, not a fallback.
    #[test]
    fn a_shaded_material_on_an_unbucketed_consumer_is_an_error() {
        let shaded = asset(
            "hero_mat",
            "Material",
            serde_json::json!({"shader": "custom_shader"}),
        );
        for (consumer_type, name) in [
            ("InstancedProp", "grass"),
            ("SkinnedMesh", "hero"),
            ("VoxelWorld", "terrain"),
        ] {
            let mut assets = render_stack();
            assets.push(shader("custom_shader"));
            assets.push(shaded.clone());
            assets.push(asset(
                name,
                consumer_type,
                serde_json::json!({"material": "hero_mat"}),
            ));
            let errs = errors_for(&assets);
            assert_eq!(errs.len(), 1, "{consumer_type}: {errs:?}");
            assert!(errs[0].contains("hero_mat"), "{errs:?}");
            assert!(errs[0].contains(name), "{errs:?}");
        }
    }

    #[test]
    fn an_unshaded_material_is_fine_on_every_consumer() {
        let mut assets = render_stack();
        assets.push(asset(
            "plain_mat",
            "Material",
            serde_json::json!({"roughness": 0.5}),
        ));
        assets.push(asset(
            "grass",
            "InstancedProp",
            serde_json::json!({"material": "plain_mat"}),
        ));
        // A Prop renders through the bucketed path, so a Shader is fine there.
        assets.push(shader("custom_shader"));
        assets.push(asset(
            "wall_mat",
            "Material",
            serde_json::json!({"shader": "custom_shader"}),
        ));
        assets.push(asset(
            "wall",
            "Prop",
            serde_json::json!({"material": "wall_mat"}),
        ));
        assert!(errors_for(&assets).is_empty());
    }
}
