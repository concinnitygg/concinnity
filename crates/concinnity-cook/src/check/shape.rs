// World-shape rules: structural constraints over the expanded world that no
// single asset's args can express. Stage-B validation (`check_world`)
// runs every rule here after expansion, so the checks see the world exactly as
// it will be baked: a rule that a build-time pass fills (the renderable
// contract, filled by companion injection) holds for any world that triggers
// the fill, and a violation always means the declared assets themselves
// conflict. Each rule documents its filler; a rule with no filler is a pure
// authoring constraint.
//
// The rules are driven by the registry's structural metadata (the `singleton`
// flag), so the expansion passes, the editor, and these assertions share one
// source of truth. Reference RESOLUTION is not a shape rule: every reference
// field the registry derives is resolved generically by the cross-reference
// validator (`validate_registry_refs`); the rules here judge relationships
// between assets that already resolve.

use std::collections::HashSet;

use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;

// A non-empty string arg, i.e. an explicit authored reference. Non-string and
// empty values are left to the per-asset arg checks.
fn str_arg<'a>(asset: &'a WorldJsonlAsset, field: &str) -> Option<&'a str> {
    asset
        .args
        .get(field)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
}

// The `$id` of every asset of the given type that declares one.
fn names_of_type(assets: &[WorldJsonlAsset], asset_type: RegisteredType) -> HashSet<&str> {
    assets
        .iter()
        .filter(|a| a.asset_type == asset_type && !a.is_anonymous())
        .map(|a| a.id.as_str())
        .collect()
}

// The screen an element belongs to: the `screen` it names, if it names one
// this world declares.
fn owning_screen<'a>(
    element: &'a WorldJsonlAsset,
    screens: &'a HashSet<&'a str>,
) -> Option<&'a str> {
    let named = str_arg(element, "screen")?;
    screens.get(named).copied()
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
        let names: Vec<&str> = assets
            .iter()
            .filter(|a| a.asset_type == *ty)
            .map(|a| a.id.as_str())
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
        .filter(|a| a.asset_type == RegisteredType::Screen)
        .filter(|a| a.args.get("initial").and_then(|v| v.as_bool()) == Some(true))
        .map(|a| a.id.as_str())
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
    let screens = names_of_type(assets, RegisteredType::Screen);
    for screen in assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::Screen)
    {
        let Some(focus) = str_arg(screen, "focus") else {
            continue;
        };
        let Some(input) = assets
            .iter()
            .find(|a| a.asset_type == RegisteredType::TextInput && a.id == focus)
        else {
            continue;
        };
        let owner = owning_screen(input, &screens);
        if let Some(owner) = owner
            && owner != screen.id
        {
            errors.push(format!(
                "Screen '{}': focus '{}' belongs to screen '{}'; a screen can only focus its own TextInput",
                screen.id, focus, owner
            ));
        }
    }
}

// A world with something to draw needs a Window to draw into. Filled by
// companion injection: any `renders`-flagged type pulls one in. This fires only
// when that injection could not run. A Shader is not required: a world that
// declares none renders through the engine's own main-pass program.
fn check_renderable_contract(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    let Some(reason) = assets
        .iter()
        .find(|a| a.asset_type.renders())
        .map(|a| a.asset_type)
    else {
        return;
    };
    let has_window = assets
        .iter()
        .any(|a| a.asset_type == RegisteredType::Window);
    if !has_window {
        errors.push(format!(
            "world renders (it declares a {}) but has no Window; declare one \
             or remove what renders",
            reason.as_str()
        ));
    }
}

// The renderer gives every Shader its own pipeline and its own indirect
// command buffer per draw pass, sized at init from a fixed bucket count. No
// filler: the count is exactly what the world declares.
fn check_shader_budget(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    let shaders = names_of_type(assets, RegisteredType::Shader);
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

// A Material naming a Shader may only be used where the renderer can honor it.
// Instanced, skinned, and voxel-chunk draws render under the world's default
// Shader, so a custom Shader on their material would silently shade them as if
// it were not there. No filler: this is a pure authoring constraint.
fn check_material_shader_consumers(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    // (consumer type, what renders it) for the draw paths with no bucket.
    const UNSUPPORTED: &[(RegisteredType, &str)] = &[
        (RegisteredType::InstancedProp, "instanced draws"),
        (RegisteredType::SkinnedMesh, "skinned draws"),
        (RegisteredType::VoxelWorld, "voxel chunk draws"),
    ];
    let shaded: HashSet<&str> = assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::Material && str_arg(a, "shader").is_some())
        .map(|a| a.id.as_str())
        .collect();
    if shaded.is_empty() {
        return;
    }
    for (consumer_type, draws) in UNSUPPORTED {
        for consumer in assets.iter().filter(|a| a.asset_type == *consumer_type) {
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
                consumer.asset_type.as_str(),
                consumer.id,
                material,
                draws,
                consumer.id
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str, asset_type: RegisteredType, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            id: name.to_string(),
            asset_type,
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
            asset("gfx", RegisteredType::GraphicsConfig, serde_json::json!({})),
            asset("win", RegisteredType::Window, serde_json::json!({})),
            asset(
                "scene_shader",
                RegisteredType::Shader,
                serde_json::json!({"fragment": "x.slang"}),
            ),
        ]
    }

    #[test]
    fn a_second_singleton_instance_is_an_error() {
        let mut assets = render_stack();
        assets.push(asset("win2", RegisteredType::Window, serde_json::json!({})));
        let errs = errors_for(&assets);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains("Window is a world singleton"));
        assert!(errs[0].contains("win") && errs[0].contains("win2"));
    }

    #[test]
    fn one_of_each_singleton_passes() {
        let mut assets = render_stack();
        assets.push(asset(
            "app",
            RegisteredType::AppConfig,
            serde_json::json!({}),
        ));
        assets.push(asset(
            "phys",
            RegisteredType::PhysicsConfig,
            serde_json::json!({}),
        ));
        assert!(errors_for(&assets).is_empty());
    }

    #[test]
    fn two_initial_screens_are_an_error() {
        let mut assets = render_stack();
        assets.push(asset(
            "menu",
            RegisteredType::Screen,
            serde_json::json!({"initial": true}),
        ));
        assets.push(asset(
            "hud",
            RegisteredType::Screen,
            serde_json::json!({"initial": true}),
        ));
        let errs = errors_for(&assets);
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(errs[0].contains("menu") && errs[0].contains("hud"));
    }

    #[test]
    fn one_initial_screen_or_none_passes() {
        let mut assets = render_stack();
        assets.push(asset(
            "menu",
            RegisteredType::Screen,
            serde_json::json!({"initial": true}),
        ));
        assets.push(asset("hud", RegisteredType::Screen, serde_json::json!({})));
        assert!(errors_for(&assets).is_empty());
    }

    #[test]
    fn focus_on_another_screens_input_is_an_error() {
        let mut assets = render_stack();
        assets.push(asset(
            "pause",
            RegisteredType::Screen,
            serde_json::json!({"focus": "menu_search"}),
        ));
        assets.push(asset("menu", RegisteredType::Screen, serde_json::json!({})));
        assets.push(asset(
            "menu_search",
            RegisteredType::TextInput,
            serde_json::json!({"screen": "menu"}),
        ));
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
            RegisteredType::Screen,
            serde_json::json!({"focus": "menu_search"}),
        ));
        assets.push(asset(
            "menu_search",
            RegisteredType::TextInput,
            serde_json::json!({"screen": "menu"}),
        ));
        // A global (unowned) input may be focused from any screen.
        assets.push(asset(
            "pause",
            RegisteredType::Screen,
            serde_json::json!({"focus": "console_line"}),
        ));
        assets.push(asset(
            "console_line",
            RegisteredType::TextInput,
            serde_json::json!({}),
        ));
        assert!(errors_for(&assets).is_empty());
    }

    // Ownership comes from the `screen` field alone: a name that reads like it
    // belongs to another screen owns nothing.
    #[test]
    fn a_name_that_reads_like_another_screens_does_not_own_an_input() {
        let mut assets = render_stack();
        assets.push(asset("menu", RegisteredType::Screen, serde_json::json!({})));
        assets.push(asset(
            "pause",
            RegisteredType::Screen,
            serde_json::json!({"focus": "menu_search"}),
        ));
        assets.push(asset(
            "menu_search",
            RegisteredType::TextInput,
            serde_json::json!({"screen": "pause"}),
        ));
        assert!(errors_for(&assets).is_empty());

        // The same input with no `screen` is global, so any screen may focus it.
        let mut assets = render_stack();
        assets.push(asset("menu", RegisteredType::Screen, serde_json::json!({})));
        assets.push(asset(
            "pause",
            RegisteredType::Screen,
            serde_json::json!({"focus": "menu_search"}),
        ));
        assets.push(asset(
            "menu_search",
            RegisteredType::TextInput,
            serde_json::json!({}),
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
            RegisteredType::Screen,
            serde_json::json!({"focus": "ghost"}),
        ));
        assert!(errors_for(&assets).is_empty());
    }

    #[test]
    fn graphics_config_without_a_window_reports_it() {
        let assets = vec![asset(
            "gfx",
            RegisteredType::GraphicsConfig,
            serde_json::json!({}),
        )];
        let errs = errors_for(&assets);
        assert!(errs.iter().any(|e| e.contains("no Window")), "{errs:?}");
    }

    #[test]
    fn graphics_config_without_a_shader_is_fine() {
        // A world that declares no Shader renders with the engine's own program.
        let assets = vec![
            asset("gfx", RegisteredType::GraphicsConfig, serde_json::json!({})),
            asset("win", RegisteredType::Window, serde_json::json!({})),
        ];
        assert!(errors_for(&assets).is_empty());
    }

    #[test]
    fn a_complete_render_stack_passes() {
        assert!(errors_for(&render_stack()).is_empty());
    }

    #[test]
    fn a_non_rendering_world_needs_no_render_stack() {
        let assets = vec![asset(
            "clip",
            RegisteredType::AudioClip,
            serde_json::json!({}),
        )];
        assert!(errors_for(&assets).is_empty());
    }

    fn shader(name: &str) -> WorldJsonlAsset {
        asset(
            name,
            RegisteredType::Shader,
            serde_json::json!({"fragment": "x.slang"}),
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
            RegisteredType::Material,
            serde_json::json!({"shader": "custom_shader"}),
        );
        for (consumer_type, name) in [
            (RegisteredType::InstancedProp, "grass"),
            (RegisteredType::SkinnedMesh, "hero"),
            (RegisteredType::VoxelWorld, "terrain"),
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
            assert_eq!(errs.len(), 1, "{consumer_type:?}: {errs:?}");
            assert!(errs[0].contains("hero_mat"), "{errs:?}");
            assert!(errs[0].contains(name), "{errs:?}");
        }
    }

    #[test]
    fn an_unshaded_material_is_fine_on_every_consumer() {
        let mut assets = render_stack();
        assets.push(asset(
            "plain_mat",
            RegisteredType::Material,
            serde_json::json!({"roughness": 0.5}),
        ));
        assets.push(asset(
            "grass",
            RegisteredType::InstancedProp,
            serde_json::json!({"material": "plain_mat"}),
        ));
        // A Prop renders through the bucketed path, so a Shader is fine there.
        assets.push(shader("custom_shader"));
        assets.push(asset(
            "wall_mat",
            RegisteredType::Material,
            serde_json::json!({"shader": "custom_shader"}),
        ));
        assets.push(asset(
            "wall",
            RegisteredType::Prop,
            serde_json::json!({"material": "wall_mat"}),
        ));
        assert!(errors_for(&assets).is_empty());
    }
}
