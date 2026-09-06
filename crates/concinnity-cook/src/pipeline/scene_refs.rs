//! The naming-convention pass, run after the declaration-order interning and
//! before the payload compile.

use crate::authoring::world::WorldJsonlAsset;
use crate::ecs::asset_id;

// Resolve scene + screen associations that the runtime can no longer derive
// from name strings, baking them into the asset args so they survive as
// AssetId ids.
//
// Naming-convention relationships handled:
//   - A Prop named `<scene>_*` belongs to Scene `<scene>`. The matched scene
//     name is written into the prop's `scene` arg.
//   - A UI element (Sprite, ImageOverlay, TextLabel, Text, TextInput,
//     HitRegion, ScrollPanel) named `<screen>_*` belongs to Screen `<screen>`.
//     The matched screen name is written into the asset's `screen` arg.
//   - A HitRegion or KeyBinding `action` of the form `scene:<name>`,
//     `screen:show:<name>`, `screen:push:<name>`, or `screen:toggle:<name>`
//     has its `<name>` part rewritten to the interned id, so `UiInputSystem`
//     can parse an integer at runtime instead of a name.
pub(in crate::pipeline) fn resolve_scene_refs(assets: &mut [WorldJsonlAsset]) {
    let norm = |s: &str| s.to_lowercase().replace('_', "");

    let scene_names: Vec<String> = assets
        .iter()
        .filter(|a| norm(&a.asset_type) == "scene")
        .map(|a| a.name.clone())
        .collect();

    let screen_names: Vec<String> = assets
        .iter()
        .filter(|a| norm(&a.asset_type) == "screen")
        .map(|a| a.name.clone())
        .collect();

    // Longest matching prefix wins so a nested name (e.g. `level_boss_*` under
    // both `level` and `level_boss`) binds to the most specific host.
    // Equivalent to first-match when no host name prefixes another.
    let longest_prefix_host = |name: &str, hosts: &[String]| -> Option<String> {
        hosts
            .iter()
            .filter(|h| name.starts_with(&format!("{h}_")))
            .max_by_key(|h| h.len())
            .cloned()
    };

    // Rewrite an action string, replacing the trailing `<name>` after the
    // given action prefix with its interned id. Returns Some(new_action) when
    // the action used the prefix with an unresolved name; None otherwise.
    let resolve_action = |action: &str| -> Option<String> {
        for prefix in ["scene:", "screen:show:", "screen:push:", "screen:toggle:"] {
            if let Some(rest) = action.strip_prefix(prefix) {
                if !rest.is_empty() && rest.parse::<u32>().is_err() {
                    return Some(format!("{prefix}{}", asset_id::intern(rest).0));
                }
                return None;
            }
        }
        None
    };

    for asset in assets.iter_mut() {
        let ty = norm(&asset.asset_type);

        // Host binding by name prefix: a Prop takes its Scene, a UI element
        // takes its Screen. An asset that already names its host is left alone.
        let host = match ty.as_str() {
            "prop" => Some(("scene", &scene_names)),
            "sprite" | "imageoverlay" | "textlabel" | "text" | "textinput" | "hitregion"
            | "scrollpanel" => Some(("screen", &screen_names)),
            _ => None,
        };
        if let Some((key, hosts)) = host
            && asset.args.get(key).is_none()
            && let Some(matched) = longest_prefix_host(&asset.name, hosts)
            && let serde_json::Value::Object(m) = &mut asset.args
        {
            m.insert(key.to_string(), serde_json::Value::String(matched));
        }

        // Resolve screen:* / scene:* action targets to interned ids.
        if matches!(ty.as_str(), "hitregion" | "keybinding") {
            let new_action = asset
                .args
                .get("action")
                .and_then(|v| v.as_str())
                .and_then(resolve_action);
            if let (Some(action), serde_json::Value::Object(m)) = (new_action, &mut asset.args) {
                m.insert("action".to_string(), serde_json::Value::String(action));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::pipeline::build_pipeline_from_str;
    use crate::pipeline::fixtures::wja;

    // `screen:show:<name>` / `screen:toggle:<name>` action targets are
    // rewritten to interned ids at build time, like `scene:<name>`.
    #[test]
    fn build_pipeline_resolves_screen_action_refs() {
        let world = concat!(
            r#"{"name":"pause_menu","type":"Screen","args":{}}"#,
            "\n",
            r#"{"name":"btn","type":"HitRegion","args":{"x":0,"y":0,"width":10,"height":10,"action":"screen:toggle:pause_menu"}}"#,
            "\n",
            r#"{"name":"esc","type":"KeyBinding","args":{"key":"Escape","action":"screen:toggle:pause_menu"}}"#,
            "\n",
        );
        let result = build_pipeline_from_str(
            world,
            None,
            None,
            concinnity_core::platform::Platform::Metal,
        )
        .expect("build");
        // pause_menu interned id = 0 (first declared name).
        let btn = result
            .defs
            .iter()
            .find(|d| d.name == Some(crate::ecs::asset_id::AssetId(1)))
            .expect("HitRegion def");
        let baked: crate::components::HitRegion = postcard::from_bytes(&btn.args_bytes).unwrap();
        assert_eq!(baked.action, "screen:toggle:0");

        let esc = result
            .defs
            .iter()
            .find(|d| d.name == Some(crate::ecs::asset_id::AssetId(2)))
            .expect("KeyBinding def");
        let baked: crate::components::KeyBinding = postcard::from_bytes(&esc.args_bytes).unwrap();
        assert_eq!(baked.action, "screen:toggle:0");
    }

    // A Sprite/TextLabel/HitRegion named `<screen>_*` has its `screen` arg
    // resolved from the prefix at build time, mirroring Prop scene refs.
    #[test]
    fn build_pipeline_resolves_screen_prefix_on_ui_assets() {
        let world = concat!(
            r#"{"name":"pause_menu","type":"Screen","args":{}}"#,
            "\n",
            r#"{"name":"pause_menu_dim","type":"Sprite","args":{"x":0,"y":0,"width":10,"height":10}}"#,
            "\n",
            r#"{"name":"pause_menu_title","type":"TextLabel","args":{"font":"f","content":"x","x":0,"y":0}}"#,
            "\n",
            r#"{"name":"pause_menu_btn","type":"HitRegion","args":{"x":0,"y":0,"width":10,"height":10,"action":"screen:hide"}}"#,
            "\n",
            r#"{"name":"f","type":"Font","args":{"size_px":16}}"#,
            "\n",
        );
        let result = build_pipeline_from_str(
            world,
            None,
            None,
            concinnity_core::platform::Platform::Metal,
        )
        .expect("build");
        // pause_menu interned id = 0; the UI assets intern in declaration order.
        let baked_view = |id: u32, expect: &str| {
            let def = result
                .defs
                .iter()
                .find(|d| d.name == Some(crate::ecs::asset_id::AssetId(id)))
                .unwrap_or_else(|| panic!("expected a def for {expect}"));
            let ct = crate::registry::RegisteredType::from_discriminant(def.discriminant)
                .unwrap_or_else(|| panic!("{expect}: unknown discriminant"));
            match ct {
                crate::registry::RegisteredType::Sprite => {
                    postcard::from_bytes::<crate::components::Sprite>(&def.args_bytes)
                        .unwrap()
                        .screen
                }
                crate::registry::RegisteredType::TextLabel => {
                    postcard::from_bytes::<crate::components::TextLabel>(&def.args_bytes)
                        .unwrap()
                        .screen
                }
                crate::registry::RegisteredType::HitRegion => {
                    postcard::from_bytes::<crate::components::HitRegion>(&def.args_bytes)
                        .unwrap()
                        .screen
                }
                other => panic!("{expect}: unexpected type {other:?}"),
            }
        };
        for (id, name) in [
            (1, "pause_menu_dim"),
            (2, "pause_menu_title"),
            (3, "pause_menu_btn"),
        ] {
            assert_eq!(
                baked_view(id, name),
                Some(crate::ecs::asset_id::AssetId(0)),
                "expected {name} to have screen=0"
            );
        }
    }

    // Nested screen names resolve by longest prefix: `<menu>_settings_*` binds
    // to the `<menu>_settings` screen, not the enclosing `<menu>` screen that is
    // declared first. (Regression: first-match claimed the nested elements,
    // so a MainMenu's settings sub-screen rendered on top of the main menu.)
    #[test]
    fn resolve_scene_refs_picks_longest_screen_prefix() {
        let mk = |name: &str, ty: &str| crate::authoring::world::WorldJsonlAsset {
            name: name.to_string(),
            asset_type: ty.to_string(),
            args: serde_json::json!({}),
        };
        let mut assets = vec![
            mk("menu", "Screen"),
            mk("menu_settings", "Screen"),
            mk("menu_title", "TextLabel"),
            mk("menu_settings_title", "TextLabel"),
        ];
        super::resolve_scene_refs(&mut assets);
        let view_of = |n: &str| {
            assets
                .iter()
                .find(|a| a.name == n)
                .and_then(|a| a.args.get("screen"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
        };
        assert_eq!(view_of("menu_title").as_deref(), Some("menu"));
        assert_eq!(
            view_of("menu_settings_title").as_deref(),
            Some("menu_settings")
        );
    }

    // An authored `screen` arg wins over the name-prefix convention, exactly
    // as an authored `scene` does on a Prop.
    #[test]
    fn resolve_scene_refs_keeps_an_authored_screen_arg() {
        let mut assets = vec![
            wja("menu", "Screen", serde_json::json!({})),
            wja("other", "Screen", serde_json::json!({})),
            wja(
                "menu_title",
                "TextLabel",
                serde_json::json!({"screen": "other"}),
            ),
        ];
        super::resolve_scene_refs(&mut assets);
        assert_eq!(assets[2].args["screen"], "other");
    }

    #[test]
    fn resolve_scene_refs_prop_scene_prefix_rules() {
        let mut assets = vec![
            wja("level", "Scene", serde_json::json!({})),
            wja("level_boss", "Scene", serde_json::json!({})),
            wja("level_boss_door", "Prop", serde_json::json!({})),
            wja("level_gate", "Prop", serde_json::json!({"scene": "other"})),
            wja("solo_thing", "Prop", serde_json::json!({})),
        ];
        super::resolve_scene_refs(&mut assets);

        // Longest scene prefix wins for the nested name.
        assert_eq!(assets[2].args["scene"], "level_boss");
        // An authored `scene` arg is never overwritten.
        assert_eq!(assets[3].args["scene"], "other");
        // No matching prefix: no `scene` arg appears.
        assert!(assets[4].args.get("scene").is_none());
    }

    #[test]
    fn resolve_scene_refs_rewrites_action_names_to_interned_ids() {
        crate::ecs::asset_id::reset_interner();
        let mut assets = vec![
            wja(
                "btn",
                "HitRegion",
                serde_json::json!({"action": "screen:show:pause"}),
            ),
            wja(
                "key",
                "KeyBinding",
                serde_json::json!({"action": "scene:day"}),
            ),
        ];
        super::resolve_scene_refs(&mut assets);

        // Names intern in resolution order on this thread's fresh interner:
        // "pause" -> 0, "day" -> 1.
        assert_eq!(assets[0].args["action"], "screen:show:0");
        assert_eq!(assets[1].args["action"], "scene:1");
    }

    #[test]
    fn resolve_scene_refs_leaves_numeric_and_foreign_actions_alone() {
        let mut assets = vec![
            wja(
                "a",
                "HitRegion",
                serde_json::json!({"action": "screen:toggle:3"}),
            ),
            wja("b", "HitRegion", serde_json::json!({"action": "quit"})),
            wja("c", "KeyBinding", serde_json::json!({"action": "scene:"})),
        ];
        super::resolve_scene_refs(&mut assets);

        // Already an id, not a recognised prefix, and an empty target: all
        // pass through unchanged.
        assert_eq!(assets[0].args["action"], "screen:toggle:3");
        assert_eq!(assets[1].args["action"], "quit");
        assert_eq!(assets[2].args["action"], "scene:");
    }
}
