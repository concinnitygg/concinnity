// The authored Screen: the runtime screen's policies plus the elements it
// holds, which the cook lifts out into world entries that name the screen.

use std::collections::HashSet;

use concinnity_engine::components::{ScreenInput, ScreenSpace, TextInput};
use concinnity_engine::ecs::{FieldTable, Ref};
use serde::de::Error as _;

use crate::authoring::registry::{Authored, RegisteredType};
use crate::authoring::world::WorldJsonlAsset;
use crate::build_only::expand::{asset_name, registered_type, set_arg_if_absent};
use crate::ui::screen_scope::SCREEN_CHILDREN;
use crate::world_builder::Diagnostic;

/// A named full-screen layer of UI drawn over the world: a pause menu, a
/// settings page, a console, a score overlay.
///
/// A screen holds its elements in `children`, each written like a one-entry
/// object naming its type: `{"Sprite": {...}}`. A [Sprite](#sprite),
/// [`TextLabel`](#textlabel), [`TextInput`](#textinput),
/// [`HitRegion`](#hitregion), [`ScrollPanel`](#scrollpanel),
/// [`KeyBinding`](#keybinding), [Panel](#panel), [Slider](#slider), or
/// [`OptionSelect`](#optionselect) can be a child. Each child becomes an asset
/// of its own that belongs to the screen, named by its `$id` or, without one,
/// `<screen>_<index>`. An element declared elsewhere joins a screen by naming
/// it in its own `screen` field instead, mirroring the [Scene](#scene) →
/// [Prop](#prop) relationship.
///
/// Active screens form a stack; each is shown / hidden via
/// [`HitRegion`](#hitregion) or [`KeyBinding`](#keybinding) actions:
/// - `{"show": "<name>"}` replaces the top of the stack (menu navigation)
/// - `{"push": "<name>"}` opens on top of what is already showing
/// - `"hide"` closes the top screen, revealing what was beneath
/// - `{"toggle": "<name>"}` closes the screen if it is on top, opens it otherwise
///
/// Screens draw in stack order (later on top); `layer` orders a screen
/// against the always-on HUD and other screens independent of stack position.
/// While any active screen has `pauses_world` set, the world freezes exactly
/// as a pause menu does. A `toggle_key` opens and closes the screen from
/// anywhere. `focus` names a [`TextInput`](#textinput) that receives keyboard
/// focus whenever the screen reaches the top of the stack. Worlds that need no
/// menus simply declare no screens.
///
/// ```rust
/// # use concinnity_cook::ui::{Element, Screen};
/// # use concinnity_engine::components::TextLabel;
/// Screen {
///     toggle_key: "Escape".into(),
///     children: vec![Element::new(
///         "paused",
///         &TextLabel {
///             content: "Paused".into(),
///             ..Default::default()
///         },
///     )
///     .unwrap()],
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    concinnity_engine::ecs::AssetFields,
    concinnity_engine::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Screen {
    /// When true, this screen is shown as soon as the world loads.
    pub initial: bool,
    /// Seconds to fade the screen in when it's shown. 0 shows it instantly.
    pub fade_in_secs: f32,
    /// `InputKey` that toggles this screen open / closed from anywhere, by the
    /// same canonical key names a [`KeyBinding`](#keybinding) uses (e.g.
    /// "Escape", "Backtick"). Empty leaves the screen action-driven only.
    pub toggle_key: String,
    /// Input policy while the screen is active.
    pub input: ScreenInput,
    /// Where the screen's elements are laid out: the 1280 x 720 reference
    /// canvas scaled to the window, or window pixels.
    pub space: ScreenSpace,
    /// Open from load and outside the stack: no action or key shows or hides
    /// it, its elements keep the visibility they are given, and its
    /// [`HitRegion`](#hitregion)s react while no capturing screen is open. A
    /// HUD is one; `initial`, `toggle_key`, `input`, `pauses_world`,
    /// `fade_in_secs`, and `focus` do not apply to it.
    pub persistent: bool,
    /// When true (the default), the world pauses beneath this screen while it
    /// is active: gameplay input, physics, and animation freeze.
    #[asset(default = true)]
    pub pauses_world: bool,
    /// [`TextInput`](#textinput) that receives keyboard focus whenever this
    /// screen reaches the top of the stack.
    pub focus: Option<Ref<TextInput>>,
    /// Draw-order bias against the always-on HUD and other screens. Screens
    /// default above the HUD in stack order; a negative layer draws beneath
    /// the HUD, a higher layer stays above later-pushed screens.
    pub layer: i32,
    /// The elements this screen holds, top of the list drawn first.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<Element>,
}

impl Screen {
    // The stack entry this screen bakes into. The children are gone by then:
    // the cook lifted them out into assets of their own.
    pub(crate) fn runtime(self) -> concinnity_engine::components::Screen {
        concinnity_engine::components::Screen {
            initial: self.initial,
            fade_in_secs: self.fade_in_secs,
            toggle_key: self.toggle_key,
            input: self.input,
            space: self.space,
            persistent: self.persistent,
            pauses_world: self.pauses_world,
            focus: self.focus,
            layer: self.layer,
        }
    }
}

/// One element of a [Screen](#screen): an asset written as a one-entry object
/// naming its type, `{"TextLabel": {"content": "Paused"}}`.
#[derive(Debug, Clone, PartialEq)]
pub struct Element {
    asset_type: String,
    args: serde_json::Map<String, serde_json::Value>,
}

impl Element {
    /// An element declaring `value`, named `name`; an empty name takes the
    /// screen's generated one.
    ///
    /// A field that references another asset holds a resolved handle rather
    /// than a name, so a value with one is better written in world.jsonl or
    /// declared on its own and pointed at the screen by its `screen` field.
    ///
    /// # Errors
    ///
    /// Returns an error if `value` does not serialize to a JSON object.
    pub fn new<T: Authored>(name: &str, value: &T) -> serde_json::Result<Self> {
        let serde_json::Value::Object(mut args) = serde_json::to_value(value)? else {
            return Err(serde_json::Error::custom("an element's args are an object"));
        };
        if !name.is_empty() {
            args.insert(
                crate::authoring::world::ID_KEY.to_string(),
                serde_json::Value::String(name.to_string()),
            );
        }
        Ok(Self {
            asset_type: T::TYPE.to_string(),
            args,
        })
    }

    /// The element's asset type, as a world line names it.
    pub fn asset_type(&self) -> &str {
        &self.asset_type
    }
}

impl serde::Serialize for Element {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(Some(1))?;
        map.serialize_entry(&self.asset_type, &self.args)?;
        map.end()
    }
}

impl<'de> serde::Deserialize<'de> for Element {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let map = serde_json::Map::<String, serde_json::Value>::deserialize(d)?;
        let mut entries = map.into_iter();
        let (Some((asset_type, args)), None) = (entries.next(), entries.next()) else {
            return Err(D::Error::custom(
                "an element is an object with its asset type as its only key",
            ));
        };
        let args = match args {
            serde_json::Value::Object(args) => args,
            serde_json::Value::Null => serde_json::Map::new(),
            _ => return Err(D::Error::custom("an element's args are an object")),
        };
        Ok(Self { asset_type, args })
    }
}

// The children are checked as the assets they become once the cook lifts them
// out, so the screen's own table describes none of their fields.
impl concinnity_engine::ecs::AssetFields for Element {
    fn collect_fields(_prefix: &str, _out: &mut FieldTable) {}
}

#[cfg(feature = "schema")]
impl concinnity_engine::ecs::schema::Described for Element {
    const TYPE: concinnity_engine::ecs::schema::FieldType =
        concinnity_engine::ecs::schema::FieldType::Object;
}

// Lift every Screen's children out into world entries of their own, each
// naming the screen, placed right after it. A generated or declared child name
// that lands on an asset already in the world is a hard error.
pub(crate) fn expand_screens(assets: &mut Vec<serde_json::Value>) -> Result<(), Diagnostic> {
    let holds_children = |v: &serde_json::Value| {
        registered_type(v) == Some(RegisteredType::Screen)
            && v.get("args").is_some_and(|a| a.get("children").is_some())
    };
    if !assets.iter().any(holds_children) {
        return Ok(());
    }

    let mut taken: HashSet<String> = assets
        .iter()
        .map(asset_name)
        .filter(|n| !n.is_empty())
        .collect();
    let mut result = Vec::with_capacity(assets.len());
    for mut value in assets.drain(..) {
        if !holds_children(&value) {
            result.push(value);
            continue;
        }
        let screen = asset_name(&value);
        if screen.is_empty() {
            return Err(Diagnostic::world(
                "Screen: children need the screen's `$id`",
            ));
        }
        let children = value
            .get_mut("args")
            .and_then(serde_json::Value::as_object_mut)
            .and_then(|args| args.remove("children"))
            .unwrap_or_default();
        let children: Vec<Element> = serde_json::from_value(children).map_err(|e| {
            Diagnostic::asset("Screen", &screen, format!("invalid args: `children`: {e}"))
        })?;
        result.push(value);
        for (i, child) in children.into_iter().enumerate() {
            result.push(lift(&screen, i, child, &mut taken)?);
        }
    }
    *assets = result;
    Ok(())
}

// The world entry child `i` of `screen` becomes: named by its `$id` or
// `<screen>_<i>`, and on the screen unless it names its own. A child of a type
// a screen cannot hold, or with a `$id` that is not a string, is refused.
fn child_entry(screen: &str, i: usize, child: Element) -> Result<serde_json::Value, String> {
    if !RegisteredType::parse(&child.asset_type).is_some_and(|ty| SCREEN_CHILDREN.contains(&ty)) {
        let held: Vec<&str> = SCREEN_CHILDREN.iter().map(|ty| ty.as_str()).collect();
        return Err(format!(
            "child {i} is a `{}`; a screen holds {}",
            child.asset_type,
            held.join(", ")
        ));
    }
    let mut args = child.args;
    let id_key = crate::authoring::world::ID_KEY;
    match args.get(id_key) {
        Some(serde_json::Value::String(_)) => {}
        Some(_) => return Err(format!("child {i}'s `{id_key}` is not a string")),
        None => {
            args.insert(
                id_key.to_string(),
                serde_json::Value::String(format!("{screen}_{i}")),
            );
        }
    }
    let mut entry = serde_json::json!({
        "type": child.asset_type,
        "args": serde_json::Value::Object(args),
    });
    set_arg_if_absent(&mut entry, "screen", screen);
    Ok(entry)
}

// The world entry child `i` of `screen` becomes, its name claimed in `taken`.
fn lift(
    screen: &str,
    i: usize,
    child: Element,
    taken: &mut HashSet<String>,
) -> Result<serde_json::Value, Diagnostic> {
    let fail = |message: String| Diagnostic::asset("Screen", screen, message);
    let entry = child_entry(screen, i, child).map_err(fail)?;
    let name = asset_name(&entry);
    if !taken.insert(name.clone()) {
        return Err(fail(format!(
            "child name '{name}' collides with an existing asset; \
             rename the child or the conflicting asset"
        )));
    }
    Ok(entry)
}

/// The assets a [Screen](#screen)'s `children` become once the cook lifts them
/// out, each named and on the screen, for a reader of the unexpanded world. Any
/// other asset holds none. A child the cook would refuse is left out; the build
/// reports it.
pub fn screen_children(screen: &WorldJsonlAsset) -> Vec<WorldJsonlAsset> {
    if screen.asset_type != RegisteredType::Screen {
        return Vec::new();
    }
    let (name, args) = (screen.id.as_str(), &screen.args);
    if name.is_empty() {
        return Vec::new();
    }
    let Some(children) = args.get("children") else {
        return Vec::new();
    };
    let Ok(children) = serde_json::from_value::<Vec<Element>>(children.clone()) else {
        return Vec::new();
    };
    children
        .into_iter()
        .enumerate()
        .filter_map(|(i, child)| child_entry(name, i, child).ok())
        .filter_map(|entry| WorldJsonlAsset::from_value(&entry).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::TextLabel;

    fn entry(v: &serde_json::Value) -> (&str, &str) {
        (
            v["type"].as_str().unwrap_or_default(),
            v["args"]["$id"].as_str().unwrap_or_default(),
        )
    }

    #[test]
    fn children_become_entries_of_the_screen() {
        let mut assets = vec![
            serde_json::json!({"type": "Window", "args": {"$id": "win"}}),
            serde_json::json!({"type": "Screen", "args": {"$id": "pause", "initial": true, "children": [
                {"Sprite": {"$id": "dim", "width": 1280.0}},
                {"TextLabel": {"content": "Paused"}},
                {"HitRegion": {"screen": "other"}},
            ]}}),
        ];
        expand_screens(&mut assets).unwrap();

        let entries: Vec<_> = assets.iter().map(entry).collect();
        assert_eq!(
            entries,
            vec![
                ("Window", "win"),
                ("Screen", "pause"),
                ("Sprite", "dim"),
                ("TextLabel", "pause_1"),
                ("HitRegion", "pause_2"),
            ]
        );
        assert!(assets[1]["args"].get("children").is_none());
        assert_eq!(assets[1]["args"]["initial"], true);
        assert_eq!(assets[2]["args"]["screen"], "pause");
        assert_eq!(assets[2]["args"]["width"], 1280.0);
        assert_eq!(assets[3]["args"]["content"], "Paused");
        assert_eq!(
            assets[4]["args"]["screen"], "other",
            "an authored screen stays"
        );
    }

    #[test]
    fn a_screen_without_children_is_untouched() {
        let screen = serde_json::json!({"type": "Screen", "args": {"$id": "hud"}});
        let mut assets = vec![screen.clone()];
        expand_screens(&mut assets).unwrap();
        assert_eq!(assets, vec![screen]);
    }

    #[test]
    fn a_child_of_another_kind_is_rejected() {
        let mut assets = vec![
            serde_json::json!({"type": "Screen", "args": {"$id": "s", "children": [
                {"Prop": {}},
            ]}}),
        ];
        let err = expand_screens(&mut assets).unwrap_err().to_string();
        assert!(err.contains("Screen 's'"), "{err}");
        assert!(err.contains("child 0 is a `Prop`"), "{err}");
    }

    #[test]
    fn a_child_name_colliding_with_the_world_is_rejected() {
        let mut assets = vec![
            serde_json::json!({"type": "Sprite", "args": {"$id": "s_0"}}),
            serde_json::json!({"type": "Screen", "args": {"$id": "s", "children": [{"Sprite": {}}]}}),
        ];
        let err = expand_screens(&mut assets).unwrap_err().to_string();
        assert!(err.contains("'s_0' collides"), "{err}");
    }

    #[test]
    fn a_malformed_child_names_the_screen() {
        for children in [
            serde_json::json!([{"Sprite": {}, "TextLabel": {}}]),
            serde_json::json!([{"Sprite": 3}]),
            serde_json::json!("Sprite"),
        ] {
            let mut assets = vec![serde_json::json!({
                "type": "Screen", "args": {"$id": "s", "children": children}
            })];
            let err = expand_screens(&mut assets).unwrap_err().to_string();
            assert!(
                err.contains("Screen 's': invalid args: `children`"),
                "{err}"
            );
        }
    }

    #[test]
    fn an_element_round_trips_as_a_one_entry_object() {
        let label = TextLabel {
            content: "Paused".into(),
            ..Default::default()
        };
        let element = Element::new("title", &label).unwrap();
        assert_eq!(element.asset_type(), "TextLabel");
        let json = serde_json::to_value(&element).unwrap();
        assert_eq!(json["TextLabel"]["$id"], "title");
        assert_eq!(json["TextLabel"]["content"], "Paused");
        let back: Element = serde_json::from_value(json).unwrap();
        assert_eq!(back, element);
    }

    #[test]
    fn the_runtime_screen_keeps_every_policy() {
        let authored: Screen = serde_json::from_value(serde_json::json!({
            "initial": true, "fade_in_secs": 0.5, "toggle_key": "Escape",
            "input": "passthrough", "space": "window", "pauses_world": false, "layer": -2,
        }))
        .unwrap();
        let screen = authored.runtime();
        assert!(screen.initial);
        assert_eq!(screen.fade_in_secs, 0.5);
        assert_eq!(screen.toggle_key, "Escape");
        assert_eq!(screen.input, ScreenInput::Passthrough);
        assert_eq!(screen.space, ScreenSpace::Window);
        assert!(!screen.pauses_world);
        assert_eq!(screen.layer, -2);
    }

    #[test]
    fn a_default_screen_pauses_the_world() {
        assert!(Screen::default().pauses_world);
        assert!(Screen::default().runtime().pauses_world);
    }

    #[test]
    fn a_non_string_child_id_is_rejected() {
        let mut assets = vec![
            serde_json::json!({"type": "Screen", "args": {"$id": "s", "children": [
                {"Sprite": {"$id": 4}},
            ]}}),
        ];
        let err = expand_screens(&mut assets).unwrap_err().to_string();
        assert!(err.contains("Screen 's'"), "{err}");
        assert!(err.contains("child 0's `$id` is not a string"), "{err}");
    }

    #[test]
    fn screen_children_reads_what_the_cook_lifts_out() {
        let args = serde_json::json!({"children": [
            {"HitRegion": {"action": "quit"}},
            {"Prop": {}},
            {"KeyBinding": {"$id": "esc", "screen": "other"}},
        ]});
        let screen = WorldJsonlAsset {
            id: "pause".to_string(),
            asset_type: RegisteredType::Screen,
            args,
        };
        let children = screen_children(&screen);
        let read: Vec<_> = children
            .iter()
            .map(|c| (c.asset_type, c.id.as_str(), c.args["screen"].as_str()))
            .collect();
        assert_eq!(
            read,
            vec![
                (RegisteredType::HitRegion, "pause_0", Some("pause")),
                (RegisteredType::KeyBinding, "esc", Some("other")),
            ]
        );
        let not_a_screen = WorldJsonlAsset {
            asset_type: RegisteredType::Panel,
            ..screen
        };
        assert!(screen_children(&not_a_screen).is_empty());
    }

    #[test]
    fn a_nameless_screen_cannot_hold_children() {
        let mut assets = vec![serde_json::json!({"type": "Screen", "args": {"children": [
            {"Sprite": {}},
        ]}})];
        let err = expand_screens(&mut assets).unwrap_err().to_string();
        assert!(
            err.contains("Screen: children need the screen's `$id`"),
            "{err}"
        );
    }
}
