// The authored Screen: the runtime screen's policies plus the elements it
// holds, which the cook lifts out into world entries that name the screen.

use std::collections::HashSet;

use concinnity_engine::components::{
    LayoutDirection, LayoutSize, ScreenInput, ScreenSpace, TextField,
};
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
/// [`TextLabel`](#textlabel), [`TextField`](#textfield),
/// [`HitRegion`](#hitregion), [`ScrollPanel`](#scrollpanel),
/// [`KeyBinding`](#keybinding), a widget ([List](#list), [Tabs](#tabs),
/// [Checkbox](#checkbox), [Button](#button), [`TextArea`](#textarea), [Menu](#menu),
/// [Modal](#modal), or [Toasts](#toasts)), [Panel](#panel),
/// [Slider](#slider), or [`OptionSelect`](#optionselect) can be a child. Each child becomes an asset
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
/// anywhere. `focus` names a [`TextField`](#textfield) that receives keyboard
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
    /// [`TextField`](#textfield) that receives keyboard focus whenever this
    /// screen reaches the top of the stack.
    pub focus: Option<Ref<TextField>>,
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
            // An authored screen is the world's own, never over it.
            over_world: false,
        }
    }
}

/// One element of a [Screen](#screen): an asset written as a one-entry object
/// naming its type, `{"TextLabel": {"content": "Paused"}}`.
///
/// A `Row`, `Column`, or `Stack` element is a [Layout](#layout) in that
/// direction, its own `children` written the same way. A child of one may size
/// itself with `width` and `height` beside its type, each `"fit"`,
/// `{"fixed": px}`, or `{"fill": weight}`:
/// `{"Sprite": {"tint": [0, 0, 0, 1]}, "width": {"fill": 1}}`.
#[derive(Debug, Clone, PartialEq)]
pub struct Element {
    asset_type: String,
    args: serde_json::Map<String, serde_json::Value>,
    width: Option<LayoutSize>,
    height: Option<LayoutSize>,
}

impl Element {
    /// An element declaring `value`, named `name`; an empty name takes the
    /// screen's generated one.
    ///
    /// A field that references another asset holds a resolved handle rather
    /// than a name, so a value with one is better written in a .jsonl file or
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
            width: None,
            height: None,
        })
    }

    /// A `Row`, `Column`, or `Stack` holding `children`, named `name`.
    pub fn layout(name: &str, direction: LayoutDirection, children: Vec<Self>) -> Self {
        let mut args = serde_json::Map::new();
        if !name.is_empty() {
            args.insert(
                crate::authoring::world::ID_KEY.to_string(),
                serde_json::Value::String(name.to_string()),
            );
        }
        args.insert(
            "children".to_string(),
            serde_json::to_value(children).unwrap_or_default(),
        );
        let asset_type = match direction {
            LayoutDirection::Row => "Row",
            LayoutDirection::Column => "Column",
            LayoutDirection::Stack => "Stack",
        };
        Self {
            asset_type: asset_type.to_string(),
            args,
            width: None,
            height: None,
        }
    }

    /// This element sized as `width` by `height` inside the layout holding it.
    #[must_use]
    pub fn sized(mut self, width: LayoutSize, height: LayoutSize) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// The element's asset type, as a world line names it.
    pub fn asset_type(&self) -> &str {
        &self.asset_type
    }
}

const SIZE_KEYS: [&str; 2] = ["width", "height"];

impl serde::Serialize for Element {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let sized = [self.width, self.height];
        let mut map = s.serialize_map(Some(1 + sized.iter().flatten().count()))?;
        map.serialize_entry(&self.asset_type, &self.args)?;
        for (key, size) in SIZE_KEYS.iter().zip(sized) {
            if let Some(size) = size {
                map.serialize_entry(key, &size)?;
            }
        }
        map.end()
    }
}

impl<'de> serde::Deserialize<'de> for Element {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let mut map = serde_json::Map::<String, serde_json::Value>::deserialize(d)?;
        let mut size = |key: &str| {
            map.remove(key)
                .map(serde_json::from_value::<LayoutSize>)
                .transpose()
                .map_err(|e| D::Error::custom(format!("`{key}`: {e}")))
        };
        let (width, height) = (size("width")?, size("height")?);
        let mut entries = map.into_iter();
        let (Some((asset_type, args)), None) = (entries.next(), entries.next()) else {
            return Err(D::Error::custom(
                "an element is an object with its asset type as its only key, \
                 beside an optional `width` and `height`",
            ));
        };
        let args = match args {
            serde_json::Value::Object(args) => args,
            serde_json::Value::Null => serde_json::Map::new(),
            _ => return Err(D::Error::custom("an element's args are an object")),
        };
        Ok(Self {
            asset_type,
            args,
            width,
            height,
        })
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
        let fail = |message: String| Diagnostic::asset("Screen", &screen, message);
        let children = value
            .get_mut("args")
            .and_then(serde_json::Value::as_object_mut)
            .and_then(|args| args.remove("children"))
            .unwrap_or_default();
        let children: Vec<Element> = serde_json::from_value(children)
            .map_err(|e| fail(format!("invalid args: `children`: {e}")))?;
        result.push(value);
        let mut lifted = Vec::new();
        for (i, child) in children.into_iter().enumerate() {
            lift(&screen, &screen, i, child, Holder::Screen, &mut lifted).map_err(fail)?;
        }
        for entry in lifted {
            let name = asset_name(&entry);
            if !taken.insert(name.clone()) {
                return Err(fail(format!(
                    "child name '{name}' collides with an existing asset; \
                     rename the child or the conflicting asset"
                )));
            }
            result.push(entry);
        }
    }
    *assets = result;
    Ok(())
}

// What holds a child: the screen itself, or a layout on it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Holder {
    Screen,
    Layout,
}

// The asset types a layout can place: the single elements, the widgets that
// take a rect, and layouts.
const LAYOUT_ITEMS: [RegisteredType; 11] = [
    RegisteredType::Sprite,
    RegisteredType::TextLabel,
    RegisteredType::TextField,
    RegisteredType::List,
    RegisteredType::Grid,
    RegisteredType::ValueSlider,
    RegisteredType::Tabs,
    RegisteredType::Checkbox,
    RegisteredType::Button,
    RegisteredType::TextArea,
    RegisteredType::Layout,
];

// The direction an authored `Row`, `Column`, or `Stack` element lays out in.
fn direction(asset_type: &str) -> Option<LayoutDirection> {
    match asset_type {
        "Row" => Some(LayoutDirection::Row),
        "Column" => Some(LayoutDirection::Column),
        "Stack" => Some(LayoutDirection::Stack),
        _ => None,
    }
}

// Lift child `i` of a `holder` named `prefix` on `screen`, appending the
// entries it becomes to `out`: named by its `$id` or `<prefix>_<i>`, and on the
// screen unless it names its own. A layout's own children follow it. Returns
// the child's name. A child the holder cannot hold, a `$id` that is not a
// string, or sizing outside a layout, is refused.
fn lift(
    screen: &str,
    prefix: &str,
    i: usize,
    child: Element,
    holder: Holder,
    out: &mut Vec<serde_json::Value>,
) -> Result<String, String> {
    let layout = direction(&child.asset_type);
    let ty = RegisteredType::parse(&child.asset_type);
    let held = match holder {
        Holder::Screen => layout.is_some() || ty.is_some_and(|t| SCREEN_CHILDREN.contains(&t)),
        Holder::Layout => layout.is_some() || ty.is_some_and(|t| LAYOUT_ITEMS.contains(&t)),
    };
    if !held {
        let what = match holder {
            Holder::Screen => "a screen holds",
            Holder::Layout => "a layout holds",
        };
        let types: Vec<&str> = match holder {
            Holder::Screen => SCREEN_CHILDREN.iter().map(|t| t.as_str()).collect(),
            Holder::Layout => LAYOUT_ITEMS.iter().map(|t| t.as_str()).collect(),
        };
        return Err(format!(
            "{prefix}'s child {i} is a `{}`; {what} Row, Column, Stack, {}",
            child.asset_type,
            types.join(", ")
        ));
    }
    if holder == Holder::Screen && (child.width.is_some() || child.height.is_some()) {
        return Err(format!(
            "{prefix}'s child {i} is sized, which only a Row, Column, or Stack child can be"
        ));
    }
    let mut args = child.args;
    let id_key = crate::authoring::world::ID_KEY;
    let name = match args.get(id_key) {
        Some(serde_json::Value::String(name)) => name.clone(),
        Some(_) => return Err(format!("{prefix}'s child {i}'s `{id_key}` is not a string")),
        None => {
            let name = format!("{prefix}_{i}");
            args.insert(id_key.to_string(), serde_json::Value::String(name.clone()));
            name
        }
    };
    let mut nested = Vec::new();
    if let Some(direction) = layout {
        let children = args.remove("children").unwrap_or_default();
        let children: Vec<Element> = serde_json::from_value(children)
            .map_err(|e| format!("{name}: invalid `children`: {e}"))?;
        let mut items = Vec::with_capacity(children.len());
        for (j, grandchild) in children.into_iter().enumerate() {
            let (width, height) = (grandchild.width, grandchild.height);
            let item = lift(screen, &name, j, grandchild, Holder::Layout, &mut nested)?;
            items.push(serde_json::json!({
                "item": item,
                "width": width.unwrap_or_default(),
                "height": height.unwrap_or_default(),
            }));
        }
        args.insert(
            "direction".to_string(),
            serde_json::to_value(direction).unwrap_or_default(),
        );
        args.insert("children".to_string(), serde_json::Value::Array(items));
    }
    let asset_type = if layout.is_some() {
        RegisteredType::Layout.as_str().to_string()
    } else {
        child.asset_type
    };
    let mut entry = serde_json::json!({
        "type": asset_type,
        "args": serde_json::Value::Object(args),
    });
    set_arg_if_absent(&mut entry, "screen", screen);
    out.push(entry);
    out.append(&mut nested);
    Ok(name)
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
    let mut out = Vec::new();
    for (i, child) in children.into_iter().enumerate() {
        let mut lifted = Vec::new();
        if lift(name, name, i, child, Holder::Screen, &mut lifted).is_ok() {
            out.append(&mut lifted);
        }
    }
    out.iter()
        .filter_map(|entry| WorldJsonlAsset::from_value(entry).ok())
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

    #[test]
    fn a_row_places_widgets_that_take_a_rect() {
        let mut assets = vec![
            serde_json::json!({"type": "Screen", "args": {"$id": "ed", "children": [
                {"Row": {"$id": "bar", "children": [
                    {"List": {"items": "files"}, "width": {"fixed": 200}},
                    {"Column": {"children": [
                        {"Tabs": {}},
                        {"TextArea": {}, "height": {"fill": 1}},
                        {"Checkbox": {"label": "Grid"}},
                    ]}},
                ]}},
            ]}}),
        ];
        expand_screens(&mut assets).unwrap();
        let entries: Vec<_> = assets.iter().map(entry).collect();
        assert_eq!(
            entries,
            vec![
                ("Screen", "ed"),
                ("Layout", "bar"),
                ("List", "bar_0"),
                ("Layout", "bar_1"),
                ("Tabs", "bar_1_0"),
                ("TextArea", "bar_1_1"),
                ("Checkbox", "bar_1_2"),
            ]
        );
    }

    #[test]
    fn a_column_lifts_into_a_layout_holding_its_children() {
        let mut assets = vec![
            serde_json::json!({"type": "Screen", "args": {"$id": "hud", "children": [
                {"Column": {"$id": "stats", "gap": 4, "anchor": "top-right", "children": [
                    {"TextLabel": {"content": "fps"}},
                    {"Row": {"children": [
                        {"Sprite": {}, "width": {"fill": 1}, "height": {"fixed": 2}},
                    ]}},
                ]}},
            ]}}),
        ];
        expand_screens(&mut assets).unwrap();
        let entries: Vec<_> = assets.iter().map(entry).collect();
        assert_eq!(
            entries,
            vec![
                ("Screen", "hud"),
                ("Layout", "stats"),
                ("TextLabel", "stats_0"),
                ("Layout", "stats_1"),
                ("Sprite", "stats_1_0"),
            ]
        );
        let column = &assets[1]["args"];
        assert_eq!(column["direction"], "column");
        assert_eq!(column["gap"], 4);
        assert_eq!(column["anchor"], "top-right");
        assert_eq!(column["screen"], "hud");
        assert_eq!(
            column["children"],
            serde_json::json!([
                {"item": "stats_0", "width": "fit", "height": "fit"},
                {"item": "stats_1", "width": "fit", "height": "fit"},
            ])
        );
        let row = &assets[3]["args"];
        assert_eq!(row["direction"], "row");
        assert_eq!(
            row["children"],
            serde_json::json!([
                {"item": "stats_1_0", "width": {"fill": 1.0}, "height": {"fixed": 2.0}},
            ])
        );
        assert!(assets.iter().skip(1).all(|e| e["args"]["screen"] == "hud"));
    }

    #[test]
    fn a_layout_places_single_elements_and_layouts_only() {
        let mut assets = vec![
            serde_json::json!({"type": "Screen", "args": {"$id": "s", "children": [
                {"Row": {"children": [{"HitRegion": {}}]}},
            ]}}),
        ];
        let err = expand_screens(&mut assets).unwrap_err().to_string();
        assert!(err.contains("s_0's child 0 is a `HitRegion`"), "{err}");
    }

    #[test]
    fn sizing_belongs_to_a_layout_child() {
        let mut assets = vec![
            serde_json::json!({"type": "Screen", "args": {"$id": "s", "children": [
                {"Sprite": {}, "width": "fit"},
            ]}}),
        ];
        let err = expand_screens(&mut assets).unwrap_err().to_string();
        assert!(err.contains("only a Row, Column, or Stack child"), "{err}");
    }

    #[test]
    fn a_sized_element_round_trips_with_its_sizes_beside_its_type() {
        let label = TextLabel::default();
        let element = Element::new("t", &label)
            .unwrap()
            .sized(LayoutSize::Fill(1.0), LayoutSize::Fixed(20.0));
        let json = serde_json::to_value(&element).unwrap();
        assert_eq!(json["width"], serde_json::json!({"fill": 1.0}));
        assert_eq!(json["height"], serde_json::json!({"fixed": 20.0}));
        let back: Element = serde_json::from_value(json).unwrap();
        assert_eq!(back, element);
        let column = Element::layout("c", LayoutDirection::Column, vec![element]);
        assert_eq!(column.asset_type(), "Column");
    }
}
