// The typed expansion every UI shorthand shares, and the cook pass that drives
// it over a world's entries.

use std::collections::{HashMap, HashSet};

use concinnity_engine::components::Font;

use crate::authoring::registry::{Authored, RegisteredType};
use crate::authoring::spec::{AssetSpec, spec_to_value};
use crate::build_only::expand::{asset_name, registered_type, schema_args};
use crate::world_builder::Diagnostic;

/// What a UI shorthand's expansion reads from the world around it.
///
/// Rows are laid out in the pixels of the font they draw with, so an expansion
/// that names a declared `Font` needs that font's size.
#[derive(Debug, Clone, Default)]
pub struct ExpandContext {
    font_px: HashMap<String, f32>,
}

impl ExpandContext {
    /// A context declaring `name` as a font of `size_px` pixels.
    #[must_use]
    pub fn with_font(mut self, name: impl Into<String>, size_px: f32) -> Self {
        self.font_px.insert(name.into(), size_px);
        self
    }

    /// The pixel size text drawn with `font` lays out at: the declared font's
    /// size, or `fallback` for an empty or undeclared name.
    pub fn font_px(&self, font: &str, fallback: f32) -> f32 {
        if font.is_empty() {
            return fallback;
        }
        self.font_px.get(font).copied().unwrap_or(fallback)
    }

    // The fonts a world's entries declare, each at its `size_px` or the type
    // default. A nameless Font cannot be referenced, so it is skipped.
    pub(crate) fn from_world(assets: &[serde_json::Value]) -> Self {
        let mut context = Self::default();
        for v in assets {
            if registered_type(v) != Some(RegisteredType::Font) {
                continue;
            }
            let name = asset_name(v);
            if name.is_empty() {
                continue;
            }
            let px = v
                .get("args")
                .and_then(|a| a.get("size_px"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_else(|| u64::from(Font::default().size_px)) as f32;
            context.font_px.insert(name, px);
        }
        context
    }
}

/// A UI shorthand: an authored asset that stands for the overlay elements it
/// expands into.
///
/// The cook replaces each one a world declares with the result of
/// [`expand`](Expand::expand), and a caller holding the typed value gets the
/// same entries by calling it directly:
///
/// ```rust
/// # use concinnity_cook::ui::{Expand, ExpandContext, Panel};
/// let card = Panel {
///     title: "Paused".into(),
///     ..Default::default()
/// };
/// let entries = card.expand("card", &ExpandContext::default()).unwrap();
/// assert_eq!(entries[0].name, "card_bg");
/// ```
pub trait Expand: Authored + serde::de::DeserializeOwned {
    /// The entries the shorthand named `name` stands for, each named after it
    /// so none collides with a hand-authored asset.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic naming the asset when its args cannot be expanded,
    /// such as a settings row naming no known setting.
    fn expand(&self, name: &str, context: &ExpandContext) -> Result<Vec<AssetSpec>, Diagnostic>;
}

// Replace every `T` entry with the entries it expands into. A generated name
// that lands on an asset already in the world is a hard error.
pub(crate) fn expand_all<T: Expand>(assets: &mut Vec<serde_json::Value>) -> Result<(), Diagnostic> {
    let Some(ty) = RegisteredType::parse(T::TYPE) else {
        return Ok(());
    };
    if !assets.iter().any(|v| registered_type(v) == Some(ty)) {
        return Ok(());
    }

    let context = ExpandContext::from_world(assets);
    let mut taken: HashSet<String> = assets
        .iter()
        .filter(|v| registered_type(v) != Some(ty))
        .map(asset_name)
        .filter(|n| !n.is_empty())
        .collect();

    let mut result = Vec::with_capacity(assets.len());
    for value in assets.drain(..) {
        if registered_type(&value) != Some(ty) {
            result.push(value);
            continue;
        }
        let name = asset_name(&value);
        if name.is_empty() {
            return Err(Diagnostic::world(format!("{}: missing `name`", T::TYPE)));
        }
        let shorthand: T = schema_args(ty, &name, value.get("args"))?;
        for spec in shorthand.expand(&name, &context)? {
            if !spec.name.is_empty() && !taken.insert(spec.name.clone()) {
                return Err(Diagnostic::asset(
                    T::TYPE,
                    &name,
                    format!(
                        "generated asset name '{}' collides with an existing \
                         asset; rename the {} or the conflicting asset",
                        spec.name,
                        T::TYPE
                    ),
                ));
            }
            result.push(spec_to_value(&spec));
        }
    }

    *assets = result;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::Panel;

    #[test]
    fn font_px_reads_declared_sizes_and_falls_back() {
        let assets = vec![
            serde_json::json!({"type":"Font","args":{"$id":"a","size_px":32}}),
            serde_json::json!({"type":"Font","args":{"$id":"b"}}),
        ];
        let context = ExpandContext::from_world(&assets);
        assert_eq!(context.font_px("a", 1.0), 32.0);
        assert_eq!(context.font_px("b", 1.0), Font::default().size_px as f32);
        assert_eq!(context.font_px("missing", 7.0), 7.0);
        assert_eq!(context.font_px("", 7.0), 7.0);
    }

    // A nameless Font cannot be referenced by an expander, so it is skipped
    // rather than mapped under the empty name.
    #[test]
    fn nameless_fonts_and_other_types_declare_nothing() {
        let assets = vec![
            serde_json::json!({"type":"Font","args":{"size_px":32}}),
            serde_json::json!({"type":"Sprite","args":{"$id":"sprite","size_px":32}}),
        ];
        assert!(ExpandContext::from_world(&assets).font_px.is_empty());
    }

    #[test]
    fn with_font_declares_a_font() {
        let context = ExpandContext::default().with_font("menu", 40.0);
        assert_eq!(context.font_px("menu", 1.0), 40.0);
    }

    #[test]
    fn a_world_without_the_type_is_untouched() {
        let mut assets = vec![serde_json::json!({"type":"Window","args":{"$id":"x"}})];
        expand_all::<Panel>(&mut assets).unwrap();
        assert_eq!(assets.len(), 1);
    }

    #[test]
    fn a_nameless_entry_is_rejected() {
        let mut assets = vec![serde_json::json!({"type":"Panel","args":{}})];
        let err = expand_all::<Panel>(&mut assets).unwrap_err().to_string();
        assert!(err.contains("Panel: missing `name`"), "{err}");
    }

    #[test]
    fn a_generated_name_colliding_with_the_world_is_rejected() {
        let mut assets = vec![
            serde_json::json!({"type":"Sprite","args":{"$id":"card_bg"}}),
            serde_json::json!({"type":"Panel","args":{"$id":"card"}}),
        ];
        let err = expand_all::<Panel>(&mut assets).unwrap_err().to_string();
        assert!(err.contains("'card_bg' collides"), "{err}");
    }

    #[test]
    fn the_pass_writes_what_the_typed_expansion_returns() {
        let mut assets = vec![serde_json::json!({
            "type": "Panel", "args": {"$id": "card", "title": "Stats"}
        })];
        expand_all::<Panel>(&mut assets).unwrap();
        let panel = Panel {
            title: "Stats".into(),
            ..Default::default()
        };
        let typed: Vec<serde_json::Value> = panel
            .expand("card", &ExpandContext::default())
            .unwrap()
            .iter()
            .map(spec_to_value)
            .collect();
        assert_eq!(assets, typed);
    }
}
