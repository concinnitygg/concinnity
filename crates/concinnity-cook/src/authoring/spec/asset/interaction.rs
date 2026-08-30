// UI interaction builder: a clickable HitRegion.

use crate::authoring::spec::AssetSpec;

/// A HitRegion over `rect` ([x, y, w, h], window pixels) that fires `action` when
/// clicked. Hover styling and references keep their defaults.
pub(crate) fn hit_region(
    name: impl Into<String>,
    rect: [f32; 4],
    action: impl Into<String>,
) -> AssetSpec {
    AssetSpec::new(name, "HitRegion")
        .set("x", rect[0])
        .set("y", rect[1])
        .set("width", rect[2])
        .set("height", rect[3])
        .set("action", action.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authoring::spec::ArgValue;

    #[test]
    fn hit_region_sets_rect_and_action() {
        let h = hit_region("resume", [10.0, 20.0, 200.0, 48.0], "menu:resume");
        assert_eq!(h.asset_type, "HitRegion");
        let field = |k: &str| h.fields.iter().find(|(key, _)| key == k).map(|(_, v)| v);
        assert_eq!(field("width"), Some(&ArgValue::Float(200.0)));
        assert_eq!(
            field("action"),
            Some(&ArgValue::Str("menu:resume".to_string()))
        );
    }
}
