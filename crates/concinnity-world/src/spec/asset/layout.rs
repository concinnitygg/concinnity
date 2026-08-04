// UI structure builders: a Panel container and a Screen layer.

use crate::spec::AssetSpec;

// A Panel (a titled background box) at `rect` ([x, y, w, h], window pixels). The
// colour, corner radius, and title styling keep their defaults.
pub fn panel(name: impl Into<String>, rect: [f32; 4], title: impl Into<String>) -> AssetSpec {
    AssetSpec::new(name, "Panel")
        .set("x", rect[0])
        .set("y", rect[1])
        .set("width", rect[2])
        .set("height", rect[3])
        .set("title", title.into())
}

// A Screen (a UI layer), shown at start when `initial` is set.
pub fn screen(name: impl Into<String>, initial: bool) -> AssetSpec {
    AssetSpec::new(name, "Screen").set("initial", initial)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ArgValue;

    #[test]
    fn panel_sets_rect_and_title() {
        let p = panel("assets", [960.0, 122.0, 320.0, 448.0], "Assets");
        assert_eq!(p.asset_type, "Panel");
        let field = |k: &str| p.fields.iter().find(|(key, _)| key == k).map(|(_, v)| v);
        assert_eq!(field("width"), Some(&ArgValue::Float(320.0)));
        assert_eq!(field("title"), Some(&ArgValue::Str("Assets".to_string())));
    }

    #[test]
    fn screen_sets_initial() {
        let v = screen("menu_root", true);
        assert_eq!(v.asset_type, "Screen");
        assert_eq!(
            v.fields
                .iter()
                .find(|(k, _)| k == "initial")
                .map(|(_, x)| x),
            Some(&ArgValue::Bool(true))
        );
    }
}
