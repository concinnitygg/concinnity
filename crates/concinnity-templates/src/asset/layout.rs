// UI structure builders: a Panel container and a View layer.

use crate::spec::AssetSpec;
use alloc::string::String;

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

// A View (a UI layer), shown at start when `initial` is set.
pub fn view(name: impl Into<String>, initial: bool) -> AssetSpec {
    AssetSpec::new(name, "View").set("initial", initial)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ArgValue;
    use alloc::string::ToString;

    #[test]
    fn panel_sets_rect_and_title() {
        let p = panel("assets", [960.0, 122.0, 320.0, 448.0], "Assets");
        assert_eq!(p.asset_type, "Panel");
        let field = |k: &str| p.fields.iter().find(|(key, _)| key == k).map(|(_, v)| v);
        assert_eq!(field("width"), Some(&ArgValue::Float(320.0)));
        assert_eq!(field("title"), Some(&ArgValue::Str("Assets".to_string())));
    }

    #[test]
    fn view_sets_initial() {
        let v = view("menu_root", true);
        assert_eq!(v.asset_type, "View");
        assert_eq!(
            v.fields
                .iter()
                .find(|(k, _)| k == "initial")
                .map(|(_, x)| x),
            Some(&ArgValue::Bool(true))
        );
    }
}
