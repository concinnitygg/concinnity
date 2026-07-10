// src/assets/application.rs

use crate::assets::Application;
use crate::ecs::{AssetOrigin, Component};

impl Component for Application {
    const NAME: &'static str = "Application";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_args_default_to_the_engine_name() {
        let a: Application = serde_json::from_str("{}").unwrap();
        assert_eq!(a.name, "Concinnity");
        assert_eq!(a.version, "0.1.0");
        assert!(a.id.is_empty() && a.author.is_empty() && a.icon.is_empty());
    }

    #[test]
    fn fields_round_trip_through_args() {
        let a: Application = serde_json::from_str(
            r#"{"name":"My Game","id":"gg.studio.mygame","version":"1.2.3","author":"Studio","icon":"art/icon.png"}"#,
        )
        .unwrap();
        assert_eq!(a.name, "My Game");
        assert_eq!(a.id, "gg.studio.mygame");
        assert_eq!(a.version, "1.2.3");
        assert_eq!(a.author, "Studio");
        assert_eq!(a.icon, "art/icon.png");
    }
}
