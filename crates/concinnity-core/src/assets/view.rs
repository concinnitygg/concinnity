// src/assets/view.rs

use crate::assets::View;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

impl Component for View {
    const NAME: &'static str = "View";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_with_defaults() {
        let v: View = serde_json::from_str("{}").unwrap();
        assert_eq!(v.fade_in_secs, 0.0);
        assert!(!v.initial);
    }

    #[test]
    fn deserializes_with_initial_true() {
        let v: View = serde_json::from_str(r#"{"initial":true}"#).unwrap();
        assert!(v.initial);
    }
}
