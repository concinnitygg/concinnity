// src/assets/prefab.rs
//
// Runtime metadata for the Prefab asset. The authored schema (Prefab,
// PrefabEntry, PrefabKind) lives in concinnity-asset; this file keeps only the
// `Component` impl. Prefab is BuildOnly: cook expands it into concrete assets
// and it never reaches the runtime, so the impl is a trivial pass-through.

use crate::assets::Prefab;
use crate::ecs::{AssetOrigin, Component};

impl Component for Prefab {
    const NAME: &'static str = "Prefab";
    const ORIGIN: AssetOrigin = AssetOrigin::BuildOnly;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}
