// src/assets/material_palette.rs
//
// Runtime metadata for the MaterialPalette asset. The authored schema
// (MaterialPalette, PaletteEntry) lives in concinnity-asset; this file keeps
// only the `Component` impl. MaterialPalette is BuildOnly: cook expands it into
// concrete Material assets and it never reaches the runtime, so the impl is a
// trivial pass-through.

use crate::assets::MaterialPalette;
use crate::ecs::{AssetOrigin, Component};

impl Component for MaterialPalette {
    const NAME: &'static str = "MaterialPalette";
    const ORIGIN: AssetOrigin = AssetOrigin::BuildOnly;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}
