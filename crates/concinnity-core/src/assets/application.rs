// src/assets/application.rs
//
// Runtime `Application` component. Its authored args live in the schema crate
// (concinnity_asset::application); only the runtime resource budgets survive
// the bake. The distribution metadata (name, id, version, author, icon) is
// read at build / export time from the authored world, so it never ships in
// the blob.

use crate::assets::AppLimits;
use crate::ecs::Component;
use concinnity_asset::ApplicationArgs;

/// Runtime half of the Application asset: the process resource budgets.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct Application {
    /// Optional overrides for the runtime's thread and memory budgets.
    pub limits: AppLimits,
}

impl Application {
    /// Translate the authored args into the runtime component: keep only the
    /// resource budgets. Run by cook at build time (the baked blob record
    /// carries the result).
    pub fn bake(args: ApplicationArgs) -> Self {
        Self {
            limits: args.limits,
        }
    }
}

impl Component for Application {
    const NAME: &'static str = "Application";

    fn from_baked(bytes: &[u8]) -> Result<Self, crate::result::CnResult> {
        Ok(postcard::from_bytes(bytes)?)
    }
}
