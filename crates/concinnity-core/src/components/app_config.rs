// src/components/app_config.rs
//
// Runtime `AppConfig` component. Its authored args live in the schema crate
// (concinnity_asset::app_config); only what the running process needs survives
// the bake: the state-tree location and the resource budgets. The distribution
// metadata (name, id, version, author, icon) is read at build / export time
// from the authored world, so it never ships in the blob.

use alloc::string::String;

use crate::ecs::Component;
use concinnity_asset::AppConfigArgs;

/// Runtime half of the AppConfig asset: where the application keeps what it
/// writes, and its process resource budgets.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AppConfig {
    /// Where the running application writes its settings, saves, crash
    /// reports, and shader caches. Empty means beside the application's data.
    pub home: String,
    /// Soft ceiling on host memory the runtime aims to stay under, in
    /// mebibytes. `0` = auto.
    pub max_memory_mb: u32,
    /// Worker threads for the shared job pool. `0` = auto.
    pub job_threads: u32,
}

impl AppConfig {
    /// Translate the authored args into the runtime component: keep the state
    /// location and the resource budgets. Run by cook at build time (the baked
    /// blob record carries the result).
    pub fn bake(args: AppConfigArgs) -> Self {
        Self {
            home: args.home,
            max_memory_mb: args.max_memory_mb,
            job_threads: args.job_threads,
        }
    }
}

impl Component for AppConfig {
    const NAME: &'static str = "AppConfig";

    fn from_baked(bytes: &[u8]) -> Result<Self, crate::result::CnResult> {
        Ok(postcard::from_bytes(bytes)?)
    }
}
