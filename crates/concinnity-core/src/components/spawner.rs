//! The `Spawner` asset: the authored args a world declares, and the runtime
//! component (with its spawn accumulator) they bake into.

use crate::components::Prop;
use crate::ecs::Ref;
use crate::ecs::asset_id::AssetId;

/// Periodically instantiates copies of an existing placement at this entity's
/// position.
///
/// A spawner clones `template` (the name of another placement in the world)
/// every `interval` seconds, giving each copy a `lifetime` after which it is
/// automatically removed. Pairing a short lifetime with a short interval keeps a
/// bounded population churning (an enemy wave, a particle of debris, a fountain
/// of props) and is what exercises GPU draw-slot recycling: each expiry frees a
/// slot the next spawn reuses.
///
/// The spawner's own `Transform` (its position) is where copies appear, so place
/// the spawner where you want the stream to originate.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct SpawnerArgs {
    /// The [Prop](#prop) placement to copy on each spawn.
    #[asset(default = Ref::new(AssetId::default()))]
    pub template: Ref<Prop>,
    /// Seconds between spawns.
    #[asset(default = 1.0)]
    pub interval: f32,
    /// Seconds each spawned copy lives before auto-removal; 0 keeps it forever.
    pub lifetime: f32,
}

/// The runtime `Spawner`: the authored fields of
/// [`cook::Spawner`](crate::components::cook::Spawner) plus its spawn accumulator.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Spawner {
    /// Name of the placement to copy on each spawn.
    pub template: AssetId,
    /// Seconds between spawns.
    pub interval: f32,
    /// Seconds each spawned copy lives before auto-removal; 0 keeps it forever.
    pub lifetime: f32,
    /// Runtime: seconds accumulated toward the next spawn.
    pub elapsed: f32,
    /// Runtime: number of copies spawned so far.
    pub count: u32,
}

impl Spawner {
    /// Translate the authored args into the runtime spawner: clamp the timing
    /// knobs and zero the runtime counters. Run by cook at build time (the
    /// baked blob record carries the result).
    pub fn bake(args: SpawnerArgs) -> Self {
        Self {
            template: args.template.id(),
            interval: args.interval.max(0.0),
            lifetime: args.lifetime.max(0.0),
            elapsed: 0.0,
            count: 0,
        }
    }
}
