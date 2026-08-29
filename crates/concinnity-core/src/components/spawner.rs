// src/components/spawner.rs
//
// The `Spawner` asset: the authored args a world declares, and the runtime
// component (with its spawn accumulator) they bake into.

use crate::ecs::Component;
use crate::ecs::asset_id::AssetId;

/// Authored fields of a `Spawner`; the runtime accumulator is not declared.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SpawnerArgs {
    /// Name of the placement to copy on each spawn.
    pub template: AssetId,
    /// Seconds between spawns.
    pub interval: f32,
    /// Seconds each spawned copy lives before auto-removal; 0 keeps it forever.
    pub lifetime: f32,
}

impl Default for SpawnerArgs {
    fn default() -> Self {
        Self {
            template: AssetId::default(),
            interval: 1.0,
            lifetime: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_spawner_ticks_once_a_second_and_never_expires_its_copies() {
        let s = SpawnerArgs::default();
        assert_eq!(s.interval, 1.0);
        // Zero lifetime means the copy lives until something despawns it.
        assert_eq!(s.lifetime, 0.0);
        assert_eq!(s.template, AssetId::default());
    }

    #[test]
    fn an_authored_spawner_parses_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let s: SpawnerArgs =
            serde_json::from_str(r#"{"template":"spark","interval":0.25,"lifetime":3}"#).unwrap();
        assert_eq!(s.template, AssetId(5));
        assert_eq!(s.interval, 0.25);

        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: SpawnerArgs = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.template, AssetId(5));
        assert_eq!(back.lifetime, 3.0);
    }
}

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
            template: args.template,
            interval: args.interval.max(0.0),
            lifetime: args.lifetime.max(0.0),
            elapsed: 0.0,
            count: 0,
        }
    }
}

impl Component for Spawner {
    const NAME: &'static str = "Spawner";

    fn from_baked(bytes: &[u8]) -> Result<Self, crate::result::CnResult> {
        Ok(crate::blob::decode_exact(bytes)?)
    }
}
