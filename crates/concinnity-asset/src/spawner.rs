// Spawner authoring schema. The runtime `Spawner` component (with its spawn
// accumulator) lives in core.

use crate::AssetId;

/// Authored fields of a [`Spawner`]; the runtime accumulator is not declared.
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
