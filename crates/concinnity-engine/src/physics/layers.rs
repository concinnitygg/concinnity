// src/physics/layers.rs
//
// Named collision layers over the simulation's 32-bit interaction groups.
// Four built-in layers cover the engine's body kinds; a world's PhysicsConfig
// may declare up to 28 more and disable pairs symmetrically via `no_collide`.
// The table resolves names to bits once at init; everything downstream works
// in plain `LayerMask` bit pairs.

use concinnity_core::assets::PhysicsConfig;
use concinnity_physics::LayerMask;

pub(crate) const LAYER_WORLD: &str = "world";
pub(crate) const LAYER_PROP: &str = "prop";
pub(crate) const LAYER_CHARACTER: &str = "character";
pub(crate) const LAYER_TRIGGER: &str = "trigger";

const BUILTIN_LAYERS: [&str; 4] = [LAYER_WORLD, LAYER_PROP, LAYER_CHARACTER, LAYER_TRIGGER];

// Layer-name resolution for one world: bit assignment plus the symmetric
// collide matrix compiled from the config's `no_collide` pairs. Unknown names
// resolve to None; cook validation rejects them before a world ships.
#[derive(Debug)]
pub(crate) struct LayerTable {
    names: Vec<String>,
    // Row per layer: the filter bits of every layer it still collides with.
    collide: Vec<u32>,
}

impl LayerTable {
    pub(crate) fn new(config: &PhysicsConfig) -> Self {
        let mut names: Vec<String> = BUILTIN_LAYERS.iter().map(|s| s.to_string()).collect();
        for layer in &config.layers {
            if names.len() >= 32 || names.iter().any(|n| n == layer) {
                continue;
            }
            names.push(layer.clone());
        }
        let all = if names.len() == 32 {
            u32::MAX
        } else {
            (1u32 << names.len()) - 1
        };
        let mut collide = vec![all; names.len()];
        let bit_of = |name: &str| names.iter().position(|n| n == name);
        for [a, b] in &config.no_collide {
            let (Some(a), Some(b)) = (bit_of(a), bit_of(b)) else {
                continue;
            };
            collide[a] &= !(1 << b);
            collide[b] &= !(1 << a);
        }
        Self { names, collide }
    }

    // The mask a collider on `layer` carries: its own bit plus the filter row
    // the collide matrix left it. Unknown names fall back to `world`.
    pub(crate) fn mask(&self, layer: &str) -> LayerMask {
        let bit = self
            .names
            .iter()
            .position(|n| n == layer)
            .unwrap_or_default();
        LayerMask {
            memberships: 1 << bit,
            filter: self.collide[bit],
        }
    }

    // A query mask: cast as a member of `layer`, hitting only `targets`.
    pub(crate) fn query_mask(&self, layer: &str, targets: &[&str]) -> LayerMask {
        let mut filter = 0u32;
        for target in targets {
            if let Some(bit) = self.names.iter().position(|n| n == target) {
                filter |= 1 << bit;
            }
        }
        LayerMask {
            memberships: self.mask(layer).memberships,
            filter,
        }
    }
}

impl Default for LayerTable {
    fn default() -> Self {
        Self::new(&PhysicsConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(layers: &[&str], no_collide: &[[&str; 2]]) -> PhysicsConfig {
        PhysicsConfig {
            layers: layers.iter().map(|s| s.to_string()).collect(),
            no_collide: no_collide
                .iter()
                .map(|[a, b]| [a.to_string(), b.to_string()])
                .collect(),
            ..PhysicsConfig::default()
        }
    }

    #[test]
    fn default_table_collides_everything_with_everything() {
        let table = LayerTable::default();
        let world = table.mask(LAYER_WORLD);
        let prop = table.mask(LAYER_PROP);
        assert_eq!(world.memberships, 1);
        assert_eq!(prop.memberships, 2);
        // Each filter covers all four built-in layers.
        assert_eq!(world.filter, 0b1111);
        assert_eq!(prop.filter, 0b1111);
    }

    #[test]
    fn no_collide_pairs_clear_bits_symmetrically() {
        let table = LayerTable::new(&config(&["debris"], &[["debris", LAYER_CHARACTER]]));
        let debris = table.mask("debris");
        let character = table.mask(LAYER_CHARACTER);
        assert_eq!(debris.memberships, 1 << 4);
        assert_eq!(debris.filter & character.memberships, 0);
        assert_eq!(character.filter & debris.memberships, 0);
        // Unrelated layers keep colliding with both.
        let prop = table.mask(LAYER_PROP);
        assert_ne!(prop.filter & debris.memberships, 0);
        assert_ne!(prop.filter & character.memberships, 0);
    }

    #[test]
    fn unknown_and_duplicate_names_are_tolerated() {
        // Duplicates of a built-in are skipped; unknown no_collide names are
        // ignored; an unknown mask lookup falls back to the world layer.
        let table = LayerTable::new(&config(&[LAYER_PROP, "debris"], &[["ghost", "debris"]]));
        assert_eq!(table.mask("debris").memberships, 1 << 4);
        assert_eq!(
            table.mask("nonexistent").memberships,
            table.mask(LAYER_WORLD).memberships
        );
    }

    #[test]
    fn query_mask_filters_to_the_requested_targets() {
        let table = LayerTable::default();
        let mask = table.query_mask(LAYER_CHARACTER, &[LAYER_WORLD, LAYER_PROP]);
        assert_eq!(mask.memberships, 1 << 2);
        assert_eq!(mask.filter, 0b0011);
    }

    #[test]
    fn layer_count_caps_at_32() {
        let extras: Vec<String> = (0..40).map(|i| format!("extra{i}")).collect();
        let refs: Vec<&str> = extras.iter().map(|s| s.as_str()).collect();
        let table = LayerTable::new(&config(&refs, &[]));
        assert_eq!(table.names.len(), 32);
        assert_eq!(table.mask("extra27").memberships, 1 << 31);
        // Layer 33+ fell off and resolves to the world fallback.
        assert_eq!(table.mask("extra28").memberships, 1);
    }
}
