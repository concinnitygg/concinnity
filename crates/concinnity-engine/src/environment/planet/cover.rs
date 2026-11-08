//! What a planet's ground looks like from place to place: [`GroundCover`]s
//! laid over its material by rule.

use crate::components::GroundRule;
use crate::ecs::MaterialHandle;

/// Most covers a planet lays over its ground; later ones are dropped.
pub const MAX_GROUND_COVERS: usize = 4;

/// A look a [Planet](#planet)'s ground takes where its `rule` holds: sand on
/// a beach, a duller seabed under deeper water, rock on steep slopes.
///
/// The cover's `material` replaces the planet's own where the rule holds and
/// fades back to it over the rule's soft edges. Its color (`tint` over its
/// `albedo` texture), `roughness` and `metallic` are used; the planet's
/// `material` keeps the normal map and every other map.
///
/// ```rust
/// # use concinnity_engine::components::{GroundCover, GroundRule};
/// GroundCover {
///     rule: GroundRule {
///         altitude: [Some(-4.0), Some(2.5)],
///         ..Default::default()
///     },
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct GroundCover {
    /// The [Material](#material) the ground takes where the rule holds.
    pub material: Option<MaterialHandle>,
    /// Where the cover lies.
    pub rule: GroundRule,
}
