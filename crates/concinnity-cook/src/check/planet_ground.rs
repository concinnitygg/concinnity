// Validation for PlanetGround: each square's args, and the squares together.

use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;
use concinnity_engine::components::cook::PlanetGround;

/// Check one `PlanetGround`'s args make a square of ground.
pub(crate) fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let ground: PlanetGround = serde_json::from_value(args.clone())
        .map_err(|e| format!("Asset '{name}': PlanetGround: {e}"))?;
    ground
        .page()
        .map(|_| ())
        .map_err(|e| format!("Asset '{name}': PlanetGround {e}"))
}

/// Refuse sculpted ground with no planet to lie on, and two squares on one
/// tile, which would leave which one the ground takes to chance.
pub(crate) fn check_squares(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    let grounds: Vec<&WorldJsonlAsset> = assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::PlanetGround)
        .collect();
    if grounds.is_empty() {
        return;
    }
    if !assets
        .iter()
        .any(|a| a.asset_type == RegisteredType::Planet)
    {
        errors.push(format!(
            "Asset '{}': PlanetGround sculpts a planet, but the world declares no Planet",
            grounds[0].id
        ));
    }
    let mut seen: std::collections::HashMap<String, &str> = std::collections::HashMap::new();
    for ground in grounds {
        let tile = ground
            .args
            .get("tile")
            .map(ToString::to_string)
            .unwrap_or_default();
        if let Some(first) = seen.insert(tile.clone(), &ground.id) {
            errors.push(format!(
                "Asset '{}': PlanetGround tile {tile} is also sculpted by '{first}'; a tile \
                 holds one square of ground",
                ground.id
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::planet::PAGE_SAMPLES;

    fn asset(id: &str, asset_type: RegisteredType, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            id: id.to_string(),
            asset_type,
            args,
        }
    }

    fn ground(id: &str, tile: [u32; 4]) -> WorldJsonlAsset {
        asset(
            id,
            RegisteredType::PlanetGround,
            serde_json::json!({ "tile": tile, "height": vec![0.5; PAGE_SAMPLES] }),
        )
    }

    #[test]
    fn a_square_s_args_are_checked() {
        assert!(check("g", &ground("g", [2, 8, 128, 127]).args).is_ok());
        let err = check("g", &ground("g", [2, 8, 300, 0]).args).unwrap_err();
        assert!(
            err.contains("Asset 'g'") && err.contains("names no square"),
            "{err}"
        );
        let err = check("g", &serde_json::json!({ "height": "high" })).unwrap_err();
        assert!(err.contains("Asset 'g'"), "{err}");
    }

    #[test]
    fn squares_need_a_planet_and_their_own_tiles() {
        let planet = asset("world", RegisteredType::Planet, serde_json::json!({}));
        let mut errors = Vec::new();
        check_squares(
            &[
                planet.clone(),
                ground("a", [2, 8, 1, 1]),
                ground("b", [2, 8, 1, 2]),
            ],
            &mut errors,
        );
        assert!(errors.is_empty(), "{errors:?}");
        check_squares(
            &[planet, ground("a", [2, 8, 1, 1]), ground("b", [2, 8, 1, 1])],
            &mut errors,
        );
        assert_eq!(errors.len(), 1);
        assert!(
            errors[0].contains("'b'") && errors[0].contains("'a'"),
            "{errors:?}"
        );
        errors.clear();
        check_squares(&[ground("a", [2, 8, 1, 1])], &mut errors);
        assert!(errors[0].contains("no Planet"), "{errors:?}");
        errors.clear();
        check_squares(&[], &mut errors);
        assert!(errors.is_empty());
    }
}
