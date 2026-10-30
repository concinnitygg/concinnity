// World-shape validation for WaterSurface: what the surface a planet's sea
// names can use.

use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;

/// Refuse a planet sea whose `WaterSurface` authors swell, which a sea does not
/// carry.
pub(crate) fn check_sea_swell(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    let seas = assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::Planet)
        .filter_map(|p| Some((p, p.args.get("sea")?.get("water")?.as_str()?)));
    for (planet, water) in seas {
        let swell = assets
            .iter()
            .filter(|a| a.asset_type == RegisteredType::WaterSurface && a.id == water)
            .filter_map(|a| a.args.get("waves")?.as_array())
            .any(|w| !w.is_empty());
        if swell {
            errors.push(format!(
                "Asset '{}': Planet's sea takes its look from WaterSurface '{water}', which \
                 authors `waves`; a planet's sea carries no swell, its waves all come from the \
                 wind, so drop the swell",
                planet.id
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::asset;

    fn planet() -> WorldJsonlAsset {
        asset(
            "world",
            RegisteredType::Planet,
            serde_json::json!({"sea": {"level": 2.0, "water": "sea"}}),
        )
    }

    #[test]
    fn a_sea_whose_surface_has_swell_is_refused() {
        let sea = asset(
            "sea",
            RegisteredType::WaterSurface,
            serde_json::json!({"waves": [{"amplitude": 0.2}]}),
        );
        let mut errors = Vec::new();
        check_sea_swell(&[planet(), sea], &mut errors);
        assert_eq!(errors.len(), 1);
        assert!(
            errors[0].contains("'sea'") && errors[0].contains("waves"),
            "{errors:?}"
        );
    }

    #[test]
    fn swell_on_a_rectangle_and_a_bare_sea_pass() {
        let swell = serde_json::json!({"waves": [{"amplitude": 0.2}]});
        let pond = asset("pond", RegisteredType::WaterSurface, swell);
        let sea = asset(
            "sea",
            RegisteredType::WaterSurface,
            serde_json::json!({"waves": []}),
        );
        let mut errors = Vec::new();
        check_sea_swell(&[planet(), pond, sea], &mut errors);
        assert!(errors.is_empty(), "{errors:?}");
    }
}
