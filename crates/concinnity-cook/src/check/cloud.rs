// World-shape validation for Cloud: the layers an Atmosphere names.

use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;
use concinnity_engine::components::Cloud;

// The most layers one atmosphere draws; `MAX_CLOUD_LAYERS` in the engine.
const MAX_LAYERS: usize = concinnity_engine::render::environment::clouds::MAX_CLOUD_LAYERS;

/// Refuse an `Atmosphere` naming more cloud layers than it draws or layers whose
/// heights overlap, and warn of a `Cloud` no atmosphere names, which is never
/// drawn.
pub(crate) fn check_layers(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    for warning in warnings(assets, errors) {
        tracing::warn!("{warning}");
    }
}

// The problems `check_layers` reports: errors pushed, warnings returned.
fn warnings(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) -> Vec<String> {
    let mut named = Vec::new();
    for a in assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::Atmosphere)
    {
        let layers: Vec<&str> = a
            .args
            .get("clouds")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str())
            .collect();
        if layers.len() > MAX_LAYERS {
            errors.push(format!(
                "Asset '{}': an Atmosphere draws at most {MAX_LAYERS} cloud layers, but its \
                 `clouds` names {}",
                a.id,
                layers.len()
            ));
        }
        overlaps(a, &layers, assets, errors);
        named.extend(layers);
    }
    assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::Cloud && !named.contains(&a.id.as_str()))
        .map(|a| {
            format!(
                "Asset '{}': no Atmosphere names this Cloud in its `clouds`, so it is not drawn",
                a.id
            )
        })
        .collect()
}

// The heights, in meters over the ground, between which the layer `cloud`
// lies, its unauthored fields taken from `Cloud`'s defaults.
fn span(cloud: &WorldJsonlAsset) -> [f64; 2] {
    let defaults = Cloud::default();
    let field = |name: &str, default: f32| {
        cloud
            .args
            .get(name)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or_else(|| f64::from(default))
    };
    let base = field("base", defaults.base);
    [base, base + field("thickness", defaults.thickness)]
}

// Refuse the layers `layers` that the atmosphere `atmosphere` names when two of
// them share heights: each layer is marched over its own stretch of a view ray,
// so the light of one inside the other is drawn out of order.
fn overlaps(
    atmosphere: &WorldJsonlAsset,
    layers: &[&str],
    assets: &[WorldJsonlAsset],
    errors: &mut Vec<String>,
) {
    let spans: Vec<(&str, [f64; 2])> = layers
        .iter()
        .filter_map(|&id| {
            assets
                .iter()
                .find(|c| c.asset_type == RegisteredType::Cloud && c.id == id)
                .map(|c| (id, span(c)))
        })
        .collect();
    for (i, (a, [a_lo, a_hi])) in spans.iter().enumerate() {
        for (b, [b_lo, b_hi]) in &spans[i + 1..] {
            if a_lo < b_hi && b_lo < a_hi {
                errors.push(format!(
                    "Asset '{}': cloud layers '{a}' and '{b}' overlap in height; each layer \
                     must lie wholly above or below the others",
                    atmosphere.id
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::asset;

    fn cloud(id: &str) -> WorldJsonlAsset {
        asset(id, RegisteredType::Cloud, serde_json::json!({}))
    }

    fn atmosphere(clouds: &[&str]) -> WorldJsonlAsset {
        asset(
            "air",
            RegisteredType::Atmosphere,
            serde_json::json!({ "clouds": clouds }),
        )
    }

    #[test]
    fn named_layers_pass() {
        let mut errors = Vec::new();
        let w = warnings(
            &[
                layer("low", 600.0, 1200.0),
                layer("high", 3200.0, 250.0),
                atmosphere(&["low", "high"]),
            ],
            &mut errors,
        );
        assert!(errors.is_empty(), "{errors:?}");
        assert!(w.is_empty(), "{w:?}");
    }

    #[test]
    fn too_many_layers_are_refused() {
        let mut errors = Vec::new();
        warnings(&[atmosphere(&["a", "b", "c"])], &mut errors);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("at most 2"), "{errors:?}");
    }

    fn layer(id: &str, base: f32, thickness: f32) -> WorldJsonlAsset {
        asset(
            id,
            RegisteredType::Cloud,
            serde_json::json!({ "base": base, "thickness": thickness }),
        )
    }

    #[test]
    fn overlapping_layers_are_refused() {
        let mut errors = Vec::new();
        warnings(
            &[
                layer("low", 600.0, 1200.0),
                layer("mid", 1500.0, 500.0),
                atmosphere(&["low", "mid"]),
            ],
            &mut errors,
        );
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("'low' and 'mid'"), "{errors:?}");
    }

    // Layers that only touch, and a default layer under an authored one, pass.
    #[test]
    fn stacked_layers_pass() {
        let mut errors = Vec::new();
        warnings(
            &[
                layer("low", 600.0, 1200.0),
                layer("high", 1800.0, 250.0),
                atmosphere(&["low", "high"]),
            ],
            &mut errors,
        );
        warnings(
            &[
                cloud("default"),
                layer("top", 3200.0, 250.0),
                atmosphere(&["default", "top"]),
            ],
            &mut errors,
        );
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn an_unnamed_cloud_is_warned_of() {
        let mut errors = Vec::new();
        let w = warnings(&[cloud("lost"), atmosphere(&[])], &mut errors);
        assert!(errors.is_empty());
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("'lost'"), "{w:?}");
    }
}
