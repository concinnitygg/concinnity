// Planet terrain detail schema.

/// How finely a planet's terrain tiles are drawn. The `planet_detail` section
/// of [GraphicsConfig](#graphicsconfig); a field left unset takes the default
/// below.
///
/// The default describes the detail capable hardware runs: the `Auto`
/// graphics quality preset resolves the detected GPU into a ceiling that
/// coarsens it tier by tier.
///
/// ```rust
/// # use concinnity_core::components::{GraphicsConfig, PlanetDetailOverrides};
/// GraphicsConfig {
///     planet_detail: PlanetDetailOverrides {
///         finest_cell_size: Some(2),
///         ..Default::default()
///     },
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[asset(section)]
#[serde(default)]
pub struct PlanetDetail {
    /// How wide a planet's terrain cells may be under the camera at the
    /// finest, in meters: the quadtree stops dividing tiles once their cells
    /// are no wider. Each level halves the cells, so the finest land between
    /// half this and this. Larger builds and draws far fewer tiles near the
    /// camera. `1` by default, coarsened by the quality preset up to
    /// 4 on the lowest tier.
    #[asset(default = 1, min = 1, max = 64, apply = live)]
    pub finest_cell_size: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_finest_cells_are_a_meter_by_default() {
        let parsed: PlanetDetail = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.finest_cell_size, 1);
        assert_eq!(PlanetDetail::default().finest_cell_size, 1);
    }
}
