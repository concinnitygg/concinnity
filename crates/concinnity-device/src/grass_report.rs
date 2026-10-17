//! How a grass frame reaching a placement limit is reported.

use concinnity_core::render::grass::GrassLimit;

/// Log the limit a grass frame reached. The field reports only the first
/// frame that reaches one, so this logs once per field.
pub(crate) fn report_grass_limit(limit: Option<GrassLimit>) {
    let what = match limit {
        Some(GrassLimit::TileRecords) => "more grass tiles in reach than a run's records hold",
        Some(GrassLimit::PlanetTiles) => "more planet tiles in reach than a grass run holds",
        None => return,
    };
    tracing::warn!("grass: {what}; the farthest are dropped");
}
