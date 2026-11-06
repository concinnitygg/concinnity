//! What a backend keeps of the world's planet beyond its streamed tiles.

/// The planet-shaped state a backend builds at init.
pub trait PlanetReshape {
    /// Bring what the backend keeps of the world's planet to `shape`, which
    /// keeps the planet's center and radius: its sea's shore, where its grass
    /// roots and its ground's far field (see
    /// [`reshape_planet`](crate::render::environment::reshape::reshape_planet)).
    /// The tiles stream again on their own. Driven by the editor's live planet
    /// seam. Default no-op, for a backend that draws no planet.
    fn reshape_planet(&mut self, shape: &crate::planet::PlanetShape) {
        let _ = shape;
    }
}
