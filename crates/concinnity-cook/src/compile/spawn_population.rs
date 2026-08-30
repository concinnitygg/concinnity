// How many copies of its template a spawner can hold alive at once, which is
// what a world has to reserve physics bodies for. Pure arithmetic over the
// authored cadence: no world, no assets, no physics.

/// How many copies of a spawner's template can be alive at the same time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpawnPopulation {
    /// The spawner never emits, so it costs nothing.
    Inert,
    /// At most this many copies are alive at once.
    Bounded(u32),
    /// Copies are never removed, so no fixed reservation covers them.
    Unbounded,
}

/// The most copies a spawner emitting one every `interval` seconds, each
/// living `lifetime` seconds, can hold alive at once.
pub(crate) fn population(interval: f32, lifetime: f32) -> SpawnPopulation {
    // The spawner's clock only fires while `elapsed >= interval` can become
    // true, which a non-positive or non-finite interval never allows.
    if !interval.is_finite() || interval <= 0.0 {
        return SpawnPopulation::Inert;
    }
    // A zero lifetime is not a countdown but "keep it forever", and a
    // non-finite one never counts down, so neither bounds the population.
    if !lifetime.is_finite() || lifetime <= 0.0 {
        return SpawnPopulation::Unbounded;
    }
    // Alive at any instant are the copies emitted within the last `lifetime`
    // seconds, plus the one emitted at that instant. Rounding the span count
    // up keeps the estimate on the reserving side of the real cadence.
    let spans = (f64::from(lifetime) / f64::from(interval)).ceil();
    // A float-to-int cast saturates, so an absurd cadence pins at the maximum
    // rather than wrapping to a tiny reservation.
    SpawnPopulation::Bounded((spans as u32).saturating_add(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cadence_holds_its_lifespan_worth_of_copies_plus_the_new_one() {
        // One every 2s, each living 4s: two spans of overlap, plus the copy
        // just emitted.
        assert_eq!(population(2.0, 4.0), SpawnPopulation::Bounded(3));
        // Ten a second for a second: ten spans plus the new one.
        assert_eq!(population(0.1, 1.0), SpawnPopulation::Bounded(11));
        // A copy that expires before the next spawn is alone.
        assert_eq!(population(5.0, 1.0), SpawnPopulation::Bounded(2));
    }

    // A lifespan that is not a whole number of intervals rounds up, never
    // down: reserving one body too few refuses a real spawn.
    #[test]
    fn a_partial_span_rounds_up() {
        assert_eq!(population(2.0, 5.0), SpawnPopulation::Bounded(4));
        assert_eq!(population(2.0, 4.001), SpawnPopulation::Bounded(4));
    }

    // The trap: a zero lifetime means the copies live forever, so the naive
    // lifetime / interval of 0 would reserve nothing for the spawner that
    // needs it most.
    #[test]
    fn a_forever_lifetime_is_unbounded_not_zero() {
        assert_eq!(population(1.0, 0.0), SpawnPopulation::Unbounded);
        assert_eq!(population(0.016, 0.0), SpawnPopulation::Unbounded);
        assert_eq!(population(1.0, -3.0), SpawnPopulation::Unbounded);
        assert_eq!(population(1.0, f32::INFINITY), SpawnPopulation::Unbounded);
        assert_eq!(population(1.0, f32::NAN), SpawnPopulation::Unbounded);
    }

    #[test]
    fn a_spawner_that_can_never_fire_costs_nothing() {
        assert_eq!(population(0.0, 4.0), SpawnPopulation::Inert);
        assert_eq!(population(-1.0, 4.0), SpawnPopulation::Inert);
        assert_eq!(population(f32::NAN, 4.0), SpawnPopulation::Inert);
        assert_eq!(population(f32::INFINITY, 4.0), SpawnPopulation::Inert);
        // A zero lifetime does not rescue an interval that never fires.
        assert_eq!(population(0.0, 0.0), SpawnPopulation::Inert);
    }

    #[test]
    fn an_absurd_cadence_saturates_instead_of_wrapping() {
        assert_eq!(
            population(f32::MIN_POSITIVE, f32::MAX),
            SpawnPopulation::Bounded(u32::MAX)
        );
    }
}
