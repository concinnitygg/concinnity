//! Authored parameters clamped into their declared ranges at load. A value
//! outside its field's `#[asset(min = ..., max = ...)]` range is moved to the
//! nearest bound, and the load reports each one rather than failing.

use concinnity_core::ecs::AssetFields;

/// `value` with every ranged field clamped into its range, passing `warn` one
/// message per value that moved.
pub(crate) fn clamp_authored<T: AssetFields>(
    asset: &str,
    mut value: T,
    mut warn: impl FnMut(String),
) -> T {
    let mut clamped = Vec::new();
    value.clamp_params(&mut clamped);
    for c in clamped {
        warn(format!(
            "{asset}.{}: {} is outside its range, clamped to {}",
            c.path(),
            c.from,
            c.to
        ));
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::{IndirectLighting, Shadows};

    #[test]
    fn an_out_of_range_value_is_clamped_with_one_warning_each() {
        let authored = IndirectLighting {
            ambient_intensity: 30.0,
            ssgi_rays: 0,
            ssgi_intensity: 0.5,
            ..Default::default()
        };
        let mut warnings = Vec::new();
        let loaded = clamp_authored("IndirectLighting", authored, |w| warnings.push(w));
        assert_eq!(loaded.ambient_intensity, 16.0);
        assert_eq!(loaded.ssgi_rays, 1);
        assert_eq!(loaded.ssgi_intensity, 0.5);
        assert_eq!(
            warnings,
            [
                "IndirectLighting.ambient_intensity: 30 is outside its range, clamped to 16",
                "IndirectLighting.ssgi_rays: 0 is outside its range, clamped to 1",
            ]
        );
    }

    #[test]
    fn the_defaults_load_without_a_warning() {
        let mut warnings = Vec::new();
        clamp_authored("Shadows", Shadows::default(), |w| warnings.push(w));
        clamp_authored("IndirectLighting", IndirectLighting::default(), |w| {
            warnings.push(w)
        });
        assert!(warnings.is_empty(), "{warnings:?}");
    }
}
