//! What an asset field declares about itself as a parameter: the range an
//! authored number is clamped into, and when a change to it takes effect.
//! `#[asset(min = ..., max = ..., apply = ...)]` states both, and
//! `#[derive(AssetFields)]` turns them into the field's [`ParamMeta`], the clamp
//! a load applies, and the comparison a settings diff groups by class.

use alloc::vec::Vec;
use concinnity_core::math::{SaturatingFrom, narrow_f32};

/// When a change to a parameter takes effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ApplyClass {
    /// A per-frame value: the next frame reads the change, at no cost.
    Live,
    /// Sizes or gates a resource the renderer owns: the change rebuilds that
    /// resource in place, with a one-frame hitch.
    Rebuild,
    /// Baked into a shader: the change selects another pipeline variant.
    Variant,
    /// Sizes something built once at launch: the change applies at the next
    /// launch.
    Restart,
}

impl ApplyClass {
    /// Every class, cheapest first.
    pub const ALL: [Self; 4] = [Self::Live, Self::Rebuild, Self::Variant, Self::Restart];

    /// The name an `#[asset(apply = ...)]` attribute and the docs use.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Rebuild => "rebuild",
            Self::Variant => "variant",
            Self::Restart => "restart",
        }
    }
}

/// The bounds an authored number is clamped into. Either end may be open.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ParamRange {
    /// The smallest accepted value.
    pub min: Option<f64>,
    /// The largest accepted value.
    pub max: Option<f64>,
}

impl ParamRange {
    /// No bound at either end.
    pub const UNBOUNDED: Self = Self {
        min: None,
        max: None,
    };

    /// Whether either end is bounded.
    #[cfg(test)]
    const fn is_bounded(&self) -> bool {
        self.min.is_some() || self.max.is_some()
    }

    /// `value` moved into the range. A NaN stays NaN.
    pub fn clamp(&self, value: f64) -> f64 {
        match (self.min, self.max) {
            (Some(min), _) if value < min => min,
            (_, Some(max)) if value > max => max,
            _ => value,
        }
    }
}

/// A field's parameter metadata.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ParamMeta {
    /// The range authored values are clamped into.
    pub range: ParamRange,
    /// When a change takes effect, for a field that declares it.
    pub apply: Option<ApplyClass>,
}

impl ParamMeta {
    /// A field declaring neither a range nor an apply class.
    pub const NONE: Self = Self {
        range: ParamRange::UNBOUNDED,
        apply: None,
    };
}

/// One field declaring parameter metadata, under its dotted path.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamField {
    /// The field's authored key, dotted through nested objects.
    pub path: alloc::string::String,
    /// What the field declares.
    pub meta: ParamMeta,
}

/// An authored value a load moved into its field's range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClampedParam {
    /// The key of the section holding the field, for a field nested in one.
    pub section: Option<&'static str>,
    /// The field's authored key.
    pub key: &'static str,
    /// The authored value.
    pub from: f64,
    /// The value it was clamped to.
    pub to: f64,
}

impl ClampedParam {
    /// The field's dotted path: `section.key`, or `key` outside a section.
    pub fn path(&self) -> alloc::string::String {
        match self.section {
            Some(section) => alloc::format!("{section}.{}", self.key),
            None => alloc::string::String::from(self.key),
        }
    }
}

/// A parameter whose value differs between two values of one asset type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamChange {
    /// The asset type's name.
    pub asset: &'static str,
    /// The field's authored key.
    pub key: &'static str,
    /// When the change takes effect.
    pub apply: ApplyClass,
}

/// The values a load clamped, in field order.
pub type ClampedParams = Vec<ClampedParam>;

/// The parameters that changed, in field order.
pub type ParamChanges = Vec<ParamChange>;

/// A number a range can bound.
pub trait ParamNumber: Copy {
    /// The value as an `f64`.
    fn to_f64(self) -> f64;
    /// The nearest value of this type to `value`.
    fn from_f64(value: f64) -> Self;
}

macro_rules! param_numbers {
    ($convert:ident: $($ty:ty),*) => {$(
        impl ParamNumber for $ty {
            fn to_f64(self) -> f64 {
                param_numbers!(@$convert self)
            }
            fn from_f64(value: f64) -> Self {
                <$ty>::saturating_from(value)
            }
        }
    )*};
    (@lossless $value:expr) => {
        f64::from($value)
    };
    (@rounded $value:expr) => {
        $value as f64
    };
}

param_numbers!(lossless: u8, u16, u32, i8, i16, i32);
param_numbers!(rounded: u64, usize, i64, isize);

impl ParamNumber for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_f64(value: f64) -> Self {
        narrow_f32(value)
    }
}

impl ParamNumber for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(value: f64) -> Self {
        value
    }
}

/// Clamp `value` into `range`, returning the authored and clamped values when
/// it moved.
#[expect(
    clippy::float_cmp,
    reason = "change detection: report only a value the clamp moved"
)]
pub fn clamp_param<T: ParamNumber>(value: &mut T, range: ParamRange) -> Option<(f64, f64)> {
    let from = value.to_f64();
    let to = range.clamp(from);
    if to == from || from.is_nan() {
        return None;
    }
    *value = T::from_f64(to);
    Some((from, to))
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT: ParamRange = ParamRange {
        min: Some(0.0),
        max: Some(1.0),
    };

    #[test]
    fn a_value_inside_the_range_is_left_alone() {
        let mut v = 0.25_f32;
        assert_eq!(clamp_param(&mut v, UNIT), None);
        assert_eq!(v, 0.25);
        let mut edge = 1.0_f32;
        assert_eq!(clamp_param(&mut edge, UNIT), None);
    }

    #[test]
    fn a_value_outside_is_moved_to_the_nearest_end() {
        let mut v = 3.0_f32;
        assert_eq!(clamp_param(&mut v, UNIT), Some((3.0, 1.0)));
        assert_eq!(v, 1.0);
        let mut n = -2_i32;
        assert_eq!(clamp_param(&mut n, UNIT), Some((-2.0, 0.0)));
        assert_eq!(n, 0);
    }

    #[test]
    fn an_open_end_bounds_nothing() {
        let floor = ParamRange {
            min: Some(1.0),
            max: None,
        };
        let mut big = u32::MAX;
        assert_eq!(clamp_param(&mut big, floor), None);
        let mut zero = 0_u32;
        assert_eq!(clamp_param(&mut zero, floor), Some((0.0, 1.0)));
        assert_eq!(zero, 1);
        assert!(!ParamRange::UNBOUNDED.is_bounded());
        assert!(floor.is_bounded());
    }

    #[test]
    fn nan_is_not_reported_as_clamped() {
        let mut v = f32::NAN;
        assert_eq!(clamp_param(&mut v, UNIT), None);
    }

    #[test]
    fn classes_are_ordered_cheapest_first() {
        assert!(ApplyClass::ALL.windows(2).all(|w| w[0] < w[1]));
        let names: Vec<_> = ApplyClass::ALL.iter().map(|c| c.as_str()).collect();
        assert_eq!(names, ["live", "rebuild", "variant", "restart"]);
    }
}
