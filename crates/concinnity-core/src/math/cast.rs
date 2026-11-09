// Numeric conversions that name what happens out of range, in place of `as`.

/// Converts a number into `Self`, clamping it to `Self`'s range.
///
/// An integer source past either end becomes `Self::MIN` or `Self::MAX`. A float source rounds
/// toward zero, saturates the same way, and NaN becomes zero.
///
/// ```
/// use concinnity_core::math::SaturatingFrom;
///
/// assert_eq!(u8::saturating_from(300_u32), 255);
/// assert_eq!(u32::saturating_from(-4_i32), 0);
/// assert_eq!(i32::saturating_from(2.9_f32), 2);
/// assert_eq!(u16::saturating_from(f32::NAN), 0);
/// ```
pub trait SaturatingFrom<T> {
    /// `value` clamped to `Self`'s range.
    fn saturating_from(value: T) -> Self;
}

macro_rules! saturating_from_unsigned {
    ($($src:ty),* => $dst:ty) => {$(
        impl SaturatingFrom<$src> for $dst {
            #[inline]
            fn saturating_from(value: $src) -> Self {
                Self::try_from(value).unwrap_or(Self::MAX)
            }
        }
    )*};
}

macro_rules! saturating_from_signed {
    ($($src:ty),* => $dst:ty) => {$(
        impl SaturatingFrom<$src> for $dst {
            #[inline]
            fn saturating_from(value: $src) -> Self {
                Self::try_from(value).unwrap_or(if value < 0 { Self::MIN } else { Self::MAX })
            }
        }
    )*};
}

macro_rules! saturating_from_int {
    ($($dst:ty),*) => {$(
        saturating_from_unsigned!(u8, u16, u32, u64, u128, usize => $dst);
        saturating_from_signed!(i8, i16, i32, i64, i128, isize => $dst);
    )*};
}

saturating_from_int!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

// Float-to-int `as` is exactly the saturating conversion: toward zero, clamped, NaN to zero.
macro_rules! saturating_from_float_unsigned {
    ($($dst:ty),*) => {$(
        impl SaturatingFrom<f32> for $dst {
            #[inline]
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "float-to-int `as` saturates and maps NaN to zero"
            )]
            fn saturating_from(value: f32) -> Self {
                value as Self
            }
        }

        impl SaturatingFrom<f64> for $dst {
            #[inline]
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "float-to-int `as` saturates and maps NaN to zero"
            )]
            fn saturating_from(value: f64) -> Self {
                value as Self
            }
        }
    )*};
}

macro_rules! saturating_from_float_signed {
    ($($dst:ty),*) => {$(
        impl SaturatingFrom<f32> for $dst {
            #[inline]
            #[expect(
                clippy::cast_possible_truncation,
                reason = "float-to-int `as` saturates and maps NaN to zero"
            )]
            fn saturating_from(value: f32) -> Self {
                value as Self
            }
        }

        impl SaturatingFrom<f64> for $dst {
            #[inline]
            #[expect(
                clippy::cast_possible_truncation,
                reason = "float-to-int `as` saturates and maps NaN to zero"
            )]
            fn saturating_from(value: f64) -> Self {
                value as Self
            }
        }
    )*};
}

saturating_from_float_unsigned!(u8, u16, u32, u64, usize);
saturating_from_float_signed!(i8, i16, i32, i64, isize);

/// [`SaturatingFrom`] from `usize` to `u32` for const contexts, such as a `size_of` stride.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is checked against u32::MAX first"
)]
pub const fn saturating_u32(value: usize) -> u32 {
    if value > u32::MAX as usize {
        u32::MAX
    } else {
        value as u32
    }
}

/// `value` rounded to the nearest `f32`; magnitudes past `f32::MAX` become infinite.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "rounding to the nearest f32 is the narrowing asked for"
)]
pub const fn narrow_f32(value: f64) -> f32 {
    value as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsigned_sources_clamp_to_max() {
        assert_eq!(u8::saturating_from(255_u32), 255);
        assert_eq!(u8::saturating_from(256_u32), 255);
        assert_eq!(u32::saturating_from(u64::MAX), u32::MAX);
        assert_eq!(i32::saturating_from(u32::MAX), i32::MAX);
        assert_eq!(u16::saturating_from(7_usize), 7);
    }

    #[test]
    fn signed_sources_clamp_to_either_end() {
        assert_eq!(u32::saturating_from(-1_i32), 0);
        assert_eq!(u8::saturating_from(i64::MAX), u8::MAX);
        assert_eq!(i16::saturating_from(i32::MIN), i16::MIN);
        assert_eq!(i16::saturating_from(i32::MAX), i16::MAX);
        assert_eq!(usize::saturating_from(-3_isize), 0);
        assert_eq!(i8::saturating_from(-5_i64), -5);
    }

    #[test]
    fn floats_round_toward_zero_and_saturate() {
        assert_eq!(u32::saturating_from(3.99_f32), 3);
        assert_eq!(i32::saturating_from(-3.99_f32), -3);
        assert_eq!(u8::saturating_from(-1.0_f32), 0);
        assert_eq!(u8::saturating_from(1000.0_f64), 255);
        assert_eq!(i64::saturating_from(f64::INFINITY), i64::MAX);
        assert_eq!(i64::saturating_from(f64::NEG_INFINITY), i64::MIN);
        assert_eq!(usize::saturating_from(f32::NAN), 0);
    }

    #[test]
    fn saturating_u32_matches_the_trait() {
        const STRIDE: u32 = saturating_u32(size_of::<[f32; 3]>());
        assert_eq!(STRIDE, 12);
        assert_eq!(saturating_u32(usize::MAX), u32::saturating_from(usize::MAX));
    }

    #[test]
    fn narrow_f32_rounds_to_nearest() {
        assert_eq!(narrow_f32(0.1), 0.1_f32);
        assert_eq!(narrow_f32(1.0e300), f32::INFINITY);
        assert_eq!(narrow_f32(-1.0e300), f32::NEG_INFINITY);
        assert!(narrow_f32(f64::NAN).is_nan());
    }
}
