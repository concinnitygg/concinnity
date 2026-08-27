//! Checked arithmetic over sizes read out of an untrusted buffer. Decoders take
//! dimensions from the payload itself, so every product and sum derived from
//! them is attacker-reachable and has to be range-checked before it is used as
//! a length.

use alloc::format;
use alloc::string::String;

/// Product of `factors`, or an error naming `label` if it overflows `usize`.
pub fn checked_product(label: &str, factors: &[usize]) -> Result<usize, String> {
    let mut acc: usize = 1;
    for f in factors {
        acc = acc
            .checked_mul(*f)
            .ok_or_else(|| format!("{} size overflow in {:?}", label, factors))?;
    }
    Ok(acc)
}

// Sum of `terms`, or an error naming `label` if it overflows `usize`.
#[cfg(test)]
pub(crate) fn checked_sum(label: &str, terms: &[usize]) -> Result<usize, String> {
    let mut acc: usize = 0;
    for t in terms {
        acc = acc
            .checked_add(*t)
            .ok_or_else(|| format!("{} offset overflow in {:?}", label, terms))?;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_multiplies() {
        assert_eq!(checked_product("t", &[2, 3, 4]).unwrap(), 24);
    }

    #[test]
    fn empty_product_is_one() {
        assert_eq!(checked_product("t", &[]).unwrap(), 1);
    }

    #[test]
    fn product_zero_short_circuits_safely() {
        assert_eq!(checked_product("t", &[0, usize::MAX]).unwrap(), 0);
    }

    #[test]
    fn product_reports_overflow() {
        let err = checked_product("atlas", &[usize::MAX, 2]).unwrap_err();
        assert!(err.contains("atlas"), "{}", err);
        assert!(err.contains("overflow"), "{}", err);
    }

    // The concrete shape that broke font decoding: a 32-bit dimension pair
    // whose product exceeds usize once the bytes-per-texel factor is applied.
    #[test]
    fn product_reports_overflow_for_max_dimensions() {
        let w = u32::MAX as usize;
        assert!(checked_product("atlas", &[w, w, 4]).is_err());
    }

    #[test]
    fn sum_adds() {
        assert_eq!(checked_sum("t", &[1, 2, 3]).unwrap(), 6);
    }

    #[test]
    fn empty_sum_is_zero() {
        assert_eq!(checked_sum("t", &[]).unwrap(), 0);
    }

    #[test]
    fn sum_reports_overflow() {
        let err = checked_sum("payload", &[usize::MAX, 1]).unwrap_err();
        assert!(err.contains("payload"), "{}", err);
        assert!(err.contains("overflow"), "{}", err);
    }
}
