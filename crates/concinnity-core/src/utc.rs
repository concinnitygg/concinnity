//! Civil UTC time.

use crate::math::SaturatingFrom;
use alloc::format;
use alloc::string::String;

/// A civil UTC timestamp, from Unix seconds without a date dependency. Only
/// meaningful for post-epoch times.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct UtcTime {
    /// The Gregorian year.
    pub year: i64,
    /// The month, 1 to 12.
    pub month: u32,
    /// The day of the month, from 1.
    pub day: u32,
    /// The hour, 0 to 23.
    pub hour: u32,
    /// The minute, 0 to 59.
    pub minute: u32,
    /// The second, 0 to 59.
    pub second: u32,
}

impl UtcTime {
    /// The civil time `secs` after the Unix epoch (Gregorian, proleptic;
    /// Hinnant's civil-from-days).
    pub fn from_unix(secs: u64) -> Self {
        let days = i64::saturating_from(secs / 86_400);
        let rem = secs % 86_400;
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097) as u64;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = u32::saturating_from(doy - (153 * mp + 2) / 5 + 1);
        let month = u32::saturating_from(if mp < 10 { mp + 3 } else { mp - 9 });
        let year = i64::saturating_from(yoe) + era * 400 + i64::from(month <= 2);
        Self {
            year,
            month,
            day,
            hour: (rem / 3600) as u32,
            minute: (rem % 3600 / 60) as u32,
            second: (rem % 60) as u32,
        }
    }

    /// `YYYY-MM-DD hh:mm:ss`.
    pub fn display(&self) -> String {
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }

    /// Compact, lexicographically sortable form for file names:
    /// `YYYYMMDD-hhmmss`.
    pub fn stem(&self) -> String {
        format!(
            "{:04}{:02}{:02}-{:02}{:02}{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_converts_to_civil_origin() {
        let t = UtcTime::from_unix(0);
        assert_eq!(t.display(), "1970-01-01 00:00:00");
    }

    #[test]
    fn known_timestamps_convert() {
        // 2026-08-08 12:34:56 UTC.
        let t = UtcTime::from_unix(1_786_192_496);
        assert_eq!(t.display(), "2026-08-08 12:34:56");
        assert_eq!(t.stem(), "20260808-123456");
        // A leap day: 2024-02-29 23:59:59 UTC.
        let t = UtcTime::from_unix(1_709_251_199);
        assert_eq!(t.display(), "2024-02-29 23:59:59");
    }
}
