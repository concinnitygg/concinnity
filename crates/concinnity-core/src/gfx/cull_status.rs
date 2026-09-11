//! The per-object outcome vocabulary the GPU cull writes into its status
//! buffer, and the host-side histogram a readback of that buffer reduces to.
//!
//! The status buffer is the only record of what the GPU-driven cull actually
//! decided: the submitted draw-call count is a CPU-side number that does not
//! move when an object is rejected on the GPU, and the Hi-Z pyramid leaves no
//! trace in the presented pixels of an object it correctly occluded. Reading
//! the buffer back and tallying it here is what gives a Hi-Z change a
//! behavioral oracle.
//!
//! Values mirror the `STATUS_*` constants in `cull.slang`; a test below reads
//! that shader source and asserts the two agree.

use alloc::vec::Vec;

/// The per-object outcomes the GPU cull records in its status buffer. Metal's
/// ICB encode kernel is told which one to draw rather than declaring them
/// itself.
pub struct CullStatus;

impl CullStatus {
    /// Visible in phase 1, drawn by the main pass.
    pub const DRAWN: u32 = 0;
    /// Hi-Z-occluded in phase 1; the only outcome phase 2 re-tests. Under
    /// single-pass occlusion no phase 2 runs, so this is the settled outcome
    /// of a Hi-Z rejection.
    pub const HIZ_CANDIDATE: u32 = 1;
    /// Frustum-, distance- or disabled-culled; settled.
    pub const CULLED: u32 = 2;
    /// A candidate phase 2 found visible, drawn by the disocclusion pass.
    pub const REDRAW: u32 = 3;
    /// A candidate phase 2 found still occluded by this frame's own depth.
    pub const HIZ_CULLED: u32 = 4;
}

/// A tally of one frame's cull-status buffer: how many objects landed in each
/// outcome. Produced by [`tally`] from a raw readback.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CullStatusCounts {
    /// Objects [`CullStatus::DRAWN`]: passed every phase-1 test.
    pub drawn: u32,
    /// Objects [`CullStatus::CULLED`]: rejected by the frustum, the per-object
    /// cull distance, or a clear enable bit. Never Hi-Z.
    pub frustum_culled: u32,
    /// Objects left at [`CullStatus::HIZ_CANDIDATE`]: Hi-Z-occluded in phase 1
    /// and not re-tested. Under two-pass occlusion a non-zero count here means
    /// phase 2 did not run over those objects.
    pub hiz_candidate: u32,
    /// Objects [`CullStatus::REDRAW`]: Hi-Z-occluded in phase 1, found visible
    /// against the rebuilt pyramid, drawn by the disocclusion pass.
    pub redrawn: u32,
    /// Objects [`CullStatus::HIZ_CULLED`]: Hi-Z-occluded in phase 1 and still
    /// occluded in phase 2.
    pub hiz_culled: u32,
    /// Entries carrying a value no `STATUS_*` constant names. Non-zero means
    /// the buffer was read past the live object count, or was never written.
    pub unknown: u32,
}

impl CullStatusCounts {
    /// Objects the cull let through to a draw, over both phases.
    pub fn visible(self) -> u32 {
        self.drawn + self.redrawn
    }

    /// Objects the Hi-Z test rejected, whether or not phase 2 re-tested them.
    /// The number a Hi-Z A/B compares.
    pub fn hiz_rejected(self) -> u32 {
        self.hiz_candidate + self.hiz_culled
    }

    /// Every entry tallied, across all outcomes.
    pub fn total(self) -> u32 {
        self.drawn
            + self.frustum_culled
            + self.hiz_candidate
            + self.redrawn
            + self.hiz_culled
            + self.unknown
    }
}

/// Reduce a raw cull-status readback to per-outcome counts.
pub fn tally(raw: &[u32]) -> CullStatusCounts {
    let mut c = CullStatusCounts::default();
    for &status in raw {
        let slot = match status {
            CullStatus::DRAWN => &mut c.drawn,
            CullStatus::CULLED => &mut c.frustum_culled,
            CullStatus::HIZ_CANDIDATE => &mut c.hiz_candidate,
            CullStatus::REDRAW => &mut c.redrawn,
            CullStatus::HIZ_CULLED => &mut c.hiz_culled,
            _ => &mut c.unknown,
        };
        *slot += 1;
    }
    c
}

/// Decode a byte-oriented readback into the `u32` statuses [`tally`] consumes,
/// truncating to `count` objects. The backends map their status buffer as raw
/// bytes; the trailing capacity past the live object count holds whatever the
/// last resize left there, so the caller's live cull count is the length that
/// matters. Returns `Err` when the mapping is too short for `count`.
pub fn decode(bytes: &[u8], count: usize) -> Result<Vec<u32>, &'static str> {
    if bytes.len() < count * core::mem::size_of::<u32>() {
        return Err("cull-status readback is shorter than the live object count");
    }
    Ok(bytes[..count * core::mem::size_of::<u32>()]
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tally_counts_every_outcome() {
        let raw = [
            CullStatus::DRAWN,
            CullStatus::DRAWN,
            CullStatus::CULLED,
            CullStatus::HIZ_CANDIDATE,
            CullStatus::REDRAW,
            CullStatus::HIZ_CULLED,
            CullStatus::HIZ_CULLED,
        ];
        let c = tally(&raw);
        assert_eq!(c.drawn, 2);
        assert_eq!(c.frustum_culled, 1);
        assert_eq!(c.hiz_candidate, 1);
        assert_eq!(c.redrawn, 1);
        assert_eq!(c.hiz_culled, 2);
        assert_eq!(c.unknown, 0);
        assert_eq!(c.total(), raw.len() as u32);
        assert_eq!(c.visible(), 3);
        assert_eq!(c.hiz_rejected(), 3);
    }

    #[test]
    fn tally_of_nothing_is_all_zero() {
        assert_eq!(tally(&[]), CullStatusCounts::default());
        assert_eq!(tally(&[]).total(), 0);
    }

    #[test]
    fn unnamed_status_values_land_in_unknown() {
        // An unwritten buffer is the case this guards: a probe reading a
        // status region the cull never dispatched over must not silently
        // report those objects as DRAWN-adjacent.
        let c = tally(&[5, 7, u32::MAX]);
        assert_eq!(c.unknown, 3);
        assert_eq!(c.total(), 3);
        assert_eq!(c.visible(), 0);
        assert_eq!(c.hiz_rejected(), 0);
    }

    #[test]
    fn decode_truncates_to_the_live_object_count() {
        let mut bytes = Vec::new();
        for v in [CullStatus::DRAWN, CullStatus::CULLED, 9u32] {
            bytes.extend_from_slice(&v.to_ne_bytes());
        }
        let decoded = decode(&bytes, 2).expect("two objects fit");
        assert_eq!(decoded, [CullStatus::DRAWN, CullStatus::CULLED]);
        assert_eq!(tally(&decoded).unknown, 0);
    }

    #[test]
    fn decode_rejects_a_short_mapping() {
        let bytes = [0u8; 4];
        assert!(decode(&bytes, 2).is_err());
        assert!(decode(&bytes, 1).is_ok());
        assert!(decode(&[], 0).is_ok());
    }

    // The shader is the authority on these values: the Rust constants exist so
    // the host can name them, and Metal's encode kernel is handed one of them
    // as a uniform. Read `cull.slang` and assert every `STATUS_*` it declares
    // matches, so a shader edit that renumbers one fails here rather than
    // silently mis-tallying a readback.
    #[test]
    fn constants_match_cull_slang() {
        let declared = parse_shader_statuses(crate::render::shaders::CULL);
        let expected = [
            ("STATUS_DRAWN", CullStatus::DRAWN),
            ("STATUS_HIZ_CANDIDATE", CullStatus::HIZ_CANDIDATE),
            ("STATUS_CULLED", CullStatus::CULLED),
            ("STATUS_REDRAW", CullStatus::REDRAW),
            ("STATUS_HIZ_CULLED", CullStatus::HIZ_CULLED),
        ];
        assert_eq!(
            declared.len(),
            expected.len(),
            "cull.slang declares {} STATUS_* constants, the host names {}: {declared:?}",
            declared.len(),
            expected.len(),
        );
        for (name, value) in expected {
            let found = declared
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("cull.slang declares no {name}"));
            assert_eq!(found.1, value, "{name} disagrees with cull.slang");
        }
    }

    // Scrape `static const uint STATUS_<NAME> = <N>u;` declarations.
    fn parse_shader_statuses(src: &str) -> Vec<(alloc::string::String, u32)> {
        src.lines()
            .filter_map(|line| {
                let rest = line.trim().strip_prefix("static const uint STATUS_")?;
                let (name, rest) = rest.split_once('=')?;
                let value = rest.split_once(';')?.0.trim().trim_end_matches('u');
                Some((
                    alloc::format!("STATUS_{}", name.trim()),
                    value.parse().ok()?,
                ))
            })
            .collect()
    }
}
