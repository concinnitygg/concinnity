//! The main-pass pipelines of a world's material-referenced shaders, one slot
//! per shader bucket past the default, generic over a backend's pipeline type.
//!
//! Bucket 0 is the world's default program, drawn under the main pass's own
//! pipeline, so it is always resident and has no slot here. Bucket `b` reads
//! slot `b - 1`, which stays empty while the scene owning its Shader has not
//! pinned.

use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;

use crate::render::error::{RenderError, RenderResult};

/// One optional pipeline per shader bucket past the default.
#[derive(Debug)]
pub struct WorldPipelines<P> {
    slots: Vec<Option<P>>,
}

impl<P> Default for WorldPipelines<P> {
    fn default() -> Self {
        Self { slots: Vec::new() }
    }
}

impl<P> WorldPipelines<P> {
    /// A table over `slots`, slot `i` holding bucket `i + 1`.
    pub fn new(slots: Vec<Option<P>>) -> Self {
        Self { slots }
    }

    /// Shader buckets, the default included.
    pub fn bucket_count(&self) -> usize {
        1 + self.slots.len()
    }

    /// The slot holding `bucket`'s pipeline. A bucket outside the table is a
    /// scene-authoring mistake, not a device failure, so it errors as
    /// [`RenderError::Other`].
    pub fn slot(&self, bucket: u32) -> RenderResult<usize> {
        let slot = (bucket as usize).checked_sub(1).ok_or_else(|| {
            RenderError::Other("shader bucket 0 is the world default program".to_string())
        })?;
        if slot >= self.slots.len() {
            return Err(RenderError::Other(format!(
                "shader bucket {bucket} is past the world's {} shader pipeline(s)",
                self.slots.len()
            )));
        }
        Ok(slot)
    }

    /// Whether `bucket`'s draws can render: the default always can, every
    /// other bucket needs its pipeline installed.
    pub fn resident(&self, bucket: usize) -> bool {
        bucket == 0 || self.get(bucket).is_some()
    }

    /// `bucket`'s installed pipeline.
    pub fn get(&self, bucket: usize) -> Option<&P> {
        self.slots.get(bucket.checked_sub(1)?)?.as_ref()
    }

    /// `bucket`'s installed pipeline, to update in place.
    pub fn get_mut(&mut self, bucket: usize) -> Option<&mut P> {
        self.slots.get_mut(bucket.checked_sub(1)?)?.as_mut()
    }

    /// Install `pipeline` for `bucket`, handing back the one it displaces.
    pub fn install(&mut self, bucket: u32, pipeline: P) -> RenderResult<Option<P>> {
        let slot = self.slot(bucket)?;
        Ok(self.slots[slot].replace(pipeline))
    }

    /// Empty `bucket`'s slot, handing back what it held. A bucket outside the
    /// table holds nothing.
    pub fn evict(&mut self, bucket: u32) -> Option<P> {
        let slot = self.slot(bucket).ok()?;
        self.slots[slot].take()
    }

    /// Every bucket past the default whose pipeline is installed, ascending.
    pub fn resident_buckets(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_some())
            .map(|(slot, _)| slot + 1)
    }
}

/// Whether a rebuilt shader bucket may replace the live one, given whether
/// each has a G-buffer pre-pass. A bucket whose pre-pass never built keeps
/// shading without one, so a rebuild that also lacks it replaces it; a rebuild
/// that would drop a pre-pass the bucket draws with is refused, so the caller
/// keeps the live pair.
pub fn check_rebuild(bucket: usize, live_prepass: bool, fresh_prepass: bool) -> RenderResult<()> {
    match fresh_prepass || !live_prepass {
        true => Ok(()),
        false => Err(RenderError::ShaderCompile(format!(
            "shader bucket {bucket}'s G-buffer pre-pass did not build; keeping its live pipelines"
        ))),
    }
}

/// The pair that replaces a live shader bucket's: `prepared` when it passes
/// [`check_rebuild`], otherwise one from `build`, which must pass it too.
/// `has_prepass` reads whether a pair has its G-buffer pre-pass.
pub fn replace_bucket<P>(
    bucket: usize,
    live_prepass: bool,
    prepared: Option<P>,
    has_prepass: impl Fn(&P) -> bool,
    build: impl FnOnce() -> RenderResult<P>,
) -> RenderResult<P> {
    let keeps = |p: &P| check_rebuild(bucket, live_prepass, has_prepass(p));
    let fresh = match prepared.filter(|p| keeps(p).is_ok()) {
        Some(pair) => pair,
        None => build()?,
    };
    keeps(&fresh)?;
    Ok(fresh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // Only a rebuild that would lose a live pre-pass is refused.
    #[test]
    fn a_rebuild_may_not_drop_a_live_prepass() {
        assert!(check_rebuild(2, true, true).is_ok());
        assert!(check_rebuild(2, false, true).is_ok());
        assert!(check_rebuild(2, false, false).is_ok());
        let refused = check_rebuild(2, true, false).expect_err("a live pre-pass is lost");
        assert!(refused.to_string().contains("shader bucket 2"), "{refused}");
    }

    // A pair as its pre-pass half: `Some` when it built.
    type Pair = Option<&'static str>;

    fn replace(live_prepass: bool, prepared: Option<Pair>, built: Pair) -> RenderResult<Pair> {
        replace_bucket(1, live_prepass, prepared, Option::is_some, || Ok(built))
    }

    #[test]
    fn a_bucket_without_a_prepass_takes_a_rebuild_without_one() {
        assert_eq!(replace(false, None, None), Ok(None));
        assert_eq!(replace(false, Some(None), Some("built")), Ok(None));
    }

    #[test]
    fn a_prepared_pair_that_drops_a_live_prepass_is_rebuilt() {
        assert_eq!(replace(true, Some(None), Some("built")), Ok(Some("built")));
        assert_eq!(
            replace(true, Some(Some("prepared")), None),
            Ok(Some("prepared"))
        );
    }

    #[test]
    fn a_rebuild_that_drops_a_live_prepass_is_refused() {
        assert!(replace(true, None, None).is_err());
        assert!(replace(true, Some(None), None).is_err());
        let failed = replace_bucket::<Pair>(1, false, None, Option::is_some, || {
            Err(RenderError::Other("no device".into()))
        });
        assert!(failed.is_err());
    }

    // Bucket 1 installed, bucket 2 deferred, bucket 3 installed.
    fn table() -> WorldPipelines<&'static str> {
        WorldPipelines::new(vec![Some("one"), None, Some("three")])
    }

    #[test]
    fn buckets_count_the_default() {
        assert_eq!(table().bucket_count(), 4);
        assert_eq!(WorldPipelines::<()>::default().bucket_count(), 1);
    }

    #[test]
    fn a_bucket_reads_the_slot_before_it() {
        let t = table();
        assert_eq!(t.slot(1).unwrap(), 0);
        assert_eq!(t.slot(3).unwrap(), 2);
        assert_eq!(t.get(3), Some(&"three"));
    }

    #[test]
    fn the_default_and_buckets_past_the_table_have_no_slot() {
        let t = table();
        for bucket in [0, 4, u32::MAX] {
            assert!(
                matches!(t.slot(bucket), Err(RenderError::Other(_))),
                "{bucket}"
            );
        }
        assert_eq!(t.get(0), None);
        assert_eq!(t.get(9), None);
    }

    #[test]
    fn the_default_is_always_resident_and_a_deferred_bucket_is_not() {
        let t = table();
        assert!(t.resident(0));
        assert!(t.resident(1));
        assert!(!t.resident(2));
        assert!(!t.resident(4));
        assert_eq!(t.resident_buckets().collect::<Vec<_>>(), [1, 3]);
    }

    #[test]
    fn get_mut_reaches_only_installed_buckets() {
        let mut t = table();
        *t.get_mut(3).unwrap() = "tres";
        assert_eq!(t.get(3), Some(&"tres"));
        assert!(t.get_mut(0).is_none());
        assert!(t.get_mut(2).is_none());
        assert!(t.get_mut(9).is_none());
    }

    #[test]
    fn install_and_evict_hand_back_what_they_replace() {
        let mut t = table();
        assert_eq!(t.install(2, "two").unwrap(), None);
        assert_eq!(t.install(1, "uno").unwrap(), Some("one"));
        assert_eq!(t.resident_buckets().collect::<Vec<_>>(), [1, 2, 3]);
        assert_eq!(t.evict(3), Some("three"));
        assert_eq!(t.evict(3), None);
        assert_eq!(t.evict(0), None);
        assert_eq!(t.evict(7), None);
        assert!(t.install(0, "zero").is_err());
        assert_eq!(t.resident_buckets().collect::<Vec<_>>(), [1, 2]);
    }
}
