//! The main-pass pipelines of a world's material-referenced shaders, one slot
//! per shader bucket past the default, generic over a backend's pipeline type.
//!
//! Bucket 0 is the world's default program, drawn under the main pass's own
//! pipeline, so it is always resident and has no slot here. Bucket `b` reads
//! slot `b - 1`, which stays empty while the scene owning its Shader has not
//! pinned.
//!
//! Only a bucket that draws a surface growing grass (one with a far band)
//! compiles the grass ground and canopy into its main fragment, since that code
//! costs every pixel of the program carrying it. A world Shader that draws one
//! compiles it into its own bucket; the default keeps its bucket without it and
//! gains a canopy twin past the last world Shader, which the far-band records
//! route to.

use alloc::format;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::components::ShaderPrograms;
use crate::gfx::render_types::MAX_SHADER_BUCKETS;
use crate::render::backend_init::WorldShader;
use crate::render::error::{RenderError, RenderResult};

/// What one shader bucket compiles: a world Shader's programs or the engine's
/// own, with or without the far-field canopy.
#[derive(Debug, Clone, Copy)]
pub struct BucketSource<'a> {
    /// The world Shader's programs; `None` for the engine's own.
    pub programs: Option<&'a ShaderPrograms>,
    /// The Shader's payload waits on a scene other than the start scene.
    pub deferred: bool,
    /// The main fragment shades a far band as grass canopy.
    pub canopy: bool,
}

/// Every shader bucket a world draws with.
#[derive(Debug, Clone)]
pub struct BucketPlan<'a> {
    /// By bucket index: one per world Shader, then the default's canopy twin.
    pub sources: Vec<BucketSource<'a>>,
    /// The bucket the default Shader's far-band draws route to, when any
    /// draws under it.
    pub canopy_twin: Option<u32>,
}

impl<'a> BucketPlan<'a> {
    /// The plan for `shaders`, indexed by ShaderHandle value with the world
    /// default at 0. Fails when the buckets exceed what a cull dispatch can
    /// route between.
    pub fn new(shaders: &[WorldShader<'a>]) -> RenderResult<Self> {
        let mut sources: Vec<BucketSource<'a>> = shaders
            .iter()
            .enumerate()
            .map(|(bucket, s)| BucketSource {
                programs: s.programs,
                deferred: s.deferred,
                canopy: bucket > 0 && s.canopy,
            })
            .collect();
        let canopy_twin = match shaders.first() {
            Some(default) if default.canopy => {
                sources.push(BucketSource {
                    programs: default.programs,
                    deferred: default.deferred,
                    canopy: true,
                });
                Some((sources.len() - 1) as u32)
            }
            _ => None,
        };
        if sources.len() > MAX_SHADER_BUCKETS {
            return Err(RenderError::Other(format!(
                "world needs {} shader buckets ({} Shaders{}) but at most \
                 {MAX_SHADER_BUCKETS} are supported",
                sources.len(),
                shaders.len(),
                match canopy_twin {
                    Some(_) => " and the default Shader's grass canopy",
                    None => "",
                }
            )));
        }
        Ok(Self {
            sources,
            canopy_twin,
        })
    }

    /// The buckets past the default.
    pub fn extra(&self) -> &[BucketSource<'a>] {
        self.sources.get(1..).unwrap_or_default()
    }
}

/// One optional pipeline per shader bucket past the default.
#[derive(Debug)]
pub struct WorldPipelines<P> {
    slots: Vec<Option<P>>,
    // Per slot: the bucket's main fragment compiles the canopy.
    canopy: Vec<bool>,
    canopy_twin: Option<u32>,
}

impl<P> Default for WorldPipelines<P> {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl<P> WorldPipelines<P> {
    /// A table over `slots`, slot `i` holding bucket `i + 1`, none of which
    /// compiles the canopy.
    pub fn new(slots: Vec<Option<P>>) -> Self {
        Self {
            canopy: vec![false; slots.len()],
            slots,
            canopy_twin: None,
        }
    }

    /// The table for `plan`, each bucket past the default built by `build`
    /// unless its Shader is deferred.
    pub fn build<E>(
        plan: &BucketPlan<'_>,
        mut build: impl FnMut(usize, &BucketSource<'_>) -> Result<P, E>,
    ) -> Result<Self, E> {
        let mut slots = Vec::with_capacity(plan.extra().len());
        for (i, source) in plan.extra().iter().enumerate() {
            slots.push(match source.deferred {
                true => None,
                false => Some(build(i + 1, source)?),
            });
        }
        Ok(Self {
            canopy: plan.extra().iter().map(|s| s.canopy).collect(),
            slots,
            canopy_twin: plan.canopy_twin,
        })
    }

    /// Whether `bucket`'s main fragment compiles the canopy. The default
    /// never does.
    pub fn canopy(&self, bucket: usize) -> bool {
        bucket
            .checked_sub(1)
            .and_then(|slot| self.canopy.get(slot))
            .is_some_and(|c| *c)
    }

    /// Every bucket's canopy flag, the default's first.
    pub fn canopy_flags(&self) -> Vec<bool> {
        (0..self.bucket_count()).map(|b| self.canopy(b)).collect()
    }

    /// The bucket the default Shader's far-band draws route to.
    pub fn canopy_twin(&self) -> Option<u32> {
        self.canopy_twin
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

    fn shader(canopy: bool, deferred: bool) -> WorldShader<'static> {
        WorldShader {
            programs: None,
            deferred,
            canopy,
        }
    }

    fn canopy_of(plan: &BucketPlan<'_>) -> Vec<bool> {
        plan.sources.iter().map(|s| s.canopy).collect()
    }

    // A world with no far band has one bucket per Shader, none with canopy.
    #[test]
    fn without_a_far_band_no_bucket_compiles_the_canopy() {
        let plan = BucketPlan::new(&[shader(false, false), shader(false, false)]).unwrap();
        assert_eq!(canopy_of(&plan), [false, false]);
        assert_eq!(plan.canopy_twin, None);
    }

    // A far band under the default adds a canopy twin past the last Shader and
    // leaves the default bucket itself without the canopy.
    #[test]
    fn a_far_band_under_the_default_adds_a_twin_past_the_last_shader() {
        let plan = BucketPlan::new(&[shader(true, false), shader(false, false)]).unwrap();
        assert_eq!(canopy_of(&plan), [false, false, true]);
        assert_eq!(plan.canopy_twin, Some(2));
        assert_eq!(plan.extra().len(), 2);
    }

    // A world Shader under a far band compiles the canopy into its own bucket.
    #[test]
    fn a_far_band_under_a_world_shader_compiles_into_its_bucket() {
        let plan = BucketPlan::new(&[
            shader(false, false),
            shader(true, false),
            shader(false, false),
        ])
        .unwrap();
        assert_eq!(canopy_of(&plan), [false, true, false]);
        assert_eq!(plan.canopy_twin, None);
    }

    // The twin counts toward the routing budget, and the error names it.
    #[test]
    fn the_twin_counts_toward_the_bucket_budget() {
        let full = vec![shader(false, false); MAX_SHADER_BUCKETS];
        assert!(BucketPlan::new(&full).is_ok());
        let mut over = full.clone();
        over[0].canopy = true;
        let err = BucketPlan::new(&over).unwrap_err().to_string();
        assert!(err.contains("grass canopy"), "{err}");
        assert!(BucketPlan::new(&over[1..]).is_ok());
    }

    // The table builds every bucket past the default except deferred ones,
    // remembers which compile the canopy, and where the twin is.
    #[test]
    fn the_table_built_from_a_plan_keeps_its_canopy_flags() {
        let plan = BucketPlan::new(&[
            shader(true, false),
            shader(true, true),
            shader(false, false),
        ])
        .unwrap();
        let mut built = Vec::new();
        let t = WorldPipelines::build(&plan, |bucket, source| {
            built.push((bucket, source.canopy));
            Ok::<_, ()>(bucket)
        })
        .unwrap();
        assert_eq!(built, [(2, false), (3, true)]);
        assert_eq!(t.bucket_count(), 4);
        assert_eq!(t.resident_buckets().collect::<Vec<_>>(), [2, 3]);
        assert_eq!(t.canopy_flags(), [false, true, false, true]);
        assert_eq!(t.canopy_twin(), Some(3));
        assert!(!t.canopy(0) && !t.canopy(9));
    }

    #[test]
    fn a_plain_table_has_no_canopy() {
        let t = table();
        assert_eq!(t.canopy_flags(), [false; 4]);
        assert_eq!(t.canopy_twin(), None);
    }
}
