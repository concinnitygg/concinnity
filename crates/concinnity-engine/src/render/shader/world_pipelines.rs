//! The main-pass pipelines of a world's material-referenced shaders, one slot
//! per shader bucket past the default, generic over a backend's pipeline type.
//!
//! Bucket 0 is the world's default program, drawn under the main pass's own
//! pipeline, so it is always resident and has no slot here. Bucket `b` reads
//! slot `b - 1`, which stays empty while the scene owning its Shader has not
//! pinned.
//!
//! Only a bucket that draws a surface growing grass (one with a far band), or a
//! planet's ground, compiles the grass ground and canopy into its main
//! fragment, since that code costs every pixel of the program carrying it. So
//! every Shader keeps a plain bucket, and one under which such a surface draws
//! gains a family of variants past the last Shader's plain bucket: a canopy
//! variant for the draws that reach their far band, a ground variant without
//! the canopy for those wholly nearer and for a planet's ground growing no
//! grass, and on a planet each of those with and without the planet's own sky.
//! The draw cull picks each draw's variant (see
//! [`GpuCullVariants`](crate::gfx::render_types::GpuCullVariants)).

use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::{format, vec};
use concinnity_core::math::SaturatingFrom;
use core::ops::Range;

use crate::components::ShaderPrograms;
use crate::gfx::render_types::{
    BucketSpans, DrawObject, GpuCullVariants, GpuVariantRoute, MAX_SHADER_BUCKETS,
    MAX_VARIANT_ROUTES, VARIANT_DAYLIGHT, VARIANT_FAR, bucket_regions, draw_args_bucket_bits,
    draw_args_route_bits,
};
use crate::render::backend::error::{RenderError, RenderResult};
use crate::render::backend::init::WorldShader;
use crate::render::shader::programs::surface::SurfaceVariant;

/// The grass variants a Shader's draws route between.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GrassVariants {
    /// None: no draw under the Shader shades its ground from the canopy block.
    #[default]
    None,
    /// The canopy and the ground, each lit from a planet's own sky where one
    /// lights the view.
    Grass,
    /// The canopy and the ground, each with and without a planet's own sky,
    /// so a draw the sky cannot reach shades without its code.
    PlanetSky,
}

impl GrassVariants {
    // The family's variants, in bucket order.
    fn variants(self) -> &'static [SurfaceVariant] {
        match self {
            Self::None => &[],
            Self::Grass => &[
                SurfaceVariant::CanopyDaylight,
                SurfaceVariant::GroundDaylight,
            ],
            Self::PlanetSky => &[
                SurfaceVariant::Canopy,
                SurfaceVariant::CanopyDaylight,
                SurfaceVariant::Ground,
                SurfaceVariant::GroundDaylight,
            ],
        }
    }

    // The route over the family from `first`: the bucket per feature
    // combination and the features tested.
    fn route(self, first: u32) -> GpuVariantRoute {
        let at = |v: SurfaceVariant| {
            first
                + u32::saturating_from(
                    self.variants()
                        .iter()
                        .position(|&x| x == v)
                        .unwrap_or_default(),
                )
        };
        use SurfaceVariant::{Canopy, CanopyDaylight, Ground, GroundDaylight};
        match self {
            Self::None => GpuVariantRoute::default(),
            Self::Grass => {
                let (ground, canopy) = (at(GroundDaylight), at(CanopyDaylight));
                GpuVariantRoute::new([ground, canopy, ground, canopy], VARIANT_FAR)
            }
            Self::PlanetSky => GpuVariantRoute::new(
                [
                    at(Ground),
                    at(Canopy),
                    at(GroundDaylight),
                    at(CanopyDaylight),
                ],
                VARIANT_FAR | VARIANT_DAYLIGHT,
            ),
        }
    }
}

/// What one shader bucket compiles: a world Shader's programs or the engine's
/// own, and which of its variants the main fragment is.
#[derive(Debug, Clone, Copy)]
pub struct BucketSource<'a> {
    /// The world Shader's programs; `None` for the engine's own.
    pub programs: Option<&'a ShaderPrograms>,
    /// The Shader's payload waits on a scene other than the start scene.
    pub deferred: bool,
    /// Which of the Shader's main fragments the bucket draws with.
    pub variant: SurfaceVariant,
}

/// Per Shader (by `ShaderHandle` value): the variant route its draws shading
/// their ground take, if it has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShaderRoutes([u8; MAX_SHADER_BUCKETS]);

impl Default for ShaderRoutes {
    fn default() -> Self {
        Self([0; MAX_SHADER_BUCKETS])
    }
}

impl ShaderRoutes {
    /// The route Shader `shader`'s draws shading their ground take.
    pub fn route(&self, shader: u32) -> Option<u32> {
        let r = *self.0.get(shader as usize)?;
        r.checked_sub(1).map(u32::from)
    }

    /// `obj`'s variant flag bits: its Shader's bucket, and the route the cull
    /// picks its bucket by when its ground shades from the canopy block.
    pub fn draw_bits(&self, obj: &DrawObject) -> u32 {
        let far_field = obj.material.far_field;
        let route = if far_field.shades_ground() {
            self.route(obj.shader_bucket)
        } else {
            None
        };
        draw_args_bucket_bits(obj.shader_bucket)
            | draw_args_route_bits(route, far_field.grows_grass())
    }
}

/// One Shader's grass variants: its buckets, and the route over them.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Family {
    buckets: Range<u32>,
    route: GpuVariantRoute,
}

/// Every shader bucket a world draws with.
#[derive(Debug, Clone)]
pub struct BucketPlan<'a> {
    /// By bucket index: one plain bucket per world Shader, then each Shader's
    /// grass variants.
    pub sources: Vec<BucketSource<'a>>,
    routes: ShaderRoutes,
    families: Vec<Family>,
}

impl<'a> BucketPlan<'a> {
    /// The plan for `shaders`, indexed by `ShaderHandle` value with the world
    /// default at 0.
    ///
    /// # Errors
    ///
    /// Returns an error if the buckets exceed what a cull dispatch can route
    /// between: [`MAX_SHADER_BUCKETS`].
    pub fn new(shaders: &[WorldShader<'a>]) -> RenderResult<Self> {
        let mut sources: Vec<BucketSource<'a>> = shaders
            .iter()
            .map(|s| BucketSource {
                programs: s.programs,
                deferred: s.deferred,
                variant: SurfaceVariant::Plain,
            })
            .collect();
        let mut routes = ShaderRoutes::default();
        let mut families = Vec::new();
        for (shader, s) in shaders.iter().enumerate() {
            let variants = s.grass.variants();
            if variants.is_empty() {
                continue;
            }
            let first = u32::saturating_from(sources.len());
            sources.extend(variants.iter().map(|&variant| BucketSource {
                programs: s.programs,
                deferred: s.deferred,
                variant,
            }));
            if let Some(slot) = routes.0.get_mut(shader) {
                *slot = u8::saturating_from(families.len() + 1);
            }
            families.push(Family {
                buckets: first..u32::saturating_from(sources.len()),
                route: s.grass.route(first),
            });
        }
        if sources.len() > MAX_SHADER_BUCKETS || families.len() > MAX_VARIANT_ROUTES {
            return Err(RenderError::Other(format!(
                "world needs {} shader buckets ({} Shaders and {} grass variants) but at most \
                 {MAX_SHADER_BUCKETS} are supported",
                sources.len(),
                shaders.len(),
                sources.len() - shaders.len(),
            )));
        }
        Ok(Self {
            sources,
            routes,
            families,
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
    // Per slot: which of its Shader's main fragments the bucket draws with.
    variants: Vec<SurfaceVariant>,
    routes: ShaderRoutes,
    families: Vec<Family>,
}

impl<P> Default for WorldPipelines<P> {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl<P> WorldPipelines<P> {
    /// A table over `slots`, slot `i` holding bucket `i + 1`, each a plain
    /// bucket.
    pub fn new(slots: Vec<Option<P>>) -> Self {
        Self {
            variants: vec![SurfaceVariant::Plain; slots.len()],
            slots,
            routes: ShaderRoutes::default(),
            families: Vec::new(),
        }
    }

    /// The table for `plan`, each bucket past the default built by `build`
    /// unless its Shader is deferred.
    ///
    /// # Errors
    ///
    /// Returns the first error `build` returns.
    pub fn build<E>(
        plan: &BucketPlan<'_>,
        mut build: impl FnMut(usize, &BucketSource<'_>) -> Result<P, E>,
    ) -> Result<Self, E> {
        let mut slots = Vec::with_capacity(plan.extra().len());
        for (i, source) in plan.extra().iter().enumerate() {
            slots.push(if source.deferred {
                None
            } else {
                Some(build(i + 1, source)?)
            });
        }
        Ok(Self {
            variants: plan.extra().iter().map(|s| s.variant).collect(),
            slots,
            routes: plan.routes,
            families: plan.families.clone(),
        })
    }

    /// Which of its Shader's main fragments `bucket` draws with. The default
    /// draws with the plain one.
    pub fn variant(&self, bucket: usize) -> SurfaceVariant {
        bucket
            .checked_sub(1)
            .and_then(|slot| self.variants.get(slot))
            .copied()
            .unwrap_or_default()
    }

    /// Which main fragment every bucket draws with, the default's first.
    pub fn variants(&self) -> Vec<SurfaceVariant> {
        (0..self.bucket_count()).map(|b| self.variant(b)).collect()
    }

    /// The route each Shader's draws shading their ground take.
    pub fn routes(&self) -> ShaderRoutes {
        self.routes
    }

    /// Shader `shader`'s grass variant buckets; none when it has no family.
    pub fn family(&self, shader: u32) -> Range<u32> {
        self.routes
            .route(shader)
            .and_then(|r| self.families.get(r as usize))
            .map_or(0..0, |f| f.buckets.clone())
    }

    /// The bucket a single command stream (a probe face, a mirror) draws every
    /// record under: the default Shader's fullest variant, the canopy lit from
    /// a planet's own sky, so a far band seen in it keeps its canopy.
    pub fn superset_bucket(&self) -> Option<u32> {
        let default = self.routes.route(0)?;
        let family = self.families.get(default as usize)?;
        family
            .buckets
            .clone()
            .find(|&b| self.variant(b as usize) == SurfaceVariant::CanopyDaylight)
    }

    /// The buckets a record of `obj` can draw under: its Shader's grass
    /// variants when its ground shades from the canopy block and the Shader
    /// has them, else its Shader's own.
    pub fn candidates(&self, obj: &DrawObject) -> Range<u32> {
        let family = obj
            .material
            .far_field
            .shades_ground()
            .then(|| self.routes.route(obj.shader_bucket))
            .flatten()
            .and_then(|r| self.families.get(r as usize));
        if let Some(f) = family {
            return f.buckets.clone();
        }
        let last = u32::saturating_from(self.bucket_count().saturating_sub(1));
        let base = obj.shader_bucket.min(last);
        base..base + 1
    }

    /// The records each bucket can take. `draws` are the build-time draws, each
    /// at its own record; `runtime` the reserve streamed chunks and clones
    /// fill, which every bucket past the default covers, since a clone takes
    /// its template's Shader and a chunk the default's. The default covers all
    /// `records`.
    pub fn bucket_spans(
        &self,
        draws: &[DrawObject],
        runtime: Range<usize>,
        records: usize,
    ) -> Vec<BucketSpans> {
        let count = self.bucket_count();
        let mut lo = vec![usize::MAX; count];
        let mut hi = vec![0usize; count];
        for (record, obj) in draws.iter().enumerate() {
            for b in self.candidates(obj) {
                if let (Some(l), Some(h)) = (lo.get_mut(b as usize), hi.get_mut(b as usize)) {
                    *l = (*l).min(record);
                    *h = (*h).max(record + 1);
                }
            }
        }
        (0..count)
            .map(|b| {
                if b == 0 {
                    return BucketSpans {
                        records: 0..records,
                        runtime: 0..0,
                    };
                }
                let records = if lo[b] < hi[b] { lo[b]..hi[b] } else { 0..0 };
                // A runtime span overlapping the build-time one would hand a
                // record two commands in one region.
                let runtime = runtime.start.max(records.end)..runtime.end;
                BucketSpans { records, runtime }
            })
            .collect()
    }

    /// The cull's variant block over `spans`: the routes and the bucket
    /// regions, with no view set yet.
    pub fn cull_variants(&self, spans: &[BucketSpans]) -> GpuCullVariants {
        let mut block = GpuCullVariants {
            regions: bucket_regions(spans),
            ..GpuCullVariants::default()
        };
        for (slot, family) in block.routes.iter_mut().zip(&self.families) {
            *slot = family.route;
        }
        block
    }

    /// Shader buckets, the default included.
    pub fn bucket_count(&self) -> usize {
        1 + self.slots.len()
    }

    /// The slot holding `bucket`'s pipeline. A bucket outside the table is a
    /// scene-authoring mistake, not a device failure, so it errors as
    /// [`RenderError::Other`].
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Other`] if `bucket` is 0 or past the table.
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
    ///
    /// # Errors
    ///
    /// Returns an error if `bucket` is 0 or past the table.
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
///
/// # Errors
///
/// Returns [`RenderError::ShaderCompile`] if the rebuild would drop a pre-pass
/// the live bucket draws with.
pub fn check_rebuild(bucket: usize, live_prepass: bool, fresh_prepass: bool) -> RenderResult<()> {
    if fresh_prepass || !live_prepass {
        Ok(())
    } else {
        Err(RenderError::ShaderCompile(format!(
            "shader bucket {bucket}'s G-buffer pre-pass did not build; keeping its live pipelines"
        )))
    }
}

/// The pair that replaces a live shader bucket's: `prepared` when it passes
/// [`check_rebuild`], otherwise one from `build`, which must pass it too.
/// `has_prepass` reads whether a pair has its G-buffer pre-pass.
///
/// # Errors
///
/// Returns an error if `build` fails or its pair does not pass
/// [`check_rebuild`].
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

    fn shader(grass: GrassVariants, deferred: bool) -> WorldShader<'static> {
        WorldShader {
            programs: None,
            deferred,
            grass,
        }
    }

    use GrassVariants::{Grass, None as NoGrass, PlanetSky};
    use SurfaceVariant::{Canopy, CanopyDaylight, Ground, GroundDaylight, Plain};

    fn variants_of(plan: &BucketPlan<'_>) -> Vec<SurfaceVariant> {
        plan.sources.iter().map(|s| s.variant).collect()
    }

    fn table_of(plan: &BucketPlan<'_>) -> WorldPipelines<usize> {
        WorldPipelines::build(plan, |bucket, _| Ok::<_, ()>(bucket)).unwrap()
    }

    // A world with no far band has one plain bucket per Shader.
    #[test]
    fn without_a_far_band_every_bucket_is_plain() {
        let plan = BucketPlan::new(&[shader(NoGrass, false), shader(NoGrass, false)]).unwrap();
        assert_eq!(variants_of(&plan), [Plain, Plain]);
        let t = table_of(&plan);
        assert_eq!(t.routes(), ShaderRoutes::default());
        assert_eq!(t.superset_bucket(), None);
    }

    // A far band under the default adds today's twins, the canopy then the
    // ground, both lit from a planet's sky, past the last Shader, and leaves
    // the default bucket plain.
    #[test]
    fn a_far_band_under_the_default_adds_its_variants_past_the_last_shader() {
        let plan = BucketPlan::new(&[shader(Grass, false), shader(NoGrass, false)]).unwrap();
        assert_eq!(
            variants_of(&plan),
            [Plain, Plain, CanopyDaylight, GroundDaylight]
        );
        let t = table_of(&plan);
        assert_eq!(t.routes().route(0), Some(0));
        assert_eq!(t.routes().route(1), None);
        assert_eq!(t.superset_bucket(), Some(2));
        let route = t.cull_variants(&[]).routes[0];
        assert_eq!(route.features, VARIANT_FAR);
        assert_eq!(
            (0..4).map(|f| route.bucket(f)).collect::<Vec<_>>(),
            [3, 2, 3, 2],
            "near -> ground, far -> canopy, whatever the sky"
        );
    }

    // On a planet the family has each variant with and without the sky, and
    // the route picks by both features.
    #[test]
    fn a_planet_family_splits_by_the_sky() {
        let plan = BucketPlan::new(&[shader(PlanetSky, false)]).unwrap();
        assert_eq!(
            variants_of(&plan),
            [Plain, Canopy, CanopyDaylight, Ground, GroundDaylight]
        );
        let t = table_of(&plan);
        assert_eq!(t.superset_bucket(), Some(2));
        let route = t.cull_variants(&[]).routes[0];
        assert_eq!(route.features, VARIANT_FAR | VARIANT_DAYLIGHT);
        let pick = |far: bool, day: bool| {
            let f = u32::from(far) * VARIANT_FAR + u32::from(day) * VARIANT_DAYLIGHT;
            t.variant(route.bucket(f) as usize)
        };
        assert_eq!(pick(false, false), Ground);
        assert_eq!(pick(true, false), Canopy);
        assert_eq!(pick(false, true), GroundDaylight);
        assert_eq!(pick(true, true), CanopyDaylight);
    }

    // A world Shader under a far band keeps a plain bucket and gains its own
    // family, as the default does.
    #[test]
    fn a_far_band_under_a_world_shader_gives_it_a_family() {
        let plan = BucketPlan::new(&[
            shader(NoGrass, false),
            shader(Grass, false),
            shader(NoGrass, false),
        ])
        .unwrap();
        assert_eq!(
            variants_of(&plan),
            [Plain, Plain, Plain, CanopyDaylight, GroundDaylight]
        );
        let t = table_of(&plan);
        assert_eq!(t.routes().route(1), Some(0));
        assert_eq!(t.routes().route(0), None);
        assert_eq!(t.family(1), 3..5);
        assert_eq!(
            t.family(1)
                .map(|b| t.variant(b as usize))
                .collect::<Vec<_>>(),
            [CanopyDaylight, GroundDaylight]
        );
        assert!(t.family(0).is_empty());
        assert_eq!(t.superset_bucket(), None, "only the default's family");
        let route = t.cull_variants(&[]).routes[0];
        assert_eq!((route.bucket(0), route.bucket(VARIANT_FAR)), (4, 3));
    }

    // The cap counts every bucket, variants included, and the error says how
    // many of each the world needs.
    #[test]
    fn the_cap_counts_every_variant() {
        let full = vec![shader(NoGrass, false); MAX_SHADER_BUCKETS];
        assert!(BucketPlan::new(&full).is_ok());
        let mut over = full.clone();
        over[0].grass = Grass;
        let err = BucketPlan::new(&over).unwrap_err().to_string();
        assert!(err.contains("needs 66 shader buckets"), "{err}");
        assert!(err.contains("64 Shaders and 2 grass variants"), "{err}");
        let fits = BucketPlan::new(&over[..MAX_SHADER_BUCKETS - 2]).unwrap();
        assert_eq!(fits.sources.len(), MAX_SHADER_BUCKETS);
        // Every Shader with a family, at the cap.
        let families = vec![shader(PlanetSky, false); MAX_SHADER_BUCKETS / 5];
        let plan = BucketPlan::new(&families).unwrap();
        assert_eq!(plan.sources.len(), MAX_SHADER_BUCKETS / 5 * 5);
    }

    // The table builds every bucket past the default except deferred ones and
    // remembers which variant each draws with.
    #[test]
    fn the_table_built_from_a_plan_keeps_its_variants() {
        let plan = BucketPlan::new(&[
            shader(Grass, false),
            shader(Grass, true),
            shader(NoGrass, false),
        ])
        .unwrap();
        let mut built = Vec::new();
        let t = WorldPipelines::build(&plan, |bucket, source| {
            built.push((bucket, source.variant));
            Ok::<_, ()>(bucket)
        })
        .unwrap();
        assert_eq!(
            built,
            [(2, Plain), (3, CanopyDaylight), (4, GroundDaylight),]
        );
        assert_eq!(t.bucket_count(), 7);
        assert_eq!(t.resident_buckets().collect::<Vec<_>>(), [2, 3, 4]);
        assert_eq!(
            t.variants(),
            [
                Plain,
                Plain,
                Plain,
                CanopyDaylight,
                GroundDaylight,
                CanopyDaylight,
                GroundDaylight
            ]
        );
        assert_eq!(t.variant(99), Plain);
    }

    #[test]
    fn a_plain_table_has_no_variants() {
        let t = table();
        assert_eq!(t.variants(), [Plain; 4]);
        assert_eq!(t.routes(), ShaderRoutes::default());
    }

    fn draw(shader: u32, far_field: u32) -> DrawObject {
        let mut obj = crate::test_support::draw_object();
        obj.shader_bucket = shader;
        obj.material.far_field = crate::gfx::render_types::FarFieldTint::of(far_field);
        obj
    }

    // A grass layer packed as the canopy block's first layer: a far band.
    const GRASS: u32 = 1 << 8;
    // A ground shading from the block with no layer: a planet's covers.
    const COVERS: u32 = 1 << 16;

    // A draw shading its ground takes its Shader's family, a draw growing grass
    // marks its far band, and any other draw keeps its Shader's bucket.
    #[test]
    fn a_draw_routes_by_its_ground_and_its_shader() {
        let plan = BucketPlan::new(&[shader(PlanetSky, false), shader(NoGrass, false)]).unwrap();
        let t = table_of(&plan);
        let routes = t.routes();
        let bits = |obj: &DrawObject| routes.draw_bits(obj);
        use crate::gfx::render_types::DrawArgsFlags;
        let route_of = |b: u32| (b >> DrawArgsFlags::ROUTE_SHIFT) & DrawArgsFlags::ROUTE_MASK;
        assert_eq!(route_of(bits(&draw(0, GRASS))), 1);
        assert_ne!(bits(&draw(0, GRASS)) & DrawArgsFlags::FAR_BAND, 0);
        assert_eq!(route_of(bits(&draw(0, COVERS))), 1);
        assert_eq!(bits(&draw(0, COVERS)) & DrawArgsFlags::FAR_BAND, 0);
        assert_eq!(bits(&draw(0, 0)), 0);
        assert_eq!(bits(&draw(1, GRASS)), 1 << DrawArgsFlags::BUCKET_SHIFT);
        assert_eq!(t.candidates(&draw(0, GRASS)), 2..6);
        assert_eq!(t.candidates(&draw(1, GRASS)), 1..2);
        assert_eq!(t.candidates(&draw(0, 0)), 0..1);
        assert_eq!(
            t.candidates(&draw(9, 0)),
            5..6,
            "clamped to the last bucket"
        );
    }

    // Each bucket past the default spans the build-time draws that can route to
    // it, then the runtime reserve; the default spans every record. Regions
    // never overlap, and every record lands in each of its candidates'
    // regions exactly once.
    #[test]
    fn regions_hold_each_record_in_its_candidates_once() {
        let plan = BucketPlan::new(&[
            shader(PlanetSky, false),
            shader(NoGrass, false),
            shader(Grass, false),
        ])
        .unwrap();
        let t = table_of(&plan);
        let draws = [
            draw(0, 0),
            draw(1, 0),
            draw(0, GRASS),
            draw(2, 0),
            draw(1, 0),
            draw(2, COVERS),
            draw(0, 0),
        ];
        let runtime = 10..14;
        let records = 16;
        let spans = t.bucket_spans(&draws, runtime.clone(), records);
        assert_eq!(spans[0].records, 0..records);
        assert_eq!(spans[1].records, 1..5);
        assert_eq!(spans[2].records, 3..4, "its covers take the family");
        assert_eq!(spans[3].records, 2..3, "the default's family");
        assert_eq!(spans[8].records, 5..6, "Shader 2's family");
        for s in &spans[1..] {
            assert_eq!(s.runtime, runtime);
        }
        let block = t.cull_variants(&spans);
        assert_regions_sound(&block.regions[..t.bucket_count()], records);
        for (record, obj) in draws.iter().enumerate() {
            for b in t.candidates(obj) {
                let r = &block.regions[b as usize];
                assert!(
                    r.command(u32::saturating_from(record)).is_some(),
                    "{record} in {b}"
                );
            }
        }
        for record in runtime {
            for r in &block.regions[..t.bucket_count()] {
                assert!(r.command(u32::saturating_from(record)).is_some());
            }
        }
    }

    // No two regions share a command, a region's commands are its spans'
    // records one to one, and the regions pack back to back.
    fn assert_regions_sound(regions: &[crate::gfx::render_types::GpuBucketRegion], records: usize) {
        let mut seen = alloc::collections::BTreeSet::new();
        let mut next = 0;
        for (b, r) in regions.iter().enumerate() {
            assert_eq!(r.offset, next, "bucket {b} packs after the last");
            next += r.len();
            let mut mine = Vec::new();
            for record in 0..u32::saturating_from(records) {
                if let Some(c) = r.command(record) {
                    assert!(seen.insert(c), "command {c} is taken twice");
                    mine.push(c);
                }
            }
            assert_eq!(
                mine.len(),
                r.len() as usize,
                "bucket {b}: one command per record"
            );
        }
    }

    // At the cap, with every bucket spanning the whole build-time list and the
    // reserve, the regions still never overlap and their total is what the
    // indirect buffers are sized by.
    #[test]
    fn regions_at_the_cap_never_overlap() {
        let shaders = vec![shader(NoGrass, false); MAX_SHADER_BUCKETS];
        let t = table_of(&BucketPlan::new(&shaders).unwrap());
        let draws: Vec<DrawObject> = (0..MAX_SHADER_BUCKETS * 3)
            .map(|i| draw(u32::saturating_from(i % MAX_SHADER_BUCKETS), 0))
            .collect();
        let records = draws.len() + 40;
        let runtime = draws.len()..draws.len() + 32;
        let spans = t.bucket_spans(&draws, runtime, records);
        let block = t.cull_variants(&spans);
        assert_regions_sound(&block.regions, records);
        let total = crate::gfx::render_types::region_commands(&block.regions, MAX_SHADER_BUCKETS);
        let per_bucket = 2 * MAX_SHADER_BUCKETS + 1 + 32;
        assert_eq!(total, records + (MAX_SHADER_BUCKETS - 1) * per_bucket);
    }
}
