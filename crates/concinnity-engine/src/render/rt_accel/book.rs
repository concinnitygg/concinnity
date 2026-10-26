use alloc::vec::Vec;
use alloc::{format, vec};
use concinnity_core::math::SaturatingFrom;

use super::plan::{self, EmptyHead, LiveState, RefreshMode, RtStep, RtUpdatePlan};
use crate::gfx::render_types::{DrawObject, InstancedCluster, RtGeomEntry, SkinnedDrawObject};
use crate::render::backend::error::{RenderError, RenderResult};
use crate::render::rt_accel::geom::{
    RtDynamicMode, cluster_geom_entry, geom_entry, models_dirty, skinned_geom_entry,
};
use crate::render::rt_accel::refit::SkinnedShape;
use crate::render::rt_accel::topology::{
    GeomSig, TopologyPlan, participates_in_bvh, plan_topology_refresh, traced_skinned,
};

type Mat4 = [[f32; 4]; 4];

/// The draw objects and instanced clusters an initial build covers.
pub struct SeedSet<'a> {
    /// Indices of the participating draw objects, in BLAS order.
    pub objects: Vec<usize>,
    /// The participating clusters, in BLAS order after the draw objects.
    pub clusters: Vec<&'a InstancedCluster>,
}

impl<'a> SeedSet<'a> {
    /// The resident, real-triangle draw objects (leaving see-through meshes out
    /// when `exclude_seethrough`) and the clusters with geometry and instances.
    pub fn new(
        draw_objects: &[DrawObject],
        clusters: &'a [InstancedCluster],
        exclude_seethrough: bool,
    ) -> Self {
        Self {
            objects: draw_objects
                .iter()
                .enumerate()
                .filter(|(_, o)| participates_in_bvh(o, exclude_seethrough))
                .map(|(i, _)| i)
                .collect(),
            clusters: clusters
                .iter()
                .filter(|c| c.index_count >= 3 && !c.instances.is_empty())
                .collect(),
        }
    }

    // Whether there is no draw or cluster geometry to build over.
    fn is_empty(&self) -> bool {
        self.objects.is_empty() && self.clusters.is_empty()
    }

    /// Whether an initial build would start out spent (see
    /// [`AccelBook::is_spent`]): no draw or cluster geometry, and
    /// `skinned_present` says no skinned geometry exists to join it, visible or
    /// not. Otherwise the build covers the seed, possibly as an empty head the
    /// skinned step fills.
    pub fn builds_nothing(&self, skinned_present: bool) -> bool {
        self.is_empty() && plan::empty_head(false, skinned_present) == EmptyHead::Drop
    }
}

/// The BLAS one TLAS instance references.
#[derive(Debug)]
pub enum InstanceBlas<'a, B> {
    /// Head BLAS `index` (a draw or cluster BLAS), already built.
    Head {
        /// Its position in the head, which is also its position in the array a
        /// TLAS indexes its instances' BLAS by.
        index: usize,
        /// The structure itself.
        blas: &'a B,
    },
    /// Head slot `index`, whose BLAS the refresh in progress builds.
    Fresh {
        /// Its position in the refreshed head.
        index: usize,
    },
    /// The `n`th of this frame's skinned BLAS, which follow the head.
    Skinned {
        /// Its position among the visible skinned objects.
        n: usize,
    },
}

// Manual so the BLAS type itself need not be `Copy`: only a reference to it is held.
impl<B> Clone for InstanceBlas<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Copy for InstanceBlas<'_, B> {}

/// A refresh of the draw BLAS head, planned against the current draw list.
pub struct HeadRefresh {
    indices: Vec<usize>,
    sigs: Vec<GeomSig>,
    plan: TopologyPlan,
}

impl HeadRefresh {
    /// The draw objects the refreshed head covers, in BLAS order.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// The slots that need a fresh BLAS, each with the draw object it covers.
    pub fn fresh_slots(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.plan
            .reuse
            .iter()
            .zip(&self.indices)
            .enumerate()
            .filter(|(_, (reuse, _))| reuse.is_none())
            .map(|(slot, (_, &idx))| (slot, idx))
    }
}

/// One instance of an instanced cluster.
struct ClusterInstance {
    // The cluster's position among the participating clusters.
    cluster: usize,
    model: Mat4,
    geom: RtGeomEntry,
}

// The scene-scaled lists a frame's update fills, kept so their capacity is
// reused from frame to frame.
struct Scratch<I> {
    skinned: Vec<usize>,
    models: Vec<Mat4>,
    shapes: Vec<SkinnedShape>,
    instances: Vec<I>,
    geom: Vec<RtGeomEntry>,
}

impl<I> Scratch<I> {
    // The instance index the next laid-out instance takes.
    fn next_id(&self) -> u32 {
        u32::saturating_from(self.instances.len())
    }

    // Lay out one instance with its geometry-table entry, unless the backend
    // could not make it.
    fn push(&mut self, instance: Option<I>, geom: RtGeomEntry) {
        if let Some(instance) = instance {
            self.instances.push(instance);
            self.geom.push(geom);
        }
    }
}

/// The CPU side of a scene acceleration structure: which draw objects and
/// clusters its BLAS cover and in what order, the transforms its TLAS was built
/// from, and the update policy over them.
///
/// `B` is the backend's BLAS handle and `I` its TLAS instance descriptor. The
/// book owns the BLAS head, one per participating draw object followed by one per
/// cluster, plus an optional tail a backend that keeps its skinned BLAS in the
/// same array appends. It lays out the TLAS instances and the geometry table the
/// trace indexes by instance, so that order is the same on every backend.
pub struct AccelBook<B, I> {
    blas: Vec<B>,
    static_blas_count: usize,
    object_indices: Vec<usize>,
    draw_sigs: Vec<GeomSig>,
    cached_models: Vec<Mat4>,
    clusters: Vec<ClusterInstance>,
    cluster_count: usize,
    has_skinned: bool,
    head_generation: u64,
    albedo_count: u32,
    clock: u64,
    // A refresh failed, so the next update refreshes again.
    refresh_owed: bool,
    // Orphans of a refresh that built no TLAS: the live TLAS may still reference
    // them until the next one publishes.
    parked: Vec<B>,
    scratch: Scratch<I>,
    // The clock the instance fills are timed by, and their total since the
    // last `take_fill_us`.
    now_us: fn() -> u64,
    fill_us: u32,
}

impl<B, I> AccelBook<B, I> {
    /// The book over `blas`, built from `seed`: one per seed object, then one per
    /// seed cluster. `albedo_count` is the shared texture pool's real-texture
    /// count the geometry table resolves texture indices against.
    ///
    /// # Errors
    ///
    /// Returns an error if `blas` does not hold one BLAS per seed object and
    /// cluster.
    pub fn new(
        seed: &SeedSet<'_>,
        blas: Vec<B>,
        draw_objects: &[DrawObject],
        albedo_count: u32,
    ) -> RenderResult<Self> {
        let cluster_count = seed.clusters.len();
        if blas.len() != seed.objects.len() + cluster_count {
            return Err(RenderError::Other(format!(
                "acceleration structure: {} BLAS for {} objects and {cluster_count} clusters",
                blas.len(),
                seed.objects.len(),
            )));
        }
        let objects = || seed.objects.iter().filter_map(|&i| draw_objects.get(i));
        let clusters = seed
            .clusters
            .iter()
            .enumerate()
            .flat_map(|(cluster, c)| {
                c.instances.iter().map(move |&model| ClusterInstance {
                    cluster,
                    model,
                    geom: cluster_geom_entry(c, model, albedo_count),
                })
            })
            .collect();
        Ok(Self {
            static_blas_count: blas.len(),
            blas,
            object_indices: seed.objects.clone(),
            draw_sigs: objects().map(GeomSig::of).collect(),
            cached_models: objects().map(|o| o.model).collect(),
            clusters,
            cluster_count,
            has_skinned: false,
            head_generation: 0,
            albedo_count,
            clock: 0,
            refresh_owed: false,
            parked: Vec::new(),
            scratch: Scratch {
                skinned: Vec::new(),
                models: Vec::new(),
                shapes: Vec::new(),
                instances: Vec::new(),
                geom: Vec::new(),
            },
            now_us: || 0,
            fill_us: 0,
        })
    }

    /// Every BLAS: the head, then the tail.
    pub fn blas(&self) -> &[B] {
        &self.blas
    }

    /// Every BLAS a live TLAS may still reference: [`Self::blas`], then the
    /// orphans parked until a TLAS without them publishes. A trace declares
    /// these resident where the API asks it to.
    pub fn traced_blas(&self) -> impl Iterator<Item = &B> {
        self.blas.iter().chain(&self.parked)
    }

    /// The draw and cluster BLAS.
    pub fn head(&self) -> &[B] {
        &self.blas[..self.static_blas_count.min(self.blas.len())]
    }

    /// How many BLAS the head holds, which is where skinned BLAS indices start.
    pub fn static_blas_count(&self) -> usize {
        self.static_blas_count
    }

    /// The participating draw objects, in BLAS order.
    pub fn object_indices(&self) -> &[usize] {
        &self.object_indices
    }

    /// Whether there is no draw or cluster geometry left.
    pub fn is_empty(&self) -> bool {
        self.static_blas_count == 0
    }

    /// Whether the live TLAS references skinned BLAS.
    pub fn has_skinned(&self) -> bool {
        self.has_skinned
    }

    /// Whether nothing is left to trace and nothing can rejoin, so the backend
    /// drops the BVH ([`EmptyHead::Drop`]): no draw or cluster geometry, no
    /// skinned BLAS published, and `skinned_present` says no skinned geometry
    /// exists to publish.
    pub fn is_spent(&self, skinned_present: bool) -> bool {
        self.is_empty()
            && !self.has_skinned
            && plan::empty_head(false, skinned_present) == EmptyHead::Drop
    }

    /// Follow a change to the shared texture pool's real-texture count.
    pub fn set_albedo_count(&mut self, albedo_count: u32) {
        self.albedo_count = albedo_count;
    }

    /// Time the instance fills by `now_us`, a monotonic clock in microseconds.
    pub fn set_timer(&mut self, now_us: fn() -> u64) {
        self.now_us = now_us;
    }

    /// The microseconds the instance fills took since the last call.
    pub fn take_fill_us(&mut self) -> u32 {
        core::mem::take(&mut self.fill_us)
    }

    // Add the time since `started` to the fill total.
    fn end_fill(&mut self, started: u64) {
        let elapsed = u32::saturating_from((self.now_us)().saturating_sub(started));
        self.fill_us = self.fill_us.saturating_add(elapsed);
    }

    /// Advance the update clock deferred frees are timed against, returning the
    /// new tick.
    pub fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// The update clock's current tick.
    pub fn clock(&self) -> u64 {
        self.clock
    }

    /// Select the skinned objects visible this frame with real triangles, in
    /// skinned-BLAS order. `None` takes none.
    pub fn select_skinned(&mut self, objects: Option<&[SkinnedDrawObject]>) {
        self.scratch.skinned.clear();
        if let Some(objects) = objects {
            self.scratch.skinned.extend(
                objects
                    .iter()
                    .enumerate()
                    .filter(|(_, o)| traced_skinned(o))
                    .map(|(i, _)| i),
            );
        }
    }

    /// The skinned objects selected this frame, as indices into the skinned
    /// draw list.
    pub fn visible_skinned(&self) -> &[usize] {
        &self.scratch.skinned
    }

    /// The first half of this frame's update, selecting the visible skinned
    /// objects (see [`Self::select_skinned`]). Plans a refresh when the draw set
    /// changed or a refresh failed since the last one succeeded. `None` when
    /// `mode` never updates.
    pub fn plan(
        &mut self,
        mode: RtDynamicMode,
        topology_dirty: bool,
        skinned: Option<&[SkinnedDrawObject]>,
    ) -> Option<RtUpdatePlan> {
        self.select_skinned(skinned);
        // A mode that never updates leaves the debt for the next one that does.
        let owed = mode.is_dynamic() && core::mem::take(&mut self.refresh_owed);
        plan::plan_update(
            mode,
            topology_dirty || owed,
            !self.scratch.skinned.is_empty(),
        )
    }

    /// The refresh this update planned failed; the next update plans another.
    pub fn owe_refresh(&mut self) {
        self.refresh_owed = true;
    }

    /// The second half of this frame's update, once any refresh has committed.
    pub fn next_step(
        &mut self,
        mode: RtDynamicMode,
        plan: &RtUpdatePlan,
        draw_objects: &[DrawObject],
    ) -> RtStep {
        let models_current =
            collect_models(&self.object_indices, draw_objects, &mut self.scratch.models);
        let live = LiveState {
            models_current,
            moved: models_current && models_dirty(&self.cached_models, &self.scratch.models),
            has_skinned: self.has_skinned,
        };
        plan::next_step(mode, *plan, live)
    }

    /// The vertex count the deformed buffer spans: up to the highest vertex any
    /// selected skinned object reaches.
    pub fn skinned_vertex_extent(&self, objects: &[SkinnedDrawObject]) -> u64 {
        self.scratch
            .skinned
            .iter()
            .filter_map(|&i| objects.get(i))
            .map(|o| u64::from(o.vertex_base) + o.vertex_count as u64)
            .max()
            .unwrap_or(0)
    }

    /// Record the geometry each selected skinned object's BLAS covers, with
    /// `vertex_extent` as the vertex range (0 for an API whose descriptors pin
    /// no vertex count). Read back with [`Self::skinned_shapes`].
    pub fn fill_skinned_shapes(&mut self, objects: &[SkinnedDrawObject], vertex_extent: u32) {
        let shapes = &mut self.scratch.shapes;
        shapes.clear();
        shapes.extend(
            self.scratch
                .skinned
                .iter()
                .filter_map(|&i| objects.get(i))
                .map(|o| SkinnedShape {
                    index_offset: o.index_offset,
                    index_count: o.index_count,
                    vertex_extent,
                }),
        );
    }

    /// The shapes [`Self::fill_skinned_shapes`] recorded.
    pub fn skinned_shapes(&self) -> &[SkinnedShape] {
        &self.scratch.shapes
    }

    /// Lay out this frame's TLAS instances and geometry table over the head,
    /// then the selected skinned objects when `skinned` is given, keeping the
    /// head's transforms for the commit that follows. `make` builds
    /// one instance from its transform, its instance index (which indexes the
    /// geometry table) and the BLAS it references, or `None` for a BLAS it
    /// cannot reference, which leaves the instance out.
    pub fn fill_instances(
        &mut self,
        draw_objects: &[DrawObject],
        skinned: Option<&[SkinnedDrawObject]>,
        mut make: impl FnMut(Mat4, u32, InstanceBlas<'_, B>) -> Option<I>,
    ) {
        let started = (self.now_us)();
        let Self {
            blas,
            object_indices,
            clusters,
            albedo_count,
            scratch,
            ..
        } = self;
        let blas: &[B] = blas;
        scratch.instances.clear();
        scratch.geom.clear();
        scratch.models.clear();
        let draw_count = object_indices.len();
        for (index, obj) in object_indices
            .iter()
            .enumerate()
            .filter_map(|(index, &i)| Some((index, draw_objects.get(i)?)))
        {
            scratch.models.push(obj.model);
            if let Some(b) = blas.get(index) {
                let id = scratch.next_id();
                scratch.push(
                    make(obj.model, id, InstanceBlas::Head { index, blas: b }),
                    geom_entry(obj, *albedo_count),
                );
            }
        }
        push_clusters(
            clusters,
            draw_count,
            scratch,
            |index| {
                blas.get(index)
                    .map(|b| InstanceBlas::Head { index, blas: b })
            },
            &mut make,
        );
        if let Some(objects) = skinned {
            for (n, obj) in scratch
                .skinned
                .iter()
                .filter_map(|&i| objects.get(i))
                .enumerate()
            {
                let id = u32::saturating_from(scratch.instances.len());
                if let Some(instance) = make(obj.model, id, InstanceBlas::Skinned { n }) {
                    scratch.instances.push(instance);
                    scratch.geom.push(skinned_geom_entry(obj, *albedo_count));
                }
            }
        }
        self.end_fill(started);
    }

    /// The TLAS instances [`Self::fill_instances`] or
    /// [`Self::fill_refresh_instances`] laid out.
    pub fn instances(&self) -> &[I] {
        &self.scratch.instances
    }

    /// The geometry table laid out with [`Self::instances`], one entry each.
    pub fn geom_table(&self) -> &[RtGeomEntry] {
        &self.scratch.geom
    }

    /// Plan a refresh of the draw BLAS head against the current draw list.
    pub fn plan_refresh(
        &self,
        draw_objects: &[DrawObject],
        exclude_seethrough: bool,
        mode: RefreshMode,
    ) -> HeadRefresh {
        let indices: Vec<usize> = draw_objects
            .iter()
            .enumerate()
            .filter(|(_, o)| participates_in_bvh(o, exclude_seethrough))
            .map(|(i, _)| i)
            .collect();
        let sigs: Vec<GeomSig> = indices
            .iter()
            .map(|&i| GeomSig::of(&draw_objects[i]))
            .collect();
        let plan = match mode {
            RefreshMode::Reuse => {
                plan_topology_refresh(&self.object_indices, &self.draw_sigs, &indices, &sigs)
            }
            RefreshMode::RebuildAll => TopologyPlan {
                reuse: vec![None; indices.len()],
                retire: (0..self.object_indices.len()).collect(),
            },
        };
        HeadRefresh {
            indices,
            sigs,
            plan,
        }
    }

    /// Whether `refresh` would leave no draw or cluster geometry.
    pub fn refresh_leaves_nothing(&self, refresh: &HeadRefresh) -> bool {
        refresh.indices.is_empty() && self.cluster_count == 0
    }

    /// Lay out the TLAS instances and geometry table over the head `refresh`
    /// would leave, with current transforms, before it commits.
    pub fn fill_refresh_instances(
        &mut self,
        refresh: &HeadRefresh,
        draw_objects: &[DrawObject],
        mut make: impl FnMut(Mat4, u32, InstanceBlas<'_, B>) -> Option<I>,
    ) {
        let started = (self.now_us)();
        let Self {
            blas,
            object_indices,
            clusters,
            albedo_count,
            scratch,
            ..
        } = self;
        let blas: &[B] = blas;
        scratch.instances.clear();
        scratch.geom.clear();
        let old_draw_count = object_indices.len();
        let draw_count = refresh.indices.len();
        for (index, (&idx, reuse)) in refresh.indices.iter().zip(&refresh.plan.reuse).enumerate() {
            let Some(obj) = draw_objects.get(idx) else {
                continue;
            };
            let target = match reuse {
                Some(k) => match blas.get(*k) {
                    Some(b) => InstanceBlas::Head { index, blas: b },
                    None => continue,
                },
                None => InstanceBlas::Fresh { index },
            };
            let id = scratch.next_id();
            scratch.push(make(obj.model, id, target), geom_entry(obj, *albedo_count));
        }
        push_clusters(
            clusters,
            draw_count,
            scratch,
            |index| {
                let old = old_draw_count + (index - draw_count);
                blas.get(old).map(|b| InstanceBlas::Head { index, blas: b })
            },
            &mut make,
        );
        self.end_fill(started);
    }

    /// Whether `fresh` holds a BLAS for every slot `refresh` builds, and the plan
    /// still matches the head, so [`Self::commit_refresh`] will apply it. Call it
    /// before recording builds the commit must follow.
    ///
    /// # Errors
    ///
    /// Returns an error if `fresh` does not match the plan or the plan no
    /// longer matches the head.
    pub fn check_refresh(&self, refresh: &HeadRefresh, fresh: &[Option<B>]) -> RenderResult<()> {
        let old_draw_count = self.object_indices.len();
        let mismatch = || RenderError::Other("topology refresh: BLAS do not match the plan".into());
        if fresh.len() != refresh.indices.len() {
            return Err(mismatch());
        }
        let mut claimed = vec![false; old_draw_count];
        for (reuse, built) in refresh.plan.reuse.iter().zip(fresh) {
            let valid = match reuse {
                Some(k) => *k < old_draw_count && !core::mem::replace(&mut claimed[*k], true),
                None => built.is_some(),
            };
            if !valid {
                return Err(mismatch());
            }
        }
        Ok(())
    }

    /// Swap in the refreshed head. `fresh[j]` is the BLAS built for slot `j` of
    /// every slot [`HeadRefresh::fresh_slots`] named. The clusters and any tail
    /// keep their BLAS, and the transforms of the refreshed set become the ones
    /// the TLAS was built from. Returns the orphaned draw BLAS, which an
    /// in-flight trace may still reach, for the caller to retire (or
    /// [`Self::park`] until a TLAS that does not reference them publishes).
    ///
    /// A refresh [`Self::check_refresh`] rejects leaves the book unchanged and
    /// hands back the fresh BLAS instead, so a caller that checked first never
    /// sees a failure here.
    pub fn commit_refresh(
        &mut self,
        refresh: HeadRefresh,
        mut fresh: Vec<Option<B>>,
        draw_objects: &[DrawObject],
    ) -> Vec<B> {
        if self.check_refresh(&refresh, &fresh).is_err() {
            return fresh.into_iter().flatten().collect();
        }
        let old_draw_count = self.object_indices.len();

        let mut old = core::mem::take(&mut self.blas);
        let rest = old.split_off(old_draw_count.min(old.len()));
        let mut old_head: Vec<Option<B>> = old.into_iter().map(Some).collect();
        let mut head = Vec::with_capacity(refresh.indices.len() + rest.len());
        for (j, reuse) in refresh.plan.reuse.iter().enumerate() {
            let b = match reuse {
                Some(k) => old_head.get_mut(*k).and_then(Option::take),
                None => fresh.get_mut(j).and_then(Option::take),
            };
            head.extend(b);
        }
        let orphans = old_head.into_iter().flatten().collect();
        head.extend(rest);

        self.blas = head;
        self.static_blas_count = refresh.indices.len() + self.cluster_count;
        self.cached_models = refresh
            .indices
            .iter()
            .filter_map(|&i| draw_objects.get(i))
            .map(|o| o.model)
            .collect();
        self.object_indices = refresh.indices;
        self.draw_sigs = refresh.sigs;
        self.head_generation = self.head_generation.wrapping_add(1);
        orphans
    }

    /// Hold `orphans` until a TLAS that does not reference them publishes: a
    /// refresh that built no TLAS of its own leaves the previous one live.
    pub fn park(&mut self, orphans: Vec<B>) {
        self.parked.extend(orphans);
    }

    /// The orphans [`Self::park`] held, for the caller to retire now that a TLAS
    /// built after them is live.
    pub fn take_parked(&mut self) -> Vec<B> {
        core::mem::take(&mut self.parked)
    }

    /// Which BLAS head the book holds, bumped by every refresh that commits:
    /// the static TLAS instances lay out the same BLAS in the same order for as
    /// long as it holds, which is when a static TLAS can be refit (see
    /// [`TlasRefit`](crate::render::rt_accel::refit::TlasRefit)).
    pub fn head_generation(&self) -> u64 {
        self.head_generation
    }

    /// The TLAS now covers the head alone, built from the transforms
    /// [`Self::fill_instances`] laid out. See [`Self::release_skinned`] for
    /// what comes back.
    pub fn commit_static(&mut self) -> Option<Vec<B>> {
        self.cached_models.clone_from(&self.scratch.models);
        self.release_skinned()
    }

    /// The TLAS no longer references skinned BLAS. When it did before, returns
    /// the tail it held (empty for a backend that keeps no tail): refit state
    /// describing those BLAS is stale, and a BLAS the previous TLAS still
    /// references must outlive the frames tracing it.
    pub fn release_skinned(&mut self) -> Option<Vec<B>> {
        let tail = self
            .blas
            .split_off(self.static_blas_count.min(self.blas.len()));
        core::mem::replace(&mut self.has_skinned, false).then_some(tail)
    }

    /// The TLAS now covers the head plus this frame's skinned BLAS, built from
    /// the transforms [`Self::fill_instances`] laid out.
    pub fn commit_skinned(&mut self) {
        self.cached_models.clone_from(&self.scratch.models);
        self.has_skinned = true;
    }

    /// Replace the tail with `tail`, the skinned BLAS the TLAS references,
    /// returning the BLAS it held.
    pub fn replace_tail(&mut self, tail: impl IntoIterator<Item = B>) -> Vec<B> {
        let old = self
            .blas
            .split_off(self.static_blas_count.min(self.blas.len()));
        self.blas.extend(tail);
        self.has_skinned |= self.blas.len() > self.static_blas_count;
        old
    }

    /// The head `refresh` would leave, in order, before it commits: each kept
    /// BLAS, or the fresh one built for its slot, then the cluster BLAS.
    pub fn refreshed_head<'a>(
        &'a self,
        refresh: &HeadRefresh,
        fresh: &'a [Option<B>],
    ) -> Vec<&'a B> {
        let old_draw_count = self.object_indices.len();
        let draws = refresh
            .plan
            .reuse
            .iter()
            .zip(fresh)
            .filter_map(|(reuse, built)| match reuse {
                Some(k) => self.blas.get(*k),
                None => built.as_ref(),
            });
        let clusters = self
            .blas
            .iter()
            .skip(old_draw_count)
            .take(self.cluster_count);
        draws.chain(clusters).collect()
    }

    /// Carry a value a backend keeps per head BLAS through `refresh`, into
    /// `out` in the order of the head it would leave: each kept BLAS keeps its
    /// value from `old`, each fresh slot `j` takes `fresh(j)`, and the clusters
    /// keep theirs. A value `old` lacks is taken from `fresh` too.
    pub fn remap_head<T: Copy>(
        &self,
        refresh: &HeadRefresh,
        old: &[T],
        mut fresh: impl FnMut(usize) -> T,
        out: &mut Vec<T>,
    ) {
        out.clear();
        for (j, reuse) in refresh.plan.reuse.iter().enumerate() {
            out.push(match reuse.and_then(|k| old.get(k)) {
                Some(&v) => v,
                None => fresh(j),
            });
        }
        let old_draw_count = self.object_indices.len();
        for c in 0..self.cluster_count {
            let j = out.len();
            out.push(match old.get(old_draw_count + c) {
                Some(&v) => v,
                None => fresh(j),
            });
        }
    }

    /// Every BLAS the book holds, parked ones included, for teardown.
    pub fn drain_blas(&mut self) -> impl Iterator<Item = B> + '_ {
        self.static_blas_count = 0;
        self.blas.append(&mut self.parked);
        self.blas.drain(..)
    }
}

// Append one instance per cluster instance, after `draw_count` draw instances.
// `target(index)` resolves head BLAS `index`; a cluster with no BLAS is skipped.
fn push_clusters<'b, B: 'b, I>(
    clusters: &[ClusterInstance],
    draw_count: usize,
    scratch: &mut Scratch<I>,
    target: impl Fn(usize) -> Option<InstanceBlas<'b, B>>,
    make: &mut impl FnMut(Mat4, u32, InstanceBlas<'b, B>) -> Option<I>,
) {
    for c in clusters {
        if let Some(t) = target(draw_count + c.cluster) {
            let id = scratch.next_id();
            scratch.push(make(c.model, id, t), c.geom);
        }
    }
}

// Re-collect the participating objects' model matrices into `out`, in BLAS
// order. `false` when an index is out of range or no longer resident with real
// triangles: the draw list changed shape, which the topology refresh handles.
fn collect_models(
    object_indices: &[usize],
    draw_objects: &[DrawObject],
    out: &mut Vec<Mat4>,
) -> bool {
    out.clear();
    for &idx in object_indices {
        match draw_objects.get(idx) {
            Some(o) if o.resident && o.index_count >= 3 => out.push(o.model),
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests;
