//! One distance-scored streaming pool of numbered items: a texture pool's slots
//! or the streamed meshes. The [`StreamPlanner`] decides what loads and evicts;
//! this scores each item by its nearest draw and moves loads through the pool's
//! [`Mailbox`].

use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::HashSet;

use super::decoded_mesh::DecodedMesh;
use super::{LoadPool, Mailbox};
use crate::bake::texture::TextureImage;
use crate::render::streaming::{StreamPlanner, StreamState};

/// What a pool streams: a name for its logs and the bytes one resident item
/// occupies.
pub trait StreamedItem {
    /// The pool's name in its logs.
    const KIND: &'static str;

    /// Device bytes this item occupies once resident.
    fn resident_bytes(&self) -> u64;
}

impl StreamedItem for TextureImage {
    const KIND: &'static str = "texture";

    fn resident_bytes(&self) -> u64 {
        self.byte_len() as u64
    }
}

impl StreamedItem for DecodedMesh {
    const KIND: &'static str = "mesh";

    fn resident_bytes(&self) -> u64 {
        self.byte_len()
    }
}

/// A finished load: the item's id and its decoded payload, or why it failed.
pub type ItemLoad<T> = (usize, Result<T, String>);

impl<T: StreamedItem> LoadPool for StreamPool<T> {
    type Request = usize;
    type Result = ItemLoad<T>;

    fn mailbox(&mut self) -> &mut Mailbox<usize, ItemLoad<T>> {
        &mut self.mail
    }

    // The item returns to `Unloaded`.
    fn request_refused(&mut self, id: usize) {
        self.planner.mark_unloaded(id);
    }

    fn load_failed(&mut self, id: usize) {
        retire::<T>(&mut self.planner, &mut self.failed, id, "the load panicked");
    }
}

// Retire pending item `id`, whose load cannot succeed: it is never requested
// again, and the pool's next `drain_failed` reports it.
fn retire<T: StreamedItem>(
    planner: &mut StreamPlanner,
    failed: &mut Vec<usize>,
    id: usize,
    why: &str,
) {
    if planner.state(id) == Some(StreamState::Pending) {
        tracing::warn!(
            "{} stream: load of {} {} failed for good: {}",
            T::KIND,
            T::KIND,
            id,
            why
        );
        planner.mark_failed(id);
        failed.push(id);
    }
}

/// Drives one streaming pool. Each frame the drive calls
/// [`update_scores`](Self::update_scores), [`plan`](Self::plan) and
/// [`drain_completed`](Self::drain_completed) in that order.
pub struct StreamPool<T> {
    planner: StreamPlanner,
    // centers[id] holds the world-space positions of every draw that uses item
    // `id`; its priority is the squared distance from the camera to the
    // nearest of them.
    centers: Vec<Vec<[f32; 3]>>,
    // Items that never load, whatever scene residency asks.
    excluded: HashSet<usize>,
    mail: Mailbox<usize, ItemLoad<T>>,
    // Items whose load failed for good since the last `drain_failed`.
    failed: Vec<usize>,
}

impl<T: StreamedItem> StreamPool<T> {
    /// A pool of `centers.len()` items. `load_budget` caps loads requested per
    /// frame; `resident_cap` caps how many stay resident at once before LRU
    /// eviction.
    pub fn new(centers: Vec<Vec<[f32; 3]>>, load_budget: usize, resident_cap: usize) -> Self {
        Self {
            planner: StreamPlanner::new(centers.len(), load_budget, resident_cap),
            centers,
            excluded: HashSet::new(),
            mail: Mailbox::default(),
            failed: Vec::new(),
        }
    }

    /// Number of streamed items.
    pub fn len(&self) -> usize {
        self.planner.len()
    }

    /// Whether the pool streams nothing.
    pub fn is_empty(&self) -> bool {
        self.planner.len() == 0
    }

    /// Set (or clear with `None`) the resident-byte budget. When set, the
    /// planner evicts farther items to hold resident bytes at or under it, on
    /// top of the item-count cap.
    pub fn set_byte_budget(&mut self, budget: Option<u64>) {
        self.planner.set_byte_budget(budget);
    }

    /// Total resident bytes.
    pub fn resident_bytes(&self) -> u64 {
        self.planner.resident_bytes()
    }

    /// The active resident-byte budget, or `None` when byte accounting is off.
    pub fn byte_budget(&self) -> Option<u64> {
        self.planner.byte_budget()
    }

    /// Block or unblock an item for scene residency: a blocked item never
    /// loads and is evicted by the next plan if resident. An excluded item
    /// stays blocked.
    pub fn set_blocked(&mut self, id: usize, blocked: bool) {
        if !blocked && self.excluded.contains(&id) {
            return;
        }
        self.planner.set_blocked(id, blocked);
    }

    /// Keep `id` from ever loading: nothing uses it.
    pub fn exclude(&mut self, id: usize) {
        self.excluded.insert(id);
        self.planner.set_blocked(id, true);
    }

    /// Where item `id` stands, or `None` past the pool.
    pub fn state(&self, id: usize) -> Option<StreamState> {
        self.planner.state(id)
    }

    /// Re-score every item from the camera position and refresh the LRU
    /// timestamp of resident ones.
    pub fn update_scores(&mut self, camera: [f32; 3], frame: u64) {
        for id in 0..self.planner.len() {
            self.planner
                .set_score(id, nearest_sq_distance(&self.centers[id], camera));
            if self.planner.state(id) == Some(StreamState::Resident) {
                self.planner.touch(id, frame);
            }
        }
    }

    /// Run the planner: request this frame's loads and return the items the
    /// caller must evict from the GPU.
    pub fn plan(&mut self) -> Vec<usize> {
        let plan = self.planner.plan();
        for &id in &plan.to_load {
            self.mail.request(id);
        }
        plan.to_evict
    }

    /// Apply every finished load through `upload`, which takes the decoded
    /// payload by value so it can carry it into a recorded backend op. A
    /// failed fetch retires the item, as a panicked load does. Returns the
    /// items brought resident.
    pub fn drain_completed(&mut self, frame: u64, mut upload: impl FnMut(usize, T)) -> usize {
        let mut applied = 0;
        for (id, decoded) in self.mail.take_results() {
            match decoded {
                Ok(item) => {
                    let bytes = item.resident_bytes();
                    upload(id, item);
                    self.planner.mark_resident(id, frame, bytes);
                    applied += 1;
                }
                Err(e) => retire::<T>(&mut self.planner, &mut self.failed, id, &e),
            }
        }
        applied
    }

    /// Hand `on_failed` each item whose load failed for good since the last
    /// call.
    pub fn drain_failed(&mut self, on_failed: impl FnMut(usize)) {
        self.failed.drain(..).for_each(on_failed);
    }

    /// Roll an item whose recorded upload was refused back to `Unloaded`, so
    /// the planner requests it again. Unloading removes its bytes from the
    /// resident sum.
    pub fn note_upload_failed(&mut self, id: usize) {
        tracing::debug!(
            "{} stream: upload of {} {} deferred, will retry",
            T::KIND,
            T::KIND,
            id
        );
        self.planner.mark_unloaded(id);
    }

    /// Restore an item whose recorded eviction failed: it is still on the GPU,
    /// so its bytes count toward the resident sum again.
    pub fn note_evict_failed(&mut self, id: usize, frame: u64) {
        self.planner.restore_resident(id, frame);
    }

    /// `(resident, pending, unloaded)` item counts.
    pub fn stats(&self) -> (usize, usize, usize) {
        self.planner.counts()
    }
}

// Squared distance from `camera` to the nearest position in `centers`. Ordering
// is all the planner needs, so no square root. An item with no draws scores 0
// so it still streams in promptly rather than stalling forever.
fn nearest_sq_distance(centers: &[[f32; 3]], camera: [f32; 3]) -> f32 {
    let mut nearest = f32::MAX;
    for c in centers {
        let dx = c[0] - camera[0];
        let dy = c[1] - camera[1];
        let dz = c[2] - camera[2];
        let d = dx * dx + dy * dy + dz * dz;
        if d < nearest {
            nearest = d;
        }
    }
    if centers.is_empty() { 0.0 } else { nearest }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    fn image() -> TextureImage {
        TextureImage::rgba8(1, 1, vec![1, 2, 3, 4])
    }

    // Answer every requested load at once, as a worker that never waits.
    fn serve(
        pool: &mut StreamPool<TextureImage>,
        load: impl Fn(usize) -> Result<TextureImage, String>,
    ) {
        let requests: Vec<usize> = pool.mailbox().take_requests().collect();
        for id in requests {
            pool.mailbox().deliver((id, load(id)));
        }
    }

    fn frame(pool: &mut StreamPool<TextureImage>, n: u64) -> Vec<usize> {
        pool.update_scores([0.0; 3], n);
        let evicted = pool.plan();
        serve(pool, |_| Ok(image()));
        pool.drain_completed(n, |_, _| {});
        evicted
    }

    #[test]
    fn nearest_sq_distance_picks_the_closest_center() {
        let centers = [[10.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]];
        assert_eq!(nearest_sq_distance(&centers, [0.0, 0.0, 0.0]), 9.0);
    }

    #[test]
    fn nearest_sq_distance_of_no_centers_is_zero() {
        assert_eq!(nearest_sq_distance(&[], [5.0, 5.0, 5.0]), 0.0);
    }

    #[test]
    fn the_nearest_items_load_first_within_the_budget() {
        let centers = vec![
            vec![[100.0, 0.0, 0.0]],
            vec![[2.0, 0.0, 0.0]],
            vec![[50.0, 0.0, 0.0]],
        ];
        let mut pool = StreamPool::<TextureImage>::new(centers, 1, 8);
        assert_eq!(pool.len(), 3);
        pool.update_scores([0.0; 3], 1);
        assert!(pool.plan().is_empty());
        assert_eq!(pool.mailbox().take_requests().collect::<Vec<_>>(), [1]);
        pool.update_scores([0.0; 3], 2);
        pool.plan();
        assert_eq!(pool.mailbox().take_requests().collect::<Vec<_>>(), [2]);
    }

    // An excluded item never loads, not even when scene residency unblocks it.
    #[test]
    fn an_excluded_item_never_loads() {
        let mut pool = StreamPool::<TextureImage>::new(vec![vec![], vec![[2.0, 0.0, 0.0]]], 4, 8);
        pool.exclude(0);
        pool.set_blocked(0, false);
        for n in 1..4 {
            frame(&mut pool, n);
        }
        assert_eq!(pool.state(0), Some(StreamState::Unloaded));
        assert_eq!(pool.state(1), Some(StreamState::Resident));
    }

    // A failed fetch retires the item: it holds no bytes, is reported once,
    // and neither a replan nor an eviction pass brings it back.
    #[test]
    fn a_failed_fetch_is_retired() {
        let mut pool = StreamPool::<TextureImage>::new(vec![vec![[1.0, 0.0, 0.0]]], 4, 8);
        pool.update_scores([0.0; 3], 1);
        pool.plan();
        serve(&mut pool, |_| Err("undecodable payload".to_string()));
        assert_eq!(pool.drain_completed(1, |_, _| {}), 0);
        assert_eq!(pool.state(0), Some(StreamState::Failed));
        assert_eq!(pool.stats(), (0, 0, 1));
        assert_eq!(pool.resident_bytes(), 0);
        let mut reported = Vec::new();
        pool.drain_failed(|id| reported.push(id));
        assert_eq!(reported, [0]);

        pool.set_blocked(0, true);
        assert!(pool.plan().is_empty(), "nothing resident to evict");
        pool.set_blocked(0, false);
        pool.update_scores([0.0; 3], 2);
        pool.plan();
        assert!(!pool.mailbox().has_requests());
        assert_eq!(pool.state(0), Some(StreamState::Failed));
    }

    #[test]
    fn upload_failure_rolls_back_to_unloaded_for_retry() {
        let mut pool = StreamPool::<TextureImage>::new(vec![vec![[1.0, 0.0, 0.0]]], 4, 8);
        frame(&mut pool, 1);
        pool.note_upload_failed(0);
        assert_eq!(pool.stats(), (0, 0, 1));
        frame(&mut pool, 2);
        assert_eq!(pool.stats().0, 1);
    }

    #[test]
    fn evict_failure_restores_the_item_to_resident() {
        let mut pool = StreamPool::<TextureImage>::new(vec![vec![[1.0, 0.0, 0.0]]], 4, 8);
        frame(&mut pool, 1);
        let resident_bytes = pool.resident_bytes();
        assert!(resident_bytes > 0);
        pool.set_blocked(0, true);
        assert_eq!(pool.plan(), vec![0]);
        assert_eq!(pool.resident_bytes(), 0);
        pool.note_evict_failed(0, 2);
        assert_eq!(pool.stats(), (1, 0, 0));
        assert_eq!(pool.resident_bytes(), resident_bytes);
    }

    // A refused request returns the item to the plan rather than pending.
    #[test]
    fn a_refused_request_is_requested_again() {
        let mut pool = StreamPool::<TextureImage>::new(vec![vec![[1.0, 0.0, 0.0]]], 4, 8);
        pool.update_scores([0.0; 3], 1);
        pool.plan();
        let refused: Vec<usize> = pool.mailbox().take_requests().collect();
        for id in refused {
            pool.request_refused(id);
        }
        assert_eq!(pool.stats(), (0, 0, 1));
        pool.plan();
        assert_eq!(pool.mailbox().take_requests().collect::<Vec<_>>(), [0]);
    }

    #[test]
    fn the_upload_receives_the_decoded_payload() {
        let mut pool = StreamPool::<DecodedMesh>::new(vec![vec![[1.0, 0.0, 0.0]]], 4, 8);
        pool.update_scores([0.0; 3], 1);
        pool.plan();
        let requests: Vec<usize> = pool.mailbox().take_requests().collect();
        let mesh = DecodedMesh {
            vertices: Vec::new(),
            indices: vec![0, 1, 2],
        };
        for id in requests {
            pool.mailbox().deliver((id, Ok(mesh.clone())));
        }
        let mut seen = None;
        pool.drain_completed(1, |id, m| seen = Some((id, m.indices)));
        assert_eq!(seen, Some((0, vec![0, 1, 2])));
        assert_eq!(pool.resident_bytes(), 6);
    }
}
