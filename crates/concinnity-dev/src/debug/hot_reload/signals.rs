//! The reload requests a dev session raises for the per-frame reload passes.
//! The `HotReloadDriver` owns one `ReloadSignals` for the session and shares it
//! with the asset watcher (on the `notify` thread) and the `reload-assets` tool
//! call (on the debug server's socket thread); `super::state::run_frame` and
//! `super::animation` take each request on the frame thread.

use concinnity_core::ecs::asset_id::AssetId;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

// Which subjects of one catalog a reload was asked for.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Pending<K> {
    // Every subject in the catalog, whatever `ids` holds.
    pub all: bool,
    pub ids: BTreeSet<K>,
}

pub(crate) type PendingShaders = Pending<AssetId>;
pub(crate) type PendingSdfVolumes = Pending<String>;

impl<K> Pending<K> {
    pub(crate) fn is_empty(&self) -> bool {
        !self.all && self.ids.is_empty()
    }
}

impl<K> Default for Pending<K> {
    fn default() -> Self {
        Self {
            all: false,
            ids: BTreeSet::new(),
        }
    }
}

impl<K: Ord> Pending<K> {
    // Whether the subject `id` is asked for.
    pub(crate) fn wants<Q>(&self, id: &Q) -> bool
    where
        K: std::borrow::Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.all || self.ids.contains(id)
    }
}

// One flag or set per reload pass, so a save only kicks the pass that serves it.
#[derive(Debug, Default)]
pub(crate) struct ReloadSignals {
    // File-backed texture / LUT / environment map / mesh payloads to decode again.
    assets: AtomicBool,
    // world.jsonl changed: regenerate ProceduralMeshes and re-apply VolumetricFog.
    world: AtomicBool,
    // Animation source changed: re-import every file-backed clip.
    animations: AtomicBool,
    // Markdown story source changed: re-expand the world's StoryImports.
    stories: AtomicBool,
    // The world Shaders to recompile, by asset id.
    shaders: Mutex<PendingShaders>,
    // The SdfVolumes whose field to recompile, by name.
    sdf_volumes: Mutex<PendingSdfVolumes>,
}

fn lock<K>(pending: &Mutex<Pending<K>>) -> MutexGuard<'_, Pending<K>> {
    pending.lock().unwrap_or_else(PoisonError::into_inner)
}

impl ReloadSignals {
    pub(crate) fn request_assets(&self) {
        self.assets.store(true, Ordering::SeqCst);
    }

    // Clear the asset request, returning whether it was raised.
    pub(crate) fn take_assets(&self) -> bool {
        self.assets.swap(false, Ordering::SeqCst)
    }

    pub(crate) fn request_world(&self) {
        self.world.store(true, Ordering::SeqCst);
    }

    pub(crate) fn take_world(&self) -> bool {
        self.world.swap(false, Ordering::SeqCst)
    }

    pub(crate) fn request_animations(&self) {
        self.animations.store(true, Ordering::SeqCst);
    }

    pub(crate) fn take_animations(&self) -> bool {
        self.animations.swap(false, Ordering::SeqCst)
    }

    pub(crate) fn request_stories(&self) {
        self.stories.store(true, Ordering::SeqCst);
    }

    pub(crate) fn take_stories(&self) -> bool {
        self.stories.swap(false, Ordering::SeqCst)
    }

    // Mark the Shaders `ids` for recompiling.
    pub(crate) fn mark_shaders(&self, ids: impl IntoIterator<Item = AssetId>) {
        lock(&self.shaders).ids.extend(ids);
    }

    // Mark the SdfVolumes `names` for recompiling.
    pub(crate) fn mark_sdf_volumes(&self, names: impl IntoIterator<Item = String>) {
        lock(&self.sdf_volumes).ids.extend(names);
    }

    // Take the pending Shader set, leaving it empty.
    pub(crate) fn take_shaders(&self) -> PendingShaders {
        std::mem::take(&mut *lock(&self.shaders))
    }

    // Take the pending SdfVolume set, leaving it empty.
    pub(crate) fn take_sdf_volumes(&self) -> PendingSdfVolumes {
        std::mem::take(&mut *lock(&self.sdf_volumes))
    }

    // Ask for every asset payload, animation clip, world.jsonl pass, Shader and
    // SdfVolume field. Stories reload only on their own `.md` watch.
    pub(crate) fn request_all(&self) {
        self.request_assets();
        self.request_animations();
        self.request_world();
        lock(&self.shaders).all = true;
        lock(&self.sdf_volumes).all = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_flag_round_trips_on_its_own() {
        let signals = ReloadSignals::default();
        signals.request_world();
        assert!(!signals.take_stories());
        assert!(!signals.take_assets());
        assert!(signals.take_world());
        assert!(!signals.take_world());

        signals.request_stories();
        assert!(signals.take_stories());
        assert!(!signals.take_stories());

        signals.request_assets();
        signals.request_animations();
        assert!(signals.take_assets());
        assert!(signals.take_animations());
        assert!(!signals.take_animations());
    }

    #[test]
    fn pending_shaders_collect_marks_until_taken() {
        let signals = ReloadSignals::default();
        assert!(signals.take_shaders().is_empty());
        signals.mark_shaders([AssetId(3), AssetId(1)]);
        signals.mark_shaders([AssetId(3)]);
        let taken = signals.take_shaders();
        assert!(!taken.all);
        assert_eq!(
            taken.ids.into_iter().collect::<Vec<_>>(),
            [AssetId(1), AssetId(3)]
        );
        assert!(signals.take_shaders().is_empty());

        signals.mark_sdf_volumes(["cloud".to_string()]);
        assert!(signals.take_shaders().is_empty(), "the sets are separate");
        assert!(signals.take_sdf_volumes().wants("cloud"));
        assert!(signals.take_sdf_volumes().is_empty());
    }

    #[test]
    fn request_all_asks_for_every_pass_but_stories() {
        let signals = ReloadSignals::default();
        signals.request_all();
        assert!(signals.take_assets());
        assert!(signals.take_animations());
        assert!(signals.take_world());
        assert!(signals.take_shaders().wants(&AssetId(99)));
        assert!(signals.take_sdf_volumes().wants("blob"));
        assert!(!signals.take_stories());
        assert!(signals.take_shaders().is_empty());
        assert!(signals.take_sdf_volumes().is_empty());
    }

    #[test]
    fn a_pending_set_wants_only_what_it_names_unless_all() {
        let some = PendingShaders {
            all: false,
            ids: [AssetId(2)].into(),
        };
        assert!(some.wants(&AssetId(2)));
        assert!(!some.wants(&AssetId(5)));
        assert!(!some.is_empty());
        assert!(PendingShaders::default().is_empty());
    }
}
