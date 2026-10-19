//! Which data blobs init frees once graphics has read its payloads.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::components::{sdf_volume, terrain};
use crate::ecs::PipelineContext;
use crate::resource::AudioClipTable;

/// Blobs another consumer still reads after graphics has consumed its own payloads.
pub fn retained_blobs(ctx: &PipelineContext<'_>) -> BTreeSet<u32> {
    // Audio reads clip payloads from the table after graphics init.
    let mut retained = ctx
        .resource::<AudioClipTable>()
        .map(AudioClipTable::blob_indices)
        .unwrap_or_default();
    // SdfVolume payloads are drained later in this same init.
    retained.extend(sdf_volume::sdf_volume_blob_indices(ctx));
    // Physics reads each terrain's grid after graphics init.
    retained.extend(terrain::terrain_blob_indices(ctx));
    retained
}

/// The consumed blobs to release, each once in first-seen order, sparing retained ones.
pub fn blobs_to_release(
    consumed: impl IntoIterator<Item = u32>,
    retained: &BTreeSet<u32>,
) -> Vec<u32> {
    let mut seen = BTreeSet::new();
    consumed
        .into_iter()
        .filter(|idx| !retained.contains(idx) && seen.insert(*idx))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn a_duplicate_blob_is_released_once_in_first_seen_order() {
        let released = blobs_to_release([3, 1, 3, 2, 1], &BTreeSet::new());
        assert_eq!(released, vec![3, 1, 2]);
    }

    #[test]
    fn a_retained_blob_is_never_released() {
        let retained = BTreeSet::from([1, 4]);
        let released = blobs_to_release([1, 2, 4, 2, 5], &retained);
        assert_eq!(released, vec![2, 5]);
    }
}
