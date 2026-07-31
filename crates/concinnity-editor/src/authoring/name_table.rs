// src/authoring/name_table.rs
//
// Name-table priming for a blob boot. The interner that resolves asset names
// to dense ids is thread-local and normally filled by an in-process cook; a
// process that only LOADS prebuilt blobs (`cn build` then `cn editor`) would
// start with an empty table, leaving every name-keyed feature (picking,
// billboards, the gizmo) silently dead until the first edit rebuilds the
// world. world-lock.json records each asset's name with the id the build
// interned it at, so the table can be reinstalled exactly.

use concinnity_cook::blob::BlobLock;

// The (id, name) pairs a build recorded in its lock: the component defs plus
// the resource-stream assets (both draw ids from the same interner).
fn recorded_names(lock: &BlobLock) -> Vec<(u32, String)> {
    lock.assets
        .iter()
        .filter_map(|a| a.id.map(|id| (id, a.name.clone())))
        .chain(
            lock.resources
                .iter()
                .filter_map(|r| r.id.map(|id| (id, r.name.clone()))),
        )
        .collect()
}

// Prime the interner from the working directory's world-lock.json. Best
// effort: a missing or old-format lock leaves the table as it was (the
// pre-existing degraded behavior), and a table an in-process cook already
// filled is left alone. Returns how many names were installed.
pub(crate) fn prime_from_lock_file() -> std::io::Result<usize> {
    let content = std::fs::read_to_string(concinnity_cook::blob::LOCK_PATH)?;
    let lock: BlobLock = serde_json::from_str(&content)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let pairs = recorded_names(&lock);
    if pairs.is_empty() || !crate::ecs::asset_id::prime_name_table(&pairs) {
        return Ok(0);
    }
    Ok(pairs.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::asset_id;

    fn lock_with(assets: &[(&str, Option<u32>)], resources: &[(&str, Option<u32>)]) -> BlobLock {
        let assets = assets
            .iter()
            .map(|(name, id)| {
                serde_json::json!({
                    "name": name, "id": id, "kind": "Component",
                    "discriminant": 0, "args_hash": "", "payload_blob": null
                })
            })
            .collect::<Vec<_>>();
        let resources = resources
            .iter()
            .map(|(name, id)| {
                serde_json::json!({
                    "name": name, "id": id, "kind": "Texture",
                    "handle": 0, "args_hash": "", "payload_blob": null
                })
            })
            .collect::<Vec<_>>();
        serde_json::from_value(serde_json::json!({
            "engine_version": "0", "built_at": "", "blobs": [],
            "assets": assets, "resources": resources, "injected": []
        }))
        .unwrap()
    }

    // The recorded pairs span both lock sections, and priming them restores
    // the build's exact id -> name mapping.
    #[test]
    fn recorded_names_prime_the_interner_at_their_baked_ids() {
        asset_id::reset_interner();
        let lock = lock_with(
            &[("cam", Some(0)), ("floor", Some(2))],
            &[("brick", Some(1))],
        );
        let pairs = recorded_names(&lock);
        assert!(asset_id::prime_name_table(&pairs));
        assert_eq!(asset_id::intern("cam"), asset_id::AssetId(0));
        assert_eq!(asset_id::intern("brick"), asset_id::AssetId(1));
        assert_eq!(asset_id::intern("floor"), asset_id::AssetId(2));
    }

    // A lock written before ids were recorded primes nothing (and the boot
    // degrades exactly as it did before the fix).
    #[test]
    fn a_lock_without_ids_is_ignored() {
        let lock = lock_with(&[("cam", None)], &[("brick", None)]);
        assert!(recorded_names(&lock).is_empty());
    }
}
