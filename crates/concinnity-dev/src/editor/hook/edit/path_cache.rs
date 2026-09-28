//! Declared file paths resolved to where they are on disk, kept once found.

use std::collections::HashMap;
use std::path::Path;

// Resolving a bare file name walks the assets tree, so a path that exists is
// kept. One that does not is kept only until `forget_missing`, so a file
// created later is found without walking on every lookup.
#[derive(Debug, Default)]
pub(in crate::editor::hook) struct PathCache {
    found: HashMap<String, String>,
    missing: HashMap<String, String>,
}

impl PathCache {
    // `declared`'s on-disk path, from `resolve` the first time it is asked
    // for or after a miss is forgotten.
    pub(in crate::editor::hook) fn get(
        &mut self,
        declared: &str,
        resolve: impl FnOnce(&str) -> String,
    ) -> String {
        if let Some(path) = self.found.get(declared).or(self.missing.get(declared)) {
            return path.clone();
        }
        let path = resolve(declared);
        let kept = if Path::new(&path).exists() {
            &mut self.found
        } else {
            &mut self.missing
        };
        kept.insert(declared.to_string(), path.clone());
        path
    }

    // Resolve the paths that did not exist again on their next lookup.
    pub(in crate::editor::hook) fn forget_missing(&mut self) {
        self.missing.clear();
    }

    pub(in crate::editor::hook) fn clear(&mut self) {
        self.found.clear();
        self.missing.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A file that exists is resolved once; one that does not is resolved again
    // once its miss is forgotten, which is how a file created later is found.
    #[test]
    fn only_a_path_that_exists_is_kept_past_a_retry() {
        let dir = tempfile::tempdir().unwrap();
        let fallback = dir.path().join("water.hlsl");
        let nested = dir.path().join("shaders").join("water.hlsl");
        let found = |_: &str| {
            if nested.exists() {
                nested.to_string_lossy().into_owned()
            } else {
                fallback.to_string_lossy().into_owned()
            }
        };
        let mut cache = PathCache::default();
        let calls = std::cell::Cell::new(0);
        let get = |cache: &mut PathCache| {
            cache.get("water.hlsl", |d| {
                calls.set(calls.get() + 1);
                found(d)
            })
        };

        assert_eq!(get(&mut cache), fallback.to_string_lossy());
        std::fs::create_dir_all(nested.parent().unwrap()).unwrap();
        std::fs::write(&nested, "").unwrap();
        assert_eq!(
            get(&mut cache),
            fallback.to_string_lossy(),
            "a miss is kept until forgotten"
        );

        cache.forget_missing();
        assert_eq!(get(&mut cache), nested.to_string_lossy());
        cache.forget_missing();
        assert_eq!(get(&mut cache), nested.to_string_lossy());
        assert_eq!(calls.get(), 2, "a found path is not resolved again");

        cache.clear();
        get(&mut cache);
        assert_eq!(calls.get(), 3);
    }
}
