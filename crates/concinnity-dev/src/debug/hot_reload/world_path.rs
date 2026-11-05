//! The world.jsonl path a dev session is running, shared between the host that
//! switches worlds and the hot-reload driver that watches the file.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

// A cloneable handle to the session's world.jsonl path. Every clone sees the
// latest `set`, so a driver re-arming after a world switch watches the new file.
#[derive(Clone, Debug)]
pub(crate) struct WorldPathHandle(Arc<Mutex<PathBuf>>);

impl WorldPathHandle {
    pub(crate) fn new(path: impl Into<PathBuf>) -> Self {
        Self(Arc::new(Mutex::new(path.into())))
    }

    pub(crate) fn set(&self, path: impl Into<PathBuf>) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = path.into();
    }

    pub(crate) fn get(&self) -> PathBuf {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::WorldPathHandle;

    #[test]
    fn a_clone_sees_a_later_set() {
        let handle = WorldPathHandle::new("a.jsonl");
        let clone = handle.clone();
        handle.set("b.jsonl");
        assert_eq!(clone.get(), std::path::Path::new("b.jsonl"));
    }
}
