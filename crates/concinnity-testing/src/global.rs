//! The one guard over the process-global state a test may touch.
//!
//! Cargo runs a binary's tests in parallel threads of one process, so anything
//! process-wide -- the working directory, the installed state root, the window
//! policy -- is shared by every test running at that moment. Each of those is
//! also something a test needs to move.
//!
//! Taking them separately is what goes wrong: one test's chdir lands under
//! another's relative write, and a state root cleared at the end of one test
//! leaves the next without one. This guard owns the lock and every move made
//! under it, and puts all of them back on drop, however the test ends.

use std::path::{Path, PathBuf};

use crate::TempTree;
use crate::access::{self, ExclusiveAccess};

/// Exclusive use of the process-global state, for as long as the value lives.
///
/// Acquiring it serialises against every other holder in the process and gives
/// the test a private [`TempTree`] to point that state at.
///
/// ```
/// # use concinnity_testing::GlobalState;
/// let state = GlobalState::acquire().with_cwd();
/// // A relative write now lands in the guard's tree, not the source checkout.
/// std::fs::write("world-lock.json", "{}").unwrap();
/// # assert!(state.root().join("world-lock.json").exists());
/// ```
pub struct GlobalState {
    // Dropped last, and only through the explicit `Drop` below: the tree is
    // removed while the lock is still held, so no other test observes a
    // half-removed root.
    _access: ExclusiveAccess,
    tree: TempTree,
    previous_cwd: Option<PathBuf>,
    clear_state_dir: Option<fn()>,
    restore_windows: bool,
}

impl GlobalState {
    /// Take exclusive access and open a private tree. Nothing global has moved
    /// yet.
    ///
    /// This is [`access::exclusive`] plus a tree, so it excludes every reader
    /// and every other writer in the binary. A test that only needs to read a
    /// global should take [`access::shared`] instead and stay parallel.
    pub fn acquire() -> Self {
        Self {
            _access: access::exclusive(),
            tree: TempTree::new(),
            previous_cwd: None,
            clear_state_dir: None,
            restore_windows: false,
        }
    }

    /// Enter the tree as the working directory, restoring the old one on drop.
    ///
    /// What a cwd-relative path resolves against for the life of the guard.
    ///
    /// # Panics
    ///
    /// If the current directory cannot be read or changed.
    #[must_use]
    pub fn with_cwd(mut self) -> Self {
        let previous = std::env::current_dir().expect("the working directory is readable");
        std::env::set_current_dir(self.tree.path()).expect("the tree is entered");
        self.previous_cwd = Some(previous);
        self
    }

    /// Point an installed state root at the tree, clearing it on drop.
    ///
    /// The root lives in a crate this one does not depend on, so the caller
    /// passes the pair that installs and clears it:
    ///
    /// ```ignore
    /// GlobalState::acquire().with_state_dir(
    ///     |path| concinnity_host::store::paths::set_state_dir(path),
    ///     concinnity_host::store::paths::clear_state_dir,
    /// )
    /// ```
    #[must_use]
    pub fn with_state_dir(mut self, install: fn(&Path), clear: fn()) -> Self {
        install(self.tree.path());
        self.clear_state_dir = Some(clear);
        self
    }

    /// Forbid window creation for the life of the guard.
    ///
    /// A backend reached under this panics naming itself, instead of standing
    /// up a window and blocking on an event loop the harness cannot end.
    #[must_use]
    pub fn without_windows(mut self) -> Self {
        concinnity_core::window_policy::forbid_windows();
        self.restore_windows = true;
        self
    }

    /// The tree the global state points at.
    pub fn tree(&self) -> &TempTree {
        &self.tree
    }

    /// The root of that tree.
    pub fn root(&self) -> &Path {
        self.tree.path()
    }
}

impl Drop for GlobalState {
    fn drop(&mut self) {
        // Leave the tree before it is removed: a working directory that no
        // longer exists is a state some hosts refuse to remove from.
        if let Some(previous) = self.previous_cwd.take() {
            let _ = std::env::set_current_dir(previous);
        }
        if let Some(clear) = self.clear_state_dir.take() {
            clear();
        }
        if self.restore_windows {
            concinnity_core::window_policy::allow_windows();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn a_relative_write_lands_in_the_tree_and_the_cwd_comes_back() {
        let before = std::env::current_dir().expect("cwd");
        let root = {
            let state = GlobalState::acquire().with_cwd();
            std::fs::write("relative.txt", "x").expect("write");
            assert!(state.root().join("relative.txt").exists());
            state.root().to_path_buf()
        };

        assert_eq!(std::env::current_dir().expect("cwd"), before);
        assert!(!root.exists(), "the tree went with the guard");
    }

    static INSTALLED: AtomicUsize = AtomicUsize::new(0);
    static CLEARED: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn a_state_root_is_installed_then_cleared() {
        INSTALLED.store(0, Ordering::SeqCst);
        CLEARED.store(0, Ordering::SeqCst);

        {
            let _state = GlobalState::acquire().with_state_dir(
                |_| {
                    INSTALLED.fetch_add(1, Ordering::SeqCst);
                },
                || {
                    CLEARED.fetch_add(1, Ordering::SeqCst);
                },
            );
            assert_eq!(INSTALLED.load(Ordering::SeqCst), 1);
            assert_eq!(CLEARED.load(Ordering::SeqCst), 0);
        }

        assert_eq!(CLEARED.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn windows_are_forbidden_only_while_the_guard_lives() {
        {
            let _state = GlobalState::acquire().without_windows();
            assert!(concinnity_core::window_policy::windows_forbidden());
        }

        assert!(!concinnity_core::window_policy::windows_forbidden());
    }

    #[test]
    fn a_poisoned_lock_is_still_acquired() {
        let panicked = std::panic::catch_unwind(|| {
            let _state = GlobalState::acquire();
            panic!("poison the lock");
        });
        assert!(panicked.is_err());

        // The point: this does not deadlock or unwrap-panic on the poison.
        let state = GlobalState::acquire();
        assert!(state.root().is_dir());
    }
}
