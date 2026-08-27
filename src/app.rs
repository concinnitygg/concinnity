//! The application: a world plus the loop that runs it.

#[cfg(feature = "std")]
use std::path::Path;

use crate::World;

// The engine's driver where there is an operating system to drive, the
// headless one where there is not. Both run the world they are handed; what
// the std one adds is the window, the frame pacing, and the wall clock the
// simulation follows.
#[cfg(feature = "std")]
pub(crate) type Inner = concinnity_engine::App;
#[cfg(not(feature = "std"))]
pub(crate) type Inner = concinnity_core::App;

/// A runnable application.
///
/// Built either from a [`World`] assembled in process, or from a world already
/// compiled into a blob file.
///
/// ```no_run
/// # use concinnity::{App, World};
/// App::from_world(World::new()).run().expect("the app runs");
/// ```
pub struct App {
    inner: Inner,
}

impl core::fmt::Debug for App {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("App").finish_non_exhaustive()
    }
}

impl App {
    /// An app that runs `world`.
    pub fn from_world(world: World) -> Self {
        Self {
            inner: Inner::from_world(world.into_inner()),
        }
    }

    /// An app that runs the world compiled into the blob file at `path`, as
    /// written by the `cook` module. Overflow payload blobs are that
    /// file's siblings named by index, so a world written to `data/0` reads
    /// `data/1`, `data/2`, ... beside it.
    ///
    /// What the app writes at runtime -- its settings, save files, crash
    /// reports, and shader caches -- lands beside the world it read, stepping
    /// out of a directory named `data`, so `from_blob("mygame/data/0")` keeps
    /// them under `mygame/`. A world that declares an `AppConfig` with a `home`
    /// chooses the location itself.
    ///
    /// ```no_run
    /// # use concinnity::App;
    /// App::from_blob("data/0")
    ///     .expect("data/0 holds a compiled world")
    ///     .run()
    ///     .expect("the app runs");
    /// ```
    #[cfg(feature = "std")]
    pub fn from_blob(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        concinnity_engine::App::from_blob(path)
            .map(|inner| Self { inner })
            .map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))
    }

    /// Run the app until its window closes, a system stops the world, or the
    /// process is interrupted.
    #[cfg(feature = "std")]
    pub fn run(self) -> std::io::Result<()> {
        self.inner.run()
    }

    /// Run the app until a system stops the world or its last system finishes.
    ///
    /// The `no_std` build has no window to close and no clock to follow: the
    /// world steps on a fixed virtual timestep, as fast as the host can step
    /// it.
    #[cfg(not(feature = "std"))]
    pub fn run(mut self) -> Result<(), concinnity_core::result::CnResult> {
        self.inner.run().map(|_| ())
    }

    #[cfg(test)]
    pub(crate) fn inner_mut(&mut self) -> &mut Inner {
        &mut self.inner
    }
}

#[cfg(all(test, feature = "cook"))]
mod tests {
    use super::App;
    use crate::components::DirectionalLight;
    use crate::cook;

    // The ahead-of-time pair, end to end: what `write_blob` puts on disk is
    // what `from_blob` reads back, with no state-tree anchor in between.
    #[test]
    fn a_written_blob_loads_back_into_a_runnable_app() {
        let dir = std::env::temp_dir().join("concinnity-app-from-blob");
        let primary = dir.join("data").join("0");
        let _ = std::fs::remove_dir_all(&dir);

        cook::world()
            .add(
                "sun",
                DirectionalLight {
                    intensity: 3.5,
                    ..Default::default()
                },
            )
            .write_blob(&primary)
            .expect("the world is written");

        let mut app = App::from_blob(&primary).expect("the written blob loads");
        assert_eq!(app.inner_mut().start(), Ok(()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // A path with no blob behind it is an error naming the file, not a panic
    // and not an empty world that fails later.
    #[test]
    fn a_missing_blob_reports_the_path_it_could_not_read() {
        let missing = std::env::temp_dir()
            .join("concinnity-no-such-blob")
            .join("0");
        let err = App::from_blob(&missing).expect_err("nothing to load");
        assert!(err.to_string().contains("concinnity-no-such-blob"), "{err}");
    }
}
