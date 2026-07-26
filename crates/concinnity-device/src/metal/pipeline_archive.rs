// Persistent GPU-binary cache for the world shader-bucket pipelines.
//
// A scene that pins a Shader builds its render pipeline mid-session, so the
// driver compiles that shader from AIR to GPU code while the loading screen is
// up. An MTLBinaryArchive keeps the compiled result under the engine's state
// directory, so later runs of the same world load the binary instead of
// recompiling it.
//
// The archive is a cache in every sense: a miss (first run, an edited shader, a
// new OS or GPU) just compiles as usual, and deleting the file is always safe.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSString, NSURL};
use objc2_metal::{
    MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLDevice, MTLRenderPipelineDescriptor,
};

// File name under the writable state directory. Versioned so a future change
// to what the engine records lands on a fresh file instead of a stale one.
const ARCHIVE_FILE: &str = "pipelines-v1.metallib-archive";

pub(super) struct PipelineArchive {
    archive: Retained<ProtocolObject<dyn MTLBinaryArchive>>,
    url: Retained<NSURL>,
    // Set by `record`, cleared by `flush`: the file is rewritten only when this
    // session actually added something.
    dirty: bool,
}

impl PipelineArchive {
    // Open the cache, loading any binaries a previous run left behind. `None`
    // when the device or the state directory will not have it, in which case
    // every pipeline compiles from AIR as it did before.
    pub(super) fn open(device: &ProtocolObject<dyn MTLDevice>) -> Option<Self> {
        let path = concinnity_core::paths::writable_state_dir().join(ARCHIVE_FILE);
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            tracing::debug!("Metal: pipeline archive directory unavailable: {}", e);
            return None;
        }
        let url = file_url(&path.to_string_lossy());

        // A descriptor carrying an existing file loads its contents; one with a
        // url the device rejects (missing, stale format) fails, and the retry
        // below starts the cache over from empty.
        let desc = MTLBinaryArchiveDescriptor::new();
        if path.exists() {
            desc.setUrl(Some(&url));
        }
        let archive = match device.newBinaryArchiveWithDescriptor_error(&desc) {
            Ok(a) => a,
            Err(e) => {
                tracing::debug!("Metal: pipeline archive not reusable ({e:?}); starting fresh");
                desc.setUrl(None);
                device.newBinaryArchiveWithDescriptor_error(&desc).ok()?
            }
        };
        archive.setLabel(Some(&NSString::from_str("concinnity world pipelines")));
        Some(Self {
            archive,
            url,
            dirty: false,
        })
    }

    // Point a pipeline descriptor at the cache, so building it looks for a
    // stored binary before compiling.
    pub(super) fn attach(&self, desc: &MTLRenderPipelineDescriptor) {
        let archives = NSArray::from_slice(&[&*self.archive]);
        desc.setBinaryArchives(Some(&archives));
    }

    // Store the binary for a pipeline that was just built, so the next run
    // finds it. A device that declines to archive this descriptor is logged
    // and otherwise ignored: the pipeline itself is already live.
    pub(super) fn record(&mut self, desc: &MTLRenderPipelineDescriptor) {
        match self
            .archive
            .addRenderPipelineFunctionsWithDescriptor_error(desc)
        {
            Ok(()) => self.dirty = true,
            Err(e) => tracing::debug!("Metal: pipeline archive add failed: {e:?}"),
        }
    }

    // Write the archive back out if this session added to it.
    pub(super) fn flush(&mut self) {
        if !self.dirty {
            return;
        }
        self.dirty = false;
        match self.archive.serializeToURL_error(&self.url) {
            Ok(()) => tracing::debug!("Metal: pipeline archive written"),
            Err(e) => tracing::debug!("Metal: pipeline archive write failed: {e:?}"),
        }
    }
}

fn file_url(path: &str) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&NSString::from_str(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The cache lands under the engine's writable state root, so a shipped app
    // writes it somewhere it actually owns.
    #[test]
    fn archive_lives_in_the_writable_state_dir() {
        let path = concinnity_core::paths::writable_state_dir().join(ARCHIVE_FILE);
        assert!(path.ends_with(ARCHIVE_FILE));
        assert_eq!(
            path.parent(),
            Some(concinnity_core::paths::writable_state_dir().as_path())
        );
    }

    #[test]
    fn file_url_round_trips_a_path() {
        let url = file_url("/tmp/concinnity/archive.bin");
        assert_eq!(
            url.path().map(|p| p.to_string()),
            Some("/tmp/concinnity/archive.bin".to_string())
        );
    }
}
