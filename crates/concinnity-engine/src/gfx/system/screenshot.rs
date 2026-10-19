// The I/O half of a `screenshot` action: the burst core's extraction names and
// schedules starts here, with its directory made and its time stamp taken.

use concinnity_core::ecs::{PipelineContext, ScreenshotRequest};
use concinnity_core::render::extract::{FrameExtractor, ScreenshotBurst};

use crate::crash::utc_now;

// Where screenshots land, relative to the working directory.
const SCREENSHOT_DIR: &str = "screenshots";

// Start the burst a `screenshot` action asked for; the frames extracted from
// here on capture it.
pub(super) fn start_requested(ctx: &mut PipelineContext, extractor: &mut FrameExtractor) {
    start_requested_in(SCREENSHOT_DIR, ctx, extractor);
}

fn start_requested_in(dir: &str, ctx: &mut PipelineContext, extractor: &mut FrameExtractor) {
    let Some(ScreenshotRequest(frames)) = ctx.resources.remove::<ScreenshotRequest>() else {
        return;
    };
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!("screenshot: cannot create {dir}: {e}");
    }
    extractor.start_screenshots(ScreenshotBurst::start(dir, utc_now().stem(), frames));
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::ecs::World;
    use concinnity_core::render::snapshot::RenderSnapshot;

    #[test]
    fn a_request_captures_its_frames_in_a_row_into_a_made_directory() {
        let tree = concinnity_testing::TempTree::new();
        let dir = tree.join("shots");
        let dir = dir.to_str().unwrap();
        let mut world = World::new();
        let mut extractor = FrameExtractor::default();
        start_requested_in(dir, &mut world.context(), &mut extractor);
        world.insert_resource(ScreenshotRequest(2));
        start_requested_in(dir, &mut world.context(), &mut extractor);
        assert!(world.resource::<ScreenshotRequest>().is_none());
        assert!(std::path::Path::new(dir).is_dir());
        let prefix = format!("{dir}/");
        for frame in [10, 11] {
            let mut snap = RenderSnapshot::default();
            extractor.extract(&mut world.context(), &mut snap, frame);
            let path = snap.capture.expect("a burst frame");
            assert!(path.starts_with(&prefix), "{path}");
            assert!(path.ends_with(&format!("of2-frame{frame}.png")), "{path}");
        }
    }
}
