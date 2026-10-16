// The frames a `screenshot` action saves: a burst of consecutive presented
// frames, each named for the moment it shows.

use concinnity_core::camera_track::CameraTrackStatus;
use concinnity_core::ecs::PipelineContext;
use concinnity_core::render::snapshot::RenderSnapshot;

use super::GraphicsSystem;
use crate::crash::UtcTime;
use crate::ecs::ScreenshotRequest;

// Where screenshots land, relative to the working directory.
const SCREENSHOT_DIR: &str = "screenshots";

impl GraphicsSystem {
    // Start the burst a `screenshot` action asked for, and hand its next frame
    // to `snap` unless the frame already captures for something else.
    pub(super) fn schedule_screenshot(
        &mut self,
        ctx: &mut PipelineContext,
        snap: &mut RenderSnapshot,
    ) {
        self.schedule_screenshot_in(SCREENSHOT_DIR, ctx, snap);
    }

    fn schedule_screenshot_in(
        &mut self,
        dir: &str,
        ctx: &mut PipelineContext,
        snap: &mut RenderSnapshot,
    ) {
        if let Some(ScreenshotRequest(frames)) = ctx.resources.remove::<ScreenshotRequest>() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                tracing::warn!("screenshot: cannot create {dir}: {e}");
            }
            self.screenshot = ScreenshotBurst::start(dir, UtcTime::now().stem(), frames);
        }
        if snap.capture.is_some() {
            return;
        }
        let Some(burst) = self.screenshot.as_mut() else {
            return;
        };
        let track = ctx
            .resource::<CameraTrackStatus>()
            .map(|status| status.elapsed_seconds);
        snap.capture = burst.next(self.frame_count, track);
        if burst.done() {
            self.screenshot = None;
        }
    }
}

// A screenshot action still saving frames.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ScreenshotBurst {
    dir: String,
    // When the action fired, shared by every frame of the burst.
    stamp: String,
    total: u32,
    taken: u32,
}

impl ScreenshotBurst {
    // A burst of `frames` frames into `dir`, or `None` for no frames.
    pub(super) fn start(dir: &str, stamp: String, frames: u32) -> Option<Self> {
        (frames > 0).then(|| Self {
            dir: dir.to_string(),
            stamp,
            total: frames,
            taken: 0,
        })
    }

    // The path to save the presented frame `frame` to, or `None` once every
    // frame of the burst has been asked for.
    pub(super) fn next(&mut self, frame: u64, track_seconds: Option<f32>) -> Option<String> {
        if self.taken == self.total {
            return None;
        }
        self.taken += 1;
        let shot = (self.total > 1).then_some((self.taken, self.total));
        let name = file_name(&self.stamp, shot, frame, track_seconds);
        Some(format!("{}/{name}", self.dir))
    }

    // Whether every frame has been asked for.
    pub(super) fn done(&self) -> bool {
        self.taken == self.total
    }
}

// `<stamp>[-<i>of<n>]-frame<frame>[-track<seconds>s].png`. The burst index is
// padded to the width of its count so a burst's files sort in order.
fn file_name(
    stamp: &str,
    shot: Option<(u32, u32)>,
    frame: u64,
    track_seconds: Option<f32>,
) -> String {
    let mut name = stamp.to_string();
    if let Some((index, count)) = shot {
        let width = count.to_string().len();
        name.push_str(&format!("-{index:0width$}of{count}"));
    }
    name.push_str(&format!("-frame{frame}"));
    if let Some(seconds) = track_seconds {
        name.push_str(&format!("-track{seconds:.2}s"));
    }
    name.push_str(".png");
    name
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::ecs::World;

    // Extract one frame's screenshot into a snapshot, as frame `frame`.
    fn scheduled(
        gs: &mut GraphicsSystem,
        world: &mut World,
        dir: &str,
        frame: u64,
    ) -> Option<String> {
        let mut snap = RenderSnapshot::default();
        gs.frame_count = frame;
        gs.schedule_screenshot_in(dir, &mut world.context(), &mut snap);
        snap.capture
    }

    #[test]
    fn a_request_captures_its_frames_in_a_row_into_a_made_directory() {
        let tree = concinnity_testing::TempTree::new();
        let dir = tree.join("shots");
        let dir = dir.to_str().unwrap();
        let mut world = World::new();
        let mut gs = GraphicsSystem::new(None);
        assert_eq!(scheduled(&mut gs, &mut world, dir, 0), None);
        world.insert_resource(ScreenshotRequest(3));
        let paths: Vec<_> = (10..15)
            .map(|frame| scheduled(&mut gs, &mut world, dir, frame))
            .collect();
        assert!(std::path::Path::new(dir).is_dir());
        let prefix = format!("{dir}/");
        for (i, path) in paths[..3].iter().enumerate() {
            let path = path.as_deref().unwrap();
            assert!(path.starts_with(&prefix), "{path}");
            assert!(
                path.ends_with(&format!("-{}of3-frame{}.png", i + 1, 10 + i)),
                "{path}"
            );
        }
        assert_eq!(paths[3..], [None, None]);
    }

    #[test]
    fn a_frame_already_capturing_defers_the_burst_a_frame() {
        let tree = concinnity_testing::TempTree::new();
        let dir = tree.join("shots");
        let dir = dir.to_str().unwrap();
        let mut world = World::new();
        let mut gs = GraphicsSystem::new(None);
        world.insert_resource(ScreenshotRequest(1));
        let mut snap = RenderSnapshot {
            capture: Some("report.png".to_string()),
            ..Default::default()
        };
        gs.schedule_screenshot_in(dir, &mut world.context(), &mut snap);
        assert_eq!(snap.capture.as_deref(), Some("report.png"));
        let next = scheduled(&mut gs, &mut world, dir, 5).unwrap();
        assert!(next.ends_with("-frame5.png"), "{next}");
    }

    #[test]
    fn a_running_track_names_its_time() {
        let tree = concinnity_testing::TempTree::new();
        let dir = tree.join("shots");
        let dir = dir.to_str().unwrap();
        let mut world = World::new();
        world.insert_resource(CameraTrackStatus {
            elapsed_seconds: 148.4,
            ..Default::default()
        });
        world.insert_resource(ScreenshotRequest(1));
        let mut gs = GraphicsSystem::new(None);
        let path = scheduled(&mut gs, &mut world, dir, 2).unwrap();
        assert!(path.ends_with("-frame2-track148.40s.png"), "{path}");
    }

    fn burst(frames: u32) -> ScreenshotBurst {
        ScreenshotBurst::start("shots", "20261016-142305".to_string(), frames).unwrap()
    }

    #[test]
    fn no_frames_starts_no_burst() {
        assert_eq!(ScreenshotBurst::start("shots", String::new(), 0), None);
    }

    #[test]
    fn a_single_screenshot_names_its_frame_and_track_time() {
        let mut shot = burst(1);
        assert_eq!(
            shot.next(1234, Some(148.4)).as_deref(),
            Some("shots/20261016-142305-frame1234-track148.40s.png")
        );
        assert!(shot.done());
        assert_eq!(shot.next(1235, Some(148.42)), None);
    }

    #[test]
    fn without_a_track_the_name_leaves_the_time_out() {
        let mut shot = burst(1);
        assert_eq!(
            shot.next(7, None).as_deref(),
            Some("shots/20261016-142305-frame7.png")
        );
    }

    #[test]
    fn a_burst_asks_for_each_of_its_frames_once_in_a_row() {
        let mut shots = burst(8);
        let paths: Vec<String> = (100..120)
            .map_while(|frame| shots.next(frame, None))
            .collect();
        assert_eq!(paths.len(), 8);
        for (i, path) in paths.iter().enumerate() {
            let expected = format!("shots/20261016-142305-{}of8-frame{}.png", i + 1, 100 + i);
            assert_eq!(path, &expected);
        }
        assert!(shots.done());
        assert_eq!(shots.next(120, None), None);
    }

    #[test]
    fn a_long_burst_pads_its_index_so_the_files_sort() {
        assert_eq!(
            file_name("s", Some((3, 12)), 9, None),
            "s-03of12-frame9.png"
        );
        let mut names: Vec<String> = (1..=12)
            .map(|i| file_name("s", Some((i, 12)), 0, None))
            .collect();
        let in_order = names.clone();
        names.sort();
        assert_eq!(names, in_order);
    }
}
