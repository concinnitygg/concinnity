//! The frames a `screenshot` action saves: a burst of consecutive presented
//! frames, each named for the moment it shows.

use alloc::format;
use alloc::string::{String, ToString};
use core::fmt;

use crate::camera_track::CameraTrackStatus;
use crate::ecs::PipelineContext;
use crate::render::snapshot::RenderSnapshot;

/// A screenshot action still saving frames.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenshotBurst {
    dir: String,
    // When the action fired, shared by every frame of the burst.
    stamp: String,
    total: u32,
    taken: u32,
}

impl ScreenshotBurst {
    /// A burst of `frames` frames into `dir`, each named after `stamp`, or
    /// `None` for no frames.
    pub fn start(dir: &str, stamp: String, frames: u32) -> Option<Self> {
        (frames > 0).then(|| Self {
            dir: dir.to_string(),
            stamp,
            total: frames,
            taken: 0,
        })
    }

    // The path to save the presented frame `frame` to, or `None` once every
    // frame of the burst has been asked for.
    fn next(&mut self, frame: u64, track_seconds: Option<f32>) -> Option<String> {
        if self.taken == self.total {
            return None;
        }
        self.taken += 1;
        let shot = (self.total > 1).then_some((self.taken, self.total));
        let name = file_name(&self.stamp, shot, frame, track_seconds);
        Some(format!("{}/{name}", self.dir))
    }

    fn done(&self) -> bool {
        self.taken == self.total
    }
}

// Hand the burst's next frame to `snap` unless the frame already captures for
// something else; the burst ends with its last frame.
pub(super) fn schedule(
    burst: &mut Option<ScreenshotBurst>,
    ctx: &PipelineContext<'_>,
    snap: &mut RenderSnapshot,
    frame: u64,
) {
    if snap.capture.is_some() {
        return;
    }
    let Some(shots) = burst.as_mut() else {
        return;
    };
    let track = ctx
        .resource::<CameraTrackStatus>()
        .map(|status| status.elapsed_seconds);
    snap.capture = shots.next(frame, track);
    if shots.done() {
        *burst = None;
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
    fmt::from_fn(|name| {
        name.write_str(stamp)?;
        if let Some((index, count)) = shot {
            let width = count.to_string().len();
            write!(name, "-{index:0width$}of{count}")?;
        }
        write!(name, "-frame{frame}")?;
        if let Some(seconds) = track_seconds {
            write!(name, "-track{seconds:.2}s")?;
        }
        name.write_str(".png")
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::World;
    use alloc::vec::Vec;

    fn burst(frames: u32) -> ScreenshotBurst {
        ScreenshotBurst::start("shots", "20261016-142305".to_string(), frames).unwrap()
    }

    fn scheduled(
        shots: &mut Option<ScreenshotBurst>,
        world: &mut World,
        frame: u64,
    ) -> Option<String> {
        let mut snap = RenderSnapshot::default();
        schedule(shots, &world.context(), &mut snap, frame);
        snap.capture
    }

    #[test]
    fn a_burst_captures_its_frames_in_a_row_then_ends() {
        let mut world = World::new();
        let mut shots = Some(burst(3));
        let paths: Vec<_> = (10..15)
            .map(|frame| scheduled(&mut shots, &mut world, frame))
            .collect();
        for (i, path) in paths[..3].iter().enumerate() {
            let path = path.as_deref().unwrap();
            assert!(
                path.ends_with(&format!("-{}of3-frame{}.png", i + 1, 10 + i)),
                "{path}"
            );
        }
        assert_eq!(paths[3..], [None, None]);
        assert!(shots.is_none());
    }

    #[test]
    fn a_frame_already_capturing_defers_the_burst_a_frame() {
        let mut world = World::new();
        let mut shots = Some(burst(1));
        let mut snap = RenderSnapshot {
            capture: Some("report.png".to_string()),
            ..Default::default()
        };
        schedule(&mut shots, &world.context(), &mut snap, 4);
        assert_eq!(snap.capture.as_deref(), Some("report.png"));
        let next = scheduled(&mut shots, &mut world, 5).unwrap();
        assert!(next.ends_with("-frame5.png"), "{next}");
    }

    #[test]
    fn a_running_track_names_its_time() {
        let mut world = World::new();
        world.insert_resource(CameraTrackStatus {
            elapsed_seconds: 148.4,
            ..Default::default()
        });
        let mut shots = Some(burst(1));
        let path = scheduled(&mut shots, &mut world, 2).unwrap();
        assert!(path.ends_with("-frame2-track148.40s.png"), "{path}");
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
