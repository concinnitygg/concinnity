//! What the clouds do each frame, apart from the GPU half every backend
//! builds: the target size they are kept at, which of the two targets a frame
//! writes, the temporal history its constants reproject through, whether the
//! noise volume still has to be built, and how many threads each kernel runs.
//! Every backend keeps one beside the textures and kernels it owns.

use core::cell::Cell;

use crate::render::environment::atmosphere::{AtmosphereFrame, AtmosphereSettings, AtmosphereView};

use super::{
    CloudHistory, CloudParams, NOISE_SIZE, PANORAMA_SIZE, SHADOW_SIZE, cloud_march_size,
    cloud_target_size,
};

/// The clouds' per-frame schedule. Interior-mutable, so a backend can step it
/// from whichever borrow of its context records the pass.
#[derive(Debug)]
pub struct CloudSchedule {
    target_size: Cell<[u32; 2]>,
    write: Cell<usize>,
    frames: Cell<u64>,
    history: Cell<CloudHistory>,
    noise_built: Cell<bool>,
}

/// How many threads each cloud kernel runs, one per texel it writes. A backend
/// that dispatches threadgroups divides each by the kernel's group size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CloudDispatch {
    /// The noise volume, built once.
    pub noise: [u32; 3],
    /// The shadow map over the ground around the camera.
    pub shadow: [u32; 3],
    /// The panorama around the camera, marched at its own march size.
    pub panorama: [u32; 3],
    /// The march, one texel of each 2x2 block of the target.
    pub march: [u32; 3],
    /// The reconstruction, every texel of the target.
    pub reconstruct: [u32; 3],
}

impl CloudSchedule {
    /// A schedule for the clouds of a scene `scene` pixels across and down.
    pub fn new(scene: [u32; 2]) -> Self {
        Self {
            target_size: Cell::new(cloud_target_size(scene[0], scene[1])),
            write: Cell::new(0),
            frames: Cell::new(0),
            history: Cell::new(CloudHistory::default()),
            noise_built: Cell::new(false),
        }
    }

    /// The size of the targets the march and the reconstruction fill.
    pub fn target_size(&self) -> [u32; 2] {
        self.target_size.get()
    }

    /// The size of the march's texture: one texel of each 2x2 block of the
    /// target's.
    pub fn march_size(&self) -> [u32; 2] {
        cloud_march_size(self.target_size())
    }

    /// Keep the targets sized for a scene `scene` pixels across and down,
    /// returning the new target size when it changed and the backend has to
    /// rebuild its targets.
    pub fn resize(&self, scene: [u32; 2]) -> Option<[u32; 2]> {
        let size = cloud_target_size(scene[0], scene[1]);
        (size != self.target_size.replace(size)).then_some(size)
    }

    /// Step to the next frame: swap the targets and compute its constants for
    /// the clouds of `settings` in `frame`, seen from `view`. `None` when the
    /// frame draws no clouds.
    pub fn advance(
        &self,
        settings: &AtmosphereSettings,
        frame: Option<&AtmosphereFrame>,
        view: &AtmosphereView,
    ) -> Option<CloudParams> {
        self.write.set(1 - self.write.get());
        let index = self.frames.get();
        self.frames.set(index.wrapping_add(1));
        let mut history = self.history.get();
        let params = history.params(settings, frame, view, index);
        self.history.set(history);
        params
    }

    /// The target this frame writes and its composite reads.
    pub fn write(&self) -> usize {
        self.write.get()
    }

    /// The target the previous frame wrote, which this frame's reconstruction
    /// reprojects.
    pub fn previous(&self) -> usize {
        1 - self.write.get()
    }

    /// Whether the noise volume still has to be built, marking it built: true
    /// once, until [`Self::rebuild_noise`].
    pub fn take_noise_build(&self) -> bool {
        !self.noise_built.replace(true)
    }

    /// Have the next frame build the noise volume again, as when its texture
    /// or kernel is rebuilt.
    pub fn rebuild_noise(&self) {
        self.noise_built.set(false);
    }

    /// How many threads each kernel runs this frame.
    pub fn dispatch(&self) -> CloudDispatch {
        let flat = |[w, h]: [u32; 2]| [w, h, 1];
        CloudDispatch {
            noise: [NOISE_SIZE; 3],
            shadow: [SHADOW_SIZE, SHADOW_SIZE, 1],
            panorama: flat(cloud_march_size(PANORAMA_SIZE)),
            march: flat(self.march_size()),
            reconstruct: flat(self.target_size()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Atmosphere;

    // An atmosphere with no cloud layers.
    fn clear_sky() -> AtmosphereSettings {
        AtmosphereSettings::resolve(&Atmosphere::default(), None)
    }

    fn view() -> AtmosphereView {
        AtmosphereView {
            inv_vp: crate::transform::IDENTITY,
            depth_inv_vp: crate::transform::IDENTITY,
            cam_pos: [0.0; 3],
            viewport: [1600.0, 900.0],
            sky_rot: crate::environment::sky::SkyOrientation::IDENTITY_ROWS,
            pre_exposure: 1.0,
            eye_gain: 1.0,
        }
    }

    #[test]
    fn a_frame_writes_the_other_target_than_the_last() {
        let schedule = CloudSchedule::new([1920, 1080]);
        let settings = clear_sky();
        let view = view();
        let first = {
            schedule.advance(&settings, None, &view);
            schedule.write()
        };
        schedule.advance(&settings, None, &view);
        assert_ne!(schedule.write(), first);
        assert_eq!(schedule.previous(), first);
    }

    #[test]
    fn a_frame_without_clouds_has_no_constants() {
        let schedule = CloudSchedule::new([64, 64]);
        let params = schedule.advance(&clear_sky(), None, &view());
        assert!(params.is_none());
    }

    #[test]
    fn the_noise_is_built_once_until_rebuilt() {
        let schedule = CloudSchedule::new([64, 64]);
        assert!(schedule.take_noise_build());
        assert!(!schedule.take_noise_build());
        schedule.rebuild_noise();
        assert!(schedule.take_noise_build());
    }

    #[test]
    fn a_resize_reports_only_a_changed_target() {
        let schedule = CloudSchedule::new([1920, 1080]);
        assert_eq!(schedule.resize([1920, 1080]), None);
        let smaller = schedule
            .resize([1280, 720])
            .expect("a smaller scene resizes");
        assert_eq!(smaller, cloud_target_size(1280, 720));
        assert_eq!(schedule.target_size(), smaller);
        assert_eq!(schedule.resize([1280, 720]), None);
    }

    // The march covers one texel of each 2x2 block, the reconstruction every
    // texel of the target.
    #[test]
    fn the_dispatch_follows_the_target() {
        let schedule = CloudSchedule::new([1920, 1080]);
        let dispatch = schedule.dispatch();
        let [w, h] = schedule.target_size();
        assert_eq!(dispatch.reconstruct, [w, h, 1]);
        assert_eq!(dispatch.march, [w.div_ceil(2), h.div_ceil(2), 1]);
        assert_eq!(dispatch.noise, [NOISE_SIZE; 3]);
        assert_eq!(dispatch.shadow, [SHADOW_SIZE, SHADOW_SIZE, 1]);
        let [pw, ph] = cloud_march_size(PANORAMA_SIZE);
        assert_eq!(dispatch.panorama, [pw, ph, 1]);
    }
}
