// src/editor/hook/cinematic_drive.rs
//
// EditorHook: the start screen's attract camera. While the sidebar is
// previewing a world, a cycle of slow shots (`editor/worlds/cinematic.rs`)
// takes the view: it frames the previewed world's renderable bounds, writes a
// pose onto the live `Camera3D` each frame, and draws the black fade its shots
// hand over through. Everything here is wiring -- the frame dt, the bounds, the
// camera, and when the cycle runs at all.
//
// It is a start-screen presentation and nothing more. The world's own pose is
// held from the frame the cycle takes over and put straight back when the
// screen hands the session a world, so the editing session opens on the camera
// the world declared. Nothing here touches the authored entries, so no shot can
// reach the world file or the session store.

use super::*;
use crate::components::Camera3D;
use framing::CameraPose;
use worlds::cinematic::{Cinematic, Framing};

impl EditorHook {
    // Advance the cycle and write its pose. Runs after the frame's routing, so
    // a click that picked another world has already restarted the cycle.
    pub(super) fn drive_cinematic(&mut self, world: &mut World) {
        if !self.start_mode {
            return self.reset_cinematic();
        }
        // A preview still owed a rebuild is showing the world on its way out,
        // so the cycle waits at its opening black rather than framing it. The
        // pose it held goes back onto that world first: a compile that fails
        // leaves it standing, and it must stand on its own camera.
        if self.rebuild_preview {
            self.stop_cinematic(world);
            self.cinematic = Some(Cinematic::new());
            return;
        }
        // The world's own pose, held before anything is written over it: the
        // spin shot turns on the spot from it, and it is what opening hands
        // back.
        if self.cinematic_restore.is_none() {
            self.cinematic_restore = camera_pose::read(world);
        }
        let Some(home) = self.cinematic_restore else {
            // No camera to take, so nothing was taken.
            return self.reset_cinematic();
        };
        let Some(framing) = self.cinematic_framing(world, home) else {
            // Nothing to frame (an empty world, or one whose camera is driven
            // by what it follows). A cycle that had already taken the camera
            // gives it back rather than leaving a shot standing on it.
            return self.stop_cinematic(world);
        };
        let dt = self.cinematic_dt();
        let pose = {
            let cine = self.cinematic.get_or_insert_with(Cinematic::new);
            cine.advance(dt);
            cine.pose(&framing)
        };
        camera_pose::write(world, &pose);
    }

    // Draw the fade at this moment's alpha, or hide it. Left out of the panel
    // layer bands, so it covers the previewed world without ever dimming the
    // sidebar listing over it.
    pub(super) fn drive_cinematic_draw(&self, world: &mut World, vp: [f32; 2], shown: bool) {
        // The loading cover owns the same black while it is up, and it leaves
        // the sidebar out of it, so the fade stands down rather than dimming
        // the listing through a compile.
        match self.cinematic {
            Some(cine) if shown && !self.loading_preview() => {
                worlds::cinematic::apply(world, vp, cine.fade_alpha())
            }
            _ => worlds::cinematic::hide(world),
        }
    }

    // End the cycle, handing the world back the pose it had when the cycle
    // took over. For the commit path: the session adopts the running world, so
    // its camera has to be the world's own again before it does.
    pub(super) fn stop_cinematic(&mut self, world: &mut World) {
        if let Some(pose) = self.cinematic_restore.take() {
            camera_pose::write(world, &pose);
        }
        self.reset_cinematic();
    }

    // Drop the cycle without restoring: the world it was framing is gone, so
    // the pose held for it no longer describes anything.
    pub(super) fn reset_cinematic(&mut self) {
        self.cinematic = None;
        self.cinematic_clock = None;
        self.cinematic_restore = None;
    }

    // Start the cycle over, keeping the pose it holds: the world it was
    // framing is being replaced, and the drive hands that pose back to it
    // before the rebuild runs, in case the rebuild fails and it stays.
    pub(super) fn restart_cinematic(&mut self) {
        self.cinematic = None;
        self.cinematic_clock = None;
    }

    // What the shots frame this frame, or `None` when the preview gets no
    // cinematic at all. The bounds are the engine's own: every renderable prop
    // GraphicsSystem indexed this frame, folded the way the backends fold a
    // scene's bounds for a probe bake.
    fn cinematic_framing(&self, world: &World, home: CameraPose) -> Option<Framing> {
        let vp = self.viewport;
        if vp[0] <= 0.0 || vp[1] <= 0.0 {
            return None;
        }
        let cam = world.query::<Camera3D>().next()?;
        // A third-person camera is placed by the character it follows, every
        // step, so a pose written here would be gone before the frame drew.
        if cam.controller.as_ref().is_some_and(|c| c.follow.is_some()) {
            return None;
        }
        let fov = cam.fov_y_degrees.to_radians();
        let index = world.resource::<crate::ecs::PickIndex>()?;
        let (mn, mx) = concinnity_core::render::reflection_probe::fold_world_bounds(
            index.entries.iter().map(|e| (e.bb_min, e.bb_max)),
        )?;
        Framing::new(mn, mx, fov, vp[0] / vp[1], home)
    }

    // This frame's dt for the shot clock. The first frame of a cycle takes no
    // time, so the cycle always opens on its own first moment.
    fn cinematic_dt(&mut self) -> f32 {
        let now = std::time::Instant::now();
        let dt = self
            .cinematic_clock
            .map(|last| now.duration_since(last).as_secs_f32())
            .unwrap_or(0.0);
        self.cinematic_clock = Some(now);
        dt
    }
}
