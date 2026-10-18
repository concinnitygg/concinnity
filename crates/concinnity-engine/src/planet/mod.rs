//! The planet system: keeps a planet world simulating around its camera.
//!
//! Each step it reads where the camera is. Once the camera strays
//! `REBASE_DISTANCE` from the simulated frame's origin, it asks the ground
//! worker for a patch around the camera in a frame recentered on it, and when
//! that arrives, moves the world into the new frame: every root transform, the
//! camera and the rigs here, and through the published `FrameRebases` the
//! physics bodies, the streamed terrain tiles and the renderer's temporal
//! history in their own systems. Between moves it keeps the ground patch the
//! physics collides with centered on the camera, and keeps the camera's up
//! pointing away from the planet's center.
//!
//! Scheduled before the world's logic, so a move lands between two ticks:
//! behaviors, spawns, physics, the controllers and this tick's frame all see
//! the world already in its new frame.

mod ground;

use concinnity_core::components::{Camera3D, Planet, SkyRotation};
use concinnity_core::ecs::{MenuActive, PipelineContext, StepResult, System};
use concinnity_core::planet::{
    CameraGeo, FrameRebases, GroundPatch, LocalFrame, PatchPurpose, PlanetFrame, PlanetGravity,
    PlanetGround, Rebase, carry_world, next_patch,
};
use concinnity_core::sky::{SkyFrame, SkyOrientation};

use ground::{GroundWorker, PatchRequest};

/// Keeps a planet world's simulated frame, gravity and ground around its
/// camera. Constructed by `World::start` when the world declares a `Planet`.
#[derive(Debug)]
pub(crate) struct PlanetSystem {
    planet: Planet,
    frame: PlanetFrame,
    rebases: FrameRebases,
    // The patch the physics collides with, by revision.
    ground: Option<PlanetGround>,
    worker: Option<GroundWorker>,
    // The outstanding request: its id and what it is for.
    pending: Option<(u64, PatchPurpose)>,
    next_id: u64,
    // Whether a `SkyRotation` publishes the sky; without one this system does.
    sky_rotates: bool,
}

impl PlanetSystem {
    pub(crate) fn new(planet: Planet) -> Self {
        let shape = planet.shape();
        Self {
            planet,
            frame: PlanetFrame {
                shape,
                frame: LocalFrame::AUTHORED,
            },
            rebases: FrameRebases::default(),
            ground: None,
            worker: None,
            pending: None,
            next_id: 0,
            sky_rotates: false,
        }
    }

    // Publish the frame, the pull, the moves and the sky's frame as they now
    // are.
    fn publish(&self, ctx: &mut PipelineContext) {
        let rotation = self.frame.frame.rotation_from_world();
        ctx.insert_resource(self.frame);
        ctx.insert_resource(PlanetGravity {
            center: self.frame.center(),
            strength: self.planet.gravity,
        });
        ctx.insert_resource(self.rebases.clone());
        ctx.insert_resource(SkyFrame(rotation));
        if !self.sky_rotates {
            ctx.insert_resource(SkyOrientation::default().in_frame(&rotation));
        }
    }

    fn publish_ground(&mut self, ctx: &mut PipelineContext, patch: GroundPatch) {
        let revision = self.ground.as_ref().map_or(0, |g| g.revision + 1);
        let ground = PlanetGround { revision, patch };
        ctx.insert_resource(ground.clone());
        self.ground = Some(ground);
    }

    // Move the world into `frame`, carried there by `rebase`.
    fn rebase(&mut self, ctx: &mut PipelineContext, frame: LocalFrame, rebase: Rebase) {
        let [x, y, z] = frame.origin();
        tracing::info!(
            "PlanetSystem: the simulated frame moved to ({x:.1}, {y:.1}, {z:.1}) on the planet"
        );
        self.frame.frame = frame;
        self.rebases.push(rebase);
        carry_world(ctx, &rebase, &self.frame);
        self.publish(ctx);
    }

    // Take a finished patch: the ground moves, and for a patch built in a new
    // frame, the world moves with it.
    fn receive(&mut self, ctx: &mut PipelineContext) {
        while let Some(result) = self.worker.as_ref().and_then(GroundWorker::try_recv) {
            let Some((id, purpose)) = self.pending else {
                continue;
            };
            if id != result.id {
                continue;
            }
            self.pending = None;
            if let PatchPurpose::Rebase { frame, rebase } = purpose {
                self.rebase(ctx, frame, rebase);
            }
            match result.patch {
                Some(patch) => self.publish_ground(ctx, patch),
                None => tracing::warn!("PlanetSystem: no ground under the camera"),
            }
        }
    }

    // Ask for what the camera needs next, one request at a time.
    fn request(&mut self, camera: [f32; 3]) {
        if self.pending.is_some() {
            return;
        }
        let patch_center = self.ground.as_ref().map(|g| g.patch.center);
        let Some((purpose, frame, around)) = next_patch(camera, &self.frame, patch_center) else {
            return;
        };
        let Some(worker) = &mut self.worker else {
            return;
        };
        let id = self.next_id;
        self.next_id += 1;
        if worker.request(PatchRequest { id, frame, around }) {
            self.pending = Some((id, purpose));
        }
    }
}

fn camera_position(ctx: &PipelineContext) -> Option<[f32; 3]> {
    ctx.query::<Camera3D>().next().map(|c| c.position)
}

// Publish where on the planet the camera is and which way it looks.
fn publish_camera_geo(ctx: &mut PipelineContext, frame: &PlanetFrame) {
    let Some((position, yaw)) = ctx.query::<Camera3D>().next().map(|c| (c.position, c.yaw)) else {
        return;
    };
    ctx.insert_resource(CameraGeo {
        position: frame.geo_at(position),
        heading: frame.heading_for_yaw(position, yaw),
    });
}

impl System for PlanetSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        self.sky_rotates = ctx.query::<SkyRotation>().next().is_some();
        self.publish(ctx);
        // The first patch is built here, so physics finds ground from its
        // first tick.
        let camera = camera_position(ctx).unwrap_or([0.0; 3]);
        let first = PatchRequest {
            id: 0,
            frame: self.frame.frame,
            around: camera,
        };
        match ground::build(&self.frame.shape, &first) {
            Some(patch) => self.publish_ground(ctx, patch),
            None => tracing::warn!("PlanetSystem: the camera starts off the planet"),
        }
        let frame = self.frame;
        for camera in ctx.query_mut::<Camera3D>() {
            camera.up = frame.up_at(camera.position);
            camera.recompose_view();
        }
        self.worker = GroundWorker::spawn(self.frame.shape)
            .map_err(|e| {
                tracing::warn!("PlanetSystem: no ground worker, the ground stays put: {e}");
            })
            .ok();
        self.next_id = 1;
        publish_camera_geo(ctx, &frame);
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        if ctx.resource::<MenuActive>().is_some_and(|m| m.0) {
            return StepResult::Continue;
        }
        self.receive(ctx);
        let Some(camera) = camera_position(ctx) else {
            return StepResult::Continue;
        };
        self.request(camera);
        let frame = self.frame;
        for camera in ctx.query_mut::<Camera3D>() {
            camera.up = frame.up_at(camera.position);
        }
        publish_camera_geo(ctx, &frame);
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::cook::Camera3D as Camera3DArgs;
    use concinnity_core::ecs::World;

    // Init publishes everything physics and the renderer read, with ground
    // under the camera and the camera's up away from the center.
    #[test]
    fn init_publishes_the_frame_gravity_and_ground() {
        let mut world = World::new();
        world.push(Camera3D::bake(Camera3DArgs {
            position: [300.0, 10.0, 0.0],
            ..Default::default()
        }));
        let planet = Planet {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            gravity: 9.0,
            ..Default::default()
        };
        let mut system = PlanetSystem::new(planet);
        system.init(&mut world.context());
        let ctx = world.context();
        let gravity = ctx.resource::<PlanetGravity>().unwrap();
        assert_eq!(gravity.strength, 9.0);
        assert_eq!(gravity.center, [0.0, -50_000.0, 0.0]);
        let ground = ctx.resource::<PlanetGround>().unwrap();
        assert_eq!(ground.revision, 0);
        assert_eq!(ground.patch.center[0], 300.0);
        assert_eq!(ctx.resource::<FrameRebases>().unwrap().count(), 0);
        let up = ctx.query::<Camera3D>().next().unwrap().up;
        assert!(up[0] > 0.0 && up[1] > 0.99, "{up:?}");
        assert!(
            ctx.resource::<SkyOrientation>().is_some(),
            "no SkyRotation, so it is ours"
        );
        // 300 m east of the start and 10 m up: on the equator, a third of a
        // degree east, looking north.
        let geo = ctx.resource::<CameraGeo>().unwrap();
        assert!(geo.position.latitude.abs() < 1e-9, "{geo:?}");
        let east = 300.0_f64.atan2(50_010.0).to_degrees();
        assert!((geo.position.longitude - east).abs() < 1e-6, "{geo:?}");
        assert!((geo.position.altitude - 10.9).abs() < 0.01, "{geo:?}");
        assert!(geo.heading.abs() < 1e-3 || (geo.heading - 360.0).abs() < 1e-3);
    }

    // A camera that strays a kilometer moves the world once the patch for
    // the new frame arrives.
    #[test]
    fn straying_far_moves_the_world_into_a_new_frame() {
        let mut world = World::new();
        world.push(Camera3D::bake(Camera3DArgs::default()));
        let mut system = PlanetSystem::new(Planet::default());
        system.init(&mut world.context());
        for camera in world.context().query_mut::<Camera3D>() {
            camera.position = [1_200.0, -14.0, 0.0];
        }
        system.step(&mut world.context());
        let before = *world.context().resource::<CameraGeo>().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            system.step(&mut world.context());
            if world.context().resource::<FrameRebases>().unwrap().count() == 1 {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the frame never moved"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let ctx = world.context();
        let camera = ctx.query::<Camera3D>().next().unwrap();
        assert!(
            camera.position.iter().all(|c| c.abs() < 1e-2),
            "{:?}",
            camera.position
        );
        let ground = ctx.resource::<PlanetGround>().unwrap();
        assert!(ground.revision >= 1);
        assert!(
            ground.patch.center[0].abs() < 1e-2,
            "built around the camera"
        );
        let gravity = ctx.resource::<PlanetGravity>().unwrap();
        assert!(gravity.center[0].abs() < 1.0 && gravity.center[1] < -49_000.0);
        // The camera stood still while the world moved under it: the same
        // place on the planet to a centimeter (1e-5 degrees is 9 mm here),
        // facing the same way.
        let after = ctx.resource::<CameraGeo>().unwrap();
        assert!((after.position.latitude - before.position.latitude).abs() < 1e-5);
        assert!((after.position.longitude - before.position.longitude).abs() < 1e-5);
        assert!((after.position.altitude - before.position.altitude).abs() < 1e-2);
        assert!(
            (after.heading - before.heading).abs() < 1e-3,
            "{before:?} {after:?}"
        );
    }
}
