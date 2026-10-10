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
//! Scheduled after the frame is drawn and before physics, so a move lands
//! between two ticks: physics, the controllers and the next frame's draw all
//! see the world already in its new frame.

mod carry;
mod ground;

use concinnity_core::components::{Camera3D, Planet, SkyRotation};
use concinnity_core::ecs::{MenuActive, PipelineContext, StepResult, System};
use concinnity_core::planet::{
    FrameRebases, GroundPatch, LocalFrame, PlanetFrame, PlanetGravity, PlanetGround,
    REBASE_DISTANCE, Rebase, needs_rebase,
};
use concinnity_core::sky::{SkyFrame, SkyOrientation};

use ground::{GroundWorker, PATCH_HALF_WIDTH, PatchRequest};

// How far the camera may walk from the ground patch's center before a new
// patch is built around it, in meters: well inside the patch's half width.
const FOLLOW_DISTANCE: f32 = PATCH_HALF_WIDTH * 0.375;

// What an outstanding patch request is for.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Purpose {
    // The camera walked from the patch; same frame.
    Follow,
    // The frame moves once the patch built in it arrives.
    Rebase { frame: LocalFrame, rebase: Rebase },
}

// What the camera at local point `camera` needs next, if anything: the frame
// moved under it once it strays too far, else the patch moved under it once
// it walks off the current one's middle.
fn next_purpose(
    camera: [f32; 3],
    frame: &PlanetFrame,
    patch_center: Option<[f32; 3]>,
) -> Option<(Purpose, LocalFrame, [f32; 3])> {
    if needs_rebase(camera, REBASE_DISTANCE) {
        let up = frame.shape.up_at(frame.frame.to_world(camera));
        let (next, rebase) = frame.frame.recentered(camera, up);
        let around = rebase.apply_point(camera);
        return Some((
            Purpose::Rebase {
                frame: next,
                rebase,
            },
            next,
            around,
        ));
    }
    let strayed = patch_center.is_none_or(|c| {
        let (dx, dz) = (camera[0] - c[0], camera[2] - c[2]);
        dx * dx + dz * dz > FOLLOW_DISTANCE * FOLLOW_DISTANCE
    });
    strayed.then_some((Purpose::Follow, frame.frame, camera))
}

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
    pending: Option<(u64, Purpose)>,
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
        carry::carry(ctx, &rebase, &self.frame);
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
            if let Purpose::Rebase { frame, rebase } = purpose {
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
        let Some((purpose, frame, around)) = next_purpose(camera, &self.frame, patch_center) else {
            return;
        };
        let Some(worker) = &self.worker else {
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
        self.worker = Some(GroundWorker::spawn(self.frame.shape));
        self.next_id = 1;
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
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::cook::Camera3D as Camera3DArgs;
    use concinnity_core::ecs::World;
    use concinnity_core::planet::PlanetShape;

    const SHAPE: PlanetShape = PlanetShape {
        center: [0.0, -50_000.0, 0.0],
        radius: 50_000.0,
        amplitude: 20.0,
        feature_size: 1_000.0,
        octaves: 4,
        seed: 1,
    };

    fn frame() -> PlanetFrame {
        PlanetFrame {
            shape: SHAPE,
            frame: LocalFrame::AUTHORED,
        }
    }

    #[test]
    fn a_camera_near_the_patch_middle_needs_nothing() {
        assert_eq!(
            next_purpose([5.0, 1.0, -3.0], &frame(), Some([0.0; 3])),
            None
        );
    }

    #[test]
    fn a_camera_off_the_patch_middle_moves_the_patch() {
        let (purpose, f, around) =
            next_purpose([30.0, 1.0, 0.0], &frame(), Some([0.0; 3])).unwrap();
        assert_eq!(purpose, Purpose::Follow);
        assert_eq!(f, LocalFrame::AUTHORED);
        assert_eq!(around, [30.0, 1.0, 0.0]);
        assert!(
            next_purpose([0.0; 3], &frame(), None).is_some(),
            "no patch yet"
        );
    }

    // Far enough out, the frame moves to the camera and the patch is built
    // around it there.
    #[test]
    fn a_camera_far_from_the_origin_moves_the_frame() {
        let camera = [1_001.0, -9.0, 0.0];
        let (purpose, f, around) =
            next_purpose(camera, &frame(), Some([1_000.0, 0.0, 0.0])).unwrap();
        let Purpose::Rebase {
            frame: next,
            rebase,
        } = purpose
        else {
            panic!("expected a rebase, got {purpose:?}");
        };
        assert_eq!(next, f);
        assert!(around.iter().all(|c| c.abs() < 1e-3), "{around:?}");
        assert_eq!(rebase.apply_point(camera), around);
    }

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
    }
}
