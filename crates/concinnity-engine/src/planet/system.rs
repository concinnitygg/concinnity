//! The planet system: keeps a planet world simulating around its camera.
//!
//! Each step it reads where the camera is. Once the camera strays
//! `REBASE_DISTANCE` from the simulated frame's origin, it asks its
//! [`GroundBuilder`] for a patch around the camera in a frame recentered on it,
//! and when that arrives, moves the world into the new frame: every root
//! transform, the camera and the rigs here, and through the published
//! `FrameRebases` the physics bodies, the streamed terrain tiles and the
//! renderer's temporal history in their own systems. Between moves it keeps the
//! ground patch the physics collides with centered on the camera, and keeps the
//! camera's up pointing away from the planet's center.
//!
//! Scheduled before the world's logic, so a move lands between two ticks:
//! behaviors, spawns, physics, the controllers and this tick's frame all see
//! the world already in its new frame.

use alloc::boxed::Box;

use crate::components::{Camera3D, Planet, SkyRotation};
use crate::ecs::{MenuActive, PipelineContext, StepResult, System};
use crate::planet::{
    CameraGeo, FrameRebases, GroundPatch, LocalFrame, PatchPurpose, PlanetFrame, PlanetGravity,
    PlanetGround, PlanetShape, Rebase, build_patch, carry_world, next_patch,
};
use crate::sky::{SkyFrame, SkyOrientation};

/// One ground patch to build: around local point `around` of `frame`, tagged
/// `id`.
#[derive(Debug, Clone, Copy)]
pub struct PatchRequest {
    /// Which request this is, echoed by its [`PatchResult`].
    pub id: u64,
    /// The frame the patch is built in.
    pub frame: LocalFrame,
    /// The local point the patch centers on.
    pub around: [f32; 3],
}

impl PatchRequest {
    /// The patch, built on the calling thread.
    pub fn build(&self, shape: &PlanetShape) -> PatchResult {
        PatchResult {
            id: self.id,
            patch: build_patch(shape, &self.frame, self.around),
        }
    }
}

/// What came back for request `id`: the patch, or `None` where the ground
/// cannot be built there (a camera off the planet) or the build failed.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchResult {
    /// The request this answers.
    pub id: u64,
    /// The built patch.
    pub patch: Option<GroundPatch>,
}

/// Where a planet's ground patches are built, off the frame.
pub trait GroundBuilder: core::fmt::Debug + Send {
    /// Queue a patch. `false` when it cannot be built.
    fn request(&mut self, request: PatchRequest) -> bool;
    /// A finished patch, if one is waiting.
    fn try_recv(&mut self) -> Option<PatchResult>;
}

/// Builds each patch on the calling thread when asked and answers on the next
/// poll: the builder for a host with no thread to lend.
#[derive(Debug)]
pub struct InlineGround {
    shape: PlanetShape,
    ready: Option<PatchResult>,
}

impl GroundBuilder for InlineGround {
    fn request(&mut self, request: PatchRequest) -> bool {
        self.ready = Some(request.build(&self.shape));
        true
    }

    fn try_recv(&mut self) -> Option<PatchResult> {
        self.ready.take()
    }
}

/// Starts an [`InlineGround`] for a planet of `shape`.
pub fn inline_ground(shape: PlanetShape) -> Option<Box<dyn GroundBuilder>> {
    Some(Box::new(InlineGround { shape, ready: None }))
}

/// A planet edited while its world runs, for the planet system to take up on
/// its next step: the ground reshaped around the camera and the pull
/// changed. Only the fields [`Planet::LIVE_FIELDS`] names may differ from the
/// running planet.
#[derive(Debug, Clone)]
pub struct PlanetEdit(pub Planet);

/// Starts a planet's [`GroundBuilder`]; `None` when it cannot, which leaves
/// the ground where init built it.
pub type StartBuilder = fn(PlanetShape) -> Option<Box<dyn GroundBuilder>>;

/// Keeps a planet world's simulated frame, gravity and ground around its
/// camera. Constructed by `World::start` when the world declares a `Planet`.
#[derive(Debug)]
pub struct PlanetSystem {
    planet: Planet,
    frame: PlanetFrame,
    rebases: FrameRebases,
    // The patch the physics collides with, by revision.
    ground: Option<PlanetGround>,
    // Starts the builder at init; without one the ground stays where init
    // built it.
    start_builder: StartBuilder,
    builder: Option<Box<dyn GroundBuilder>>,
    // The outstanding request: its id and what it is for.
    pending: Option<(u64, PatchPurpose)>,
    next_id: u64,
    // Whether a `SkyRotation` publishes the sky; without one this system does.
    sky_rotates: bool,
}

impl PlanetSystem {
    /// The system for a world that declares a `Planet`.
    pub fn for_world(world: &crate::ecs::World, start_builder: StartBuilder) -> Option<Self> {
        let planet = world.query::<Planet>().next().cloned()?;
        Some(Self::new(planet, start_builder))
    }

    /// The system for `planet`, building its moving ground on the builder
    /// `start_builder` starts at init.
    pub fn new(planet: Planet, start_builder: StartBuilder) -> Self {
        let shape = planet.shape();
        Self {
            planet,
            frame: PlanetFrame {
                shape,
                frame: LocalFrame::AUTHORED,
            },
            rebases: FrameRebases::default(),
            ground: None,
            start_builder,
            builder: None,
            pending: None,
            next_id: 0,
            sky_rotates: false,
        }
    }

    // Publish the frame, the pull, the moves and the sky's frame as they now
    // are.
    fn publish(&self, ctx: &mut PipelineContext<'_>) {
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

    fn publish_ground(&mut self, ctx: &mut PipelineContext<'_>, patch: GroundPatch) {
        let revision = self.ground.as_ref().map_or(0, |g| g.revision + 1);
        let ground = PlanetGround { revision, patch };
        ctx.insert_resource(ground.clone());
        self.ground = Some(ground);
    }

    // Move the world into `frame`, carried there by `rebase`.
    fn rebase(&mut self, ctx: &mut PipelineContext<'_>, frame: LocalFrame, rebase: Rebase) {
        let [x, y, z] = frame.origin();
        tracing::info!("planet: the simulated frame moved to ({x:.1}, {y:.1}, {z:.1})");
        self.frame.frame = frame;
        self.rebases.push(rebase);
        carry_world(ctx, &rebase, &self.frame);
        self.publish(ctx);
    }

    // Take up an edited planet: its pull, and for a new shape, the ground
    // around the camera built again on it, on a builder started for it.
    fn adopt(&mut self, ctx: &mut PipelineContext<'_>, planet: Planet) {
        let planet = crate::components::validate::planet(planet);
        let shape = planet.shape();
        let reshaped = shape != self.frame.shape;
        self.planet = planet;
        self.frame.shape = shape;
        self.publish(ctx);
        if !reshaped {
            return;
        }
        self.pending = None;
        self.builder = (self.start_builder)(shape);
        let around = camera_position(ctx)
            .or_else(|| self.ground.as_ref().map(|g| g.patch.center))
            .unwrap_or([0.0; 3]);
        let request = PatchRequest {
            id: self.next_id,
            frame: self.frame.frame,
            around,
        };
        self.next_id += 1;
        if let Some(patch) = request.build(&shape).patch {
            self.publish_ground(ctx, patch);
        }
    }

    // Take a finished patch: the ground moves, and for a patch built in a new
    // frame, the world moves with it.
    fn receive(&mut self, ctx: &mut PipelineContext<'_>) {
        while let Some(result) = self.builder.as_mut().and_then(|b| b.try_recv()) {
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
            if let Some(patch) = result.patch {
                self.publish_ground(ctx, patch);
            } else {
                tracing::warn!("planet: no ground under the camera");
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
        let Some(builder) = &mut self.builder else {
            return;
        };
        let id = self.next_id;
        self.next_id += 1;
        if builder.request(PatchRequest { id, frame, around }) {
            self.pending = Some((id, purpose));
        }
    }
}

fn camera_position(ctx: &PipelineContext<'_>) -> Option<[f32; 3]> {
    ctx.query::<Camera3D>().next().map(|c| c.position)
}

// Publish where on the planet the camera is and which way it looks.
fn publish_camera_geo(ctx: &mut PipelineContext<'_>, frame: &PlanetFrame) {
    let Some((position, yaw)) = ctx.query::<Camera3D>().next().map(|c| (c.position, c.yaw)) else {
        return;
    };
    ctx.insert_resource(CameraGeo {
        position: frame.geo_at(position),
        heading: frame.heading_for_yaw(position, yaw),
    });
}

impl System for PlanetSystem {
    fn init(&mut self, ctx: &mut PipelineContext<'_>) {
        self.sky_rotates = ctx.query::<SkyRotation>().next().is_some();
        self.publish(ctx);
        // The first patch is built here, so physics finds ground from its
        // first tick.
        let first = PatchRequest {
            id: 0,
            frame: self.frame.frame,
            around: camera_position(ctx).unwrap_or([0.0; 3]),
        };
        if let Some(patch) = first.build(&self.frame.shape).patch {
            self.publish_ground(ctx, patch);
        } else {
            tracing::warn!("planet: the camera starts off the planet");
        }
        let frame = self.frame;
        for camera in ctx.query_mut::<Camera3D>() {
            camera.up = frame.up_at(camera.position);
            camera.recompose_view();
        }
        self.builder = (self.start_builder)(self.frame.shape);
        self.next_id = 1;
        publish_camera_geo(ctx, &frame);
    }

    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        if let Some(PlanetEdit(planet)) = ctx.remove_resource::<PlanetEdit>() {
            self.adopt(ctx, planet);
        }
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
mod tests;
