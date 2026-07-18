// concinnity-physics/src/system.rs
//
// The Rapier rigid-body simulation. An internal system (not a declarable
// asset): the engine schedule constructs one when the world declares a
// `PhysicsConfig`, a `RigidBody`, or a `PropBody`, reading the optional
// `PhysicsConfig` for the floor / terrain.

use crate::{BodyHandle, ColliderShape, PhysicsWorld};
use concinnity_core::assets::{
    Camera3D, Collider, Held, Joint, PhysicsConfig, Pickup, PropBody, RigidBody, Transform,
};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{
    Entity, EntityByName, EventCursor, MenuActive, PipelineContext, StepResult, System,
};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// Acceleration due to gravity in world units per second squared. Shared with
// the third-person controller so its jump takeoff matches the rig's fall.
pub const GRAVITY: f32 = 20.0;
// Friction coefficient for static (non-PropBody) prop colliders.
const STATIC_FRICTION: f32 = 0.8;
// Largest physics timestep; longer frames are clamped for solver stability.
const MAX_DT: f32 = 1.0 / 30.0;

// Reach distance for picking up a Prop, in world units.
const PICKUP_REACH: f32 = 3.0;
// Minimum facing dot product (~60-degree cone) for a pickup.
const PICKUP_MIN_DOT: f32 = 0.5;
// Distance ahead of the camera a carried prop hovers.
const HOLD_DISTANCE: f32 = 1.8;
// Drop of a carried prop below eye level.
const HOLD_DROP: f32 = 0.35;
// Launch speed applied to a prop when it is dropped/thrown.
const THROW_SPEED: f32 = 6.0;

// Rapier rigid-body simulation behavior. Constructed internally by
// `World::start` from the world's `PhysicsConfig`; never a declarable asset.
#[derive(Debug)]
pub struct PhysicsSystem {
    last_step: Option<Instant>,
    // Camera eye Y at spawn; the flat-floor fallback derives nothing from it,
    // but it seeds a sensible fallback camera position.
    floor_y: f32,
    // Terrain parameters. None when terrain_subdivisions == 0 (flat floor).
    terrain: Option<TerrainParams>,
    // Reference to a `ProceduralMesh` asset whose `heightfield` generator
    // drives the physics collider. Resolved against the live component list
    // at `init`. Takes precedence over `terrain` when both are set.
    terrain_mesh: Option<AssetId>,
    // World-space Y offset applied to whichever terrain source is active
    // (procedural noise or heightfield mesh). Matches the rendering Prop's
    // `position[1]`.
    terrain_offset_y: f32,
    // The Rapier simulation, built in init().
    world: Option<PhysicsWorld>,
    // The player capsule, when the world has a Camera3D + RigidBody.
    player: Option<PlayerPhysics>,
    // One capsule per root-motion character rig (see `super::rig`).
    rigs: Vec<super::rig::RigPhysics>,
    // Reader cursor over the `RootMotion` event queue.
    root_cursor: EventCursor,
    // One entry per Prop that carries a collider.
    prop_bodies: Vec<PropPhysics>,
    // Index into `prop_bodies` of the prop currently being carried.
    held: Option<usize>,
}

#[derive(Debug, Clone)]
struct TerrainParams {
    half_width: f32,
    half_depth: f32,
    subdivisions: u32,
    amplitude: f32,
    offset_y: f32,
}

// Runtime physics state for the player camera capsule.
#[derive(Debug)]
struct PlayerPhysics {
    handle: BodyHandle,
    // Capsule cylinder half-height (excludes the hemisphere caps).
    half_height: f32,
    radius: f32,
    // Camera eye Y minus capsule-centre Y.
    eye_offset: f32,
    // False for a free-flying camera (no RigidBody): no gravity, no jump.
    has_gravity: bool,
    gravity_scale: f32,
    jump_height: f32,
    // Current vertical velocity (world units/second).
    vy: f32,
    // Whether the capsule rested on a surface last frame.
    grounded: bool,
}

// Links a prop entity to its body in the simulation.
#[derive(Debug)]
struct PropPhysics {
    // The prop's entity, used to read/write its Transform and toggle its Held tag.
    entity: Entity,
    handle: BodyHandle,
    // False for static (immovable) props.
    dynamic: bool,
    // Whether the prop can be picked up and carried.
    pickup: bool,
}

impl PhysicsSystem {
    // Number of rigid bodies the physics world holds. Test-only observable for
    // the body-reaping path.
    #[cfg(test)]
    fn physics_body_count(&self) -> usize {
        self.world.as_ref().map_or(0, |w| w.body_count())
    }

    // Build the simulation from the world's `PhysicsConfig` (floor / terrain).
    // Bodies and colliders are added from the ECS in [`System::init`].
    pub fn new(config: PhysicsConfig) -> Self {
        let terrain = if config.terrain_subdivisions > 0 {
            Some(TerrainParams {
                half_width: config.terrain_half_width,
                half_depth: config.terrain_half_depth,
                subdivisions: config.terrain_subdivisions,
                amplitude: config.terrain_amplitude,
                offset_y: config.terrain_offset_y,
            })
        } else {
            None
        };
        Self {
            last_step: None,
            floor_y: config.floor_y,
            terrain,
            terrain_mesh: config.terrain_mesh,
            terrain_offset_y: config.terrain_offset_y,
            world: None,
            player: None,
            rigs: Vec::new(),
            root_cursor: EventCursor::default(),
            prop_bodies: Vec::new(),
            held: None,
        }
    }

    // Build one body per collider-bearing entity from its per-instance components
    // (Transform + Collider + the Pickup tag), resolving PropBody owners through
    // the name index and keying `body_handles` by AssetId (via the name index's
    // inverse) so the joint wiring resolves.
    fn build_prop_bodies(
        &mut self,
        ctx: &PipelineContext,
        world: &mut PhysicsWorld,
        body_handles: &mut HashMap<AssetId, BodyHandle>,
    ) {
        let (propbody_of, entity_name): (HashMap<Entity, PropBody>, HashMap<Entity, AssetId>) = {
            let name_index = ctx.resource::<EntityByName>();
            let inverse = name_index
                .map(|n| n.0.iter().map(|(&id, &e)| (e, id)).collect())
                .unwrap_or_default();
            let owners = ctx
                .query::<PropBody>()
                .filter_map(|b| {
                    let name = b.prop_name?;
                    let entity = name_index?.get(name)?;
                    Some((entity, b.clone()))
                })
                .collect();
            (owners, inverse)
        };

        let pickup: HashSet<Entity> = ctx.query_with_entity::<Pickup>().map(|(e, _)| e).collect();
        let snaps: Vec<(Entity, PropCollSnap)> = ctx
            .join2::<Collider, Transform>()
            .map(|(entity, collider, transform)| {
                (
                    entity,
                    PropCollSnap {
                        shape: crate::collider_shape(&collider.0, transform.scale),
                        position: transform.position,
                        rotation_deg: transform.rotation_deg,
                        pickup: pickup.contains(&entity),
                    },
                )
            })
            .collect();

        for (entity, snap) in snaps {
            let handle = if let Some(prop_body) = propbody_of.get(&entity) {
                let handle = world.add_dynamic(
                    &snap.shape,
                    snap.position,
                    snap.rotation_deg,
                    crate::dynamic_params(prop_body),
                );
                self.prop_bodies.push(PropPhysics {
                    entity,
                    handle,
                    dynamic: true,
                    pickup: snap.pickup,
                });
                handle
            } else {
                let handle = world.add_fixed(
                    &snap.shape,
                    snap.position,
                    snap.rotation_deg,
                    STATIC_FRICTION,
                );
                self.prop_bodies.push(PropPhysics {
                    entity,
                    handle,
                    dynamic: false,
                    pickup: false,
                });
                handle
            };
            if let Some(&id) = entity_name.get(&entity) {
                body_handles.insert(id, handle);
            }
        }
    }
}

impl System for PhysicsSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        self.last_step = Some(Instant::now());

        let mut world = PhysicsWorld::new(GRAVITY);

        // floor: heightfield-mesh-driven, procedural noise, or flat slab
        let mut floor_built = false;
        if let Some(mesh_id) = self.terrain_mesh {
            let mesh_snap = ctx
                .query::<concinnity_core::assets::ProceduralMesh>()
                .find(|m| m.asset_id == mesh_id)
                .cloned();
            match mesh_snap {
                Some(m) if m.generator == "heightfield" => {
                    match build_heightfield_collider(&mut world, &m, self.terrain_offset_y, ctx) {
                        Ok(()) => floor_built = true,
                        Err(e) => tracing::warn!(
                            "physics: heightfield collider load failed ({}); falling back to flat slab",
                            e
                        ),
                    }
                }
                Some(m) => {
                    tracing::warn!(
                        "physics: terrain_mesh '{}' has generator '{}', expected 'heightfield'; falling back",
                        mesh_id,
                        m.generator
                    );
                }
                None => {
                    tracing::warn!(
                        "physics: terrain_mesh asset {} not found; falling back",
                        mesh_id
                    );
                }
            }
        }
        if !floor_built {
            if let Some(terrain) = self.terrain.clone() {
                build_heightfield(&mut world, &terrain);
            } else {
                // A large thin slab whose top face sits at Y = 0.
                world.add_fixed(
                    &crate::ColliderShape::Cuboid {
                        half_extents: [500.0, 5.0, 500.0],
                    },
                    [0.0, -5.0, 0.0],
                    [0.0; 3],
                    STATIC_FRICTION,
                );
            }
        }

        // Prop name -> BodyHandle, populated alongside `self.prop_bodies`.
        // Joints resolve their `body_a`/`body_b` references through this map.
        let mut body_handles: HashMap<AssetId, BodyHandle> = HashMap::new();
        self.build_prop_bodies(ctx, &mut world, &mut body_handles);
        tracing::debug!(
            "PhysicsSystem: {} prop bodies ({} dynamic)",
            self.prop_bodies.len(),
            self.prop_bodies.iter().filter(|p| p.dynamic).count(),
        );

        // joints
        // Each Joint references one or two Props by AssetId. Cross-reference
        // validation already guarantees the Prop exists; here we additionally
        // require the Prop to own a collider (and therefore a body). A Joint
        // with body_b empty anchors body_a to a hidden static body created on
        // demand at the world-space `anchor_b`.
        let joints: Vec<Joint> = ctx.drain::<Joint>();
        let mut wired = 0usize;
        for joint in joints {
            let Some(body_a_id) = joint.body_a else {
                tracing::warn!("Joint '{}': body_a is required; skipping", joint.asset_id);
                continue;
            };
            let Some(handle_a) = body_handles.get(&body_a_id).copied() else {
                tracing::warn!(
                    "Joint '{}': body_a Prop has no collider; skipping",
                    joint.asset_id
                );
                continue;
            };
            let handle_b = if let Some(body_b_id) = joint.body_b {
                match body_handles.get(&body_b_id).copied() {
                    Some(h) => h,
                    None => {
                        tracing::warn!(
                            "Joint '{}': body_b Prop has no collider; skipping",
                            joint.asset_id
                        );
                        continue;
                    }
                }
            } else {
                // Static world anchor at anchor_b. Sub-millimetre ball so it
                // takes effectively no space in the broad-phase BVH.
                world.add_fixed(
                    &ColliderShape::Ball { radius: 0.001 },
                    joint.anchor_b,
                    [0.0; 3],
                    0.0,
                )
            };
            // When body_b is the implicit world anchor, the anchor sits at the
            // origin of that hidden body, not at the authored offset.
            let anchor_b = if joint.body_b.is_some() {
                joint.anchor_b
            } else {
                [0.0, 0.0, 0.0]
            };
            world.add_joint(
                handle_a,
                handle_b,
                joint.anchor_a,
                anchor_b,
                crate::joint_spec(&joint),
            );
            wired += 1;
        }
        if wired > 0 {
            tracing::debug!("PhysicsSystem: wired {} joint(s)", wired);
        }

        // player capsule for the Camera3D
        // Every first-person camera is collided as a capsule. A RigidBody
        // upgrades it from a free-flying spectator to a grounded,
        // gravity-bound character. A third-person camera (a controller with
        // a `follow` block) gets no capsule: it is a virtual orbit around the
        // followed character, whose own rig capsule is the collided body.
        let camera_pos = ctx
            .query::<Camera3D>()
            .next()
            .filter(|c| {
                c.controller
                    .as_ref()
                    .is_none_or(|ctrl| ctrl.follow.is_none())
            })
            .map(|c| c.position);
        if let Some(cam_pos) = camera_pos {
            let rb_opt = ctx.query::<RigidBody>().next().cloned();
            let has_gravity = rb_opt.is_some();
            let rb = rb_opt.unwrap_or_default();
            if self.floor_y == 0.0 {
                self.floor_y = cam_pos[1];
            }
            let radius = rb.capsule_radius.max(0.05);
            let half_height = ((rb.capsule_height * 0.5) - radius).max(0.05);
            // a grounded character's eye sits at the capsule top; a flying
            // camera's capsule is centred on the eye.
            let eye_offset = if has_gravity {
                (rb.capsule_height * 0.5).max(radius + 0.05)
            } else {
                0.0
            };
            let center = [cam_pos[0], cam_pos[1] - eye_offset, cam_pos[2]];
            world.configure_character(rb.max_slope_deg, rb.step_height, has_gravity);
            let handle = world.add_character(half_height, radius, center);
            self.player = Some(PlayerPhysics {
                handle,
                half_height,
                radius,
                eye_offset,
                has_gravity,
                gravity_scale: rb.gravity_scale.max(0.0),
                jump_height: rb.jump_height.max(0.0),
                vy: 0.0,
                grounded: true,
            });
            tracing::debug!(
                "PhysicsSystem: player capsule r={:.2} h={:.2} gravity={}",
                radius,
                half_height,
                has_gravity,
            );
        }

        // Kinematic capsules for the root-motion character rigs published by
        // GraphicsSystem (which ran init first this tick).
        self.rigs = super::rig::init_rigs(&mut world, ctx);

        self.world = Some(world);
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        let now = Instant::now();
        let dt = self
            .last_step
            .map(|t| now.duration_since(t).as_secs_f32().min(MAX_DT))
            .unwrap_or(0.0);
        self.last_step = Some(now);

        // Freeze while a menu is open: skip the solve so the world truly pauses.
        // `last_step` is advanced above regardless, so the next live frame sees
        // one normal interval rather than the whole paused span -- no catch-up
        // burst on resume. The flag is published by GraphicsSystem, which runs
        // first in the schedule, so it reflects this same tick.
        if ctx.resource::<MenuActive>().is_some_and(|m| m.0) {
            return StepResult::Continue;
        }

        if self.world.is_none() || dt <= 0.0 {
            return StepResult::Continue;
        }

        // snapshot reads (released before any query_mut below)
        let (cam_pos, cam_yaw, cam_pitch, desired_move, jump_req, interact_req) = ctx
            .query::<Camera3D>()
            .next()
            .map(|c| {
                (
                    c.position,
                    c.yaw,
                    c.pitch,
                    c.desired_move,
                    c.jump_requested,
                    c.interact_requested,
                )
            })
            .unwrap_or(([0.0, self.floor_y, 0.0], 0.0, 0.0, [0.0; 3], false, false));

        // camera-space basis vectors
        let fwd_flat = [-cam_yaw.sin(), 0.0, -cam_yaw.cos()];
        let fwd_full = [
            -(cam_yaw.sin() * cam_pitch.cos()),
            -cam_pitch.sin(),
            -(cam_yaw.cos() * cam_pitch.cos()),
        ];

        let world = self.world.as_mut().expect("world checked above");

        // Reap bodies whose entity was despawned: GraphicsSystem runs before
        // PhysicsSystem and removes the entity from the ECS, so a body left behind
        // would keep simulating - and colliding with live bodies - invisibly.
        // Remove it from Rapier and drop its record before the step, keeping
        // self.held a valid index into the compacted list.
        let dead: Vec<BodyHandle> = self
            .prop_bodies
            .iter()
            .filter(|p| !ctx.is_alive(p.entity))
            .map(|p| p.handle)
            .collect();
        if !dead.is_empty() {
            let held_entity = self
                .held
                .and_then(|i| self.prop_bodies.get(i))
                .map(|p| p.entity);
            for handle in dead {
                world.remove_body(handle);
            }
            self.prop_bodies.retain(|p| ctx.is_alive(p.entity));
            self.held =
                held_entity.and_then(|e| self.prop_bodies.iter().position(|p| p.entity == e));
        }

        // pickup / drop on the interact edge; held_changed carries the entity to
        // toggle the Held tag on in the write-back.
        let mut held_changed: Option<(Entity, bool)> = None;
        if interact_req {
            if let Some(held_idx) = self.held.take() {
                // drop: hand the prop back to dynamic simulation with a throw.
                let pp = &self.prop_bodies[held_idx];
                let throw = [
                    fwd_full[0] * THROW_SPEED,
                    fwd_full[1] * THROW_SPEED + 1.0,
                    fwd_full[2] * THROW_SPEED,
                ];
                world.make_dynamic(pp.handle, throw);
                held_changed = Some((pp.entity, false));
            } else {
                // pickup: nearest carriable prop within reach the player faces.
                // Entity positions for the reach test, read from the Transform
                // column only on the interact edge (not every frame).
                let entity_positions: HashMap<Entity, [f32; 3]> = ctx
                    .query_with_entity::<Transform>()
                    .map(|(e, t)| (e, t.position))
                    .collect();
                let mut best: Option<(f32, usize)> = None;
                for (idx, pp) in self.prop_bodies.iter().enumerate() {
                    if !pp.pickup {
                        continue;
                    }
                    let pos = entity_positions.get(&pp.entity).copied().unwrap_or(cam_pos);
                    let dx = pos[0] - cam_pos[0];
                    let dz = pos[2] - cam_pos[2];
                    let dist = (dx * dx + dz * dz).sqrt();
                    if dist >= PICKUP_REACH || dist <= 0.0 {
                        continue;
                    }
                    let dot = (fwd_flat[0] * dx + fwd_flat[2] * dz) / dist;
                    if dot > PICKUP_MIN_DOT && best.is_none_or(|(d, _)| dist < d) {
                        best = Some((dist, idx));
                    }
                }
                if let Some((_, idx)) = best {
                    let pp = &self.prop_bodies[idx];
                    world.make_kinematic(pp.handle);
                    held_changed = Some((pp.entity, true));
                    self.held = Some(idx);
                }
            }
        }

        // carried prop hovers in front of the camera
        if let Some(held_idx) = self.held {
            let pp = &self.prop_bodies[held_idx];
            let hold_pos = [
                cam_pos[0] + fwd_full[0] * HOLD_DISTANCE,
                cam_pos[1] + fwd_full[1] * HOLD_DISTANCE - HOLD_DROP,
                cam_pos[2] + fwd_full[2] * HOLD_DISTANCE,
            ];
            world.set_kinematic_translation(pp.handle, hold_pos);
        }

        // move the player capsule
        let mut new_cam_pos = cam_pos;
        let mut grounded = true;
        if let Some(player) = self.player.as_mut() {
            if player.has_gravity {
                if jump_req && player.grounded && player.jump_height > 0.0 {
                    player.vy = (2.0 * GRAVITY * player.gravity_scale * player.jump_height).sqrt();
                }
                player.vy -= GRAVITY * player.gravity_scale * dt;
            }

            let center = [cam_pos[0], cam_pos[1] - player.eye_offset, cam_pos[2]];
            let desired = [desired_move[0] * dt, player.vy * dt, desired_move[2] * dt];
            let moved = world.move_character(
                player.half_height,
                player.radius,
                center,
                desired,
                dt,
                player.handle,
            );
            let new_center = [
                center[0] + moved.translation[0],
                center[1] + moved.translation[1],
                center[2] + moved.translation[2],
            ];
            world.set_kinematic_translation(player.handle, new_center);

            player.grounded = moved.grounded;
            grounded = moved.grounded;
            if moved.grounded && player.vy < 0.0 {
                player.vy = 0.0;
            }
            new_cam_pos = [
                new_center[0],
                new_center[1] + player.eye_offset,
                new_center[2],
            ];
        }

        // move the root-motion character rig capsules
        super::rig::step_rigs(
            world,
            ctx,
            &mut self.rigs,
            &mut self.root_cursor,
            dt,
            GRAVITY,
        );

        // answer the IK ground probes and the follow camera's occlusion probe
        super::probes::step_probes(world, ctx, &self.rigs);

        // advance the simulation
        world.step(dt);

        // read dynamic prop transforms back out, keyed by entity
        let prop_updates: Vec<(Entity, [f32; 3], [f32; 3])> = self
            .prop_bodies
            .iter()
            .filter(|p| p.dynamic)
            .map(|p| {
                let (pos, rot) = world.body_pose(p.handle);
                (p.entity, pos, rot)
            })
            .collect();

        // write the simulated pose back to each entity's Transform, and the
        // carried flag to its Held tag.
        for (entity, pos, rot) in prop_updates {
            if let Some(t) = ctx.get_mut::<Transform>(entity) {
                t.position = pos;
                t.rotation_deg = rot;
            }
        }
        if let Some((entity, is_held)) = held_changed {
            if is_held {
                if ctx.get::<Held>(entity).is_none() {
                    ctx.insert(entity, Held);
                }
            } else {
                ctx.remove::<Held>(entity);
            }
        }

        // write camera position + view matrix
        for camera in ctx.query_mut::<Camera3D>() {
            camera.position = new_cam_pos;
            camera.view_matrix = concinnity_core::gfx::camera::view_matrix(
                camera.position,
                camera.yaw,
                camera.pitch,
            );
        }

        // publish grounded state for jump gating
        for body in ctx.query_mut::<RigidBody>() {
            body.is_grounded = grounded;
        }

        StepResult::Continue
    }
}

// A Prop's collider plus transform, snapshotted at init time.
struct PropCollSnap {
    shape: crate::ColliderShape,
    position: [f32; 3],
    rotation_deg: [f32; 3],
    pickup: bool,
}

// Build a Rapier heightfield collider for a heightfield-generator
// `ProceduralMesh` from the collider grid baked into its compiled payload. The
// build step stores the mesh's own per-vertex heights (an `n x n` row-major
// world-Y grid) as a trailer on the payload, so the collider tracks the
// rendered surface vertex-for-vertex without decoding the source image at
// runtime. The terrain mesh's blob is held resident past GraphicsSystem init
// for exactly this read (see the release sweep in `graphics_system::init`).
fn build_heightfield_collider(
    world: &mut PhysicsWorld,
    mesh: &concinnity_core::assets::ProceduralMesh,
    offset_y: f32,
    ctx: &mut PipelineContext,
) -> Result<(), String> {
    let locator = mesh
        .locator
        .as_ref()
        .ok_or("heightfield ProceduralMesh has no compiled payload")?;
    let bytes = ctx
        .read_payload(locator)
        .map_err(|e| format!("read terrain payload: {e:?}"))?;
    let grid = concinnity_core::gfx::mesh_payload::deserialise_heightfield(bytes)?
        .ok_or("terrain mesh payload has no baked heightfield collider")?;
    if grid.rows < 2 || grid.cols < 2 {
        return Err(format!(
            "heightfield collider grid too small ({}x{})",
            grid.rows, grid.cols
        ));
    }
    let width = mesh.half_width * 2.0;
    let depth = mesh.half_depth * 2.0;
    world.add_heightfield(
        grid.rows,
        grid.cols,
        grid.heights,
        [width, 1.0, depth],
        [0.0, offset_y, 0.0],
    );
    Ok(())
}

// Build a Rapier heightfield collider matching the procedural terrain mesh.
fn build_heightfield(world: &mut PhysicsWorld, terrain: &TerrainParams) {
    let n = (terrain.subdivisions as usize) + 1;
    let width = terrain.half_width * 2.0;
    let depth = terrain.half_depth * 2.0;
    let mut heights = Vec::with_capacity(n * n);
    for i in 0..n {
        // row i spans Z
        let z = (i as f32 / (n - 1) as f32 - 0.5) * depth;
        for j in 0..n {
            // col j spans X
            let x = (j as f32 / (n - 1) as f32 - 0.5) * width;
            heights.push(terrain_height_at(x, z, terrain));
        }
    }
    world.add_heightfield(
        n,
        n,
        heights,
        [width, 1.0, depth],
        [0.0, terrain.offset_y, 0.0],
    );
}

// Compute terrain surface height at world-space (x, z) using the same bilinear
// noise as the "terrain" mesh generator in build_mesh.rs. Converting world XZ
// to a fractional grid position and bilinearly interpolating between lattice
// samples gives a continuous height field that matches the rendered mesh exactly.
fn terrain_height_at(world_x: f32, world_z: f32, t: &TerrainParams) -> f32 {
    // clamp to terrain footprint; out-of-bounds positions use the edge height
    let x = world_x.clamp(-t.half_width, t.half_width);
    let z = world_z.clamp(-t.half_depth, t.half_depth);

    // fractional grid position in [0, subdivisions]
    let s = (x + t.half_width) / (t.half_width * 2.0) * t.subdivisions as f32;
    let g = (z + t.half_depth) / (t.half_depth * 2.0) * t.subdivisions as f32;

    let octaves: &[(u32, f32)] = &[
        (1, 1.00), // coarse hills
        (3, 0.40), // medium bumps
        (9, 0.15), // fine surface variation
    ];

    let mut sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;

    for &(divisor, weight) in octaves {
        let scale = (t.subdivisions / divisor).max(1) as f32;
        let gs = s / scale;
        let gt = g / scale;
        let gx = gs.floor() as u32;
        let gy = gt.floor() as u32;
        let fx = gs - gx as f32;
        let fy = gt - gy as f32;

        let h00 = lattice_val(gx, gy);
        let h10 = lattice_val(gx + 1, gy);
        let h01 = lattice_val(gx, gy + 1);
        let h11 = lattice_val(gx + 1, gy + 1);
        let top = h00 + (h10 - h00) * fx;
        let bot = h01 + (h11 - h01) * fx;
        sum += (top + (bot - top) * fy) * weight;
        weight_sum += weight;
    }

    let normalised = sum / weight_sum;
    (normalised - 0.05).max(0.0) * t.amplitude
}

fn lattice_val(x: u32, y: u32) -> f32 {
    let h = lcg_hash(x.wrapping_mul(1619).wrapping_add(y.wrapping_mul(31337)));
    (h & 0xFF) as f32 / 255.0
}

fn lcg_hash(mut v: u32) -> u32 {
    v = v.wrapping_mul(1664525).wrapping_add(1013904223);
    v ^= v >> 16;
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::assets::{CameraController, FollowController, PropCollider};
    use concinnity_core::blob::BlobData;
    use concinnity_core::ecs::{ComponentStorage, Resources, SkinnedMeshHandle};
    use concinnity_core::gfx::profile::FrameProfile;

    // Hand-assembled stand-in for the engine's `World`: owns the storage a
    // `PipelineContext` borrows so a `PhysicsSystem` can be init/stepped in
    // isolation. `World` stays in the engine, so this crate cannot use it.
    struct TestWorld {
        components: ComponentStorage,
        blob: BlobData,
        profile: FrameProfile,
        resources: Resources,
    }

    impl TestWorld {
        fn new() -> Self {
            Self {
                components: ComponentStorage::default(),
                blob: BlobData::empty(),
                profile: FrameProfile::default(),
                resources: Resources::new(),
            }
        }

        fn ctx(&mut self) -> PipelineContext<'_> {
            PipelineContext {
                components: &mut self.components,
                blob: &mut self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
            }
        }

        // Push a decomposed prop (Transform + Collider, plus the Pickup tag when
        // asked) exactly as the load-time decomposition would, and register it in
        // the name index under `id` so a `PropBody` owner resolves to it.
        fn spawn_prop(&mut self, id: AssetId, position: [f32; 3], pickup: bool) -> Entity {
            let entity = self.components.push_typed(Transform {
                position,
                ..Default::default()
            });
            self.components.insert_typed(
                entity,
                Collider(PropCollider {
                    shape: "ball".to_string(),
                    half_extents: [0.5; 3],
                    radius: 0.5,
                    half_height: 0.0,
                }),
            );
            if pickup {
                self.components.insert_typed(entity, Pickup);
            }
            match self.resources.get_mut::<EntityByName>() {
                Some(index) => {
                    index.0.insert(id, entity);
                }
                None => {
                    let mut index = HashMap::new();
                    index.insert(id, entity);
                    self.resources.insert(EntityByName(index));
                }
            }
            entity
        }
    }

    #[test]
    fn flat_terrain_is_zero_height() {
        let t = TerrainParams {
            half_width: 32.0,
            half_depth: 32.0,
            subdivisions: 32,
            amplitude: 0.0,
            offset_y: 0.0,
        };
        assert_eq!(terrain_height_at(0.0, 0.0, &t), 0.0);
        assert_eq!(terrain_height_at(10.0, -5.0, &t), 0.0);
    }

    #[test]
    fn terrain_height_is_continuous_and_bounded() {
        let t = TerrainParams {
            half_width: 32.0,
            half_depth: 32.0,
            subdivisions: 32,
            amplitude: 4.0,
            offset_y: 0.0,
        };
        // Height never exceeds the amplitude and neighbouring samples are close.
        let mut prev = terrain_height_at(-32.0, 0.0, &t);
        let mut x = -32.0;
        while x <= 32.0 {
            let h = terrain_height_at(x, 0.0, &t);
            assert!((0.0..=4.0).contains(&h), "height {h} out of range at x={x}");
            assert!((h - prev).abs() < 1.0, "terrain jumped at x={x}");
            prev = h;
            x += 0.5;
        }
    }

    fn controlled_camera() -> Camera3D {
        Camera3D {
            fov_y_degrees: 75.0,
            near: 0.05,
            far: 200.0,
            view_matrix: [[0.0; 4]; 4],
            position: [0.0, 1.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            desired_move: [0.0; 3],
            jump_requested: false,
            interact_requested: false,
            controller: Some(CameraController::default()),
        }
    }

    // A third-person camera is a virtual orbit: no player capsule is created
    // for it. (Regression: the spectator capsule spawned at the camera eye
    // overlapped the followed rig's capsule and squeezed it through the
    // floor.) A first-person camera keeps its capsule. The schedule gate that
    // builds the system at all is covered by the engine's schedule tests.
    #[test]
    fn third_person_camera_gets_no_player_capsule() {
        // Third-person (follow) camera: a virtual orbit, so no player capsule.
        let mut world = TestWorld::new();
        let mut camera = controlled_camera();
        camera.controller = Some(CameraController {
            follow: Some(FollowController {
                target: Some(SkinnedMeshHandle(1)),
                ..FollowController::default()
            }),
            ..CameraController::default()
        });
        world.components.push_typed(camera);
        let mut physics = PhysicsSystem::new(PhysicsConfig::default());
        physics.init(&mut world.ctx());
        assert!(physics.player.is_none(), "no capsule for the orbit camera");

        // First-person camera keeps its capsule.
        let mut world = TestWorld::new();
        world.components.push_typed(controlled_camera());
        let mut physics = PhysicsSystem::new(PhysicsConfig::default());
        physics.init(&mut world.ctx());
        assert!(
            physics.player.is_some(),
            "first-person camera keeps its capsule"
        );
    }

    fn ball_body(id: AssetId) -> PropBody {
        PropBody {
            prop_name: Some(id),
            mass: 1.0,
            friction: 0.5,
            restitution: 0.0,
            gravity_scale: 1.0,
            linear_damping: 0.0,
        }
    }

    // A camera looking down -Z with the interact field latched on, so the
    // PhysicsSystem (which reads Camera3D.interact_requested) triggers a pickup.
    fn interacting_camera(position: [f32; 3]) -> Camera3D {
        Camera3D {
            interact_requested: true,
            controller: None,
            position,
            ..controlled_camera()
        }
    }

    // The simulated pose is written back to the prop's Transform.
    #[test]
    fn dynamic_prop_writes_transform() {
        use std::{thread, time::Duration};
        let id = AssetId(1);
        let mut world = TestWorld::new();
        let entity = world.spawn_prop(id, [0.0, 5.0, 0.0], false);
        world.components.push_typed(ball_body(id));

        let mut physics = PhysicsSystem::new(PhysicsConfig::default());
        physics.init(&mut world.ctx());
        for _ in 0..3 {
            thread::sleep(Duration::from_millis(20));
            physics.step(&mut world.ctx());
        }

        let transform_y = world.components.get::<Transform>(entity).unwrap().position[1];
        assert!(
            transform_y < 5.0,
            "the simulated pose falls the Transform (y={transform_y})"
        );
    }

    // A menu freezes the solve: while `MenuActive(true)` is published the body
    // does not fall, and clearing it resumes the fall from where it froze. This
    // is the "true pause" the menu relies on; `last_step` advancing under the
    // freeze keeps the resume to one normal interval (no catch-up burst).
    #[test]
    fn menu_active_freezes_then_resumes_physics() {
        use std::{thread, time::Duration};
        let id = AssetId(1);
        let mut world = TestWorld::new();
        let entity = world.spawn_prop(id, [0.0, 5.0, 0.0], false);
        world.components.push_typed(ball_body(id));

        let mut physics = PhysicsSystem::new(PhysicsConfig::default());
        physics.init(&mut world.ctx());

        // Paused: step several frames with real wall-clock dt; the body stays put.
        world.resources.insert(MenuActive(true));
        for _ in 0..5 {
            thread::sleep(Duration::from_millis(20));
            physics.step(&mut world.ctx());
        }
        let y_paused = world.components.get::<Transform>(entity).unwrap().position[1];
        assert!(
            (y_paused - 5.0).abs() < 1e-3,
            "the body must not fall while a menu is active (y={y_paused})"
        );

        // Resumed: the body falls again.
        world.resources.insert(MenuActive(false));
        for _ in 0..3 {
            thread::sleep(Duration::from_millis(20));
            physics.step(&mut world.ctx());
        }
        let y_resumed = world.components.get::<Transform>(entity).unwrap().position[1];
        assert!(
            y_resumed < y_paused - 1e-3,
            "the body must fall once the menu closes (y={y_resumed})"
        );
    }

    // Despawning a decomposed prop reaps its Rapier body, so it stops simulating
    // (and colliding) once its entity is gone.
    #[test]
    fn despawning_a_prop_reaps_its_physics_body() {
        use std::{thread, time::Duration};
        let id = AssetId(1);
        let mut world = TestWorld::new();
        let ball = world.spawn_prop(id, [0.0, 5.0, 0.0], false);
        world.components.push_typed(ball_body(id));

        let mut physics = PhysicsSystem::new(PhysicsConfig::default());
        physics.init(&mut world.ctx());

        // Settle so the body is live and falling.
        for _ in 0..2 {
            thread::sleep(Duration::from_millis(20));
            physics.step(&mut world.ctx());
        }
        let before = physics.physics_body_count();

        // Despawn the ball (stand-in for GraphicsSystem) and step: PhysicsSystem
        // reaps the orphaned body.
        world.components.despawn(ball);
        thread::sleep(Duration::from_millis(20));
        physics.step(&mut world.ctx());
        let after = physics.physics_body_count();
        assert_eq!(after, before - 1, "the despawned prop's body was removed");

        // The sim keeps running cleanly with the body gone (no further removals).
        thread::sleep(Duration::from_millis(20));
        physics.step(&mut world.ctx());
        assert_eq!(
            physics.physics_body_count(),
            after,
            "no further bodies removed"
        );
    }

    // Picking up a carriable prop tags its entity with Held.
    #[test]
    fn pickup_sets_held_tag() {
        use std::{thread, time::Duration};
        let id = AssetId(1);
        let mut world = TestWorld::new();
        world.spawn_prop(id, [0.0, 1.0, -2.0], true);
        world.components.push_typed(ball_body(id));
        world
            .components
            .push_typed(interacting_camera([0.0, 1.0, 0.0]));

        let mut physics = PhysicsSystem::new(PhysicsConfig::default());
        physics.init(&mut world.ctx());

        thread::sleep(Duration::from_millis(5));
        physics.step(&mut world.ctx());

        assert_eq!(
            world.ctx().query::<Held>().count(),
            1,
            "pickup inserts the Held tag on the entity"
        );
    }
}
