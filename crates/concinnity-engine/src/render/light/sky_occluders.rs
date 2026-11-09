//! The spheres that keep part of the sky's light off the surfaces around them:
//! the drawn props whose collider is a ball, the nearest to the camera kept,
//! so a ball resting on the ground darkens it where it touches, as it would
//! under an overcast sky.

use crate::components::{
    Collider, GlobalTransform, Hidden, PropColliderShape, RenderHandle, SceneMember,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Entity, PipelineContext};
use crate::gfx::render_types::MAX_SKY_OCCLUDERS;
use concinnity_core::math::vec3;

/// Up to [`MAX_SKY_OCCLUDERS`] spheres, each its center (xyz) and radius (w)
/// in the space the frame draws in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyOccluders {
    spheres: [[f32; 4]; MAX_SKY_OCCLUDERS],
    count: usize,
}

impl Default for SkyOccluders {
    fn default() -> Self {
        Self::NONE
    }
}

impl SkyOccluders {
    /// No sphere.
    pub const NONE: Self = Self {
        spheres: [[0.0; 4]; MAX_SKY_OCCLUDERS],
        count: 0,
    };

    /// The live spheres.
    pub fn as_slice(&self) -> &[[f32; 4]] {
        &self.spheres[..self.count]
    }

    /// The ball-collider props drawn in the `active` scene, the nearest to
    /// `eye` kept, moved by `offset` from absolute world space into the space
    /// the frame draws in. A ball that draws nothing, or is hidden, casts no
    /// occlusion.
    pub fn gather(
        ctx: &PipelineContext<'_>,
        active: Option<AssetId>,
        eye: [f32; 3],
        offset: [f32; 3],
    ) -> Self {
        let mut occluders = Self::NONE;
        for (entity, Collider(collider)) in ctx.query_with_entity::<Collider>() {
            if collider.shape != PropColliderShape::Ball || !drawn(ctx, entity) {
                continue;
            }
            if ctx
                .get::<SceneMember>(entity)
                .is_some_and(|member| Some(member.0) != active)
            {
                continue;
            }
            let Some(GlobalTransform(model)) = ctx.get::<GlobalTransform>(entity) else {
                continue;
            };
            occluders.offer(ball(model, collider.radius), eye);
        }
        for sphere in &mut occluders.spheres[..occluders.count] {
            for (c, o) in sphere.iter_mut().zip(offset) {
                *c += o;
            }
        }
        occluders
    }

    // Keep `sphere` when there is room, or in place of the farthest from `eye`
    // when it is nearer.
    fn offer(&mut self, sphere: [f32; 4], eye: [f32; 3]) {
        if sphere[3] <= 0.0 {
            return;
        }
        if self.count < MAX_SKY_OCCLUDERS {
            self.spheres[self.count] = sphere;
            self.count += 1;
            return;
        }
        let gap = gap(sphere, eye);
        let (farthest, farthest_gap) = self
            .spheres
            .iter()
            .map(|s| self::gap(*s, eye))
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
        if gap < farthest_gap {
            self.spheres[farthest] = sphere;
        }
    }
}

// Whether `entity` draws anything: it has draw slots and no hide request has
// switched them off.
fn drawn(ctx: &PipelineContext<'_>, entity: Entity) -> bool {
    ctx.get::<Hidden>(entity).is_none()
        && ctx
            .get::<RenderHandle>(entity)
            .is_some_and(|handle| !handle.draws.is_empty())
}

// The sphere a ball collider of `radius` fills under `model`: its center at the
// model's origin, its radius scaled by the model's X axis as the collider is.
fn ball(model: &[[f32; 4]; 4], radius: f32) -> [f32; 4] {
    let scale = vec3::length([model[0][0], model[0][1], model[0][2]]);
    [model[3][0], model[3][1], model[3][2], radius * scale]
}

// How far `eye` is from the surface of `sphere`.
fn gap(sphere: [f32; 4], eye: [f32; 3]) -> f32 {
    vec3::length(vec3::sub([sphere[0], sphere[1], sphere[2]], eye)) - sphere[3]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::PropCollider;
    use crate::ecs::World;
    use crate::gfx::render_types::DrawIndex;

    fn at(position: [f32; 3], scale: f32) -> GlobalTransform {
        let mut m = crate::transform::IDENTITY;
        for (i, row) in m.iter_mut().take(3).enumerate() {
            row[i] = scale;
        }
        m[3] = [position[0], position[1], position[2], 1.0];
        GlobalTransform(m)
    }

    fn collider(shape: PropColliderShape, radius: f32) -> Collider {
        Collider(PropCollider {
            shape,
            radius,
            ..PropCollider::default()
        })
    }

    fn drawn_with(world: &mut World, collider: Collider, transform: GlobalTransform) -> Entity {
        let e = world.push(collider);
        world.insert(e, transform);
        world.insert(
            e,
            RenderHandle {
                draws: [DrawIndex(0)].into(),
            },
        );
        e
    }

    fn place(world: &mut World, shape: PropColliderShape, position: [f32; 3]) -> Entity {
        drawn_with(world, collider(shape, 0.5), at(position, 1.0))
    }

    #[test]
    fn a_ball_collider_is_a_sphere_at_its_prop_scaled_with_it() {
        let mut world = World::new();
        drawn_with(
            &mut world,
            collider(PropColliderShape::Ball, 0.6),
            at([1.0, 2.0, 3.0], 2.0),
        );
        let occluders = SkyOccluders::gather(&world.context(), None, [0.0; 3], [0.0; 3]);
        assert_eq!(occluders.as_slice(), &[[1.0, 2.0, 3.0, 1.2]]);
    }

    #[test]
    fn only_ball_colliders_occlude() {
        let mut world = World::new();
        place(&mut world, PropColliderShape::Cuboid, [0.0; 3]);
        place(&mut world, PropColliderShape::Capsule, [0.0; 3]);
        let occluders = SkyOccluders::gather(&world.context(), None, [0.0; 3], [0.0; 3]);
        assert!(occluders.as_slice().is_empty());
    }

    #[test]
    fn a_ball_in_another_scene_does_not_occlude() {
        let mut world = World::new();
        let shown = place(&mut world, PropColliderShape::Ball, [1.0, 0.0, 0.0]);
        world.insert(shown, SceneMember(AssetId(1)));
        let elsewhere = place(&mut world, PropColliderShape::Ball, [2.0, 0.0, 0.0]);
        world.insert(elsewhere, SceneMember(AssetId(2)));
        let occluders =
            SkyOccluders::gather(&world.context(), Some(AssetId(1)), [0.0; 3], [0.0; 3]);
        assert_eq!(occluders.as_slice(), &[[1.0, 0.0, 0.0, 0.5]]);
    }

    #[test]
    fn a_hidden_or_undrawn_ball_does_not_occlude() {
        let mut world = World::new();
        let hidden = place(&mut world, PropColliderShape::Ball, [1.0, 0.0, 0.0]);
        world.insert(hidden, Hidden);
        let undrawn = world.push(collider(PropColliderShape::Ball, 0.5));
        world.insert(undrawn, at([2.0, 0.0, 0.0], 1.0));
        let occluders = SkyOccluders::gather(&world.context(), None, [0.0; 3], [0.0; 3]);
        assert!(occluders.as_slice().is_empty());
    }

    #[test]
    fn the_spheres_move_into_the_drawn_space() {
        let mut world = World::new();
        place(&mut world, PropColliderShape::Ball, [10.0, 0.0, 0.0]);
        let occluders = SkyOccluders::gather(&world.context(), None, [0.0; 3], [-10.0, 1.0, 0.0]);
        assert_eq!(occluders.as_slice(), &[[0.0, 1.0, 0.0, 0.5]]);
    }

    #[test]
    fn past_capacity_the_nearest_to_the_eye_are_kept() {
        let mut world = World::new();
        let count = MAX_SKY_OCCLUDERS + 3;
        for i in (0..count).rev() {
            place(
                &mut world,
                PropColliderShape::Ball,
                [i as f32 * 10.0, 0.0, 0.0],
            );
        }
        let occluders = SkyOccluders::gather(&world.context(), None, [0.0; 3], [0.0; 3]);
        let mut kept: alloc::vec::Vec<f32> = occluders.as_slice().iter().map(|s| s[0]).collect();
        kept.sort_by(f32::total_cmp);
        let nearest: alloc::vec::Vec<f32> =
            (0..MAX_SKY_OCCLUDERS).map(|i| i as f32 * 10.0).collect();
        assert_eq!(kept, nearest);
    }
}
