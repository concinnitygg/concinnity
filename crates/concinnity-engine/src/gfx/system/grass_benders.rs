// What tramples the grass each frame, as the upright shapes its bend field
// stamps: every root-motion character's capsule, the player's capsule, and
// every dynamic body's collider. The backend culls them to the field around
// the camera and to the ground they stand on.

use concinnity_core::components::{
    BodyDynamics, Camera3D, CharacterRig, Collider, GlobalTransform, PropCollider,
    PropColliderShape, RigidBody,
};
use concinnity_core::ecs::PipelineContext;
use concinnity_core::gfx::frustum::transform_aabb;
use concinnity_core::render::grass::GrassBender;

// Refill `out` with this frame's benders.
pub(super) fn gather_into(ctx: &PipelineContext, out: &mut Vec<GrassBender>) {
    out.clear();
    out.extend(ctx.query::<CharacterRig>().map(of_rig));
    if let (Some(camera), Some(body)) = (
        ctx.query::<Camera3D>().next(),
        ctx.query::<RigidBody>().next(),
    ) && camera
        .controller
        .as_ref()
        .is_none_or(|c| c.follow.is_none())
    {
        out.push(of_player(camera.position, body));
    }
    for (entity, collider, _) in ctx.join2::<Collider, BodyDynamics>() {
        if let Some(global) = ctx.get::<GlobalTransform>(entity) {
            out.push(of_body(&collider.0, global.0));
        }
    }
}

// A character's capsule, standing on its mesh origin.
fn of_rig(rig: &CharacterRig) -> GrassBender {
    GrassBender {
        base: rig.position,
        radius: rig.radius,
        height: 2.0 * (rig.half_height + rig.radius),
    }
}

// The player's capsule, under the eye at its top.
fn of_player(eye: [f32; 3], body: &RigidBody) -> GrassBender {
    GrassBender {
        base: [eye[0], eye[1] - body.capsule_height, eye[2]],
        radius: body.capsule_radius,
        height: body.capsule_height,
    }
}

// A dynamic body's collider: a ball as itself, whichever way it has rolled;
// any other shape by its world bounds, as wide as their wider horizontal half
// and standing on their bottom.
fn of_body(collider: &PropCollider, model: [[f32; 4]; 4]) -> GrassBender {
    if collider.shape == PropColliderShape::Ball {
        // Balls scale by the X axis, like the collider they mirror.
        let x = model[0];
        let r = collider.radius * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        let c = model[3];
        return GrassBender {
            base: [c[0], c[1] - r, c[2]],
            radius: r,
            height: 2.0 * r,
        };
    }
    let half = match collider.shape {
        PropColliderShape::Cuboid | PropColliderShape::Ball => collider.half_extents,
        PropColliderShape::Capsule => [
            collider.radius,
            collider.half_height + collider.radius,
            collider.radius,
        ],
    };
    let (min, max) = transform_aabb(half.map(|h| -h), half, model);
    GrassBender {
        base: [0.5 * (min[0] + max[0]), min[1], 0.5 * (min[2] + max[2])],
        radius: 0.5 * (max[0] - min[0]).max(max[2] - min[2]),
        height: max[1] - min[1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translated(scale: f32, at: [f32; 3]) -> [[f32; 4]; 4] {
        [
            [scale, 0.0, 0.0, 0.0],
            [0.0, scale, 0.0, 0.0],
            [0.0, 0.0, scale, 0.0],
            [at[0], at[1], at[2], 1.0],
        ]
    }

    #[test]
    fn a_character_stands_its_capsule_on_its_origin() {
        let mut rig = CharacterRig::new(
            Default::default(),
            concinnity_core::gfx::render_types::SkinnedIndex(0),
            translated(1.0, [1.0, 2.0, 3.0]),
            0.6,
            0.3,
        );
        rig.position = [4.0, 2.0, -1.0];
        let b = of_rig(&rig);
        assert_eq!(b.base, [4.0, 2.0, -1.0]);
        assert_eq!(b.radius, 0.3);
        assert!((b.height - 1.8).abs() < 1e-6);
    }

    #[test]
    fn the_player_stands_under_the_eye() {
        let body = RigidBody::default();
        let b = of_player([1.0, 5.0, 2.0], &body);
        assert_eq!(b.base, [1.0, 5.0 - body.capsule_height, 2.0]);
        assert_eq!(b.radius, body.capsule_radius);
        assert_eq!(b.height, body.capsule_height);
    }

    #[test]
    fn a_body_stands_on_the_bottom_of_its_scaled_bounds() {
        let ball = PropCollider {
            shape: PropColliderShape::Ball,
            radius: 0.5,
            ..PropCollider::default()
        };
        let b = of_body(&ball, translated(2.0, [3.0, 1.0, -2.0]));
        assert_eq!(b.base, [3.0, 0.0, -2.0]);
        assert_eq!(b.radius, 1.0);
        assert_eq!(b.height, 2.0);
        // Rolled a quarter turn about Z, it is the same ball.
        let rolled = [
            [0.0, 2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [3.0, 1.0, -2.0, 1.0],
        ];
        assert_eq!(of_body(&ball, rolled).radius, 1.0);

        let crate_box = PropCollider {
            shape: PropColliderShape::Cuboid,
            half_extents: [0.4, 0.25, 0.9],
            ..PropCollider::default()
        };
        let b = of_body(&crate_box, translated(1.0, [0.0, 0.25, 0.0]));
        assert_eq!(b.base, [0.0, 0.0, 0.0]);
        assert_eq!(b.radius, 0.9);
        assert_eq!(b.height, 0.5);

        let capsule = PropCollider {
            shape: PropColliderShape::Capsule,
            radius: 0.2,
            half_height: 0.5,
            ..PropCollider::default()
        };
        let b = of_body(&capsule, translated(1.0, [0.0, 0.7, 0.0]));
        assert!(b.base[1].abs() < 1e-6);
        assert_eq!(b.radius, 0.2);
        assert!((b.height - 1.4).abs() < 1e-6);
    }
}
