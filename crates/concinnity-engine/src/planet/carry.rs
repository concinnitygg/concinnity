//! Carries the world's own positions into a new simulated frame: every root
//! entity's transform, the camera, the character rigs, and the sky this tick
//! already turned. Children ride their parents; the sky's pivot only turns.

use alloc::vec::Vec;

use super::{PlanetFrame, Rebase};
use crate::components::{Camera3D, CharacterRig, Parent, SkyRotation, Transform};
use crate::ecs::PipelineContext;
use crate::sky::SkyOrientation;

/// Carry everything the world positions itself into the frame `rebase` moves
/// it to, `frame` being the frame after the move.
pub fn carry_world(ctx: &mut PipelineContext<'_>, rebase: &Rebase, frame: &PlanetFrame) {
    let roots: Vec<_> = ctx
        .query_with_entity::<Transform>()
        .map(|(entity, _)| entity)
        .filter(|&e| ctx.get::<Parent>(e).is_none() && ctx.get::<SkyRotation>(e).is_none())
        .collect();
    for entity in roots {
        if let Some(t) = ctx.get_mut::<Transform>(entity) {
            t.position = rebase.apply_point(t.position);
            t.rotation_deg = rebase.apply_euler_deg(t.rotation_deg);
        }
    }
    for camera in ctx.query_mut::<Camera3D>() {
        camera.position = rebase.apply_point(camera.position);
        camera.desired_move = rebase.apply_vector(camera.desired_move);
        camera.up = frame.up_at(camera.position);
        camera.recompose_view();
    }
    for rig in ctx.query_mut::<CharacterRig>() {
        rig.position = rebase.apply_point(rig.position);
    }
    if let Some(sky) = ctx.resource::<SkyOrientation>().copied() {
        let sky = rebase.apply_sky(sky);
        let pivots: Vec<_> = ctx
            .query_with_entity::<SkyRotation>()
            .map(|(entity, _)| entity)
            .collect();
        for pivot in pivots {
            if let Some(t) = ctx.get_mut::<Transform>(pivot) {
                t.rotation_deg = sky.euler_deg();
            }
        }
        ctx.insert_resource(sky);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::cook::Camera3D as Camera3DArgs;
    use crate::ecs::World;
    use crate::planet::{LocalFrame, PlanetShape};

    fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < tol)
    }

    const SHAPE: PlanetShape = PlanetShape {
        center: [0.0, -50_000.0, 0.0],
        radius: 50_000.0,
        amplitude: 0.0,
        feature_size: 1_000.0,
        octaves: 1,
        seed: 0,
        continents: crate::planet::Continents::NONE,
        edits: None,
    };

    // A walk of a kilometer and the frame recentered under it.
    fn moved() -> (Rebase, PlanetFrame) {
        let at = [1_000.0, -10.0, 0.0];
        let up = SHAPE.up_at(LocalFrame::AUTHORED.to_world(at));
        let (frame, rebase) = LocalFrame::AUTHORED.recentered(at, up);
        (
            rebase,
            PlanetFrame {
                shape: SHAPE,
                frame,
            },
        )
    }

    #[test]
    fn roots_and_the_camera_move_while_children_and_the_sky_stay() {
        let mut world = World::new();
        let root = world.push(Transform {
            position: [1_001.0, -9.0, 2.0],
            ..Default::default()
        });
        let child = world.push(Transform {
            position: [0.0, 1.0, 0.0],
            ..Default::default()
        });
        world.insert(child, Parent(root));
        let pivot = world.push(SkyRotation::default());
        world.insert(pivot, Transform::default());
        world.push(Camera3D::bake(Camera3DArgs {
            position: [1_000.0, -8.0, 0.0],
            yaw: 0.4,
            pitch: -0.1,
            ..Default::default()
        }));
        let (rebase, frame) = moved();
        carry_world(&mut world.context(), &rebase, &frame);

        let ctx = world.context();
        let r = ctx.get::<Transform>(root).unwrap();
        assert!(close(
            r.position,
            rebase.apply_point([1_001.0, -9.0, 2.0]),
            1e-3
        ));
        assert!(close(r.position, [1.0, 1.0, 2.0], 0.05), "{:?}", r.position);
        assert_eq!(
            ctx.get::<Transform>(child).unwrap().position,
            [0.0, 1.0, 0.0]
        );
        assert_eq!(ctx.get::<Transform>(pivot).unwrap().position, [0.0; 3]);
        let cam = ctx.query::<Camera3D>().next().unwrap();
        assert!(
            close(cam.position, [0.0, 2.0, 0.0], 0.05),
            "{:?}",
            cam.position
        );
        assert!(close(cam.up, [0.0, 1.0, 0.0], 1e-4), "{:?}", cam.up);
        assert_eq!(
            (cam.yaw, cam.pitch),
            (0.4, -0.1),
            "the heading carries across"
        );
    }

    // The sky turned before the move is carried with it, so the sun lights the
    // carried world from where it lit the old one on the very tick of the move.
    #[test]
    fn the_sky_published_before_the_move_is_carried_with_it() {
        let mut world = World::new();
        let pivot = world.push(SkyRotation::default());
        world.insert(pivot, Transform::default());
        let sky = SkyOrientation::new([0.0, 0.0, -1.0], 286.5);
        world.insert_resource(sky);
        let (rebase, frame) = moved();
        carry_world(&mut world.context(), &rebase, &frame);

        let ctx = world.context();
        let carried = *ctx.resource::<SkyOrientation>().unwrap();
        let sun = [1.0, 0.0, 0.35];
        let want = rebase.apply_vector(sky.rotate(sun));
        assert!(
            close(carried.rotate(sun), want, 1e-5),
            "{:?}",
            carried.rotate(sun)
        );
        assert_eq!(
            ctx.get::<Transform>(pivot).unwrap().rotation_deg,
            carried.euler_deg()
        );
    }

    // The camera sees the carried world exactly as it saw the old one: the
    // view of a carried point is the old view of the point.
    #[test]
    fn the_carried_camera_sees_the_same_picture() {
        let mut world = World::new();
        let eye = [1_000.0, -8.0, 0.0];
        let mut camera = Camera3D::bake(Camera3DArgs {
            position: eye,
            yaw: 1.2,
            pitch: 0.2,
            ..Default::default()
        });
        camera.up = PlanetFrame {
            shape: SHAPE,
            frame: LocalFrame::AUTHORED,
        }
        .up_at(eye);
        camera.recompose_view();
        let before = camera.view_matrix;
        world.push(camera);
        let (rebase, frame) = moved();
        carry_world(&mut world.context(), &rebase, &frame);
        let after = world
            .context()
            .query::<Camera3D>()
            .next()
            .unwrap()
            .view_matrix;
        let view = |m: [[f32; 4]; 4], p: [f32; 3]| -> [f32; 3] {
            core::array::from_fn(|i| m[0][i] * p[0] + m[1][i] * p[1] + m[2][i] * p[2] + m[3][i])
        };
        for p in [[1_010.0, -9.0, 4.0], [990.0, -7.5, -20.0]] {
            let a = view(before, p);
            let b = view(after, rebase.apply_point(p));
            assert!(close(a, b, 2e-3), "{a:?} {b:?}");
        }
    }
}
