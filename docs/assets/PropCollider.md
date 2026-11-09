<!-- Auto-generated - do not edit. -->

# PropCollider

Collision volume attached to a [Prop](Prop.md).

The shape dimensions are in the prop's local space and are scaled by the
prop's `scale`. `ball` and `capsule` use the X scale component (they assume
uniform scaling).

A `ball` also keeps the sky's light off what lies around it, the way a ball
resting on the ground darkens the ground where it touches. The eight balls
nearest the camera do.

## Parameters

- `shape`: A string (see [PropColliderShape](PropColliderShape.md)). Collision shape: `aabb` (alias `cuboid`), `ball` (alias `sphere`), or `capsule`. See [`PropColliderShape`]. Defaults to `"cuboid"`.
- `half_extents`: An array of 3 floats. Box half-extents in local space [x, y, z]. Used by cuboid shapes. Defaults to `[0.5, 0.5, 0.5]`.
- `radius`: A float. Radius in local space. Used by ball and capsule shapes. Defaults to `0.5`.
- `half_height`: A float. Half the cylinder height in local space. Used by capsule shapes. Defaults to `0.5`.
- `layer`: A string. Collision layer name. Built-in layers are `world`, `prop`, `character`, and `trigger`; extra names come from [`PhysicsConfig`](PhysicsConfig.md) `layers`. Empty derives the layer from the body kind: `world` for a static prop, `prop` when a [`PropBody`](PropBody.md) makes it dynamic. Defaults to `""`.
