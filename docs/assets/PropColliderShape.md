<!-- Auto-generated - do not edit. -->

# PropColliderShape

The collision volume a [PropCollider](PropCollider.md)'s `shape` names. The
single accepted vocabulary: the build rejects an authored name this does not
recognize, and the runtime resolves the same name through it.

## Values

- `Cuboid`: Box sized by `half_extents`. Authored as `aabb` or `cuboid`.
- `Ball`: Sphere sized by `radius`. Authored as `ball` or `sphere`.
- `Capsule`: Capsule sized by `radius` and `half_height`.
