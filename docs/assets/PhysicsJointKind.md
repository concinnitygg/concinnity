<!-- Auto-generated - do not edit. -->

# PhysicsJointKind

The constraint shape a `PhysicsJoint` declares.

## Values

- `Fixed`: All 6 degrees of freedom locked. The bodies move and rotate as one rigid assembly relative to their anchors. Use to weld two props together.
- `Revolute`: Single rotational axis. Rotation around `axis` (in each body's local frame) is free; everything else is locked. The canonical door hinge.
- `Spherical`: Three rotational axes free, all translation locked. Ball-and-socket joint: the canonical rope link or a hip socket.
- `Prismatic`: Single translational axis. Sliding along `axis` is free; rotation and the other two translational axes are locked. The canonical slider / piston.
