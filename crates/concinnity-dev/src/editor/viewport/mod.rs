// src/editor/viewport/mod.rs
//
// The editor's viewport interaction: what the pointer does over the scene
// itself, as opposed to over the HUD's panels. Each module is the pure half of
// one interaction -- the geometry, the hit-test, the math -- plus the reserved
// overlay assets it draws through. The state each one needs, and the frame that
// feeds it input, belong to the hook (`hook/drag/` for the gestures,
// `hook/drive/` for the per-frame furniture).
//
// `gizmo`, `snap` and `group_transform` are the transform manipulator: the
// handles, the step math, and the multi-member pivot. `marquee` and `highlight`
// are box-select and what a selection looks like. `axes` and `billboards` are
// the world-space furniture. `framing` and `orbit` are the camera math behind
// frame-selected and Alt+drag. `cursor` owns the in-engine mouse pointer and
// `resize` the panel edge that answers a drag.

pub(crate) mod axes;
pub(crate) mod billboards;
pub(crate) mod cursor;
pub(crate) mod framing;
pub(crate) mod gizmo;
pub(crate) mod group_transform;
pub(crate) mod highlight;
pub(crate) mod marquee;
pub(crate) mod orbit;
pub(crate) mod resize;
pub(crate) mod snap;
