// src/editor/hook/drag/mod.rs
//
// EditorHook: the pointer gestures. Each module owns one press-to-release drag
// over the viewport -- the state it arms on the press, what it does with the
// cursor each frame it is held, and what it commits on the release. The press
// that arms one and the priority between them is the hook's routing
// (`hook/routing.rs`); what a gesture edits, it edits through the same commit
// path as any panel (`hook/edits.rs`).
//
// `gizmo` drags a selected asset's transform along the manipulator's axes,
// `marquee` sweeps a rectangle over the viewport to replace the selection,
// `content` carries an asset out of the Content panel to a spot in the world,
// and `shape` drags one of the character-shape sliders.

pub(super) mod content;
pub(super) mod gizmo;
pub(super) mod marquee;
pub(super) mod shape;
