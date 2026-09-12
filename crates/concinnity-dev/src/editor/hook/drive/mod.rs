// src/editor/hook/drive/mod.rs
//
// EditorHook: the per-frame drives. Each module takes one piece of the editor's
// live furniture and, every frame, reconciles what the world shows with what
// the hook's state says it should show: it places or hides the furniture's
// reserved assets, reads this frame's input against them, and reports what was
// hit. What state a drive needs it declares here and the hook holds as a field
// (`CreateMenu`, `CameraGlide`, `ModalState`, `OrbitDrag`), so a drive is
// re-entered fresh each frame and commits through `hook/edits.rs` like any
// other edit.
//
// `axes`, `billboard` and `outline` draw world-space furniture (the origin
// axes, the pickable billboards for assets with no mesh, the selection
// outlines); `modal`, `notify`, `create_menu` and `view_menu` drive the
// overlays that are not registered panels; `orbit`, `glide` and `cinematic`
// move the camera; `trace` exchanges the running world's behavior trace with
// the Behavior panel.

pub(super) mod axes;
pub(super) mod billboard;
pub(super) mod cinematic;
pub(super) mod create_menu;
pub(super) mod glide;
pub(super) mod modal;
pub(super) mod notify;
pub(super) mod orbit;
pub(super) mod outline;
pub(super) mod trace;
pub(super) mod view_menu;
