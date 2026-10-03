// Stats HUD schema.

use crate::components::TextLabel;
use crate::ecs::Ref;

/// Requests the default on-screen stats HUD. Drives a set of
/// [TextLabel](#textlabel) chips with live engine stats, refreshed on a fixed
/// interval.
///
/// Each label field, when set, receives one chip: `fps_label` the averaged
/// frame rate, `gpu_wait_label` the part of the frame the CPU spent blocked on
/// the GPU, `vram_label` the GPU-memory use, `ram_label` the host process
/// memory (resident set size, against the memory budget when known), `ev_label`
/// the auto-exposure value, and `edr_label` the HDR headroom multiplier. Chips
/// whose stat is unavailable stay blank. The frame-rate, blocked-on-GPU, and
/// GPU-memory chips are shown or hidden from the in-game video settings
/// ("Display performance stats"); the host-memory, exposure, and HDR chips show
/// whenever their reading is available.
///
/// `GPU WAIT` is the frame's fence / semaphore and swapchain-acquire time, the
/// share of the frame the CPU spent waiting rather than working. It reads near
/// the frame time on a GPU-bound scene and near zero on a CPU-bound one.
///
/// The chips are packed into a tight strip anchored at the top-left of the
/// window, left to right in the order fps, gpu wait, vram, ram, ev, edr; a blank chip
/// reserves no width, so hidden readouts leave no gap. Their on-screen position
/// is fixed by the engine rather than the authored coordinates.
///
/// Developer-facing readouts (per-pass GPU timings, cursor position, live
/// camera pose) live on the separate [DebugHud](#debughud), toggled with F1.
///
/// A world that declares a [MainMenu](#mainmenu) receives a `StatHud` from the
/// build when it declares none, since the menu's performance-stats toggles
/// drive the chips, and any label field left unset receives a chip at start.
/// So the example below is only needed to restyle the chips or run a HUD
/// without a menu. Declare an [EngineDefaults](#enginedefaults) with
/// `"hud": false` to leave the chips unfilled.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct StatHud {
    /// [TextLabel](#textlabel) that receives the frame-rate chip text.
    pub fps_label: Option<Ref<TextLabel>>,
    /// [TextLabel](#textlabel) that receives the blocked-on-GPU chip text.
    pub gpu_wait_label: Option<Ref<TextLabel>>,
    /// [TextLabel](#textlabel) that receives the GPU-memory chip text.
    pub vram_label: Option<Ref<TextLabel>>,
    /// [TextLabel](#textlabel) that receives the host-memory (RSS) chip text.
    pub ram_label: Option<Ref<TextLabel>>,
    /// [TextLabel](#textlabel) that receives the auto-exposure chip text.
    pub ev_label: Option<Ref<TextLabel>>,
    /// [TextLabel](#textlabel) that receives the HDR-headroom chip text.
    pub edr_label: Option<Ref<TextLabel>>,
}
