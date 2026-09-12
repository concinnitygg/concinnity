//! What a backend reports about itself and its device.
//!
//! The capability flags and the coarse GPU class the settings resolver reads
//! once at init, the per-frame counters the profiler reads, and the two
//! readbacks (`screenshot`, `read_cull_status`) that pull a rendered result
//! back to the CPU. The classification rule itself is here, shared by all
//! three backends so they agree on the same GPU, and unit-testable without
//! one.

use crate::gfx::profile::RenderStats;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

/// GPU/device capability flags, queried from the backend once it is built.
/// Surfaced so the settings menu can gray out (and make inert) toggles the
/// device cannot honor -- e.g. ray-traced reflections on a GPU without hardware
/// ray tracing. Mirrors an RHI-style capability set: a handful of bools held in
/// memory and re-queried each launch, never persisted, so it is always correct
/// for the current device + driver.
#[derive(Clone, Copy, Debug)]
pub struct DeviceCapabilities {
    /// Hardware ray tracing for the RT-reflections pass: DXR 1.1 on DirectX, the
    /// ray-query device extensions on Vulkan (and not under XeSS), and
    /// `MTLDevice::supportsRaytracing` on Metal.
    pub ray_tracing: bool,
    /// Whether the upscaler implementation is a choice (FSR3 / DLSS / XeSS)
    /// rather than fixed. DirectX and Vulkan offer the selection; Metal always
    /// upscales through MetalFX, so the row has nothing to pick.
    pub selectable_upscaler: bool,
    /// Whether a retired build-time draw slot may be recycled by a runtime
    /// clone. Metal's per-frame RT topology refresh re-admits recycled
    /// build-time slots; DirectX / Vulkan key their cull BVH + RT tables to
    /// fixed build-time indices and cannot refit, so only the runtime-append
    /// region recycles there. Read by the engine's draw-slot allocator.
    pub reuses_build_slots: bool,
    /// Whether a built draw slot's material and cull distance may be rewritten
    /// in place ([`LiveEdit::set_draw_material`] /
    /// [`LiveEdit::set_draw_cull_distance`]). Metal rebuilds its per-object
    /// buffer from the draw list every frame, so a rewritten slot draws with the
    /// new material next frame; DirectX / Vulkan bake per-object material state
    /// at build time and would keep drawing the old one. Read by the editor's
    /// live draw seam, which sends the edit to a world rebuild instead.
    pub rewrites_draws: bool,
}

impl DeviceCapabilities {
    /// Every capability present. The trait default, so a backend that does not
    /// report capabilities never wrongly disables a toggle (it keeps the prior
    /// behavior: the feature no-ops with a warning on an incapable device).
    pub const ALL: Self = Self {
        ray_tracing: true,
        selectable_upscaler: true,
        reuses_build_slots: true,
        rewrites_draws: true,
    };
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self::ALL
    }
}

/// Coarse GPU vendor class, derived per backend from the adapter's reported
/// vendor id (DirectX / Vulkan) or unified-memory / Apple-family signals (Metal).
/// Used only to pick default quality and to gate vendor-specific options (e.g.
/// which upscalers to offer); never persisted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuVendor {
    /// Apple silicon.
    Apple,
    /// NVIDIA.
    Nvidia,
    /// AMD.
    Amd,
    /// Intel.
    Intel,
    /// A vendor the probe does not recognize.
    Other,
}

/// Coarse performance class for default-quality selection, ordered low -> high so
/// callers can compare with `>=`. Each backend maps its native signals (memory
/// budget, discrete / integrated, Apple GPU family) onto this via `classify_tier`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuTier {
    /// Unknown hardware: the conservative default, never the top preset. Sorts
    /// lowest so a comparison-based resolver treats it as the floor.
    Unknown,
    /// Integrated / low-power GPU: the lowest quality tier.
    Integrated,
    /// Older or small discrete GPU, or an Apple base M-series: entry quality.
    EntryDiscrete,
    /// Mainstream discrete GPU, or an Apple Pro: mid quality.
    MidDiscrete,
    /// Enthusiast discrete GPU, or an Apple Max / Ultra: high quality.
    HighDiscrete,
}

/// A coarse, Copy snapshot of the active GPU's class, queried from the backend
/// once it is built (mirrors `DeviceCapabilities`). Read at init to choose
/// sensible default graphics quality; never persisted, re-queried each launch so
/// it is always correct for the current device + driver. The GPU *name* is
/// deliberately omitted (it is not `Copy`); a backend exposes the name separately
/// when a UI needs it.
#[derive(Clone, Copy, Debug)]
pub struct GpuProfile {
    /// The GPU's vendor.
    pub vendor: GpuVendor,
    /// The performance tier the probe placed the GPU in.
    pub tier: GpuTier,
    /// Dedicated VRAM on a discrete GPU, or the recommended working-set on a
    /// unified-memory GPU. 0 when the backend / driver cannot report it.
    pub memory_budget_bytes: u64,
    /// Whether the GPU shares memory with the host.
    pub unified_memory: bool,
    /// Whether the GPU is a discrete card.
    pub discrete: bool,
}

impl GpuProfile {
    /// Conservative fallback for a backend that does not report a profile:
    /// unknown hardware picks the cautious baseline, never a high preset. The
    /// opposite default from `DeviceCapabilities::ALL` -- a feature gate fails
    /// open (assume capable, no-op with a warning if not), but quality
    /// auto-config fails safe (assume modest, never overdrive a weak GPU).
    pub const UNKNOWN: Self = Self {
        vendor: GpuVendor::Other,
        tier: GpuTier::Unknown,
        memory_budget_bytes: 0,
        unified_memory: false,
        discrete: false,
    };
}

impl Default for GpuProfile {
    fn default() -> Self {
        Self::UNKNOWN
    }
}

/// The cheap signals every backend can gather about its GPU, mapped to a coarse
/// `GpuTier` by one shared rule so the three backends classify consistently and
/// the mapping is unit-testable without a GPU. The backends differ in what they
/// can report (Apple exposes a GPU family; DirectX / Vulkan expose a VRAM figure
/// and a discrete / integrated flag), so this carries the union and the rule
/// uses whichever signals are present.
pub struct GpuClassInput {
    /// The GPU's vendor.
    pub vendor: GpuVendor,
    /// Device memory the driver reports as budgeted for this process.
    pub memory_budget_bytes: u64,
    /// Whether the GPU is a discrete card.
    pub discrete: bool,
    /// Apple GPU family generation rank (7 = M1 .. 10 = M4), or 0 for a non-Apple
    /// GPU. Apple silicon classifies by generation; everything else by VRAM.
    pub apple_family: u8,
}

/// The Apple GPU family generation rank a device name implies, or 0 when the name
/// is not an Apple silicon GPU. Metal reads the rank straight off the device
/// (`MTLDevice::supportsFamily`); Vulkan has no equivalent query, so a MoltenVK
/// build recovers it from the reported device name ("Apple M2 Max"). Without it
/// Apple silicon falls through `classify_tier`'s integrated branch and the two
/// backends disagree on the same GPU. `M<n>` maps to `n + 6`, matching Metal's
/// `MTLGPUFamily::Apple7` = M1.
pub fn apple_family_from_device_name(name: &str) -> u8 {
    let Some(rest) = name.strip_prefix("Apple M") else {
        return 0;
    };
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    match digits.parse::<u8>() {
        Ok(n) if n >= 1 => n.saturating_add(6),
        _ => 0,
    }
}

/// Map the gathered GPU signals to a coarse performance tier. Apple silicon is
/// classified by GPU family generation (family alone cannot separate base from
/// Pro / Max / Ultra within a generation -- a working-set refinement can split
/// them later); a non-Apple integrated / low-power GPU is the lowest tier; a
/// discrete GPU is bucketed by dedicated VRAM. An unreporting device (no memory,
/// not discrete) stays `Unknown` so the resolver uses the conservative baseline.
pub fn classify_tier(input: &GpuClassInput) -> GpuTier {
    const GB: u64 = 1 << 30;
    // Apple silicon: classify by GPU family generation.
    if input.vendor == GpuVendor::Apple && input.apple_family >= 7 {
        return match input.apple_family {
            7 => GpuTier::EntryDiscrete, // M1 class
            8 => GpuTier::MidDiscrete,   // M2 class
            _ => GpuTier::HighDiscrete,  // M3 / M4 and newer
        };
    }
    // Any non-Apple integrated / low-power GPU is the lowest tier (Apple silicon
    // is unified too, but it returned above via its family branch).
    if !input.discrete {
        return GpuTier::Integrated;
    }
    // Discrete GPU: bucket by dedicated VRAM.
    match input.memory_budget_bytes {
        0 => GpuTier::Unknown,
        b if b >= 12 * GB => GpuTier::HighDiscrete,
        b if b >= 6 * GB => GpuTier::MidDiscrete,
        _ => GpuTier::EntryDiscrete,
    }
}

/// The backend's own report: device capabilities, GPU class, frame counters,
/// and the CPU-side readbacks.
///
/// All defaulted to the conservative answer (no capability, `Unknown` tier,
/// zeroed counters, an unsupported readback), so a query never has to ask
/// whether the backend implements it.
pub trait BackendProbe {
    /// Device capability flags, queried from the GPU once the backend is built.
    /// Read by GraphicsSystem to gray out + disable settings rows the device
    /// cannot honor. Default: all capable, so a backend that does not report
    /// capabilities keeps every toggle live (the feature then no-ops with a
    /// warning on an incapable device, as before).
    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::ALL
    }

    /// Coarse GPU performance profile, queried once the backend is built. Read at
    /// init to pick default graphics quality on first launch. Default: `UNKNOWN`
    /// (the conservative tier), so a backend that does not report a profile never
    /// makes the resolver auto-select a high preset.
    fn gpu_profile(&self) -> GpuProfile {
        GpuProfile::UNKNOWN
    }
    /// Per-frame draw-call / object counters. Default no-op so a backend that
    /// tracks none still satisfies the trait; all three shipping backends
    /// override it.
    fn render_stats(&self) -> RenderStats {
        RenderStats::default()
    }

    /// Capture the last presented frame to a PNG at `path` and return the saved
    /// path. Driven by the `cn debug` WS `screenshot` command for headless
    /// on-GPU render verification. Default `Err`: a backend without a capture
    /// path reports it unsupported (all current backends override this).
    fn screenshot(&mut self, path: &str) -> Result<String, String> {
        let _ = path;
        Err("screenshot capture not supported on this backend".to_string())
    }

    /// Read the GPU-driven cull's per-object status buffer back to the host,
    /// one [`crate::gfx::cull_status::CullStatus`] value per live cull record,
    /// for the most recently submitted frame.
    ///
    /// The submitted draw-call count is a CPU-side number that does not move
    /// when the GPU rejects an object, and an object the Hi-Z test correctly
    /// occluded leaves no trace in the presented pixels, so this buffer is the
    /// only observable record of what the cull decided. Driven by the `cn
    /// debug` WS `cull-status` command; synchronous (it idles the device).
    ///
    /// Default `Err`: a backend with no GPU-driven cull, or one whose readback
    /// path is not implemented, reports it unsupported.
    fn read_cull_status(&mut self) -> Result<Vec<u32>, String> {
        Err("cull-status readback not supported on this backend".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    fn input(
        vendor: GpuVendor,
        memory_budget_bytes: u64,
        discrete: bool,
        apple_family: u8,
    ) -> GpuClassInput {
        GpuClassInput {
            vendor,
            memory_budget_bytes,
            discrete,
            apple_family,
        }
    }

    #[test]
    fn unknown_profile_is_the_conservative_default() {
        // The opposite default from capabilities: quality auto-config fails safe.
        let p = GpuProfile::default();
        assert_eq!(p.tier, GpuTier::Unknown);
        assert_eq!(p.vendor, GpuVendor::Other);
        assert_eq!(p.memory_budget_bytes, 0);
        // Unknown sorts below every real tier, so a `>=` resolver treats it as
        // the floor.
        assert!(GpuTier::Unknown < GpuTier::Integrated);
        assert!(GpuTier::Integrated < GpuTier::EntryDiscrete);
        assert!(GpuTier::EntryDiscrete < GpuTier::MidDiscrete);
        assert!(GpuTier::MidDiscrete < GpuTier::HighDiscrete);
    }

    #[test]
    fn apple_family_reads_the_generation_out_of_the_device_name() {
        // The names MoltenVK reports, mapped onto Metal's family ranks.
        assert_eq!(apple_family_from_device_name("Apple M1"), 7);
        assert_eq!(apple_family_from_device_name("Apple M2 Max"), 8);
        assert_eq!(apple_family_from_device_name("Apple M3 Pro"), 9);
        assert_eq!(apple_family_from_device_name("Apple M4 Ultra"), 10);
        // A generation past what Metal's SDK names yet still ranks above M3, so
        // a newer Mac is not demoted.
        assert!(apple_family_from_device_name("Apple M9") > 9);
    }

    #[test]
    fn non_apple_device_names_report_no_family() {
        for name in [
            "NVIDIA GeForce RTX 4090",
            "AMD Radeon RX 7900 XTX",
            "Intel(R) Arc(tm) A770",
            // Apple's own non-M naming, and a truncated / malformed report.
            "Apple A17 Pro",
            "Apple M",
            "Apple MX",
            "",
        ] {
            assert_eq!(apple_family_from_device_name(name), 0, "{name}");
        }
    }

    #[test]
    fn apple_family_from_a_name_reaches_the_same_tier_metal_does() {
        // The whole point of the name probe: a MoltenVK build must land on the
        // tier the Metal backend reports for the same silicon, not on the
        // integrated floor a zero family falls through to.
        let family = apple_family_from_device_name("Apple M2 Max");
        assert_eq!(
            classify_tier(&input(GpuVendor::Apple, 32 * GB, false, family)),
            GpuTier::MidDiscrete
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Apple, 32 * GB, false, 0)),
            GpuTier::Integrated
        );
    }

    #[test]
    fn apple_silicon_classifies_by_generation() {
        // Unified memory is large on Apple silicon, but the family generation
        // (not the working-set) decides the tier, so the huge shared budget does
        // not read as a high-VRAM discrete card.
        assert_eq!(
            classify_tier(&input(GpuVendor::Apple, 16 * GB, false, 7)),
            GpuTier::EntryDiscrete // M1
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Apple, 24 * GB, false, 8)),
            GpuTier::MidDiscrete // M2
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Apple, 48 * GB, false, 9)),
            GpuTier::HighDiscrete // M3
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Apple, 64 * GB, false, 10)),
            GpuTier::HighDiscrete // M4 and newer cap at high
        );
    }

    #[test]
    fn discrete_gpu_classifies_by_vram() {
        // An Intel-Mac AMD dGPU or a PC discrete card: vendor is not Apple and
        // there is no Apple family, so VRAM buckets the tier.
        assert_eq!(
            classify_tier(&input(GpuVendor::Nvidia, 24 * GB, true, 0)),
            GpuTier::HighDiscrete
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Amd, 8 * GB, true, 0)),
            GpuTier::MidDiscrete
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Nvidia, 4 * GB, true, 0)),
            GpuTier::EntryDiscrete
        );
        // A discrete card that reports no memory budget is left Unknown rather
        // than guessed high.
        assert_eq!(
            classify_tier(&input(GpuVendor::Amd, 0, true, 0)),
            GpuTier::Unknown
        );
    }

    #[test]
    fn integrated_gpu_is_the_lowest_tier() {
        // Non-Apple integrated part: no dedicated memory, not unified, no Apple
        // family.
        assert_eq!(
            classify_tier(&input(GpuVendor::Intel, 0, false, 0)),
            GpuTier::Integrated
        );
    }

    #[test]
    fn vram_bucket_boundaries() {
        // Boundaries are inclusive lower bounds (>= 12 GB high, >= 6 GB mid).
        assert_eq!(
            classify_tier(&input(GpuVendor::Nvidia, 12 * GB, true, 0)),
            GpuTier::HighDiscrete
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Nvidia, 12 * GB - 1, true, 0)),
            GpuTier::MidDiscrete
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Nvidia, 6 * GB, true, 0)),
            GpuTier::MidDiscrete
        );
        assert_eq!(
            classify_tier(&input(GpuVendor::Nvidia, 6 * GB - 1, true, 0)),
            GpuTier::EntryDiscrete
        );
    }
}
