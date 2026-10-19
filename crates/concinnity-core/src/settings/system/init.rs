// The settings state graphics init resolves, and the init-time work on the
// settings menu: syncing every row's value label to the live value, adopting
// the display modes the built backend reports, and graying the rows the device
// cannot provide. Runs while the menu HitRegions / TextLabels / ScrollPanels are
// still present, before UiInputSystem drains them.

use alloc::string::String;
use alloc::vec::Vec;

use crate::components::{HitRegion, ScrollPanel, TextLabel, UiAction, Window};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{DisplayModes, FrameRateCap, PipelineContext};
use crate::platform::{CompiledPlatforms, GraphicsApiStatus};
use crate::render::backend;
use crate::settings::quality::preset_label;
use crate::settings::registry::{self, DEFAULT_VOLUME};
use crate::settings::rows::{quality_cycle, quality_toggle};
use crate::settings::{
    QualityCeiling, QualityPreset, SettingKey, StoredSettings, UserGraphicsSettings, WorldSettings,
    levels, options,
};
use crate::window::display_mode;

use super::SettingsState;
use super::rows::{DISABLED_ROW_COLOR, expand_dim_set};

/// What a built backend reports about the display it presents on.
pub struct DisplayReport {
    /// The modes the display supports, in the backend's order.
    pub modes: Vec<display_mode::DisplayMode>,
    /// The display's own mode, when the backend can read it.
    pub current: Option<display_mode::DisplayMode>,
}

/// What graphics init has decided before it builds the settings state.
pub struct SettingsInit<'a> {
    /// The stored settings-menu choices.
    pub stored: &'a StoredSettings,
    /// The detected GPU.
    pub gpu_profile: backend::GpuProfile,
    /// What the probed device can run.
    pub capabilities: backend::DeviceCapabilities,
    /// The master quality preset this launch runs.
    pub preset: QualityPreset,
    /// The preset's ceiling on this GPU.
    pub ceiling: &'a QualityCeiling,
    /// The world's window, when it declares one.
    pub window: Option<Window>,
    /// The world's authored graphics values.
    pub authored: WorldSettings,
    /// This launch's overrides, layered over the user's choices and never
    /// persisted.
    pub session: UserGraphicsSettings,
}

impl SettingsState {
    /// The state graphics init starts from: the world's authored values under
    /// the stored choices, this launch's overrides and the preset's ceiling,
    /// clamped to what the device can run.
    pub fn from_init(init: SettingsInit<'_>) -> Self {
        let user = &init.stored.graphics;
        let mut state = Self::new();
        state.persisted_graphics = user.clone();
        state.gpu_profile = init.gpu_profile;
        state.capabilities = init.capabilities;
        state.quality_preset = init.preset;
        if let Some(window) = init.window {
            state.window_args = window;
        }
        if let Some(mode) = user.window_mode {
            state.window_args.mode = mode;
        }
        // Fullscreen-only: the windowed size stays the world's authored value.
        if let Some([width, height, refresh_hz]) = user.resolution {
            state.resolution = Some(display_mode::DisplayMode {
                width,
                height,
                refresh_hz,
            });
        }
        state.authored = init.authored;
        state.session = init.session;
        state.resolved = state.resolve(user, init.ceiling);
        state.keymap = init.stored.controls.keymap.unwrap_or_default();
        state.gamepad_map = init.stored.controls.gamepad_map.unwrap_or_default();
        state
    }

    /// The world's authored graphics values.
    pub fn authored(&self) -> &WorldSettings {
        &self.authored
    }

    /// Set every settings-menu row to its live value before the first render,
    /// so a persisted or authored choice shows instead of the build's
    /// placeholder, and capture the rows a runtime change relabels.
    pub fn sync_menu_labels(&mut self, ctx: &mut PipelineContext<'_>, stored: &StoredSettings) {
        let volume_of = |gain: Option<f32>| levels::VOLUME.index_of(gain.unwrap_or(DEFAULT_VOLUME));
        let audio = &stored.audio;
        let volumes = [
            volume_of(audio.master_volume),
            volume_of(audio.music_volume),
            volume_of(audio.sfx_volume),
            volume_of(audio.voice_volume),
        ];
        let r = &self.resolved;
        let g = &r.graphics;
        let window_mode = self.window_args.mode;
        sync_setting_value_labels(ctx, |key| match key {
            SettingKey::Vsync => Some(usize::from(g.frame_pacing.vsync)),
            SettingKey::FpsCap => Some(levels::FPS_CAP.index_of(g.frame_pacing.fps_cap)),
            SettingKey::WindowMode => Some(levels::WINDOW_MODE.index_of(window_mode)),
            // Resolution is a dynamic dropdown, labeled once the backend has
            // enumerated the display modes.
            SettingKey::RenderScale => {
                Some(levels::RENDER_SCALE.index_of(g.anti_aliasing.upscale_quality))
            }
            SettingKey::UpscaleBackend => {
                Some(levels::UPSCALE_BACKEND.index_of(g.anti_aliasing.upscaler))
            }
            SettingKey::DlssPreset => Some(levels::DLSS_PRESET.index_of(g.dlss_preset)),
            SettingKey::MasterVolume => Some(volumes[0]),
            SettingKey::MusicVolume => Some(volumes[1]),
            SettingKey::SfxVolume => Some(volumes[2]),
            SettingKey::VoiceVolume => Some(volumes[3]),
            SettingKey::TemporalUpscaling => Some(usize::from(g.anti_aliasing.upscaling)),
            SettingKey::HdrDisplay => Some(usize::from(g.display.hdr)),
            SettingKey::HdrPq => Some(usize::from(g.display.pq)),
            SettingKey::PerfStats => Some(usize::from(r.perf_stats)),
            SettingKey::ShowFps => Some(usize::from(r.show_fps)),
            SettingKey::ShowVram => Some(usize::from(r.show_vram)),
            SettingKey::ShadowMapSize => {
                Some(levels::SHADOW_RESOLUTION.index_of(g.shadows.map_size))
            }
            SettingKey::ShadowUpdate => Some(levels::SHADOW_UPDATE.index_of(g.shadows.update)),
            SettingKey::ShadowDistance => {
                Some(levels::SHADOW_DISTANCE.index_of(g.shadows.distance))
            }
            SettingKey::ShadowCascades => {
                Some(levels::SHADOW_CASCADES.index_of(g.shadows.cascades))
            }
            SettingKey::Anisotropy => {
                Some(levels::ANISOTROPY.index_of(g.texture_filtering.anisotropy))
            }
            SettingKey::FramesInFlight => {
                Some(levels::FRAMES_IN_FLIGHT.index_of(g.frame_pacing.frames_in_flight))
            }
            SettingKey::OcclusionTwoPass => Some(usize::from(g.culling.two_pass_occlusion)),
            SettingKey::TextureQuality => {
                Some(levels::TEXTURE_QUALITY.index_of((r.texture_cap, r.texture_budget)))
            }
            key => quality_toggle(key)
                .map(|row| usize::from((row.get)(g)))
                .or_else(|| quality_cycle(key).map(|row| (row.index)(g))),
        });
        // After the generic sync: under Auto the master row names the resolved
        // tier ("Auto (High)"), which the static option table cannot express.
        let label = preset_label(self.quality_preset, &self.gpu_profile);
        set_setting_row_label(ctx, SettingKey::GraphicsQuality, &label);
        if let Some(status) = ctx.resource::<GraphicsApiStatus>().copied() {
            let label = registry::graphics_api_label(status.current, status.reverted);
            set_setting_row_label(ctx, SettingKey::GraphicsApi, &label);
        }
        self.init_sliders(ctx, stored);
        self.init_rebind_rows(ctx);
        self.init_cycle_value_labels(ctx);
        self.capture_perf_sub_rows(ctx);
        self.capture_resolution_row(ctx);
    }

    /// Adopt what the built backend reports about the display, when there is a
    /// backend: its modes (possibly none, falling back to the static presets)
    /// and the display's own mode. Then publish the Resolution row's mode list
    /// and value label and the resolved frame-rate cap for the runtime-level
    /// pacer.
    pub fn adopt_display(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        reported: Option<DisplayReport>,
    ) {
        if let Some(DisplayReport { modes, current }) = reported {
            self.display_modes = if modes.is_empty() {
                display_mode::fallback_modes()
            } else {
                display_mode::normalize(modes)
            };
            self.current_mode = current;
        }
        ctx.insert_resource(DisplayModes(self.display_modes.clone()));
        ctx.insert_resource(FrameRateCap(self.resolved.graphics.frame_pacing.fps_cap));
        let idx = display_mode::index_of(&self.display_modes, self.effective_resolution());
        if let Some(mode) = self.display_modes.get(idx) {
            set_setting_row_label(ctx, SettingKey::Resolution, &mode.label());
        }
    }
}

// Set the value TextLabel of every settings-row HitRegion to the live value of
// that setting. `current_index` maps a setting key to the index of its active
// option (None for an unknown key).
fn sync_setting_value_labels(
    ctx: &mut PipelineContext<'_>,
    current_index: impl Fn(SettingKey) -> Option<usize>,
) {
    let rows: Vec<(SettingKey, AssetId)> = ctx
        .query::<HitRegion>()
        .filter_map(|r| match r.action {
            Some(UiAction::Setting { key, .. }) => Some((key, r.label?.id())),
            _ => None,
        })
        .collect();
    for (key, label_id) in rows {
        let (Some(opts), Some(idx)) = (options(key), current_index(key)) else {
            continue;
        };
        if let Some(text) = opts.get(idx).copied() {
            crate::ecs::by_asset_id::set_text(ctx, label_id, text);
        }
    }
}

// Set the value label of the settings row bound to `key` to `text` directly,
// for a label that is not one of the row's static `options` (the master preset
// row's "Auto (High)", the Graphics API row, a display mode).
fn set_setting_row_label(ctx: &mut PipelineContext<'_>, key: SettingKey, text: &str) {
    let label_id = ctx.query::<HitRegion>().find_map(|r| match r.action {
        Some(UiAction::Setting { key: row_key, .. }) if row_key == key => r.label,
        _ => None,
    });
    if let Some(id) = label_id {
        crate::ecs::by_asset_id::set_text(ctx, id.id(), text);
    }
}

/// Gray out and disable every settings row whose feature the device cannot
/// provide (e.g. ray-traced reflections on a GPU without hardware ray
/// tracing). A disabled `HitRegion` is dropped by `UiInputSystem` so it never
/// hovers or fires; the row's labels are recolored to a muted gray, and the
/// value label is marked as unsupported.
pub fn gate_unavailable_rows(
    ctx: &mut PipelineContext<'_>,
    caps: &backend::DeviceCapabilities,
    vendor: backend::GpuVendor,
) {
    let compiled = ctx.resource::<CompiledPlatforms>().map_or(&[][..], |c| c.0);
    // Mark each unavailable setting's region(s) disabled and collect their
    // value-label ids (both stepper regions of a row reference its value
    // label, so this is the row's anchor into the scroll element list).
    let mut gated: hashbrown::HashSet<AssetId> = hashbrown::HashSet::new();
    // A single-API build's Graphics API row names the API it renders with,
    // which the device does support: it is grayed without the suffix.
    let mut unsuffixed: hashbrown::HashSet<AssetId> = hashbrown::HashSet::new();
    let unavailable: Vec<(SettingKey, Option<AssetId>)> = ctx
        .query::<HitRegion>()
        .filter_map(|r| match r.action {
            Some(UiAction::Setting { key, .. })
                if !registry::setting_available(key, caps, vendor, compiled) =>
            {
                Some((key, r.label.map(crate::ecs::reference::Ref::id)))
            }
            _ => None,
        })
        .collect();
    for r in ctx.query_mut::<HitRegion>() {
        if let Some(UiAction::Setting { key, .. }) = r.action
            && unavailable.iter().any(|(k, _)| *k == key)
        {
            r.disabled = true;
        }
    }
    for (key, label) in unavailable {
        if let Some(label) = label {
            gated.insert(label);
            if key == SettingKey::GraphicsApi {
                unsuffixed.insert(label);
            }
        }
    }
    if gated.is_empty() {
        return;
    }
    let rows: Vec<Vec<AssetId>> = ctx
        .query::<ScrollPanel>()
        .flat_map(|p| {
            p.rows
                .iter()
                .map(|r| r.elements.iter().map(|e| e.id()).collect())
        })
        .collect();
    for &id in gated.difference(&unsuffixed) {
        crate::ecs::by_asset_id::update::<TextLabel>(ctx, Some(id), |l| {
            let marked: String = registry::unsupported_label(&l.content);
            l.content = marked;
        });
    }
    for id in expand_dim_set(&gated, &rows) {
        crate::ecs::by_asset_id::update::<TextLabel>(ctx, Some(id), |l| {
            l.color = DISABLED_ROW_COLOR;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::{Ref, World};
    use crate::platform::Platform;
    use alloc::string::ToString;

    fn region(key: &str, label: u32) -> HitRegion {
        HitRegion {
            action: Some(
                serde_json::from_value(
                    serde_json::json!({"setting": {"key": key, "verb": "next"}}),
                )
                .unwrap(),
            ),
            label: Some(Ref::new(AssetId(label))),
            ..Default::default()
        }
    }

    fn labeled(world: &mut World, id: u32, content: &str) {
        world.push_identified(
            AssetId(id),
            TextLabel {
                content: content.to_string(),
                ..Default::default()
            },
        );
    }

    fn content(world: &mut World, id: u32) -> String {
        world
            .context()
            .get_by_id::<TextLabel>(AssetId(id))
            .map(|l| l.content.clone())
            .unwrap()
    }

    // The state resolves the stored choices over the authored values and takes
    // the stored window mode, resolution and bindings.
    #[test]
    fn from_init_layers_the_stored_choices() {
        let mut stored = StoredSettings::default();
        stored.graphics.vsync = Some(true);
        stored.graphics.window_mode = Some(crate::components::WindowMode::Borderless);
        stored.graphics.resolution = Some([1920, 1080, 60]);
        let ceiling = QualityCeiling::NONE;
        let state = SettingsState::from_init(SettingsInit {
            stored: &stored,
            gpu_profile: backend::GpuProfile::UNKNOWN,
            capabilities: backend::DeviceCapabilities::ALL,
            preset: QualityPreset::Custom,
            ceiling: &ceiling,
            window: None,
            authored: WorldSettings::default(),
            session: UserGraphicsSettings::default(),
        });
        assert!(state.resolved().graphics.frame_pacing.vsync);
        assert_eq!(
            state.window().mode,
            crate::components::WindowMode::Borderless
        );
        assert_eq!(state.resolution().map(|m| m.width), Some(1920));
        assert_eq!(state.persisted_graphics().vsync, Some(true));
    }

    // Each row's value label shows its live option, and the Graphics API row
    // names the running API.
    #[test]
    fn menu_labels_sync_to_the_live_values() {
        let mut world = World::new();
        labeled(&mut world, 1, "placeholder");
        labeled(&mut world, 2, "placeholder");
        world.push(region("fps_cap", 1));
        world.push(region("graphics_api", 2));
        world.insert_resource(GraphicsApiStatus {
            current: Platform::Vulkan,
            reverted: None,
            confirm_pending: false,
        });
        let mut state = SettingsState::for_tests();
        state.resolved.graphics.frame_pacing.fps_cap = 144;
        state.sync_menu_labels(&mut world.context(), &StoredSettings::default());
        assert_eq!(content(&mut world, 1), "144");
        assert_eq!(content(&mut world, 2), "Vulkan");
    }

    // An empty mode list falls back to the presets, and the Resolution row
    // names the effective mode.
    #[test]
    fn adopted_display_modes_publish_and_label_the_row() {
        let mut world = World::new();
        labeled(&mut world, 1, "placeholder");
        world.push(region("resolution", 1));
        let mut state = SettingsState::for_tests();
        let current = display_mode::DisplayMode {
            width: 2560,
            height: 1440,
            refresh_hz: 144,
        };
        state.adopt_display(
            &mut world.context(),
            Some(DisplayReport {
                modes: Vec::new(),
                current: Some(current),
            }),
        );
        assert!(!state.display_modes.is_empty());
        assert!(world.resource::<DisplayModes>().is_some());
        assert_eq!(world.resource::<FrameRateCap>().map(|c| c.0), Some(0));
        let idx = display_mode::index_of(&state.display_modes, current);
        assert_eq!(content(&mut world, 1), state.display_modes[idx].label());
    }

    // A row the device cannot provide is disabled, grayed, and marked; one it
    // can stays live.
    #[test]
    fn unavailable_rows_are_gated() {
        let mut world = World::new();
        labeled(&mut world, 1, "On");
        labeled(&mut world, 2, "On");
        world.push(region("ray_traced_reflections", 1));
        world.push(region("vsync", 2));
        let caps = backend::DeviceCapabilities {
            ray_tracing: false,
            ..backend::DeviceCapabilities::ALL
        };
        gate_unavailable_rows(&mut world.context(), &caps, backend::GpuVendor::Other);
        assert_eq!(content(&mut world, 1), "On (not supported)");
        assert_eq!(content(&mut world, 2), "On");
        let disabled: Vec<bool> = world.query::<HitRegion>().map(|r| r.disabled).collect();
        assert_eq!(disabled, [true, false]);
    }
}
