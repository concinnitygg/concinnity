// Init-time work on the settings menu: publish live values, adopt the
// backend's display modes and gray what the device lacks, before UiInputSystem
// drains the menu.

use alloc::vec::Vec;

use crate::components::{HitRegion, ScrollPanel, TextLabel, Window};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::protocol::FrameRateCap;
use crate::render::backend;
use crate::settings::quality::{auto_upscaling, ceiling_preset, preset_label};
use crate::settings::registry::{self, DEFAULT_VOLUME};
use crate::settings::rows::{quality_cycle, quality_toggle};
use crate::settings::{
    QualityCeiling, QualityPreset, SettingKey, SettingValues, StoredSettings, UserGraphicsSettings,
    WorldSettings, levels, options,
};
use crate::window::display_mode;
use crate::window::display_mode::DisplayModes;
use concinnity_core::platform::{CompiledPlatforms, GraphicsApiStatus};

use super::SettingsState;
use super::rows::{DISABLED_ROW_COLOR, expand_dim_set, menu_keys, show};

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
    /// The ceiling on this GPU of the preset's [`ceiling_preset`].
    pub ceiling: &'a QualityCeiling,
    /// The world's window, when it declares one.
    pub window: Option<Window>,
    /// The display the window would open on, when it can be read before the
    /// window exists.
    pub display: Option<display_mode::DisplayProbe>,
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
        if let Some(mode) = init.session.window_mode.or(user.window_mode) {
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
        let fullscreen = state.resolution;
        state.output = display_mode::predicted_output(&state.window_args, fullscreen, init.display);
        let tier = ceiling_preset(init.preset, user.customized_from);
        state.auto_upscaling = auto_upscaling(tier, &init.gpu_profile, state.output);
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

    /// Publish the live value of every setting the overlay shows or operates
    /// as [`SettingValues`] before the first render, so a persisted or
    /// authored choice shows instead of the build's placeholder, and capture
    /// the rows a runtime change grays.
    pub fn sync_menu_labels(&mut self, ctx: &mut PipelineContext<'_>, stored: &StoredSettings) {
        if ctx.resource::<SettingValues>().is_none() {
            ctx.insert_resource(SettingValues::default());
        }
        let keys = menu_keys(ctx);
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
        publish_option_values(ctx, &keys, |key| match key {
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
        show(ctx, SettingKey::GraphicsQuality, &label);
        if let Some(status) = ctx.resource::<GraphicsApiStatus>().copied() {
            let label = registry::graphics_api_label(status.current, status.reverted);
            show(ctx, SettingKey::GraphicsApi, &label);
        }
        self.publish_rows(ctx, stored, &keys);
        self.capture_perf_sub_rows(ctx);
        self.capture_resolution_row(ctx);
        self.capture_ssr_row(ctx);
        self.gate_launch_fixed_rows(ctx);
    }

    /// Adopt what the built backend reports about the display, when there is a
    /// backend: its modes (possibly none, falling back to the static presets)
    /// and the display's own mode. Then publish the Resolution row's mode list
    /// and value and the resolved frame-rate cap for the runtime-level pacer.
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
            show(ctx, SettingKey::Resolution, &mode.label());
        }
    }
}

// Publish the option text of each of `keys` that has a static option list.
// `current_index` maps a setting key to the index of its active option (None
// for a key without one).
fn publish_option_values(
    ctx: &mut PipelineContext<'_>,
    keys: &[SettingKey],
    current_index: impl Fn(SettingKey) -> Option<usize>,
) {
    for &key in keys {
        if let (Some(opts), Some(idx)) = (options(key), current_index(key))
            && let Some(text) = opts.get(idx)
        {
            show(ctx, key, text);
        }
    }
}

/// Gray out and disable every settings row whose feature the device cannot
/// provide (e.g. ray-traced reflections on a GPU without hardware ray
/// tracing). A disabled `HitRegion` is dropped by `UiInputSystem` so it never
/// hovers or fires; the row's labels are recolored to a muted gray, and its
/// shown value is marked as unsupported.
pub fn gate_unavailable_rows(
    ctx: &mut PipelineContext<'_>,
    caps: &backend::DeviceCapabilities,
    vendor: backend::GpuVendor,
) {
    let compiled = ctx.resource::<CompiledPlatforms>().map_or(&[][..], |c| c.0);
    // A single-API build's Graphics API row names the API it renders with,
    // which the device does support: it is grayed without the suffix.
    disable_rows(
        ctx,
        |key| !registry::setting_available(key, caps, vendor, compiled),
        |key| key != SettingKey::GraphicsApi,
    );
}

// Disable every settings row whose key is `disabled`, gray its labels, and mark
// its shown value unsupported where `marked`.
pub(super) fn disable_rows(
    ctx: &mut PipelineContext<'_>,
    disabled: impl Fn(SettingKey) -> bool,
    marked: impl Fn(SettingKey) -> bool,
) {
    // Both stepper regions of a row reference its value label, so the value
    // label is the row's anchor into the scroll element list.
    let mut gated: hashbrown::HashSet<AssetId> = hashbrown::HashSet::new();
    let mut marked_keys: Vec<SettingKey> = Vec::new();
    for r in ctx.query_mut::<HitRegion>() {
        let Some(key) = r.setting().filter(|&key| disabled(key)) else {
            continue;
        };
        r.disabled = true;
        if marked(key) && !marked_keys.contains(&key) {
            marked_keys.push(key);
        }
        if let Some(label) = r.label.map(crate::ecs::reference::Ref::id) {
            gated.insert(label);
        }
    }
    if let Some(values) = ctx.resource_mut::<SettingValues>() {
        for key in marked_keys {
            if let Some(text) = values.text(key) {
                let marked = registry::unsupported_label(text);
                values.set_text(key, &marked);
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
    use concinnity_core::platform::Platform;

    fn region(key: &str, label: u32) -> HitRegion {
        HitRegion {
            action: Some(
                serde_json::from_value(
                    serde_json::json!({"step": {"binding": alloc::format!("setting:{key}"), "by": 1}}),
                )
                .unwrap(),
            ),
            label: Some(Ref::new(AssetId(label))),
            ..Default::default()
        }
    }

    fn shown(world: &World, key: SettingKey) -> &str {
        world
            .resource::<SettingValues>()
            .and_then(|v| v.text(key))
            .unwrap_or_default()
    }

    fn color(world: &mut World, id: u32) -> [f32; 3] {
        world
            .context()
            .get_by_id::<TextLabel>(AssetId(id))
            .map(|l| l.color)
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
            display: None,
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

    // Each shown setting publishes its live option, and the Graphics API row
    // names the running API.
    #[test]
    fn menu_labels_sync_to_the_live_values() {
        let mut world = World::new();
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
        assert_eq!(shown(&world, SettingKey::FpsCap), "144");
        assert_eq!(shown(&world, SettingKey::GraphicsApi), "Vulkan");
    }

    // An empty mode list falls back to the presets, and the Resolution row
    // names the effective mode.
    #[test]
    fn adopted_display_modes_publish_and_label_the_row() {
        let mut world = World::new();
        world.insert_resource(SettingValues::default());
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
        assert_eq!(
            shown(&world, SettingKey::Resolution),
            state.display_modes[idx].label()
        );
    }

    // A row the device cannot provide is disabled, grayed, and marked; one it
    // can stays live.
    #[test]
    fn unavailable_rows_are_gated() {
        let mut world = World::new();
        let mut values = SettingValues::default();
        values.set_text(SettingKey::RayTracedReflections, "On");
        values.set_text(SettingKey::Vsync, "On");
        world.insert_resource(values);
        for id in [1, 2] {
            world.push_identified(AssetId(id), TextLabel::default());
        }
        world.push(region("ray_traced_reflections", 1));
        world.push(region("vsync", 2));
        let caps = backend::DeviceCapabilities {
            ray_tracing: false,
            ..backend::DeviceCapabilities::ALL
        };
        gate_unavailable_rows(&mut world.context(), &caps, backend::GpuVendor::Other);
        assert_eq!(
            shown(&world, SettingKey::RayTracedReflections),
            "On (not supported)"
        );
        assert_eq!(shown(&world, SettingKey::Vsync), "On");
        assert_eq!(color(&mut world, 1), DISABLED_ROW_COLOR);
        assert_ne!(color(&mut world, 2), DISABLED_ROW_COLOR);
        let disabled: Vec<bool> = world.query::<HitRegion>().map(|r| r.disabled).collect();
        assert_eq!(disabled, [true, false]);
    }
}
