//! The Streamline declarations the engine calls through, transcribed from the
//! `include/sl*.h` headers of Streamline 2.14.16. Every structure is a
//! `sl::BaseStructure` (a chain pointer, a GUID and a version) followed by its
//! own fields, and C++ references in the API are passed as pointers. The tests
//! pin each layout to the offsets the SDK's own `static_assert`s state.

use std::ffi::{c_char, c_void};

/// `sl::Result`: zero is success.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlResult(pub(crate) i32);

impl SlResult {
    pub(crate) const OK: Self = Self(0);
    /// `eErrorInvalidParameter`, which freeing a viewport that never
    /// evaluated also returns.
    pub(crate) const ERROR_INVALID_PARAMETER: Self = Self(25);

    pub(crate) fn is_ok(self) -> bool {
        self == Self::OK
    }

    /// The `sl::Result` enumerator's name, for logs.
    pub(crate) fn name(self) -> &'static str {
        const NAMES: [&str; 40] = [
            "eOk",
            "eErrorIO",
            "eErrorDriverOutOfDate",
            "eErrorOSOutOfDate",
            "eErrorOSDisabledHWS",
            "eErrorDeviceNotCreated",
            "eErrorNoSupportedAdapterFound",
            "eErrorAdapterNotSupported",
            "eErrorNoPlugins",
            "eErrorVulkanAPI",
            "eErrorDXGIAPI",
            "eErrorD3DAPI",
            "eErrorNRDAPI",
            "eErrorNVAPI",
            "eErrorReflexAPI",
            "eErrorNGXFailed",
            "eErrorJSONParsing",
            "eErrorMissingProxy",
            "eErrorMissingResourceState",
            "eErrorInvalidIntegration",
            "eErrorMissingInputParameter",
            "eErrorNotInitialized",
            "eErrorComputeFailed",
            "eErrorInitNotCalled",
            "eErrorExceptionHandler",
            "eErrorInvalidParameter",
            "eErrorMissingConstants",
            "eErrorDuplicatedConstants",
            "eErrorMissingOrInvalidAPI",
            "eErrorCommonConstantsMissing",
            "eErrorUnsupportedInterface",
            "eErrorFeatureMissing",
            "eErrorFeatureNotSupported",
            "eErrorFeatureMissingHooks",
            "eErrorFeatureFailedToLoad",
            "eErrorFeatureWrongPriority",
            "eErrorFeatureMissingDependency",
            "eErrorFeatureManagerInvalidState",
            "eErrorInvalidState",
            "eWarnOutOfVRAM",
        ];
        usize::try_from(self.0)
            .ok()
            .and_then(|i| NAMES.get(i))
            .copied()
            .unwrap_or("unknown")
    }
}

/// `sl::Feature`.
pub(crate) type Feature = u32;
pub(crate) const FEATURE_DLSS: Feature = 0;

/// `sl::BufferType` values the super-resolution evaluate is tagged with.
pub(crate) type BufferType = u32;
pub(crate) const BUFFER_DEPTH: BufferType = 0;
pub(crate) const BUFFER_MOTION_VECTORS: BufferType = 1;
pub(crate) const BUFFER_SCALING_INPUT_COLOR: BufferType = 3;
pub(crate) const BUFFER_SCALING_OUTPUT_COLOR: BufferType = 4;

/// `kSDKVersion` of the headers these declarations follow.
pub(crate) const SDK_VERSION: u64 = (2 << 48) | (14 << 32) | (16 << 16) | 0xfedc;

/// `sl::LogLevel::eDefault`.
pub(crate) const LOG_LEVEL_DEFAULT: u32 = 1;
/// `sl::LogType`.
pub(crate) const LOG_TYPE_WARN: u32 = 1;
pub(crate) const LOG_TYPE_ERROR: u32 = 2;

/// `sl::PreferenceFlags`.
pub(crate) const PREFERENCE_DISABLE_CL_STATE_TRACKING: u64 = 1 << 0;
pub(crate) const PREFERENCE_USE_MANUAL_HOOKING: u64 = 1 << 2;

/// `sl::EngineType::eCustom`.
pub(crate) const ENGINE_CUSTOM: u32 = 0;

/// `sl::RenderAPI`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RenderApi {
    D3D12 = 1,
    #[cfg_attr(
        all(not(backend_vk), not(test)),
        expect(dead_code, reason = "only the Vulkan backend initializes for Vulkan")
    )]
    Vulkan = 2,
}

/// `sl::ResourceType::eTex2d` (the enum is a `char`).
pub(crate) const RESOURCE_TEX2D: u8 = 0;

/// `sl::ResourceLifecycle::eValidUntilPresent`: the resource stays as tagged
/// for the rest of the frame, so Streamline reads it in place.
pub(crate) const LIFECYCLE_VALID_UNTIL_PRESENT: i32 = 1;

/// `sl::Boolean` (a `char`).
pub(crate) const FALSE: u8 = 0;
pub(crate) const TRUE: u8 = 1;

/// `sl::StructType`, a GUID.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StructType {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

impl StructType {
    const fn new(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Self {
        Self {
            data1,
            data2,
            data3,
            data4,
        }
    }
}

/// `sl::BaseStructure`: what every Streamline structure opens with.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct BaseStructure {
    next: *mut Self,
    struct_type: StructType,
    struct_version: usize,
}

impl BaseStructure {
    const fn new(struct_type: StructType, struct_version: usize) -> Self {
        Self {
            next: std::ptr::null_mut(),
            struct_type,
            struct_version,
        }
    }
}

/// `sl::Version`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Version {
    pub(crate) major: u32,
    pub(crate) minor: u32,
    pub(crate) build: u32,
}

/// `sl::PFun_LogMessageCallback`.
pub(crate) type LogCallback = unsafe extern "C" fn(ty: u32, msg: *const c_char);

/// `sl::Preferences` (version 1).
#[repr(C)]
pub(crate) struct Preferences {
    base: BaseStructure,
    pub(crate) show_console: bool,
    pub(crate) log_level: u32,
    pub(crate) paths_to_plugins: *const *const u16,
    pub(crate) num_paths_to_plugins: u32,
    pub(crate) path_to_logs_and_data: *const u16,
    pub(crate) allocate_callback: *const c_void,
    pub(crate) release_callback: *const c_void,
    pub(crate) log_message_callback: Option<LogCallback>,
    pub(crate) flags: u64,
    pub(crate) features_to_load: *const Feature,
    pub(crate) num_features_to_load: u32,
    pub(crate) application_id: u32,
    pub(crate) engine: u32,
    pub(crate) engine_version: *const c_char,
    pub(crate) project_id: *const c_char,
    pub(crate) render_api: RenderApi,
}

impl Preferences {
    const TYPE: StructType = StructType::new(
        0x1ca1_0965,
        0xbf8e,
        0x432b,
        [0x8d, 0xa1, 0x67, 0x16, 0xd8, 0x79, 0xfb, 0x14],
    );

    /// Nothing loaded, logged or flagged yet, for `render_api`.
    pub(crate) fn new(render_api: RenderApi) -> Self {
        Self {
            base: BaseStructure::new(Self::TYPE, 1),
            show_console: false,
            log_level: LOG_LEVEL_DEFAULT,
            paths_to_plugins: std::ptr::null(),
            num_paths_to_plugins: 0,
            path_to_logs_and_data: std::ptr::null(),
            allocate_callback: std::ptr::null(),
            release_callback: std::ptr::null(),
            log_message_callback: None,
            flags: 0,
            features_to_load: std::ptr::null(),
            num_features_to_load: 0,
            application_id: 0,
            engine: ENGINE_CUSTOM,
            engine_version: std::ptr::null(),
            project_id: std::ptr::null(),
            render_api,
        }
    }
}

/// `sl::FrameToken`: an SDK-owned object only ever handled by pointer.
#[repr(C)]
pub(crate) struct FrameToken {
    _opaque: [u8; 0],
}

/// `sl::ViewportHandle`.
#[repr(C)]
pub(crate) struct ViewportHandle {
    base: BaseStructure,
    value: u32,
}

impl ViewportHandle {
    const TYPE: StructType = StructType::new(
        0x171b_6435,
        0x9b3c,
        0x4fc8,
        [0x99, 0x94, 0xfb, 0xe5, 0x25, 0x69, 0xaa, 0xa4],
    );

    pub(crate) fn new(value: u32) -> Self {
        Self {
            base: BaseStructure::new(Self::TYPE, 1),
            value,
        }
    }

    /// The handle as an evaluate input.
    pub(crate) fn as_base(&self) -> *const BaseStructure {
        (&raw const self.base).cast()
    }
}

/// `sl::AdapterInfo`.
#[repr(C)]
pub(crate) struct AdapterInfo {
    base: BaseStructure,
    pub(crate) device_luid: *const u8,
    pub(crate) device_luid_size_in_bytes: u32,
    pub(crate) vk_physical_device: *mut c_void,
}

impl AdapterInfo {
    const TYPE: StructType = StructType::new(
        0x0677_315f,
        0xa746,
        0x4492,
        [0x9f, 0x42, 0xcb, 0x61, 0x42, 0xc9, 0xc3, 0xd4],
    );

    /// The adapter a DXGI LUID names.
    pub(crate) fn luid(luid: &[u8; 8]) -> Self {
        Self {
            base: BaseStructure::new(Self::TYPE, 1),
            device_luid: luid.as_ptr(),
            device_luid_size_in_bytes: 8,
            vk_physical_device: std::ptr::null_mut(),
        }
    }

    /// A Vulkan physical device.
    #[cfg(backend_vk)]
    pub(crate) fn vk_physical_device(physical_device: *mut c_void) -> Self {
        Self {
            base: BaseStructure::new(Self::TYPE, 1),
            device_luid: std::ptr::null(),
            device_luid_size_in_bytes: 0,
            vk_physical_device: physical_device,
        }
    }
}

/// `sl::FeatureVersion`.
#[repr(C)]
pub(crate) struct FeatureVersion {
    base: BaseStructure,
    pub(crate) version_sl: Version,
    pub(crate) version_ngx: Version,
}

impl FeatureVersion {
    const TYPE: StructType = StructType::new(
        0x6d5b_51f0,
        0x076b,
        0x486d,
        [0x99, 0x95, 0x5a, 0x56, 0x10, 0x43, 0xf5, 0xc1],
    );
}

impl Default for FeatureVersion {
    fn default() -> Self {
        Self {
            base: BaseStructure::new(Self::TYPE, 1),
            version_sl: Version::default(),
            version_ngx: Version::default(),
        }
    }
}

/// `sl::Extent`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Extent {
    pub(crate) top: u32,
    pub(crate) left: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl Extent {
    /// The `width` x `height` area at the origin.
    pub(crate) fn sized((width, height): (u32, u32)) -> Self {
        Self {
            top: 0,
            left: 0,
            width,
            height,
        }
    }
}

/// `sl::Resource`. D3D12 fills the native resource and its state; Vulkan also
/// describes the image, since Vulkan has no query for it.
#[repr(C)]
pub(crate) struct Resource {
    base: BaseStructure,
    pub(crate) ty: u8,
    pub(crate) native: *mut c_void,
    pub(crate) memory: *mut c_void,
    pub(crate) view: *mut c_void,
    pub(crate) state: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) native_format: u32,
    pub(crate) mip_levels: u32,
    pub(crate) array_layers: u32,
    pub(crate) gpu_virtual_address: u64,
    pub(crate) flags: u32,
    pub(crate) usage: u32,
    internal_flags: u16,
    reserved: u16,
}

impl Resource {
    const TYPE: StructType = StructType::new(
        0x3a9d_70cf,
        0x2418,
        0x4b72,
        [0x83, 0x91, 0x13, 0xf8, 0x72, 0x1c, 0x72, 0x61],
    );

    /// A 2D texture `native` in `state`, described no further.
    pub(crate) fn texture(native: *mut c_void, state: u32) -> Self {
        Self {
            base: BaseStructure::new(Self::TYPE, 1),
            ty: RESOURCE_TEX2D,
            native,
            memory: std::ptr::null_mut(),
            view: std::ptr::null_mut(),
            state,
            width: 0,
            height: 0,
            native_format: 0,
            mip_levels: 0,
            array_layers: 0,
            gpu_virtual_address: 0,
            flags: 0,
            usage: 0,
            internal_flags: 0,
            reserved: 0,
        }
    }

    /// Chain `range` onto the resource; the resource then borrows it by
    /// address.
    pub(crate) fn with_subresource(mut self, range: &SubresourceRange) -> Self {
        self.base.next = (&raw const range.base).cast_mut();
        self
    }
}

/// `sl::SubresourceRange`: the Vulkan subresource a tagged image's view
/// covers, chained onto its `Resource`. Without one Streamline assumes the
/// whole color aspect.
#[repr(C)]
pub(crate) struct SubresourceRange {
    base: BaseStructure,
    pub(crate) aspect_mask: u32,
    pub(crate) base_mip_level: u32,
    pub(crate) level_count: u32,
    pub(crate) base_array_layer: u32,
    pub(crate) layer_count: u32,
}

impl SubresourceRange {
    const TYPE: StructType = StructType::new(
        0x8d4c_316c,
        0xd402,
        0x4524,
        [0x89, 0xa7, 0x14, 0xe7, 0x9e, 0x63, 0x8e, 0x3a],
    );

    /// The first mip and layer of `aspect_mask`.
    pub(crate) fn single(aspect_mask: u32) -> Self {
        Self {
            base: BaseStructure::new(Self::TYPE, 1),
            aspect_mask,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
}

/// `sl::ResourceTag`.
#[repr(C)]
pub(crate) struct ResourceTag {
    base: BaseStructure,
    resource: *const Resource,
    ty: BufferType,
    lifecycle: i32,
    extent: Extent,
}

impl ResourceTag {
    const TYPE: StructType = StructType::new(
        0x4c6a_5aad,
        0xb445,
        0x496c,
        [0x87, 0xff, 0x1a, 0xf3, 0x84, 0x5b, 0xe6, 0x53],
    );

    /// `resource` tagged as `ty` over `extent`, valid for the rest of the
    /// frame. The tag borrows `resource` by address.
    pub(crate) fn new(resource: &Resource, ty: BufferType, extent: Extent) -> Self {
        Self {
            base: BaseStructure::new(Self::TYPE, 1),
            resource,
            ty,
            lifecycle: LIFECYCLE_VALID_UNTIL_PRESENT,
            extent,
        }
    }

    /// The tag as an evaluate input.
    pub(crate) fn as_base(&self) -> *const BaseStructure {
        (&raw const self.base).cast()
    }
}

/// `sl::float2`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Float2 {
    pub(crate) x: f32,
    pub(crate) y: f32,
}

/// `sl::float3`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Float3 {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,
}

impl From<[f32; 3]> for Float3 {
    fn from([x, y, z]: [f32; 3]) -> Self {
        Self { x, y, z }
    }
}

/// `sl::float4x4`: row-major, for row vectors. A column-major matrix for
/// column vectors has the same bytes, so the engine's matrices copy straight
/// in.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Float4x4(pub(crate) [[f32; 4]; 4]);

/// `sl::Constants` (version 2): the camera and motion description every
/// feature reads.
#[repr(C)]
pub(crate) struct Constants {
    pub(crate) base: BaseStructure,
    pub(crate) camera_view_to_clip: Float4x4,
    pub(crate) clip_to_camera_view: Float4x4,
    pub(crate) clip_to_lens_clip: Float4x4,
    pub(crate) clip_to_prev_clip: Float4x4,
    pub(crate) prev_clip_to_clip: Float4x4,
    pub(crate) jitter_offset: Float2,
    pub(crate) mvec_scale: Float2,
    pub(crate) camera_pinhole_offset: Float2,
    pub(crate) camera_pos: Float3,
    pub(crate) camera_up: Float3,
    pub(crate) camera_right: Float3,
    pub(crate) camera_fwd: Float3,
    pub(crate) camera_near: f32,
    pub(crate) camera_far: f32,
    pub(crate) camera_fov: f32,
    pub(crate) camera_aspect_ratio: f32,
    pub(crate) motion_vectors_invalid_value: f32,
    pub(crate) depth_inverted: u8,
    pub(crate) camera_motion_included: u8,
    pub(crate) motion_vectors_3d: u8,
    pub(crate) reset: u8,
    pub(crate) orthographic_projection: u8,
    pub(crate) motion_vectors_dilated: u8,
    pub(crate) motion_vectors_jittered: u8,
    pub(crate) min_relative_linear_depth_object_separation: f32,
}

impl Constants {
    const TYPE: StructType = StructType::new(
        0xdcd3_5ad7,
        0x4e4a,
        0x4bad,
        [0xa9, 0x0c, 0xe0, 0xc4, 0x9e, 0xb2, 0x3a, 0xfe],
    );
}

impl Default for Constants {
    // Identity transforms at the origin, every flag false and the SDK's own
    // default depth separation.
    fn default() -> Self {
        const IDENTITY: Float4x4 = Float4x4([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let zero2 = Float2 { x: 0.0, y: 0.0 };
        let zero3 = Float3::from([0.0; 3]);
        Self {
            base: BaseStructure::new(Self::TYPE, 2),
            camera_view_to_clip: IDENTITY,
            clip_to_camera_view: IDENTITY,
            clip_to_lens_clip: IDENTITY,
            clip_to_prev_clip: IDENTITY,
            prev_clip_to_clip: IDENTITY,
            jitter_offset: zero2,
            mvec_scale: zero2,
            camera_pinhole_offset: zero2,
            camera_pos: zero3,
            camera_up: zero3,
            camera_right: zero3,
            camera_fwd: zero3,
            camera_near: 0.0,
            camera_far: 0.0,
            camera_fov: 0.0,
            camera_aspect_ratio: 0.0,
            motion_vectors_invalid_value: 0.0,
            depth_inverted: FALSE,
            camera_motion_included: FALSE,
            motion_vectors_3d: FALSE,
            reset: FALSE,
            orthographic_projection: FALSE,
            motion_vectors_dilated: FALSE,
            motion_vectors_jittered: FALSE,
            min_relative_linear_depth_object_separation: 40.0,
        }
    }
}

/// `sl::DLSSMode`.
pub(crate) const DLSS_MODE_MAX_PERFORMANCE: u32 = 1;
pub(crate) const DLSS_MODE_BALANCED: u32 = 2;
pub(crate) const DLSS_MODE_MAX_QUALITY: u32 = 3;
pub(crate) const DLSS_MODE_ULTRA_PERFORMANCE: u32 = 4;
pub(crate) const DLSS_MODE_DLAA: u32 = 6;

/// `sl::DLSSPreset`.
pub(crate) const DLSS_PRESET_DEFAULT: u32 = 0;
pub(crate) const DLSS_PRESET_K: u32 = 11;
pub(crate) const DLSS_PRESET_L: u32 = 12;
pub(crate) const DLSS_PRESET_M: u32 = 13;

/// `sl::DLSSOptions` (version 3).
#[repr(C)]
pub(crate) struct DlssOptions {
    pub(crate) base: BaseStructure,
    pub(crate) mode: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) sharpness: f32,
    pub(crate) pre_exposure: f32,
    pub(crate) exposure_scale: f32,
    pub(crate) color_buffers_hdr: u8,
    pub(crate) indicator_invert_axis_x: u8,
    pub(crate) indicator_invert_axis_y: u8,
    pub(crate) dlaa_preset: u32,
    pub(crate) quality_preset: u32,
    pub(crate) balanced_preset: u32,
    pub(crate) performance_preset: u32,
    pub(crate) ultra_performance_preset: u32,
    pub(crate) ultra_quality_preset: u32,
    pub(crate) use_auto_exposure: u8,
    pub(crate) alpha_upscaling_enabled: u8,
}

impl DlssOptions {
    const TYPE: StructType = StructType::new(
        0x6ac8_26e4,
        0x4c61,
        0x4101,
        [0xa9, 0x2d, 0x63, 0x8d, 0x42, 0x10, 0x57, 0xb8],
    );
}

impl Default for DlssOptions {
    // The header's defaults: off, HDR color, unit exposure, default presets.
    fn default() -> Self {
        Self {
            base: BaseStructure::new(Self::TYPE, 3),
            mode: 0,
            output_width: u32::MAX,
            output_height: u32::MAX,
            sharpness: 0.0,
            pre_exposure: 1.0,
            exposure_scale: 1.0,
            color_buffers_hdr: TRUE,
            indicator_invert_axis_x: FALSE,
            indicator_invert_axis_y: FALSE,
            dlaa_preset: DLSS_PRESET_DEFAULT,
            quality_preset: DLSS_PRESET_DEFAULT,
            balanced_preset: DLSS_PRESET_DEFAULT,
            performance_preset: DLSS_PRESET_DEFAULT,
            ultra_performance_preset: DLSS_PRESET_DEFAULT,
            ultra_quality_preset: DLSS_PRESET_DEFAULT,
            use_auto_exposure: FALSE,
            alpha_upscaling_enabled: FALSE,
        }
    }
}

pub(crate) type PfnSlInit =
    unsafe extern "C" fn(pref: *const Preferences, sdk_version: u64) -> SlResult;
pub(crate) type PfnSlShutdown = unsafe extern "C" fn() -> SlResult;
pub(crate) type PfnSlIsFeatureSupported =
    unsafe extern "C" fn(feature: Feature, adapter: *const AdapterInfo) -> SlResult;
pub(crate) type PfnSlEvaluateFeature = unsafe extern "C" fn(
    feature: Feature,
    frame: *const FrameToken,
    inputs: *const *const BaseStructure,
    num_inputs: u32,
    cmd: *mut c_void,
) -> SlResult;
pub(crate) type PfnSlFreeResources =
    unsafe extern "C" fn(feature: Feature, viewport: *const ViewportHandle) -> SlResult;
pub(crate) type PfnSlGetFeatureVersion =
    unsafe extern "C" fn(feature: Feature, version: *mut FeatureVersion) -> SlResult;
pub(crate) type PfnSlUpgradeInterface =
    unsafe extern "C" fn(base_interface: *mut *mut c_void) -> SlResult;
pub(crate) type PfnSlSetConstants = unsafe extern "C" fn(
    values: *const Constants,
    frame: *const FrameToken,
    viewport: *const ViewportHandle,
) -> SlResult;
pub(crate) type PfnSlGetFeatureFunction = unsafe extern "C" fn(
    feature: Feature,
    name: *const c_char,
    function: *mut *mut c_void,
) -> SlResult;
pub(crate) type PfnSlGetNewFrameToken =
    unsafe extern "C" fn(token: *mut *mut FrameToken, frame_index: *const u32) -> SlResult;
pub(crate) type PfnSlSetD3DDevice = unsafe extern "C" fn(device: *mut c_void) -> SlResult;
pub(crate) type PfnSlDlssSetOptions =
    unsafe extern "C" fn(viewport: *const ViewportHandle, options: *const DlssOptions) -> SlResult;

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // The SDK asserts these offsets itself (`sl.cpp`); the rest follow from
    // the C layout rules the tests below spell out.
    #[test]
    fn layouts_match_the_sdk_static_asserts() {
        assert_eq!(offset_of!(Preferences, render_api), 136);
        assert_eq!(offset_of!(ResourceTag, extent), 48);
        assert_eq!(offset_of!(Resource, internal_flags), 104);
        assert_eq!(offset_of!(Resource, reserved), 106);
        assert_eq!(offset_of!(Constants, motion_vectors_jittered), 450);
        assert_eq!(offset_of!(AdapterInfo, vk_physical_device), 48);
        assert_eq!(offset_of!(FeatureVersion, version_ngx), 44);
    }

    #[test]
    fn base_structure_layout() {
        assert_eq!(size_of::<StructType>(), 16);
        assert_eq!(size_of::<BaseStructure>(), 32);
        assert_eq!(offset_of!(BaseStructure, struct_type), 8);
        assert_eq!(offset_of!(BaseStructure, struct_version), 24);
    }

    #[test]
    fn preferences_layout() {
        assert_eq!(offset_of!(Preferences, show_console), 32);
        assert_eq!(offset_of!(Preferences, log_level), 36);
        assert_eq!(offset_of!(Preferences, paths_to_plugins), 40);
        assert_eq!(offset_of!(Preferences, num_paths_to_plugins), 48);
        assert_eq!(offset_of!(Preferences, path_to_logs_and_data), 56);
        assert_eq!(offset_of!(Preferences, log_message_callback), 80);
        assert_eq!(offset_of!(Preferences, flags), 88);
        assert_eq!(offset_of!(Preferences, features_to_load), 96);
        assert_eq!(offset_of!(Preferences, num_features_to_load), 104);
        assert_eq!(offset_of!(Preferences, application_id), 108);
        assert_eq!(offset_of!(Preferences, engine), 112);
        assert_eq!(offset_of!(Preferences, engine_version), 120);
        assert_eq!(offset_of!(Preferences, project_id), 128);
        assert_eq!(size_of::<Preferences>(), 144);
    }

    #[test]
    fn resource_and_tag_layouts() {
        assert_eq!(offset_of!(Resource, ty), 32);
        assert_eq!(offset_of!(Resource, native), 40);
        assert_eq!(offset_of!(Resource, view), 56);
        assert_eq!(offset_of!(Resource, state), 64);
        assert_eq!(offset_of!(Resource, width), 68);
        assert_eq!(offset_of!(Resource, native_format), 76);
        assert_eq!(offset_of!(Resource, gpu_virtual_address), 88);
        assert_eq!(offset_of!(Resource, flags), 96);
        assert_eq!(offset_of!(Resource, usage), 100);
        assert_eq!(size_of::<Resource>(), 112);
        assert_eq!(offset_of!(ResourceTag, resource), 32);
        assert_eq!(offset_of!(ResourceTag, ty), 40);
        assert_eq!(offset_of!(ResourceTag, lifecycle), 44);
        assert_eq!(size_of::<ResourceTag>(), 64);
        assert_eq!(size_of::<ViewportHandle>(), 40);
        assert_eq!(offset_of!(SubresourceRange, aspect_mask), 32);
        assert_eq!(offset_of!(SubresourceRange, layer_count), 48);
        assert_eq!(size_of::<SubresourceRange>(), 56);
    }

    #[test]
    fn constants_layout() {
        assert_eq!(size_of::<Float4x4>(), 64);
        assert_eq!(offset_of!(Constants, camera_view_to_clip), 32);
        assert_eq!(offset_of!(Constants, prev_clip_to_clip), 288);
        assert_eq!(offset_of!(Constants, jitter_offset), 352);
        assert_eq!(offset_of!(Constants, mvec_scale), 360);
        assert_eq!(offset_of!(Constants, camera_pos), 376);
        assert_eq!(offset_of!(Constants, camera_fwd), 412);
        assert_eq!(offset_of!(Constants, camera_near), 424);
        assert_eq!(offset_of!(Constants, motion_vectors_invalid_value), 440);
        assert_eq!(offset_of!(Constants, depth_inverted), 444);
        assert_eq!(offset_of!(Constants, reset), 447);
        assert_eq!(
            offset_of!(Constants, min_relative_linear_depth_object_separation),
            452
        );
        assert_eq!(size_of::<Constants>(), 456);
    }

    #[test]
    fn dlss_options_layout() {
        assert_eq!(offset_of!(DlssOptions, mode), 32);
        assert_eq!(offset_of!(DlssOptions, sharpness), 44);
        assert_eq!(offset_of!(DlssOptions, pre_exposure), 48);
        assert_eq!(offset_of!(DlssOptions, color_buffers_hdr), 56);
        assert_eq!(offset_of!(DlssOptions, dlaa_preset), 60);
        assert_eq!(offset_of!(DlssOptions, ultra_quality_preset), 80);
        assert_eq!(offset_of!(DlssOptions, use_auto_exposure), 84);
        assert_eq!(size_of::<DlssOptions>(), 88);
    }

    #[test]
    fn the_sdk_version_packs_major_minor_patch_and_magic() {
        assert_eq!(SDK_VERSION >> 48, 2);
        assert_eq!((SDK_VERSION >> 32) & 0xffff, 14);
        assert_eq!((SDK_VERSION >> 16) & 0xffff, 16);
        assert_eq!(SDK_VERSION & 0xffff, 0xfedc);
    }

    #[test]
    fn results_name_their_enumerator() {
        assert!(SlResult::OK.is_ok());
        assert_eq!(SlResult(15).name(), "eErrorNGXFailed");
        assert_eq!(
            SlResult::ERROR_INVALID_PARAMETER.name(),
            "eErrorInvalidParameter"
        );
        assert_eq!(SlResult(39).name(), "eWarnOutOfVRAM");
        assert_eq!(SlResult(-1).name(), "unknown");
        assert_eq!(SlResult(40).name(), "unknown");
    }

    #[test]
    fn struct_versions_follow_the_headers() {
        assert_eq!(Constants::default().base.struct_version, 2);
        assert_eq!(DlssOptions::default().base.struct_version, 3);
        assert_eq!(ViewportHandle::new(7).base.struct_version, 1);
        assert_eq!(ViewportHandle::new(7).value, 7);
    }
}
