//! AMD FidelityFX FSR through the FidelityFX SDK's unified `ffx_api`, which the
//! D3D12 and Vulkan runtimes expose identically: an upscale context created
//! from a chain of descriptions, queried for its jitter sequence, and
//! dispatched onto a command list. A backend contributes the backend-create
//! description that heads the chain and the raw handles of its resources.
//!
//! Layouts match `ffx-api/include/ffx_api/{ffx_api.h,ffx_api_types.h,ffx_upscale.h}`
//! of FidelityFX SDK v1.1.4; the tests pin every size and field offset.
#![expect(
    non_camel_case_types,
    reason = "the FFX bindings keep the SDK's own C type names"
)]

use std::ffi::c_void;
use std::ptr;

use concinnity_core::render::depth::{CAMERA_DEPTH, DepthMapping};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::history_reset::UpscalerResetLatch;

use super::camera::FrameClock;
use super::{SdkLibrary, UpscaleCamera, UpscaleExtent, entry_point};

type ffxContext = *mut c_void;
type ffxReturnCode_t = u32;

const FFX_API_RETURN_OK: ffxReturnCode_t = 0;

const FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE: u64 = 0x0001_0000;
const FFX_API_DISPATCH_DESC_TYPE_UPSCALE: u64 = 0x0001_0001;
const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: u64 = 0x0001_0004;
const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: u64 = 0x0001_0005;
const FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1: u64 = 0x000_0001;
const FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE: u32 = 0xfff_ffff;

// FfxApiCreateContextUpscaleFlags.
const FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE: u32 = 1 << 0;
const FFX_UPSCALE_ENABLE_DEPTH_INVERTED: u32 = 1 << 3;
const FFX_UPSCALE_ENABLE_DEPTH_INFINITE: u32 = 1 << 4;
const FFX_UPSCALE_ENABLE_AUTO_EXPOSURE: u32 = 1 << 5;

const FFX_API_RESOURCE_TYPE_TEXTURE2D: u32 = 2;
const FFX_API_RESOURCE_USAGE_READ_ONLY: u32 = 0;
const FFX_API_RESOURCE_USAGE_UAV: u32 = 1 << 1;
const FFX_API_RESOURCE_USAGE_DEPTHTARGET: u32 = 1 << 2;
const FFX_API_RESOURCE_STATE_UNORDERED_ACCESS: u32 = 1 << 1;
const FFX_API_RESOURCE_STATE_COMPUTE_READ: u32 = 1 << 2;
const FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT: u32 = 4;
const FFX_API_SURFACE_FORMAT_R16G16_FLOAT: u32 = 18;
const FFX_API_SURFACE_FORMAT_R32_FLOAT: u32 = 28;
const FFX_API_SURFACE_FORMAT_R8_UNORM: u32 = 25;

// FfxApiMessageType.
#[cfg(windows)]
const FFX_API_MESSAGE_TYPE_ERROR: u32 = 0;
#[cfg(windows)]
const FFX_API_MESSAGE_TYPE_WARNING: u32 = 1;

// The phase count used when the jitter-phase query fails.
const FALLBACK_JITTER_PHASE_COUNT: i32 = 8;

/// The header every `ffx_api` description starts with: its type, and the next
/// description in the chain.
#[repr(C)]
pub(crate) struct ffxApiHeader {
    ty: u64,
    p_next: *mut ffxApiHeader,
}

impl ffxApiHeader {
    /// The header of a description of type `ty`, ending its chain.
    pub(crate) const fn new(ty: u64) -> Self {
        Self {
            ty,
            p_next: ptr::null_mut(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FfxApiDimensions2D {
    width: u32,
    height: u32,
}

impl From<(u32, u32)> for FfxApiDimensions2D {
    fn from((width, height): (u32, u32)) -> Self {
        Self { width, height }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FfxApiFloatCoords2D {
    x: f32,
    y: f32,
}

#[repr(C)]
struct FfxApiResourceDescription {
    ty: u32,
    format: u32,
    width_or_size: u32,
    height_or_stride: u32,
    depth_or_alignment: u32,
    mip_count: u32,
    flags: u32,
    usage: u32,
}

#[repr(C)]
struct FfxApiResource {
    resource: *mut c_void,
    description: FfxApiResourceDescription,
    state: u32,
}

impl FfxApiResource {
    // An optional input left unbound.
    fn empty() -> Self {
        Self {
            resource: ptr::null_mut(),
            description: FfxApiResourceDescription {
                ty: 0,
                format: 0,
                width_or_size: 0,
                height_or_stride: 0,
                depth_or_alignment: 0,
                mip_count: 0,
                flags: 0,
                usage: 0,
            },
            state: 0,
        }
    }

    // A single-mip 2D texture of `size` claimed in `state`.
    fn texture(
        resource: *mut c_void,
        format: u32,
        usage: u32,
        state: u32,
        size: (u32, u32),
    ) -> Self {
        Self {
            resource,
            description: FfxApiResourceDescription {
                ty: FFX_API_RESOURCE_TYPE_TEXTURE2D,
                format,
                width_or_size: size.0,
                height_or_stride: size.1,
                depth_or_alignment: 1,
                mip_count: 1,
                flags: 0,
                usage,
            },
            state,
        }
    }
}

type FfxApiMessage = Option<extern "C" fn(ty: u32, message: *const u16)>;

#[repr(C)]
struct ffxCreateContextDescUpscale {
    header: ffxApiHeader,
    flags: u32,
    max_render_size: FfxApiDimensions2D,
    max_upscale_size: FfxApiDimensions2D,
    fp_message: FfxApiMessage,
}

#[repr(C)]
struct ffxDispatchDescUpscale {
    header: ffxApiHeader,
    command_list: *mut c_void,
    color: FfxApiResource,
    depth: FfxApiResource,
    motion_vectors: FfxApiResource,
    exposure: FfxApiResource,
    reactive: FfxApiResource,
    transparency_and_composition: FfxApiResource,
    output: FfxApiResource,
    jitter_offset: FfxApiFloatCoords2D,
    motion_vector_scale: FfxApiFloatCoords2D,
    render_size: FfxApiDimensions2D,
    upscale_size: FfxApiDimensions2D,
    enable_sharpening: bool,
    sharpness: f32,
    frame_time_delta: f32,
    pre_exposure: f32,
    reset: bool,
    camera_near: f32,
    camera_far: f32,
    camera_fov_angle_vertical: f32,
    view_space_to_meters_factor: f32,
    flags: u32,
}

#[repr(C)]
struct ffxQueryDescUpscaleGetJitterPhaseCount {
    header: ffxApiHeader,
    render_width: u32,
    display_width: u32,
    out_phase_count: *mut i32,
}

#[repr(C)]
struct ffxQueryDescUpscaleGetJitterOffset {
    header: ffxApiHeader,
    index: i32,
    phase_count: i32,
    out_x: *mut f32,
    out_y: *mut f32,
}

#[repr(C)]
struct ffxConfigureDescGlobalDebug1 {
    header: ffxApiHeader,
    fp_message: FfxApiMessage,
    debug_level: u32,
}

#[repr(C)]
struct ffxAllocationCallbacks {
    user_data: *mut c_void,
    alloc: *mut c_void,
    dealloc: *mut c_void,
}

type PfnFfxCreateContext = unsafe extern "C" fn(
    context: *mut ffxContext,
    desc: *mut ffxApiHeader,
    mem_cb: *const ffxAllocationCallbacks,
) -> ffxReturnCode_t;
type PfnFfxDestroyContext = unsafe extern "C" fn(
    context: *mut ffxContext,
    mem_cb: *const ffxAllocationCallbacks,
) -> ffxReturnCode_t;
type PfnFfxConfigure =
    unsafe extern "C" fn(context: *mut ffxContext, desc: *const ffxApiHeader) -> ffxReturnCode_t;
type PfnFfxQuery =
    unsafe extern "C" fn(context: *mut ffxContext, desc: *mut ffxApiHeader) -> ffxReturnCode_t;
type PfnFfxDispatch =
    unsafe extern "C" fn(context: *mut ffxContext, desc: *const ffxApiHeader) -> ffxReturnCode_t;

// The create flags for a camera whose depth maps as `depth`: HDR linear input
// and FFX's own auto-exposure (the scene is un-exposed until after the
// upscale), inverted depth when the near plane is device depth 1, and infinite
// depth when the projection has no far plane.
const fn ffx_create_flags(depth: DepthMapping) -> u32 {
    let mut flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    if depth.reversed {
        flags |= FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
    }
    if depth.infinite {
        flags |= FFX_UPSCALE_ENABLE_DEPTH_INFINITE;
    }
    flags
}

// The `(camera_near, camera_far)` pair the dispatch hands FFX for a camera
// whose near plane is `near` and whose projection has no far plane: FFX takes
// that far plane as FLT_MAX, and with inverted depth the two planes swapped.
const fn ffx_camera_planes(depth: DepthMapping, near: f32) -> (f32, f32) {
    let far = f32::MAX;
    if depth.reversed {
        (far, near)
    } else {
        (near, far)
    }
}

// Routes FFX's diagnostics into the log. FFX passes `wchar_t` text, which is
// UTF-16 only on Windows, so elsewhere no sink is installed.
#[cfg(windows)]
extern "C" fn ffx_message_sink(ty: u32, message: *const u16) {
    if message.is_null() {
        return;
    }
    let mut len = 0usize;
    // SAFETY: FFX passes a NUL-terminated string that stays alive for the callback, so every unit
    // up to and including the terminator is readable.
    while unsafe { *message.add(len) } != 0 {
        len += 1;
    }
    // SAFETY: the scan above stopped at the terminator, so `len` initialized units start at
    // `message`.
    let text = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(message, len) });
    match ty {
        FFX_API_MESSAGE_TYPE_ERROR => tracing::error!("FFX: {text}"),
        FFX_API_MESSAGE_TYPE_WARNING => tracing::warn!("FFX: {text}"),
        other => tracing::info!("FFX[{other}]: {text}"),
    }
}

fn message_callback() -> FfxApiMessage {
    #[cfg(windows)]
    {
        Some(ffx_message_sink)
    }
    #[cfg(not(windows))]
    {
        None
    }
}

struct FfxEntryPoints {
    create_context: PfnFfxCreateContext,
    destroy_context: PfnFfxDestroyContext,
    configure: PfnFfxConfigure,
    query: PfnFfxQuery,
    dispatch: PfnFfxDispatch,
}

impl FfxEntryPoints {
    fn resolve(library: &impl SdkLibrary) -> Option<Self> {
        // SAFETY: each type is the prototype `ffx_api.h` declares for that export.
        unsafe {
            Some(Self {
                create_context: entry_point(library, c"ffxCreateContext")?,
                destroy_context: entry_point(library, c"ffxDestroyContext")?,
                configure: entry_point(library, c"ffxConfigure")?,
                query: entry_point(library, c"ffxQuery")?,
                dispatch: entry_point(library, c"ffxDispatch")?,
            })
        }
    }
}

/// The raw API handles of one upscale dispatch: the command list it records
/// onto and the textures it reads and writes, each as the pointer-sized value
/// `ffx_api` takes.
pub(crate) struct FfxDispatchHandles {
    pub(crate) command_list: *mut c_void,
    pub(crate) color: *mut c_void,
    pub(crate) depth: *mut c_void,
    pub(crate) motion_vectors: *mut c_void,
    /// The reactive mask, or null on a frame no pass wrote it.
    pub(crate) reactive: *mut c_void,
    pub(crate) output: *mut c_void,
}

/// An FFX upscale context for one render-to-output extent, with the runtime
/// library its entry points come from. Destroyed on drop, before the library
/// is released.
pub(crate) struct FfxContext<L> {
    api: FfxEntryPoints,
    handle: ffxContext,
    extent: UpscaleExtent,
    jitter_phase_count: i32,
    reset: UpscalerResetLatch,
    clock: FrameClock,
    _library: L,
}

impl<L: SdkLibrary> FfxContext<L> {
    /// Create the upscale context for `extent` on the device `backend`
    /// describes. `None` (logged under `label`) when the library lacks an entry
    /// point or FFX refuses the context.
    ///
    /// # Safety
    ///
    /// `backend` must head a valid backend-create description whose device
    /// handles outlive the context.
    pub(crate) unsafe fn create(
        library: L,
        backend: &mut ffxApiHeader,
        extent: UpscaleExtent,
        label: &str,
    ) -> Option<Self> {
        let Some(api) = FfxEntryPoints::resolve(&library) else {
            tracing::warn!("{label}: the runtime library lacks an ffx_api entry point");
            return None;
        };
        // The context is recreated whenever either size changes, so the render
        // size is the exact maximum.
        let mut desc = ffxCreateContextDescUpscale {
            header: ffxApiHeader {
                ty: FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE,
                p_next: backend,
            },
            flags: ffx_create_flags(CAMERA_DEPTH),
            max_render_size: extent.render.into(),
            max_upscale_size: extent.output.into(),
            fp_message: message_callback(),
        };
        let mut handle: ffxContext = ptr::null_mut();
        // SAFETY: the entry point came from the loaded library with the header's prototype, `desc`
        // and `handle` are live locals, and the caller guarantees the backend description `desc`
        // chains to.
        let rc = unsafe { (api.create_context)(&mut handle, &mut desc.header, ptr::null()) };
        if rc != FFX_API_RETURN_OK || handle.is_null() {
            tracing::warn!("{label}: ffxCreateContext returned {rc}; trying the next backend");
            return None;
        }
        let mut context = Self {
            api,
            handle,
            extent,
            jitter_phase_count: FALLBACK_JITTER_PHASE_COUNT,
            reset: UpscalerResetLatch::default(),
            clock: FrameClock::default(),
            _library: library,
        };
        context.raise_debug_level(label);
        context.jitter_phase_count = context.query_jitter_phase_count(label);
        tracing::info!("{label}: context created: {extent}");
        Some(context)
    }

    // `ffxConfigure` dereferences the context's provider, so this runs only
    // once the context exists. The message sink itself is installed by the
    // create description.
    fn raise_debug_level(&mut self, label: &str) {
        let desc = ffxConfigureDescGlobalDebug1 {
            header: ffxApiHeader::new(FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1),
            fp_message: message_callback(),
            debug_level: FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE,
        };
        // SAFETY: `self.handle` is the live context and `desc` a live local the call only reads.
        let rc = unsafe { (self.api.configure)(&mut self.handle, &desc.header) };
        if rc != FFX_API_RETURN_OK {
            tracing::warn!("{label}: global debug configure returned {rc} (non-fatal)");
        }
    }

    fn query_jitter_phase_count(&mut self, label: &str) -> i32 {
        let mut phase_count: i32 = 0;
        let mut desc = ffxQueryDescUpscaleGetJitterPhaseCount {
            header: ffxApiHeader::new(FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT),
            render_width: self.extent.render.0,
            display_width: self.extent.output.0,
            out_phase_count: &mut phase_count,
        };
        // SAFETY: `self.handle` is the live context, and `desc` is a live local whose out-pointer
        // borrows another live local.
        let rc = unsafe { (self.api.query)(&mut self.handle, &mut desc.header) };
        if rc != FFX_API_RETURN_OK || phase_count <= 0 {
            tracing::warn!(
                "{label}: jitter-phase-count query returned {rc} (phase_count={phase_count})"
            );
            return FALLBACK_JITTER_PHASE_COUNT;
        }
        phase_count
    }

    /// The extent the context was created for.
    pub(crate) fn extent(&self) -> UpscaleExtent {
        self.extent
    }

    /// FFX's sub-pixel jitter for `frame_index`, in render pixels; zero when
    /// the query fails.
    pub(crate) fn jitter_offset(&self, frame_index: u32) -> [f32; 2] {
        let (mut x, mut y) = (0.0_f32, 0.0_f32);
        let phase_count = self.jitter_phase_count.max(1);
        let mut desc = ffxQueryDescUpscaleGetJitterOffset {
            header: ffxApiHeader::new(FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET),
            index: (frame_index as i32).rem_euclid(phase_count),
            phase_count: self.jitter_phase_count,
            out_x: &mut x,
            out_y: &mut y,
        };
        // SAFETY: `self.handle` is the live context and `desc` a live local whose out-pointers
        // borrow live locals. The C prototype takes the handle by mutable pointer, but only
        // create and destroy write through it.
        let rc = unsafe { (self.api.query)(self.handle_ptr(), &mut desc.header) };
        if rc != FFX_API_RETURN_OK {
            return [0.0, 0.0];
        }
        [x, y]
    }

    /// Discard FFX's history on the next dispatch.
    pub(crate) fn request_history_reset(&self) {
        self.reset.request();
    }

    /// Record the upscale of this frame onto `handles.command_list`, reading
    /// the inputs in the compute-read state and writing the output in the
    /// unordered-access state. The first dispatch after creation, and the
    /// first after [`Self::request_history_reset`], resets FFX's history.
    ///
    /// # Safety
    ///
    /// The command list must be recording, and every texture handle must name a
    /// live resource of the context's extent in the state above, kept alive
    /// until the commands execute.
    pub(crate) unsafe fn dispatch(
        &self,
        handles: FfxDispatchHandles,
        camera: UpscaleCamera,
    ) -> RenderResult<()> {
        let UpscaleExtent { render, output, .. } = self.extent;
        let (camera_near, camera_far) = ffx_camera_planes(CAMERA_DEPTH, camera.near);
        let input = |resource, format, usage| {
            FfxApiResource::texture(
                resource,
                format,
                usage,
                FFX_API_RESOURCE_STATE_COMPUTE_READ,
                render,
            )
        };
        let desc = ffxDispatchDescUpscale {
            header: ffxApiHeader::new(FFX_API_DISPATCH_DESC_TYPE_UPSCALE),
            command_list: handles.command_list,
            color: input(
                handles.color,
                FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
                FFX_API_RESOURCE_USAGE_READ_ONLY,
            ),
            depth: input(
                handles.depth,
                FFX_API_SURFACE_FORMAT_R32_FLOAT,
                FFX_API_RESOURCE_USAGE_DEPTHTARGET,
            ),
            motion_vectors: input(
                handles.motion_vectors,
                FFX_API_SURFACE_FORMAT_R16G16_FLOAT,
                FFX_API_RESOURCE_USAGE_READ_ONLY,
            ),
            exposure: FfxApiResource::empty(),
            reactive: if handles.reactive.is_null() {
                FfxApiResource::empty()
            } else {
                input(
                    handles.reactive,
                    FFX_API_SURFACE_FORMAT_R8_UNORM,
                    FFX_API_RESOURCE_USAGE_READ_ONLY,
                )
            },
            transparency_and_composition: FfxApiResource::empty(),
            output: FfxApiResource::texture(
                handles.output,
                FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
                FFX_API_RESOURCE_USAGE_UAV,
                FFX_API_RESOURCE_STATE_UNORDERED_ACCESS,
                output,
            ),
            jitter_offset: FfxApiFloatCoords2D {
                x: camera.jitter_offset[0],
                y: camera.jitter_offset[1],
            },
            // Motion vectors are `prev_uv - cur_uv` in UV space; FFX takes them
            // in render pixels.
            motion_vector_scale: FfxApiFloatCoords2D {
                x: render.0 as f32,
                y: render.1 as f32,
            },
            render_size: render.into(),
            upscale_size: output.into(),
            enable_sharpening: false,
            sharpness: 0.0,
            frame_time_delta: self.clock.delta_ms(camera.elapsed),
            pre_exposure: 1.0,
            reset: crate::upscale_reset::consume(&self.reset),
            camera_near,
            camera_far,
            camera_fov_angle_vertical: camera.fov_y_radians,
            view_space_to_meters_factor: 1.0,
            flags: 0,
        };
        // SAFETY: `self.handle` is the live context and `desc` a live local; the caller guarantees
        // the command list and resources it names. Only create and destroy write through the
        // handle pointer.
        let rc = unsafe { (self.api.dispatch)(self.handle_ptr(), &desc.header) };
        if rc != FFX_API_RETURN_OK {
            return Err(RenderError::Other(format!(
                "ffxDispatch (upscale) returned {rc}"
            )));
        }
        Ok(())
    }
}

impl<L> FfxContext<L> {
    // The handle by the mutable pointer the C prototypes take.
    fn handle_ptr(&self) -> *mut ffxContext {
        &self.handle as *const ffxContext as *mut ffxContext
    }

    /// Destroy the context now; dropping it afterwards does nothing more. The
    /// device it was created on must be idle and still alive.
    pub(crate) fn destroy(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: `self.handle` is the live, non-null context, destroyed exactly once (nulled
        // straight after) while the library its entry point came from is still loaded.
        unsafe { (self.api.destroy_context)(&mut self.handle, ptr::null()) };
        self.handle = ptr::null_mut();
    }
}

impl<L> Drop for FfxContext<L> {
    fn drop(&mut self) {
        self.destroy();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn ffx_layouts_match_sdk_v114() {
        assert_eq!(size_of::<ffxApiHeader>(), 16);
        assert_eq!(offset_of!(ffxApiHeader, ty), 0);
        assert_eq!(offset_of!(ffxApiHeader, p_next), 8);

        assert_eq!(size_of::<FfxApiDimensions2D>(), 8);
        assert_eq!(offset_of!(FfxApiDimensions2D, width), 0);
        assert_eq!(offset_of!(FfxApiDimensions2D, height), 4);
        assert_eq!(size_of::<FfxApiFloatCoords2D>(), 8);
        assert_eq!(offset_of!(FfxApiFloatCoords2D, x), 0);
        assert_eq!(offset_of!(FfxApiFloatCoords2D, y), 4);

        assert_eq!(size_of::<FfxApiResourceDescription>(), 32);
        assert_eq!(offset_of!(FfxApiResourceDescription, ty), 0);
        assert_eq!(offset_of!(FfxApiResourceDescription, format), 4);
        assert_eq!(offset_of!(FfxApiResourceDescription, width_or_size), 8);
        assert_eq!(offset_of!(FfxApiResourceDescription, height_or_stride), 12);
        assert_eq!(
            offset_of!(FfxApiResourceDescription, depth_or_alignment),
            16
        );
        assert_eq!(offset_of!(FfxApiResourceDescription, mip_count), 20);
        assert_eq!(offset_of!(FfxApiResourceDescription, flags), 24);
        assert_eq!(offset_of!(FfxApiResourceDescription, usage), 28);

        assert_eq!(size_of::<FfxApiResource>(), 48);
        assert_eq!(offset_of!(FfxApiResource, resource), 0);
        assert_eq!(offset_of!(FfxApiResource, description), 8);
        assert_eq!(offset_of!(FfxApiResource, state), 40);

        assert_eq!(size_of::<ffxAllocationCallbacks>(), 24);
        assert_eq!(offset_of!(ffxAllocationCallbacks, user_data), 0);
        assert_eq!(offset_of!(ffxAllocationCallbacks, alloc), 8);
        assert_eq!(offset_of!(ffxAllocationCallbacks, dealloc), 16);
    }

    #[test]
    fn ffx_description_layouts_match_sdk_v114() {
        assert_eq!(size_of::<ffxCreateContextDescUpscale>(), 48);
        assert_eq!(offset_of!(ffxCreateContextDescUpscale, header), 0);
        assert_eq!(offset_of!(ffxCreateContextDescUpscale, flags), 16);
        assert_eq!(offset_of!(ffxCreateContextDescUpscale, max_render_size), 20);
        assert_eq!(
            offset_of!(ffxCreateContextDescUpscale, max_upscale_size),
            28
        );
        assert_eq!(offset_of!(ffxCreateContextDescUpscale, fp_message), 40);

        type D = ffxDispatchDescUpscale;
        assert_eq!(size_of::<D>(), 432);
        assert_eq!(offset_of!(D, header), 0);
        assert_eq!(offset_of!(D, command_list), 16);
        assert_eq!(offset_of!(D, color), 24);
        assert_eq!(offset_of!(D, depth), 72);
        assert_eq!(offset_of!(D, motion_vectors), 120);
        assert_eq!(offset_of!(D, exposure), 168);
        assert_eq!(offset_of!(D, reactive), 216);
        assert_eq!(offset_of!(D, transparency_and_composition), 264);
        assert_eq!(offset_of!(D, output), 312);
        assert_eq!(offset_of!(D, jitter_offset), 360);
        assert_eq!(offset_of!(D, motion_vector_scale), 368);
        assert_eq!(offset_of!(D, render_size), 376);
        assert_eq!(offset_of!(D, upscale_size), 384);
        assert_eq!(offset_of!(D, enable_sharpening), 392);
        assert_eq!(offset_of!(D, sharpness), 396);
        assert_eq!(offset_of!(D, frame_time_delta), 400);
        assert_eq!(offset_of!(D, pre_exposure), 404);
        assert_eq!(offset_of!(D, reset), 408);
        assert_eq!(offset_of!(D, camera_near), 412);
        assert_eq!(offset_of!(D, camera_far), 416);
        assert_eq!(offset_of!(D, camera_fov_angle_vertical), 420);
        assert_eq!(offset_of!(D, view_space_to_meters_factor), 424);
        assert_eq!(offset_of!(D, flags), 428);

        type P = ffxQueryDescUpscaleGetJitterPhaseCount;
        assert_eq!(size_of::<P>(), 32);
        assert_eq!(offset_of!(P, header), 0);
        assert_eq!(offset_of!(P, render_width), 16);
        assert_eq!(offset_of!(P, display_width), 20);
        assert_eq!(offset_of!(P, out_phase_count), 24);

        type J = ffxQueryDescUpscaleGetJitterOffset;
        assert_eq!(size_of::<J>(), 40);
        assert_eq!(offset_of!(J, header), 0);
        assert_eq!(offset_of!(J, index), 16);
        assert_eq!(offset_of!(J, phase_count), 20);
        assert_eq!(offset_of!(J, out_x), 24);
        assert_eq!(offset_of!(J, out_y), 32);

        type G = ffxConfigureDescGlobalDebug1;
        assert_eq!(size_of::<G>(), 32);
        assert_eq!(offset_of!(G, header), 0);
        assert_eq!(offset_of!(G, fp_message), 16);
        assert_eq!(offset_of!(G, debug_level), 24);
    }

    #[test]
    fn bound_and_unbound_textures_describe_themselves() {
        let unbound = FfxApiResource::empty();
        assert!(unbound.resource.is_null());
        assert_eq!(unbound.description.ty, 0);
        assert_eq!(unbound.description.mip_count, 0);

        let bound = FfxApiResource::texture(
            ptr::null_mut(),
            FFX_API_SURFACE_FORMAT_R32_FLOAT,
            FFX_API_RESOURCE_USAGE_DEPTHTARGET,
            FFX_API_RESOURCE_STATE_COMPUTE_READ,
            (640, 360),
        );
        let d = &bound.description;
        assert_eq!(d.ty, FFX_API_RESOURCE_TYPE_TEXTURE2D);
        assert_eq!((d.width_or_size, d.height_or_stride), (640, 360));
        assert_eq!((d.depth_or_alignment, d.mip_count), (1, 1));
        assert_eq!(bound.state, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    }

    // Each depth flag follows its own field of the mapping, the planes swap
    // under inverted depth, and the camera's depth sets both flags.
    #[test]
    fn the_depth_contract_follows_the_depth_mapping() {
        for reversed in [false, true] {
            for infinite in [false, true] {
                let depth = DepthMapping { reversed, infinite };
                let flags = ffx_create_flags(depth);
                assert_eq!((flags & FFX_UPSCALE_ENABLE_DEPTH_INVERTED) != 0, reversed);
                assert_eq!((flags & FFX_UPSCALE_ENABLE_DEPTH_INFINITE) != 0, infinite);
                let expected = if reversed {
                    (f32::MAX, 0.1)
                } else {
                    (0.1, f32::MAX)
                };
                assert_eq!(ffx_camera_planes(depth, 0.1), expected);
            }
        }
        // The planes pass the far plane as FLT_MAX, which FFX reads as no far
        // plane only under the infinite-depth flag.
        let camera = ffx_create_flags(CAMERA_DEPTH);
        assert_ne!(camera & FFX_UPSCALE_ENABLE_DEPTH_INVERTED, 0);
        assert_ne!(camera & FFX_UPSCALE_ENABLE_DEPTH_INFINITE, 0);
        assert_ne!(camera & FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, 0);
        assert_ne!(camera & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, 0);
    }
}
