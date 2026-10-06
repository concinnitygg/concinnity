//! Temporal upscaling for the Vulkan backend. The engine renders the 3D scene at
//! a fraction of the swapchain extent (`render_extent`) and the
//! `PassId::Upscale` pass reconstructs a swapchain-resolution image the bloom +
//! composite stack consumes.
//!
//! Three interchangeable backends sit behind the `VkUpscaleBackend` trait:
//!   fsr   AMD FidelityFX FSR (cross-vendor; ffx_api VK)
//!   dlss  NVIDIA DLSS via raw NGX (RTX only; cfg(ngx_sdk_bundled))
//!   xess  Intel XeSS (cross-vendor; runtime libxess)
//! The API-independent half of each lives in `crate::upscale_sdk`; these files
//! hold the Vulkan resources, library loading and command recording.
//! `build_upscaler` constructs the first backend that initializes, in the shared
//! fallback order, and `VkContext::encode_upscale` drives whichever is active.
//!
//! DLSS and XeSS additionally need instance / device extensions (and, for XeSS,
//! device features) enabled before the upscaler context exists. `UpscaleSdk` is
//! queried up front (in `init.rs`, before `create_instance`) and threaded into
//! `device::create_logical_device`; see its docs.

use std::cell::Cell;
use std::ffi::{CStr, CString, c_char, c_void};

use ash::vk;
use concinnity_core::components::UpscalerBackend;
use concinnity_core::render::error::{RenderError, RenderResult};

use crate::upscale_sdk::{
    Availability, SdkLibrary, UpscaleCamera, UpscaleExtent, build_first_available, preferred,
};
use crate::vulkan::allocator::DeviceAllocator;
use crate::vulkan::context::{HDR_FORMAT, VkContext};
use crate::vulkan::graph_exec::GraphFrameParams;
use crate::vulkan::owned::VkDevice;
use crate::vulkan::texture::{
    GpuImage, ImageSpec, create_image, create_image_view, one_shot_submit,
};

pub(in crate::vulkan) use crate::upscale_sdk::ResolvedBackend;

#[cfg(ngx_sdk_bundled)]
mod dlss;
mod fsr;
mod xess;

// One render-resolution image handed to a backend's `dispatch`. FFX takes only
// the image; NGX and XeSS describe each with its view, format and size.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct UpscaleImage {
    pub(in crate::vulkan) image: vk::Image,
    pub(in crate::vulkan) view: vk::ImageView,
    pub(in crate::vulkan) format: vk::Format,
    pub(in crate::vulkan) width: u32,
    pub(in crate::vulkan) height: u32,
    pub(in crate::vulkan) aspect: vk::ImageAspectFlags,
}

// The three inputs of one frame's upscale, each in SHADER_READ_ONLY_OPTIMAL.
pub(in crate::vulkan) struct UpscaleInputs<'a> {
    pub(in crate::vulkan) color: &'a UpscaleImage,
    pub(in crate::vulkan) depth: &'a UpscaleImage,
    pub(in crate::vulkan) motion: &'a UpscaleImage,
}

// One temporal-upscaling backend. `encode_upscale` transitions the inputs and
// the output, then calls `dispatch`; each backend records its vendor upscale
// onto the supplied command buffer.
pub(in crate::vulkan) trait VkUpscaleBackend: Send {
    // The render and output sizes the backend was created for.
    fn extent(&self) -> UpscaleExtent;
    // The output image the bloom + composite stack samples as the scene.
    fn output(&self) -> &UpscaleOutput;
    // Sub-pixel jitter for this frame's index, shared with the camera
    // projection so the jittered VP and the upscale agree (render pixels).
    fn jitter_offset(&self, frame_index: u32) -> [f32; 2];
    // This frame's jitter, set on the main thread before the parallel fan-out
    // and read back on the worker in `encode_upscale`.
    fn jitter(&self) -> &Cell<[f32; 2]>;
    // Record the upscale onto `cmd`, with the inputs in
    // SHADER_READ_ONLY_OPTIMAL and the output in GENERAL.
    fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        inputs: UpscaleInputs<'_>,
        camera: UpscaleCamera,
    ) -> RenderResult<()>;
    // Tear down the SDK context and the owned images. Called after
    // `device_wait_idle`, before the device is destroyed.
    fn destroy(&mut self);
}

// `xess_vk_image_view_info` (`xess_vk.h`) and `NVSDK_NGX_ImageViewInfo_VK`
// (`nvsdk_ngx_defs_vk.h`), which XeSS 3.0.1 and NGX 1.5.0 lay out identically.
#[repr(C)]
#[derive(Clone, Copy)]
struct ImageViewInfo {
    image_view: vk::ImageView,
    image: vk::Image,
    subresource_range: vk::ImageSubresourceRange,
    format: vk::Format,
    width: u32,
    height: u32,
}

impl ImageViewInfo {
    // An optional input left unbound.
    fn empty() -> Self {
        Self {
            image_view: vk::ImageView::null(),
            image: vk::Image::null(),
            subresource_range: vk::ImageSubresourceRange::default(),
            format: vk::Format::UNDEFINED,
            width: 0,
            height: 0,
        }
    }

    fn of(img: &UpscaleImage) -> Self {
        Self {
            image_view: img.view,
            image: img.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: img.aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            format: img.format,
            width: img.width,
            height: img.height,
        }
    }
}

// How a backend's vendor dispatch writes its output image: the stages and
// accesses the barriers around the dispatch must declare, and the usage those
// writes need. NGX clears the DLSS output inside `EvaluateFeature`, so that
// backend writes through a transfer clear as well as a storage write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::vulkan) struct OutputWrites {
    stage: vk::PipelineStageFlags,
    access: vk::AccessFlags,
    usage: vk::ImageUsageFlags,
}

impl OutputWrites {
    // Storage writes from the backend's compute dispatch only.
    fn storage() -> Self {
        Self {
            stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            access: vk::AccessFlags::SHADER_WRITE,
            usage: vk::ImageUsageFlags::STORAGE,
        }
    }

    // Storage writes plus a `vkCmdClearColorImage` on the output.
    #[cfg(any(ngx_sdk_bundled, test))]
    fn storage_and_clear() -> Self {
        Self {
            stage: vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            access: vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
        }
    }

    // The output's image usage: these writes, plus the sampling bloom + composite do.
    fn image_usage(self) -> vk::ImageUsageFlags {
        self.usage | vk::ImageUsageFlags::SAMPLED
    }

    // Sync for the barrier that hands the output to the dispatch after `src_stage`
    // last touched it with `src_access`.
    fn acquire(
        self,
        src_stage: vk::PipelineStageFlags,
        src_access: vk::AccessFlags,
    ) -> BarrierSync {
        BarrierSync {
            src_stage,
            src_access,
            dst_stage: self.stage,
            dst_access: self.access,
        }
    }

    // Sync for the barrier that hands the written output to the fragment sampling.
    fn release_to_sampling(self) -> BarrierSync {
        BarrierSync {
            src_stage: self.stage,
            src_access: self.access,
            dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
            dst_access: vk::AccessFlags::SHADER_READ,
        }
    }
}

// The output-resolution RGBA16F image a backend writes and the post stack
// samples. It rests in GENERAL while written and in SHADER_READ_ONLY_OPTIMAL
// while sampled; `layout` tracks which across frames.
pub(in crate::vulkan) struct UpscaleOutput {
    image: GpuImage,
    size: (u32, u32),
    writes: OutputWrites,
    layout: Cell<vk::ImageLayout>,
}

impl UpscaleOutput {
    // Create the image with the usage `writes` needs, transitioned
    // UNDEFINED -> GENERAL so the first dispatch finds it writable.
    fn create(gpu: UpscalerGpu<'_>, size: (u32, u32), writes: OutputWrites) -> RenderResult<Self> {
        let pooled = create_image(
            gpu.alloc,
            &ImageSpec {
                width: size.0.max(1),
                height: size.1.max(1),
                format: HDR_FORMAT,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: writes.image_usage(),
                mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: vk::SampleCountFlags::TYPE_1,
            },
        )?;
        let image = pooled.image();
        let view = create_image_view(gpu.device, image, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
        one_shot_submit(gpu.device, gpu.command_pool, gpu.queue, |cmd| {
            image_barrier(
                gpu.device,
                cmd,
                image,
                vk::ImageAspectFlags::COLOR,
                LayoutTransition {
                    from: vk::ImageLayout::UNDEFINED,
                    to: vk::ImageLayout::GENERAL,
                },
                writes.acquire(
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::AccessFlags::empty(),
                ),
            );
        })?;
        Ok(Self {
            image: GpuImage::from_pooled(pooled, view),
            size,
            writes,
            layout: Cell::new(vk::ImageLayout::GENERAL),
        })
    }

    pub(in crate::vulkan) fn image(&self) -> &GpuImage {
        &self.image
    }

    // The output described like an input.
    fn as_upscale_image(&self) -> UpscaleImage {
        UpscaleImage {
            image: self.image.image,
            view: self.image.view,
            format: HDR_FORMAT,
            width: self.size.0,
            height: self.size.1,
            aspect: vk::ImageAspectFlags::COLOR,
        }
    }

    fn release(&mut self) {
        self.image = GpuImage::null();
    }
}

// The layout change one `image_barrier` records.
#[derive(Clone, Copy)]
struct LayoutTransition {
    from: vk::ImageLayout,
    to: vk::ImageLayout,
}

// The source / destination stage + access scopes one `image_barrier`
// synchronizes.
#[derive(Clone, Copy)]
struct BarrierSync {
    src_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_stage: vk::PipelineStageFlags,
    dst_access: vk::AccessFlags,
}

// One image barrier with explicit stages/access (the upscalers read their
// inputs in the COMPUTE stage; the generic `transition_image_layout` helper
// targets FRAGMENT, which would not synchronize the compute reads).
fn image_barrier(
    device: &VkDevice,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    transition: LayoutTransition,
    sync: BarrierSync,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(transition.from)
        .new_layout(transition.to)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(sync.src_access)
        .dst_access_mask(sync.dst_access);
    // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice these
    // commands name is live for the call.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            sync.src_stage,
            sync.dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&barrier),
        );
    }
}

// The SDKs this build bundles. DLSS links NGX statically; XeSS and FFX are
// runtime libraries tried only when `build.rs` bundled them.
fn availability() -> Availability {
    Availability {
        dlss: cfg!(ngx_sdk_bundled),
        xess: cfg!(xess_sdk_bundled),
        fsr: cfg!(ffx_sdk_bundled),
    }
}

// Load a vendor runtime library by file name from the executable's directory
// or the system search path.
fn open_library(name: &str) -> Option<libloading::Library> {
    // SAFETY: loading runs the library's initializers, and the vendor runtimes' have no
    // preconditions; a failed load is returned as an error.
    unsafe { libloading::Library::new(name) }.ok()
}

impl SdkLibrary for libloading::Library {
    fn symbol(&self, name: &CStr) -> Option<*const c_void> {
        // SAFETY: the export is read as an address only; `entry_point` gives it a prototype.
        unsafe { self.get::<*const c_void>(name.to_bytes_with_nul()) }
            .ok()
            .map(|address| *address)
    }
}

// The GPU handles a backend needs to create its output image + run one-shot
// init transitions.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct UpscalerGpu<'a> {
    pub(in crate::vulkan) alloc: &'a DeviceAllocator,
    pub(in crate::vulkan) instance: &'a ash::Instance,
    pub(in crate::vulkan) device: &'a VkDevice,
    pub(in crate::vulkan) physical_device: vk::PhysicalDevice,
    pub(in crate::vulkan) command_pool: vk::CommandPool,
    pub(in crate::vulkan) queue: vk::Queue,
}

// Construct the upscaler for the requested backend at `upscale_scale` of
// `output`, falling through the shared order whenever one cannot initialize
// (library miss, unsupported GPU, context-init failure). Returns the backend
// (`None` renders at native resolution) and the candidate that built. The
// extensions of the first candidate were enabled at device creation (see
// `UpscaleSdk`); a fallback past it can only land on FSR or native, which need
// none.
pub(in crate::vulkan) fn build_upscaler(
    gpu: UpscalerGpu<'_>,
    output: (u32, u32),
    upscale_scale: f32,
    requested: UpscalerBackend,
) -> RenderResult<(Option<Box<dyn VkUpscaleBackend>>, ResolvedBackend)> {
    let extent = UpscaleExtent::resolve(output, upscale_scale);
    build_first_available(requested, availability(), output, |candidate| {
        Ok(match candidate {
            ResolvedBackend::Fsr => fsr::FsrUpscaler::try_new(gpu, extent)?.map(boxed),
            ResolvedBackend::Xess => xess::XessUpscaler::try_new(gpu, extent)?.map(boxed),
            #[cfg(ngx_sdk_bundled)]
            ResolvedBackend::Dlss => dlss::DlssUpscaler::try_new(gpu, extent)?.map(boxed),
            _ => None,
        })
    })
}

fn boxed(backend: impl VkUpscaleBackend + 'static) -> Box<dyn VkUpscaleBackend> {
    Box::new(backend)
}

// Vulkan instance / device extension requirements for DLSS / XeSS, resolved
// before `create_instance`. `prepare` loads only the chosen SDK and calls its
// extension-enumeration entry points; the instance extensions feed
// `create_instance`, and the struct is then threaded into
// `device::create_logical_device` for the device extensions / features. Inert
// (`choice == Native`, empty lists) when upscaling is off or the chosen backend
// needs nothing.
pub(in crate::vulkan) struct UpscaleSdk {
    pub(in crate::vulkan) choice: ResolvedBackend,
    // Held so XeSS's SDK-owned device-feature chain stays mapped through
    // `vkCreateDevice`. `None` for every other backend.
    xess: Option<xess::XessExtQuery>,
    // Owned instance-extension names, kept alive until `create_instance`
    // consumes the pointers `instance_extension_ptrs` hands out.
    instance_exts: Vec<CString>,
    // DLSS device extensions, captured with the instance list (NGX yields both
    // in one call). XeSS queries its device extensions later, from
    // `create_logical_device`, since they need the physical device.
    dlss_device_exts: Vec<CString>,
    // Minimum Vulkan instance `apiVersion` the chosen backend needs (XeSS 3.x
    // requires 1.3 for SPV_KHR_integer_dot_product). 0 = no requirement beyond
    // the engine default. The caller clamps to loader support.
    min_api_version: u32,
}

impl UpscaleSdk {
    // Resolve which backend's extensions to enable and query the instance
    // extensions for it. Never fails: any SDK miss / query error degrades the
    // choice to a backend that needs no extra extensions, so instance / device
    // creation proceeds exactly as without upscaling.
    pub(in crate::vulkan) fn prepare(temporal_upscaling: bool, requested: UpscalerBackend) -> Self {
        let mut sdk = UpscaleSdk {
            choice: ResolvedBackend::Native,
            xess: None,
            instance_exts: Vec::new(),
            dlss_device_exts: Vec::new(),
            min_api_version: 0,
        };
        if !temporal_upscaling {
            return sdk;
        }
        sdk.choice = preferred(requested, availability());
        match sdk.choice {
            ResolvedBackend::Dlss =>
            {
                #[cfg(ngx_sdk_bundled)]
                match dlss::required_extensions() {
                    Some((inst, dev)) => {
                        sdk.instance_exts = inst;
                        sdk.dlss_device_exts = dev;
                    }
                    None => {
                        tracing::warn!(
                            "temporal upscaling: DLSS required-extensions query failed; \
                             device creation will skip DLSS extensions (build_upscaler will \
                             fall back to FSR / native)"
                        );
                        sdk.choice = ResolvedBackend::Fsr;
                    }
                }
            }
            ResolvedBackend::Xess => match xess::XessExtQuery::load() {
                Some(q) => {
                    let (exts, min_api) = q.instance_extensions();
                    sdk.instance_exts = exts;
                    sdk.min_api_version = min_api;
                    sdk.xess = Some(q);
                }
                None => {
                    tracing::warn!(
                        "temporal upscaling: XeSS library / extension query unavailable; device \
                         creation will skip XeSS extensions (build_upscaler will fall back to \
                         FSR / native)"
                    );
                    sdk.choice = ResolvedBackend::Fsr;
                }
            },
            _ => {}
        }
        sdk
    }

    // Raw instance-extension name pointers for `create_instance`. Valid as long
    // as `self` lives (the `CString`s are owned by `self.instance_exts`).
    pub(in crate::vulkan) fn instance_extension_ptrs(&self) -> Vec<*const c_char> {
        self.instance_exts.iter().map(|c| c.as_ptr()).collect()
    }

    // Minimum Vulkan instance `apiVersion` the chosen backend needs (0 = none).
    pub(in crate::vulkan) fn min_api_version(&self) -> u32 {
        self.min_api_version
    }

    // Device-extension names required by the chosen backend, filtered to those
    // the physical device actually exposes and not already requested in
    // `already`. DLSS returns its up-front list; XeSS queries now (needs the
    // instance + physical device).
    pub(in crate::vulkan) fn device_extensions(
        &self,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        already: &[CString],
    ) -> Vec<CString> {
        let supported = supported_device_extensions(instance, physical_device);
        let raw: Vec<CString> = match self.choice {
            ResolvedBackend::Dlss => self.dlss_device_exts.clone(),
            ResolvedBackend::Xess => self
                .xess
                .as_ref()
                .map(|q| q.device_extensions(instance, physical_device))
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        raw.into_iter()
            .filter(|name| supported.iter().any(|s| s == name))
            .filter(|name| !already.iter().any(|a| a == name))
            .collect()
    }

    // The XeSS-required device-feature chain head (an SDK-owned `pNext` chain to
    // splice into `VkDeviceCreateInfo`), or `head` unchanged for every other
    // backend. The chain memory is owned by the XeSS library and stays valid
    // while `self` lives (it holds the loaded library), which spans
    // `vkCreateDevice`. `head` is the caller's existing `pNext` chain that the
    // XeSS chain is appended in front of, so the SDK can also patch fields on
    // the caller's structs.
    pub(in crate::vulkan) fn xess_device_features(
        &self,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        head: *mut c_void,
    ) -> *mut c_void {
        match (self.choice, self.xess.as_ref()) {
            (ResolvedBackend::Xess, Some(q)) => q.device_features(instance, physical_device, head),
            _ => head,
        }
    }
}

// Copy a `const char* const*` array (`count` entries) returned by an SDK
// extension query into owned `CString`s, severing the dependence on the
// SDK-owned memory. Shared by the DLSS + XeSS extension queries.
//
// SAFETY: `exts` must be null or point to `count` valid, null-terminated C
// strings (the SDK contract).
unsafe fn copy_ext_names(count: u32, exts: *const *const c_char) -> Vec<CString> {
    if exts.is_null() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        // SAFETY: `exts` points at `count` entries the SDK just wrote, and `i` is below `count`.
        let p = unsafe { *exts.add(i) };
        if !p.is_null() {
            // SAFETY: `p` is non-null per the check above and the SDK owns the NUL-terminated name
            // for the lifetime of the library; `to_owned` copies it out.
            out.push(unsafe { CStr::from_ptr(p) }.to_owned());
        }
    }
    out
}

// Names of every device extension the physical device exposes, as owned
// `CString`s for equality checks against SDK-requested names.
fn supported_device_extensions(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Vec<CString> {
    // SAFETY: an enumeration query on a live instance handle; it only reads, and ash sizes the
    // output vector from the count the driver reports.
    let props = unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .unwrap_or_default();
    props
        .iter()
        .map(|e| {
            // SAFETY: Vulkan fills `extension_name` with a NUL-terminated string, and the borrow
            // does not outlive the properties entry it points into.
            let name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
            CString::from(name)
        })
        .collect()
}

impl VkContext {
    // Encode the temporal upscale onto `cmd`. Runs after SSR resolve / fog /
    // particles / transparent (so the scene input is the fully decorated
    // post-SSR color) and before Bloom + Composite (which sample the
    // upscaler's output, rewired at init / resize). Recorded onto the
    // `PassId::Upscale` per-pass command buffer by the executor. The barrier
    // choreography (output GENERAL, color / motion / depth SHADER_READ_ONLY) is
    // the same for every backend; only the inner `dispatch` differs.
    pub(in crate::vulkan) fn encode_upscale(
        &self,
        cmd: vk::CommandBuffer,
        params: &GraphFrameParams<'_>,
    ) -> RenderResult<()> {
        let Some(upscaler) = &self.upscale else {
            return Ok(());
        };
        let frame = params.frame_idx;
        let extent = upscaler.extent();

        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!("temporal upscaling: first encode_upscale firing ({extent})");
        }

        // Render-res motion + depth the upscalers consume. The unified G-buffer
        // pre-pass owns these (the `velocity` MRT target + the pre-pass's private
        // depth, both STORE'd shader-readable / depth attachment). Init builds the
        // merged pre-pass whenever upscaling is on (it forces `taa_enabled`), so it
        // is always present here; velocity rests in SHADER_READ_ONLY and depth in
        // DEPTH_STENCIL_ATTACHMENT so the barriers below are unchanged.
        let gb = self.gbuffer.as_ref().ok_or_else(|| {
            RenderError::Other(
                "upscale: enabled but the unified G-buffer pre-pass is absent".into(),
            )
        })?;
        let velocity = gb.velocity_images.get(frame).ok_or_else(|| {
            RenderError::Other("upscale: gbuffer velocity slot out of range".into())
        })?;
        let depth = gb
            .depth_images
            .get(frame)
            .ok_or_else(|| RenderError::Other("upscale: gbuffer depth slot out of range".into()))?;

        // Scene color: the reflection composite's output when a reflection
        // resolve ran, else this slot's HDR resolve. Either rests in
        // SHADER_READ_ONLY_OPTIMAL after its last render pass writer.
        let scene = self.post_scene_image(frame);
        let (rw, rh) = extent.render;
        let render_image = |image, view, format, aspect| UpscaleImage {
            image,
            view,
            format,
            width: rw,
            height: rh,
            aspect,
        };
        let color = render_image(
            scene.image,
            scene.view,
            HDR_FORMAT,
            vk::ImageAspectFlags::COLOR,
        );
        let motion = render_image(
            velocity.image,
            velocity.view,
            vk::Format::R16G16_SFLOAT,
            vk::ImageAspectFlags::COLOR,
        );
        let depth_in = render_image(
            depth.image,
            depth.view,
            vk::Format::D32_SFLOAT,
            vk::ImageAspectFlags::DEPTH,
        );

        // Make the producer writes (color / velocity = COLOR_ATTACHMENT_WRITE,
        // depth = DEPTH_STENCIL_ATTACHMENT_WRITE) visible to the upscaler's
        // COMPUTE reads, and transition the depth from its attachment layout to
        // SHADER_READ_ONLY. The color + velocity already rest in
        // SHADER_READ_ONLY (their render-pass final layout), so those are
        // same-layout execution+memory barriers.
        let color_read = BarrierSync {
            src_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            dst_access: vk::AccessFlags::SHADER_READ,
        };
        let stay_read_only = LayoutTransition {
            from: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            to: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        for image in [color.image, motion.image] {
            image_barrier(
                &self.hw.device,
                cmd,
                image,
                vk::ImageAspectFlags::COLOR,
                stay_read_only,
                color_read,
            );
        }
        image_barrier(
            &self.hw.device,
            cmd,
            depth_in.image,
            vk::ImageAspectFlags::DEPTH,
            LayoutTransition {
                from: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                to: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            BarrierSync {
                src_stage: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                src_access: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                dst_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
                dst_access: vk::AccessFlags::SHADER_READ,
            },
        );
        // The output rests in SHADER_READ_ONLY after the previous frame's
        // bloom + composite sampled it; flip it back to GENERAL for the write
        // (skipped on the first frame, where it starts in GENERAL).
        let output = upscaler.output();
        if output.layout.get() != vk::ImageLayout::GENERAL {
            image_barrier(
                &self.hw.device,
                cmd,
                output.image.image,
                vk::ImageAspectFlags::COLOR,
                LayoutTransition {
                    from: output.layout.get(),
                    to: vk::ImageLayout::GENERAL,
                },
                output.writes.acquire(
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::AccessFlags::SHADER_READ,
                ),
            );
            output.layout.set(vk::ImageLayout::GENERAL);
        }

        upscaler.dispatch(
            cmd,
            UpscaleInputs {
                color: &color,
                depth: &depth_in,
                motion: &motion,
            },
            UpscaleCamera::new(
                upscaler.jitter().get(),
                params.elapsed,
                params.near,
                params.fov_y_radians,
            ),
        )?;

        // Flip the output GENERAL -> SHADER_READ_ONLY so bloom + composite can
        // sample it. (Inputs are left where the upscaler leaves them; the next
        // frame's render passes reset them.)
        image_barrier(
            &self.hw.device,
            cmd,
            output.image.image,
            vk::ImageAspectFlags::COLOR,
            LayoutTransition {
                from: vk::ImageLayout::GENERAL,
                to: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            output.writes.release_to_sampling(),
        );
        output.layout.set(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn image_view_info_layout_matches_xess_v301_and_ngx_v150() {
        assert_eq!(size_of::<ImageViewInfo>(), 48);
        assert_eq!(offset_of!(ImageViewInfo, image_view), 0);
        assert_eq!(offset_of!(ImageViewInfo, image), 8);
        assert_eq!(offset_of!(ImageViewInfo, subresource_range), 16);
        assert_eq!(offset_of!(ImageViewInfo, format), 36);
        assert_eq!(offset_of!(ImageViewInfo, width), 40);
        assert_eq!(offset_of!(ImageViewInfo, height), 44);
    }

    #[test]
    fn storage_writes_need_no_transfer_usage() {
        let writes = OutputWrites::storage();
        assert_eq!(
            writes.image_usage(),
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED
        );
        let sync = writes.acquire(
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        );
        assert_eq!(sync.dst_stage, vk::PipelineStageFlags::COMPUTE_SHADER);
        assert_eq!(sync.dst_access, vk::AccessFlags::SHADER_WRITE);
    }

    #[test]
    fn clearing_writes_declare_the_transfer_write() {
        let writes = OutputWrites::storage_and_clear();
        assert!(
            writes
                .image_usage()
                .contains(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE)
        );
        let sync = writes.acquire(
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        );
        assert!(sync.dst_stage.contains(vk::PipelineStageFlags::TRANSFER));
        assert!(sync.dst_access.contains(vk::AccessFlags::TRANSFER_WRITE));
    }

    #[test]
    fn release_covers_every_write_the_acquire_allowed() {
        for writes in [OutputWrites::storage(), OutputWrites::storage_and_clear()] {
            let acquire = writes.acquire(
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::SHADER_READ,
            );
            let release = writes.release_to_sampling();
            assert_eq!(release.src_stage, acquire.dst_stage);
            assert_eq!(release.src_access, acquire.dst_access);
            assert_eq!(release.dst_stage, vk::PipelineStageFlags::FRAGMENT_SHADER);
            assert_eq!(release.dst_access, vk::AccessFlags::SHADER_READ);
        }
    }

    #[test]
    fn image_view_info_covers_one_mip_and_layer_of_the_aspect() {
        let img = UpscaleImage {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            format: vk::Format::D32_SFLOAT,
            width: 64,
            height: 32,
            aspect: vk::ImageAspectFlags::DEPTH,
        };
        let info = ImageViewInfo::of(&img);
        assert_eq!(
            info.subresource_range.aspect_mask,
            vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(
            (
                info.subresource_range.level_count,
                info.subresource_range.layer_count
            ),
            (1, 1)
        );
        assert_eq!(
            (info.width, info.height, info.format),
            (64, 32, vk::Format::D32_SFLOAT)
        );
        assert_eq!(ImageViewInfo::empty().subresource_range.layer_count, 0);
    }
}
