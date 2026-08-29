// src/render_graph/types.rs
//
// Shared, backend-agnostic types for the render graph: resource handles,
// resource descriptions, state / access enums, and the small structs the
// compile pass emits. The graph tracks *order*, *barriers*, and
// *lifetimes*: it does not allocate transient GPU resources (those stay
// backend-owned).

use crate::math::floor;
use core::num::NonZeroU32;

// The set operations every flag newtype in this module shares. `$noun` names
// what a bit means so the generated rustdoc reads naturally per type.
macro_rules! flag_set_ops {
    ($ty:ident, $noun:literal) => {
        impl $ty {
            #[doc = concat!("The empty ", $noun, " set.")]
            pub const fn empty() -> Self {
                Self(0)
            }

            #[doc = concat!("Every ", $noun, " in either set.")]
            pub const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }

            #[doc = concat!("Whether every ", $noun, " in `other` is set here.")]
            pub const fn contains(self, other: Self) -> bool {
                (self.0 & other.0) == other.0
            }
        }

        impl core::ops::BitOr for $ty {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                self.union(rhs)
            }
        }
    };
}

/// One side of a `PassBuilder::read_*` / `write_*` declaration. The
/// resource is a small dense index into the graph's resource arena; the
/// `version` increments on every write so a read-after-write chain
/// (`main → decals → fog` writing the same hdr_resolve) is an unambiguous
/// DAG.
///
/// Each `TextureHandle` / `BufferHandle` pairs the resource id with the
/// version it refers to. `write_*` returns a new handle pointing at the
/// post-write version; the old handle stays valid (and refers to the
/// pre-write version) so a pass can still legally read the prior content
/// if it wants.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct TextureHandle {
    pub(super) resource: ResourceId,
    pub(super) version: u32,
}

impl TextureHandle {
    // Sentinel for "no texture", used by the per-frame graph builder
    // for conditional passes (SSR off, TAA off, ...) so the call sites
    // stay branchless. The compile pass treats reads / writes of an
    // invalid handle as no-ops.
    pub(crate) const INVALID: Self = Self {
        resource: ResourceId::INVALID,
        version: 0,
    };

    // `true` when this handle was produced by a valid `create_*` /
    // `import_*` call; `false` when it's the `INVALID` sentinel.
    pub(crate) fn is_valid(self) -> bool {
        self.resource.is_valid()
    }
}

// Buffer counterpart to [`TextureHandle`]. Same handle / version model.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct BufferHandle {
    pub(super) resource: ResourceId,
    pub(super) version: u32,
}

impl BufferHandle {
    pub(crate) const INVALID: Self = Self {
        resource: ResourceId::INVALID,
        version: 0,
    };

    pub(crate) fn is_valid(self) -> bool {
        self.resource.is_valid()
    }
}

/// Dense resource identifier. `u32::MAX` reserved as the "invalid"
/// sentinel; everything else is a valid index into the compiled graph's
/// `resources` Vec. The executor uses `index()` to look up a resource's
/// realised GPU object.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ResourceId(pub(super) u32);

impl ResourceId {
    pub(crate) const INVALID: Self = Self(u32::MAX);

    pub(crate) fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }

    /// The resource's stable index into `CompiledGraph.resources`.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// What kind of work an executor encodes for a pass: render vs compute.
/// The graph cares about this only enough to pick the right
/// `MTLRenderPassDescriptor` / `MTLComputePassDescriptor` analogue per
/// backend; the actual encoding stays in the per-backend `encode_*`
/// methods. Blit passes are not yet in scope (today's engine has none).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PassKind {
    /// A render pass.
    Render,
    /// A compute pass.
    Compute,
}

// Whether a resource is engine-owned (the graph references it) or
// declared inside the graph (the graph tracks its lifetime; the graph
// does not yet own its allocation).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResourceOrigin {
    // Engine owns the GPU object; the graph just references it by
    // handle. Most of today's `MtlContext` targets enter the graph via
    // `import_texture` / `import_buffer`.
    Imported,
    // Graph-tracked resource declared via `create_texture` /
    // `create_buffer`. The backend asserts it has a target of matching
    // shape; the graph does not yet allocate from a pool with aliasing.
    Transient,
}

/// Coarse per-resource state used by the barrier deriver. The
/// executor maps each to the backend's concrete state: for Vulkan a
/// `VkImageLayout` + `VkAccessFlags` pair, for DirectX a
/// `D3D12_RESOURCE_STATES`, for Metal mostly a no-op except `useResource`
/// on the ICB-driven cull pass.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResourceState {
    /// Initial state before any pass uses the resource. Reading from
    /// `Undefined` is a compile error (a read with no producer); writing
    /// to it is always legal and transitions the state to a writer
    /// variant below.
    Undefined,
    /// A pass reads the resource (sampled texture, uniform / SSBO,
    /// indirect-args buffer, depth read, ...). Multiple consecutive
    /// reads are coalesced: they don't insert barriers between
    /// themselves.
    Read,
    /// A pass writes the resource (render target, depth-stencil target,
    /// storage write, blend write). A second write after a read inserts
    /// a Write→Read→Write barrier chain; consecutive writes by the same
    /// pass do not, but consecutive writes across passes do.
    Write,
}

/// How a graph resource a backend drives from `barriers_before` is used, so
/// the backend can translate the coarse `ResourceState` into a concrete native
/// state: the same `Write` means a colour render target for one resource and a
/// depth-stencil target for another, which map to different
/// `D3D12_RESOURCE_STATES` / `vk::ImageLayout`s. The backend resolver assigns a
/// class to each migrated resource; the backend's barrier translator maps
/// `(class, state)` to its native state. Extend as resources of new kinds
/// (storage / compute targets, ...) migrate.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum GraphResourceClass {
    /// Sampled colour render target (e.g. the SSAO occlusion `ao_output`).
    ColorTarget,
    /// Sampled depth-stencil render target (e.g. the CSM `shadow_map`).
    DepthTarget,
    /// Compute-written, shader-sampled storage image (e.g. the volumetric-fog
    /// `fog_froxel_volume`): a compute pass writes it (DirectX UNORDERED_ACCESS /
    /// Vulkan GENERAL) and a later fragment pass samples it. Unlike the two
    /// target classes its `Write` happens in the compute stage, so the translated
    /// state pairs a storage-write layout with the compute pipeline stage.
    StorageImage,
    /// Compute-written buffer consumed as draw arguments (e.g. the GPU cull pass's
    /// `draw_args`): a compute pass writes it and the drawing pass reads it through
    /// the indirect-draw stage, which is neither of the shader stages `ReadStages`
    /// models. Buffers carry no layout, so a transition on this class is a pure
    /// execution + memory dependency.
    IndirectBuffer,
    /// Compute-written, shader-read buffer (e.g. the clustered-lighting
    /// `cluster_light_list`). Like `IndirectBuffer` it has no layout; its read side
    /// follows the consuming stage union, so it reads as a shader resource.
    StorageBuffer,
    /// Buffer both written and read through an unordered-access view (e.g. the
    /// two-pass cull's `cull_status`, which phase 1 writes and phase 2 reads with
    /// the same binding). Distinct from `StorageBuffer` because its read is not a
    /// shader-resource read: on DirectX it never leaves `UNORDERED_ACCESS`, so its
    /// ordering comes from a UAV barrier rather than a state transition.
    UnorderedBuffer,
}

impl GraphResourceClass {
    /// The class a texture of `usage` belongs to. Declared usage is the single
    /// source of truth for it: a backend resolves a resource label to its GPU
    /// object, but never restates what kind of resource it is, so the two
    /// executors cannot disagree about (say) whether the shadow map is a depth
    /// target. Depth-stencil wins over storage wins over render target, since a
    /// target declaring several is used in the most constrained of them.
    pub(crate) const fn for_texture_usage(usage: TextureUsage) -> Self {
        if usage.contains(TextureUsage::DEPTH_STENCIL) {
            GraphResourceClass::DepthTarget
        } else if usage.contains(TextureUsage::STORAGE) {
            GraphResourceClass::StorageImage
        } else {
            GraphResourceClass::ColorTarget
        }
    }

    /// The class a buffer of `usage` belongs to, on the same declared-usage rule
    /// as [`Self::for_texture_usage`]. Indirect arguments win over the read-write
    /// binding, since a buffer declaring both is consumed as draw arguments.
    pub(crate) const fn for_buffer_usage(usage: BufferUsage) -> Self {
        if usage.contains(BufferUsage::INDIRECT) {
            GraphResourceClass::IndirectBuffer
        } else if usage.contains(BufferUsage::UNORDERED) {
            GraphResourceClass::UnorderedBuffer
        } else {
            GraphResourceClass::StorageBuffer
        }
    }

    /// Whether this class names a buffer rather than an image. Buffers have no
    /// layout, so a backend emits a buffer / global memory barrier for them and an
    /// image barrier for everything else.
    pub const fn is_buffer(self) -> bool {
        matches!(
            self,
            GraphResourceClass::IndirectBuffer
                | GraphResourceClass::StorageBuffer
                | GraphResourceClass::UnorderedBuffer
        )
    }
}

/// Which shader stage(s) read a graph resource across a contiguous read-run
/// (the passes that read one resource version before the next writer). Carried
/// on a barrier whose Read side spans this run so a backend can satisfy it in a
/// single transition: a write made visible to both a compute consumer and a
/// fragment consumer needs one barrier covering both stages, not a per-consumer
/// read-to-read barrier (which would not carry the producing write). Derived
/// from each reading pass's `PassKind` (a render pass samples in the fragment
/// stage, a compute pass in the compute stage); empty on a barrier with no Read
/// side (a write-only producer transition). Add bits as passes read in stages
/// the two current ones do not model.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ReadStages(u32);

flag_set_ops!(ReadStages, "stage");

impl ReadStages {
    /// A render pass's sampled read (DirectX PIXEL_SHADER_RESOURCE / Vulkan
    /// FRAGMENT_SHADER stage).
    pub const FRAGMENT: Self = Self(1 << 0);
    /// A compute pass's read (DirectX NON_PIXEL_SHADER_RESOURCE / Vulkan
    /// COMPUTE_SHADER stage).
    pub const COMPUTE: Self = Self(1 << 1);

    /// Whether no stage is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The stage a pass of `kind` reads a resource in: render passes sample in
    /// the fragment stage, compute passes in the compute stage. This is the one
    /// place a `PassKind` becomes a read stage, so the approximation lives here:
    /// a render pass that sampled in the vertex / geometry stage would be
    /// labelled FRAGMENT. No graph-driven resource is read that way today; if
    /// one ever is, carry an explicit per-read stage instead of deriving it.
    pub(crate) const fn for_pass_kind(kind: PassKind) -> Self {
        match kind {
            PassKind::Render => Self::FRAGMENT,
            PassKind::Compute => Self::COMPUTE,
        }
    }
}

/// One barrier the executor must insert before a pass runs. Per-backend
/// interpretation: Vulkan emits `vkCmdPipelineBarrier`; DirectX emits
/// `D3D12_RESOURCE_BARRIER`; Metal mostly ignores them (implicit hazard
/// tracking) but may translate `from: Write, to: Read` on the cull ICB
/// path into an explicit `useResource(.Write)` declaration.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BarrierOp {
    pub(super) resource: ResourceId,
    pub(super) from: ResourceState,
    pub(super) to: ResourceState,
    // Stage union of this barrier's Read side (see `ReadStages`): the consuming
    // run's stages for a `* -> Read` transition, the prior run's stages for a
    // `Read -> Write` (WAR), empty when neither side is Read. The backend
    // translator targets this union so one transition covers every consuming
    // stage.
    pub(super) read_stages: ReadStages,
}

impl BarrierOp {
    /// Pass-local accessors so callers don't have to import `ResourceId`.
    /// Returns the resource's stable index, the same value the executor
    /// uses to look the resource up in `CompiledGraph.resources`.
    pub fn resource_index(self) -> usize {
        self.resource.index()
    }
    /// Accessor for the transition's source state, paired with `to_state`.
    pub fn source_state(self) -> ResourceState {
        self.from
    }
    /// The transition's destination state.
    pub fn to_state(self) -> ResourceState {
        self.to
    }
    /// Stage union of this barrier's Read side: the consuming run's stages for a
    /// `* -> Read` transition (the backend must make the producing write visible
    /// to all of them), the prior run's stages for a `Read -> Write` (WAR).
    /// Empty when neither side is Read.
    pub fn read_stages(self) -> ReadStages {
        self.read_stages
    }
}

// Inclusive `[first, last]` range over pass indices in the compiled
// graph's `passes` Vec. Used to describe a transient resource's
// lifetime so an aliaser can overlap non-overlapping lifetimes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PassRange {
    pub first: usize,
    pub last: usize,
}

/// Texture-shape description carried by both imported and transient
/// resources. This is the authoritative shape: the aliaser sizes a
/// resource from it and each backend's transient pool translates it into
/// a native descriptor, so a desc that disagrees with what the backend
/// would have created is a defect rather than a documentation slip.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TextureDesc {
    /// Width in pixels, or a fraction of the drawable.
    pub width: TextureSize,
    /// Height in pixels, or a fraction of the drawable.
    pub height: TextureSize,
    /// Depth extent of a 3D texture, 1 for 2D and 2D-array textures. Distinct
    /// from `array_layers`: 3D slices mip down with the other two axes, array
    /// layers do not.
    pub depth: u32,
    /// Texel format.
    pub format: PixelFormat,
    /// MSAA sample count, 1 for non-multisample. The graph doesn't care
    /// what value this is; the backend executor maps it to its API's
    /// sample-count enum.
    pub sample_count: u32,
    /// Number of array layers. 1 for plain 2D, 6 for cube, N for CSM
    /// shadow-map arrays.
    pub array_layers: u32,
    /// Mip levels in the chain, 1 for a single-level target. A resource whose
    /// consumers sample coarser levels (the Hi-Z pyramid, a bloom octave chain)
    /// carries its real count, since the levels past 0 are a third of the
    /// footprint the aliaser packs.
    pub mip_levels: u32,
    /// How passes bind the texture.
    pub usage: TextureUsage,
    /// The value this target is cleared to at the head of the pass that writes
    /// it. Part of the shape rather than the encoder's business because D3D12
    /// bakes an *optimized* clear value into the resource at creation: a placed
    /// resource created with one value and cleared to another is both a
    /// debug-layer warning and a real decompression cost, and the pool creates
    /// the resource while the feature owns the clear. A target whose background
    /// means something (roughness clears to 1.0 = fully rough, so untouched
    /// pixels reflect nothing) reads wrong on its first frame if these disagree.
    pub clear: ClearValue,
}

/// What a target's clear resolves to. Split by kind rather than carried as four
/// floats so a depth target cannot silently be given a colour.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ClearValue {
    /// Colour clear value, linear RGBA.
    Color([f32; 4]),
    /// Depth clear value; stencil is always 0 (no engine target has stencil).
    Depth(f32),
}

// Buffer-shape description. Size is optional because some
// imported buffers grow dynamically per-frame (the GPU object data
// buffer, the per-emitter spawn ring, ...): the graph then just
// tracks the dependency, not the size.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct BufferDesc {
    pub size_bytes: Option<NonZeroU32>,
    pub usage: BufferUsage,
}

// How a texture is sized. Two non-absolute variants let bloom mips and
// full-resolution targets express their size without the graph needing
// to know the swapchain dimensions at declaration time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TextureSize {
    // Fixed pixel count. CSM shadow-map slices use this
    // (`Absolute(2048)`); the rest of the engine's targets follow the
    // drawable.
    Absolute(u32),
    // Tracks the swapchain drawable's width or height.
    Drawable,
    // Scaled fraction of the drawable, floored to >= 1 by the executor.
    // Bloom mips chain through this (`DrawableScaled(0.5)^n`).
    DrawableScaled(f32),
}

/// Backend-agnostic pixel format. Maps to `MTLPixelFormat` /
/// `vk::Format` / `DXGI_FORMAT` per executor. Only the formats the
/// engine actually uses are enumerated; extend as new passes need new
/// targets.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// 16-bit float RGBA, the HDR working format.
    Rgba16Float,
    /// 8-bit unorm RGBA.
    Rgba8Unorm,
    /// 16-bit float RG.
    Rg16Float,
    /// 8-bit unorm single channel.
    R8Unorm,
    /// 32-bit float single channel.
    R32Float,
    /// 32-bit float depth.
    Depth32Float,
    /// Whatever format the swapchain presents.
    BgraSwapchain,
}

impl PixelFormat {
    /// Bytes per texel (per sample). Used by the aliasing planner to size a
    /// resource's memory footprint. The engine uses only single-plane,
    /// power-of-two formats, so this is one byte count per variant.
    pub(crate) const fn bytes_per_texel(self) -> u32 {
        match self {
            PixelFormat::Rgba16Float => 8,
            PixelFormat::Rgba8Unorm
            | PixelFormat::Rg16Float
            | PixelFormat::R32Float
            | PixelFormat::Depth32Float
            | PixelFormat::BgraSwapchain => 4,
            PixelFormat::R8Unorm => 1,
        }
    }

    /// Whether this is a depth format. The aliasing planner keeps depth and
    /// colour resources in separate memory pools because their backend memory
    /// requirements (heap flags / memory type) differ; the finer per-usage
    /// compatibility is the backend's concern when it realises the plan.
    pub const fn is_depth(self) -> bool {
        matches!(self, PixelFormat::Depth32Float)
    }
}

impl TextureSize {
    // Resolve to a concrete pixel count against the current drawable extent.
    // `DrawableScaled` floors to >= 1 so a mip-scaled target never degenerates
    // to zero.
    pub fn resolve(self, drawable: u32) -> u32 {
        match self {
            TextureSize::Absolute(n) => n.max(1),
            TextureSize::Drawable => drawable.max(1),
            TextureSize::DrawableScaled(f) => (floor(drawable as f32 * f) as u32).max(1),
        }
    }
}

// Levels in a full mip chain for a target of `width` x `height`:
// `floor(log2(max(w, h))) + 1`. Power-of-two sources end exactly at 1x1;
// non-power-of-two sources stop one level short of 1x1 on the smaller axis,
// which is what each backend's Hi-Z build already does.
pub(crate) const fn full_mip_levels(width: u32, height: u32) -> u32 {
    let m = if width > height { width } else { height };
    let m = if m < 1 { 1 } else { m };
    32 - m.leading_zeros()
}

impl TextureDesc {
    /// A single-sample, single-mip, single-layer 2D texture: the shape almost
    /// every graph target has. The axes that differ are added by the `with_*`
    /// methods below, so a desc that names one is saying something.
    pub(crate) const fn texture_2d(
        width: TextureSize,
        height: TextureSize,
        format: PixelFormat,
        usage: TextureUsage,
    ) -> Self {
        Self {
            width,
            height,
            depth: 1,
            format,
            sample_count: 1,
            array_layers: 1,
            mip_levels: 1,
            usage,
            // Zero / far, which is what every target the graph models clears to
            // unless it says otherwise via `with_clear_color`.
            clear: if format.is_depth() {
                ClearValue::Depth(1.0)
            } else {
                ClearValue::Color([0.0; 4])
            },
        }
    }

    /// A single-mip 3D texture of `depth` slices (the volumetric-fog froxel
    /// volume). Distinct from an array: the slices are a sampled third axis.
    pub(crate) const fn volume_3d(
        width: TextureSize,
        height: TextureSize,
        depth: u32,
        format: PixelFormat,
        usage: TextureUsage,
    ) -> Self {
        Self {
            depth,
            ..Self::texture_2d(width, height, format, usage)
        }
    }

    /// This desc with the MSAA sample count replaced.
    pub(crate) const fn with_sample_count(self, sample_count: u32) -> Self {
        Self {
            sample_count,
            ..self
        }
    }

    /// This desc with the array-layer count replaced.
    pub(crate) const fn with_array_layers(self, array_layers: u32) -> Self {
        Self {
            array_layers,
            ..self
        }
    }

    /// This desc with the mip-level count replaced.
    pub(crate) const fn with_mip_levels(self, mip_levels: u32) -> Self {
        Self { mip_levels, ..self }
    }

    /// Override the colour a target clears to. Only for a target whose cleared
    /// background carries meaning; see `clear`.
    pub(crate) const fn with_clear_color(self, color: [f32; 4]) -> Self {
        Self {
            clear: ClearValue::Color(color),
            ..self
        }
    }

    /// Resolved pixel extent at the given drawable extent, as the backend must
    /// create it: `(width, height, depth)`.
    pub fn extent(&self, drawable_w: u32, drawable_h: u32) -> (u32, u32, u32) {
        (
            self.width.resolve(drawable_w),
            self.height.resolve(drawable_h),
            self.depth.max(1),
        )
    }

    /// The resource's memory footprint in bytes at the given drawable extent:
    /// the summed texel count of every mip level, times bytes-per-texel,
    /// sample count and array layers. The aliasing planner sums and packs
    /// these. Sample count multiplies the whole chain, which is exact for the
    /// only shape that carries both (a multisample target is single-mip).
    pub fn byte_size(&self, drawable_w: u32, drawable_h: u32) -> u64 {
        let (w, h, d) = self.extent(drawable_w, drawable_h);
        let mut texels: u64 = 0;
        for level in 0..self.mip_levels.max(1) {
            let at = |n: u32| n.checked_shr(level).unwrap_or(0).max(1) as u64;
            texels += at(w) * at(h) * at(d);
        }
        texels
            * self.format.bytes_per_texel() as u64
            * self.sample_count.max(1) as u64
            * self.array_layers.max(1) as u64
    }
}

/// Bitset describing how a texture can be used. The graph doesn't
/// enforce these against declared reads / writes (executors do);
/// the field exists so the aliaser can match transient resources to
/// pool entries with the right usage flags.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TextureUsage(pub u32);

impl TextureUsage {
    /// Sampled or otherwise read from a shader.
    pub const SHADER_READ: Self = Self(1 << 0);
    /// Bound as a colour render target.
    pub const RENDER_TARGET: Self = Self(1 << 1);
    /// Bound as a depth / stencil target.
    pub const DEPTH_STENCIL: Self = Self(1 << 2);
    /// Bound for read-write shader storage.
    pub const STORAGE: Self = Self(1 << 3);
    /// Source of a copy.
    pub const TRANSFER_SRC: Self = Self(1 << 4);
    /// Destination of a copy.
    pub const TRANSFER_DST: Self = Self(1 << 5);
}

flag_set_ops!(TextureUsage, "usage");

/// Buffer-side counterpart to [`TextureUsage`]. Same bitset shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BufferUsage(pub u32);

impl BufferUsage {
    /// Bound for read-write shader storage.
    pub const STORAGE: Self = Self(1 << 1);
    /// Read as indirect draw / dispatch arguments.
    pub(crate) const INDIRECT: Self = Self(1 << 4);
    /// Consumers access this buffer through the same read-write binding the
    /// producer wrote through, rather than a read-only view of it. It therefore
    /// never transitions to a read state, and ordering between the producer and
    /// the consumer comes from an execution barrier instead of a state change.
    /// The two-pass cull's `cull_status` is the case: both phases bind it the
    /// same way.
    pub(crate) const UNORDERED: Self = Self(1 << 7);
}

flag_set_ops!(BufferUsage, "usage");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_handle_is_invalid() {
        assert!(!TextureHandle::INVALID.is_valid());
        assert!(!BufferHandle::INVALID.is_valid());
    }

    #[test]
    fn texture_usage_bitset_round_trips() {
        let u = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        assert!(u.contains(TextureUsage::SHADER_READ));
        assert!(u.contains(TextureUsage::RENDER_TARGET));
        assert!(!u.contains(TextureUsage::STORAGE));
        assert_eq!(
            u.union(TextureUsage::STORAGE).0,
            TextureUsage::SHADER_READ.0 | TextureUsage::RENDER_TARGET.0 | TextureUsage::STORAGE.0
        );
    }

    #[test]
    fn pass_range_is_inclusive() {
        let r = PassRange { first: 2, last: 5 };
        assert_eq!(r.first, 2);
        assert_eq!(r.last, 5);
    }

    #[test]
    fn pixel_format_texel_size_and_depth() {
        assert_eq!(PixelFormat::Rgba16Float.bytes_per_texel(), 8);
        assert_eq!(PixelFormat::Rgba8Unorm.bytes_per_texel(), 4);
        assert_eq!(PixelFormat::Rg16Float.bytes_per_texel(), 4);
        assert_eq!(PixelFormat::R32Float.bytes_per_texel(), 4);
        assert_eq!(PixelFormat::Depth32Float.bytes_per_texel(), 4);
        assert_eq!(PixelFormat::BgraSwapchain.bytes_per_texel(), 4);
        assert_eq!(PixelFormat::R8Unorm.bytes_per_texel(), 1);
        assert!(PixelFormat::Depth32Float.is_depth());
        assert!(!PixelFormat::Rgba8Unorm.is_depth());
        assert!(!PixelFormat::R8Unorm.is_depth());
    }

    #[test]
    fn texture_size_resolves_and_floors_to_one() {
        assert_eq!(TextureSize::Absolute(2048).resolve(720), 2048);
        assert_eq!(TextureSize::Absolute(0).resolve(720), 1);
        assert_eq!(TextureSize::Drawable.resolve(720), 720);
        assert_eq!(TextureSize::Drawable.resolve(0), 1);
        assert_eq!(TextureSize::DrawableScaled(0.5).resolve(720), 360);
        // floor(1 * 0.5) = 0, floored back up to 1 so a mip never degenerates.
        assert_eq!(TextureSize::DrawableScaled(0.5).resolve(1), 1);
    }

    #[test]
    fn texture_desc_byte_size_multiplies_every_factor() {
        let base = TextureDesc::texture_2d(
            TextureSize::Drawable,
            TextureSize::Drawable,
            PixelFormat::Rgba16Float, // 8 bytes / texel
            TextureUsage::RENDER_TARGET,
        );
        assert_eq!(base.byte_size(4, 2), 4 * 2 * 8);
        // Sample count and array layers both multiply in.
        let multi = base.with_sample_count(4).with_array_layers(6);
        assert_eq!(multi.byte_size(4, 2), 4 * 2 * 8 * 4 * 6);
        // A zero sample count / layer count clamps to 1 rather than zeroing.
        let degenerate = TextureDesc {
            sample_count: 0,
            array_layers: 0,
            mip_levels: 0,
            ..base
        };
        assert_eq!(degenerate.byte_size(4, 2), 4 * 2 * 8);
    }

    #[test]
    fn byte_size_sums_the_mip_chain() {
        // A 3-level chain over 8x8: 64 + 16 + 4 texels, not 64. Under-counting
        // the tail is what would let the aliaser undersize a shared slot.
        let chain = TextureDesc::texture_2d(
            TextureSize::Drawable,
            TextureSize::Drawable,
            PixelFormat::R8Unorm,
            TextureUsage::SHADER_READ,
        )
        .with_mip_levels(3);
        assert_eq!(chain.byte_size(8, 8), 64 + 16 + 4);
        // Levels floor at one texel rather than vanishing, so an over-long
        // chain keeps adding 1 instead of 0.
        let over_long = chain.with_mip_levels(6);
        assert_eq!(over_long.byte_size(8, 8), 64 + 16 + 4 + 1 + 1 + 1);
    }

    #[test]
    fn byte_size_counts_volume_slices_and_mips_them() {
        // 3D depth multiplies like the other two axes, and mips down with
        // them -- which is what separates it from `array_layers`.
        let volume = TextureDesc::volume_3d(
            TextureSize::Drawable,
            TextureSize::Drawable,
            4,
            PixelFormat::R8Unorm,
            TextureUsage::STORAGE,
        );
        assert_eq!(volume.byte_size(8, 8), 8 * 8 * 4);
        assert_eq!(
            volume.with_mip_levels(2).byte_size(8, 8),
            8 * 8 * 4 + 4 * 4 * 2
        );
        // An array of the same nominal size does not shrink its layer count.
        let array = TextureDesc::texture_2d(
            TextureSize::Drawable,
            TextureSize::Drawable,
            PixelFormat::R8Unorm,
            TextureUsage::STORAGE,
        )
        .with_array_layers(4)
        .with_mip_levels(2);
        assert_eq!(array.byte_size(8, 8), (8 * 8 + 4 * 4) * 4);
    }

    #[test]
    fn constructors_default_the_uninteresting_axes() {
        let d = TextureDesc::texture_2d(
            TextureSize::Drawable,
            TextureSize::Absolute(7),
            PixelFormat::Rgba8Unorm,
            TextureUsage::SHADER_READ,
        );
        assert_eq!(
            (d.depth, d.sample_count, d.array_layers, d.mip_levels),
            (1, 1, 1, 1)
        );
        assert_eq!(d.extent(3, 99), (3, 7, 1));
        let v = TextureDesc::volume_3d(
            TextureSize::Absolute(2),
            TextureSize::Absolute(3),
            5,
            PixelFormat::Rgba8Unorm,
            TextureUsage::STORAGE,
        );
        assert_eq!(v.extent(0, 0), (2, 3, 5));
        assert_eq!(v.array_layers, 1, "a volume is not an array");
    }

    #[test]
    fn full_mip_levels_matches_the_backends_hiz_chain() {
        // Mirrors each backend's `hiz_mip_count`: floor(log2(max)) + 1.
        assert_eq!(full_mip_levels(1, 1), 1);
        assert_eq!(full_mip_levels(2, 1), 2);
        assert_eq!(full_mip_levels(1920, 1080), 11);
        assert_eq!(full_mip_levels(1024, 1024), 11);
        // Zero on either axis still yields a one-level chain.
        assert_eq!(full_mip_levels(0, 0), 1);
    }

    #[test]
    fn read_stages_bitset_and_pass_kind() {
        let both = ReadStages::FRAGMENT | ReadStages::COMPUTE;
        assert!(both.contains(ReadStages::FRAGMENT));
        assert!(both.contains(ReadStages::COMPUTE));
        assert!(!both.is_empty());
        assert!(ReadStages::empty().is_empty());
        assert!(!ReadStages::empty().contains(ReadStages::FRAGMENT));
        // union with empty is the identity.
        assert_eq!(both.union(ReadStages::empty()), both);
        // A render pass reads in the fragment stage, a compute pass in compute.
        assert_eq!(
            ReadStages::for_pass_kind(PassKind::Render),
            ReadStages::FRAGMENT
        );
        assert_eq!(
            ReadStages::for_pass_kind(PassKind::Compute),
            ReadStages::COMPUTE
        );
    }

    #[test]
    fn class_follows_declared_usage() {
        // Declared usage is the single source of truth for a resource's barrier
        // class, so the precedence between overlapping bits is pinned here rather
        // than restated by each backend.
        let tex = |usage| GraphResourceClass::for_texture_usage(usage);
        assert_eq!(
            tex(TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ),
            GraphResourceClass::ColorTarget
        );
        assert_eq!(
            tex(TextureUsage::DEPTH_STENCIL | TextureUsage::SHADER_READ),
            GraphResourceClass::DepthTarget
        );
        assert_eq!(
            tex(TextureUsage::STORAGE | TextureUsage::SHADER_READ),
            GraphResourceClass::StorageImage
        );
        // Depth wins over storage, storage over render target: a target declaring
        // several is used in the most constrained of them.
        assert_eq!(
            tex(TextureUsage::DEPTH_STENCIL | TextureUsage::STORAGE | TextureUsage::RENDER_TARGET),
            GraphResourceClass::DepthTarget
        );
        assert_eq!(
            tex(TextureUsage::STORAGE | TextureUsage::RENDER_TARGET),
            GraphResourceClass::StorageImage
        );

        let buf = |usage| GraphResourceClass::for_buffer_usage(usage);
        assert_eq!(buf(BufferUsage::STORAGE), GraphResourceClass::StorageBuffer);
        assert_eq!(
            buf(BufferUsage::STORAGE | BufferUsage::UNORDERED),
            GraphResourceClass::UnorderedBuffer
        );
        assert_eq!(
            buf(BufferUsage::STORAGE | BufferUsage::INDIRECT),
            GraphResourceClass::IndirectBuffer
        );
        // Indirect wins over the read-write binding: a buffer declaring both is
        // consumed as draw arguments.
        assert_eq!(
            buf(BufferUsage::INDIRECT | BufferUsage::UNORDERED),
            GraphResourceClass::IndirectBuffer
        );
        // Every buffer class reports as a buffer; no image class does.
        for c in [
            GraphResourceClass::IndirectBuffer,
            GraphResourceClass::StorageBuffer,
            GraphResourceClass::UnorderedBuffer,
        ] {
            assert!(c.is_buffer(), "{c:?}");
        }
        for c in [
            GraphResourceClass::ColorTarget,
            GraphResourceClass::DepthTarget,
            GraphResourceClass::StorageImage,
        ] {
            assert!(!c.is_buffer(), "{c:?}");
        }
    }

    #[test]
    fn buffer_usage_bitset_ops() {
        let u = BufferUsage::STORAGE | BufferUsage::INDIRECT;
        assert!(u.contains(BufferUsage::STORAGE));
        assert!(u.contains(BufferUsage::INDIRECT));
        assert!(!u.contains(BufferUsage::UNORDERED));
        assert_eq!(BufferUsage::empty().0, 0);
        assert_eq!(
            u.union(BufferUsage::UNORDERED).0,
            BufferUsage::STORAGE.0 | BufferUsage::INDIRECT.0 | BufferUsage::UNORDERED.0
        );
    }

    #[test]
    fn barrier_op_accessors_expose_the_transition() {
        let op = BarrierOp {
            resource: ResourceId(4),
            from: ResourceState::Read,
            to: ResourceState::Write,
            read_stages: ReadStages::FRAGMENT,
        };
        assert_eq!(op.resource_index(), 4);
        assert_eq!(op.source_state(), ResourceState::Read);
        assert_eq!(op.to_state(), ResourceState::Write);
        assert_eq!(op.read_stages(), ReadStages::FRAGMENT);
    }
}
