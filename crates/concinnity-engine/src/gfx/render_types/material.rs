//! Material layouts: material uniforms, parameter tables, and texture pool slots.

use concinnity_core::math::SaturatingFrom;

/// A draw object's material scalars. A CPU record: the cull packs it into the
/// `GpuObjectData` the GPU-driven pass reads, and no shader declares it.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct MaterialUniforms {
    /// Perceptual roughness [0, 1]: 0 = mirror, 1 = fully diffuse.
    pub roughness: f32,
    /// Metallic factor [0, 1]: 0 = dielectric, 1 = metal.
    pub metallic: f32,
    /// Alpha-cutout threshold in [0, 1]; 0 disables the test. Non-zero makes the
    /// fragment shader discard any texel whose albedo alpha falls below it, which
    /// is how foliage and decal cards punch holes while staying in the opaque pass.
    pub alpha_cutoff: f32,
    /// Base surface opacity in [0, 1]; 1 = fully opaque (the default). Only
    /// meaningful with `transparent`: it drives the glass alpha in the
    /// transparent pass. Carried on the material so it rides Material ->
    /// DrawObject.material to the backend; the opaque main-pass shader ignores it.
    pub opacity: f32,
    /// Linear-space RGB multiplier on the albedo sample.
    pub tint: [f32; 3],
    /// Padding so the field layout matches the shader-side struct.
    pub _pad0: f32,
    /// Additive emission color in linear space.
    pub emissive: [f32; 3],
    /// Padding so the field layout matches the shader-side struct.
    pub _pad1: f32,
    /// Bindless-pool index for the emissive map. `0` means no map: the shader
    /// gates on a non-zero index, falling back to the scalar `emissive` factor.
    pub emissive_map_index: u32,
    /// Bindless-pool index for the packed occlusion/roughness/metalness map
    /// (R = occlusion, G = roughness, B = metalness). `0` means no map: the
    /// shader keeps the scalar `roughness`/`metallic` and full occlusion.
    pub orm_map_index: u32,
    /// 1 when this surface routes through the transparent pass instead of the
    /// opaque one (a glass MESH on an RT-capable device); 0 for opaque. A CPU
    /// routing flag: the backend reads it to skip the draw in the opaque pass +
    /// the RT BLAS and to feed the per-pixel-RT transparent producer. No GPU
    /// shader reads it (the opaque skip is decided CPU-side).
    pub transparent: u32,
    /// 1 when a `transparent` glass MESH renders as genuinely see-through (Layer
    /// 2: the scene behind shows through plus a sharp per-pixel reflection); 0
    /// keeps it as opaque low-roughness reflective glass (Layer 1). Another CPU
    /// routing flag (no GPU shader reads it): the see-through producer, the
    /// opaque-pass skip, and the RT-BLAS exclude all key off it, so Layer 2 is
    /// opt-in per material. Always 0 unless `transparent` is also 1.
    pub see_through: u32,
    /// Row of the material parameter table holding this material's `params`;
    /// row 0 is the all-zero row a draw without a material reads.
    pub params_index: u32,
    /// How a terrain under a grass field blends toward its far-field color;
    /// [`FarFieldTint::NONE`] for every other surface.
    pub far_field: FarFieldTint,
}

/// Which of the canopy block's layers a surface under grass shades with: the
/// ground under its blades, and past where they end, the canopy they make.
/// Where the blades hand over is the frame's (`CanopyParams::band`). A
/// terrain with a far field carries each layer's density in its vertex color,
/// rather than a multiplier on its albedo. A planet's ground shades from the
/// block with no layers too, for its covers and wet sand.
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct FarFieldTint {
    /// The canopy block's layers the surface shades with, packed by
    /// `CanopyRange::packed`; 0 for a surface the block does not shade.
    pub canopy: u32,
    /// Keeps the record's later members where the shaders read them.
    pub _pad: [u32; 2],
}

impl FarFieldTint {
    /// No far field: the vertex color multiplies the albedo.
    pub const NONE: Self = Self {
        canopy: 0,
        _pad: [0; 2],
    };

    /// The far field shading with the canopy layers `canopy` packs.
    pub fn of(canopy: u32) -> Self {
        Self {
            canopy,
            _pad: [0; 2],
        }
    }

    /// Whether the surface's ground shades from the canopy block.
    pub fn shades_ground(&self) -> bool {
        self.canopy != 0
    }

    /// Whether the surface grows grass, and so has a far field past it.
    pub fn grows_grass(&self) -> bool {
        crate::render::environment::grass::CanopyRange::unpacked(self.canopy).count != 0
    }
}

impl MaterialUniforms {
    /// Neutral material: matte, non-metallic, white tint, no emission.
    pub const DEFAULT: Self = Self {
        roughness: 0.8,
        metallic: 0.0,
        alpha_cutoff: 0.0,
        opacity: 1.0,
        tint: [1.0, 1.0, 1.0],
        _pad0: 0.0,
        emissive: [0.0, 0.0, 0.0],
        _pad1: 0.0,
        emissive_map_index: 0,
        orm_map_index: 0,
        transparent: 0,
        see_through: 0,
        params_index: 0,
        far_field: FarFieldTint::NONE,
    };
}

/// Sentinel `normal_map_slot` meaning "this object has no normal map." The
/// texture pool is a single handle-indexed table shared by albedo and normal
/// maps, so a real normal map carries its own texture handle (which may be 0);
/// `usize::MAX` cannot collide with any handle and lets a backend substitute the
/// synthesized flat-normal fallback (tangent-space (0,0,1)) without a shader
/// branch. Backends map this to the fallback's reserved pool index.
pub const NO_NORMAL_MAP_SLOT: usize = usize::MAX;

/// Sentinel albedo `texture_slot` meaning "this object has no albedo texture."
/// Slot 0 cannot say that: it is a real handle, held by whichever texture the
/// world declares first, so an unset albedo encoded as 0 sampled that texture
/// and tinted it. `usize::MAX` collides with no handle and lets a backend
/// substitute the synthesized white fallback, which leaves `tint` as the
/// object's color without a shader branch.
pub const NO_ALBEDO_SLOT: usize = usize::MAX;

/// Pool entries reserved past the real textures: the flat-normal fallback at
/// `texture_count`, then the white albedo fallback at `texture_count + 1`.
/// Every backend appends them in this order, so the two `*_pool_index`
/// resolvers below agree with each backend's pool construction.
pub const FALLBACK_TEXTURE_COUNT: usize = 2;

/// Number of generic floats a `Material` carries for its Shader.
pub const MATERIAL_PARAM_COUNT: usize = 8;

/// One row of the material parameter table: a material's `params`, as the
/// main pass reads them. Declared in `shaders/main_bindless.hlsl` as two
/// `float4`s; row 0 is all zeros, for draws without a material.
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct GpuMaterialParams {
    /// The material's parameters, in order.
    pub values: [f32; MATERIAL_PARAM_COUNT],
}

/// Resolve an albedo `texture_slot` to its index in the handle-indexed texture
/// pool. A real albedo addresses its own texture slot (clamped into the
/// real-texture range so a stale slot reads the last valid entry rather than
/// out of bounds); `NO_ALBEDO_SLOT` addresses the white fallback at
/// `texture_count + 1`, so an untextured material samples white and shows its
/// `tint` rather than whichever texture happens to hold slot 0.
pub fn albedo_pool_index(texture_slot: usize, texture_count: u32) -> u32 {
    if texture_slot == NO_ALBEDO_SLOT {
        texture_count + 1
    } else {
        u32::saturating_from(texture_slot).min(texture_count.saturating_sub(1))
    }
}

/// Resolve a `normal_map_slot` to its index in the handle-indexed texture pool.
/// A real normal map addresses its own texture slot (clamped into the
/// real-texture range); `NO_NORMAL_MAP_SLOT` addresses the flat-normal fallback
/// at `texture_count` (the first reserved entry past the last real texture), so
/// the shader never needs a "has a normal map?" branch.
pub fn normal_pool_index(normal_map_slot: usize, texture_count: u32) -> u32 {
    if normal_map_slot == NO_NORMAL_MAP_SLOT {
        texture_count
    } else {
        u32::saturating_from(normal_map_slot).min(texture_count.saturating_sub(1))
    }
}
