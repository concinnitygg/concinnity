//! Post-process layouts: the parameter blocks of the screen-space and composite passes.

/// The authored half of [`PostProcessParams`]: everything the `Bloom`,
/// `Exposure`, `Vignette`, `ColorLut` and `AntiAliasing` assets (or a settings
/// slider) resolve to. Carries no display-output flag, so a
/// live push can never disturb the EDR path the backend negotiated at init.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PostProcessTunables {
    /// Additive bloom strength. 0 disables the bloom passes entirely.
    pub bloom_intensity: f32,
    /// Luminance threshold for the bloom prefilter.
    pub bloom_threshold: f32,
    /// Quadratic soft-knee width below the threshold.
    pub bloom_knee: f32,
    /// Linear exposure multiplier applied to HDR radiance before the bloom
    /// prefilter and the composite tonemap. Resolved from `Exposure.ev` as
    /// `2^ev`, so 1.0 is neutral.
    pub exposure: f32,
    /// Vignette strength in `[0, 1]`. 0 disables the corner darkening.
    pub vignette: f32,
    /// Color-LUT blend in `[0, 1]`: `mix(scene, graded, lut_strength)` in the
    /// composite pass. 0 skips the LUT, and is what a world without a
    /// `ColorLut` resolves to.
    pub lut_strength: f32,
    /// FXAA edge-filter flag on the SDR path. `1.0` runs the composite's FXAA
    /// pass; `0.0` skips it (the `Off` anti-aliasing mode). Resolved from
    /// `AntiAliasing.mode`: on for `Fxaa` and `Taa`, off for `Off`.
    pub fxaa: f32,
}

impl PostProcessTunables {
    /// What the assets' defaults resolve to: used when none is declared.
    pub const DEFAULT: Self = Self {
        bloom_intensity: 0.6,
        bloom_threshold: 1.0,
        bloom_knee: 0.5,
        exposure: 1.0,
        vignette: 0.0,
        lut_strength: 0.0,
        fxaa: 1.0,
    };
}

/// Post-process tunables resolved from the settings assets (or their
/// defaults), plus the two display-output flags the backend resolves from the
/// display's EDR capability at init. Pushed verbatim to the bloom prefilter and
/// composite fragment shaders, so the layout must stay in sync with the
/// `PostUniforms` struct in those shaders. 36 bytes.
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct PostProcessParams {
    /// Additive bloom strength. 0 disables the bloom passes entirely.
    pub bloom_intensity: f32,
    /// Luminance threshold for the bloom prefilter.
    pub bloom_threshold: f32,
    /// Quadratic soft-knee width below the threshold.
    pub bloom_knee: f32,
    /// Linear exposure multiplier applied to HDR radiance before the bloom
    /// prefilter and the composite tonemap. Resolved from `Exposure.ev` as
    /// `2^ev`, so 1.0 is neutral.
    pub exposure: f32,
    /// Vignette strength in `[0, 1]`. 0 disables the corner darkening.
    pub vignette: f32,
    /// Color-LUT blend in `[0, 1]`: `mix(scene, graded, lut_strength)` in the
    /// composite pass. 0 skips the LUT, and is what a world without a
    /// `ColorLut` resolves to.
    pub lut_strength: f32,
    /// HDR display output flag. `0.0` = SDR path (ACES tonemap + gamma 2.2 +
    /// FXAA + `ColorLut`, output `BGRA8Unorm`). `1.0` = HDR EDR path (no tonemap
    /// or gamma; the exposed HDR scene is emitted into a `RGBA16Float`
    /// swapchain attached to a Display P3 EDR layer). FXAA + `ColorLut` are
    /// skipped on the HDR path because both depend on display-referred
    /// values. Stored as a float (not a uint) so the MSL shader can branch
    /// on `> 0.5` without a cast.
    pub hdr_output: f32,
    /// Output-encoding flag inside the HDR branch. `0.0` = scRGB-linear
    /// passthrough (the OS handles the encode for the panel). `1.0` =
    /// PQ-encode (SMPTE ST 2084) in-shader so the swapchain ships
    /// PQ-encoded values directly to an HDR10 panel. Only read when
    /// `hdr_output > 0.5`; always 0.0 on the SDR path.
    pub pq_output: f32,
    /// FXAA edge-filter flag on the SDR path. `1.0` runs the composite's
    /// FXAA pass; `0.0` skips it (the `Off` anti-aliasing mode). Resolved
    /// from `AntiAliasing.mode`: on for `Fxaa` and `Taa`, off for
    /// `Off`. Always ignored on the HDR path, which never runs FXAA. Stored
    /// as a float so the shader branches on `> 0.5` without a cast.
    pub fxaa: f32,
}

impl PostProcessParams {
    /// Overwrite the authored tunables, keeping `hdr_output` / `pq_output`.
    /// Those two are a property of the display the backend negotiated with, not
    /// an authored value, so a live push from the settings menu leaves them
    /// alone.
    pub fn set_tunables(&mut self, tunables: PostProcessTunables) {
        let PostProcessTunables {
            bloom_intensity,
            bloom_threshold,
            bloom_knee,
            exposure,
            vignette,
            lut_strength,
            fxaa,
        } = tunables;
        self.bloom_intensity = bloom_intensity;
        self.bloom_threshold = bloom_threshold;
        self.bloom_knee = bloom_knee;
        self.exposure = exposure;
        self.vignette = vignette;
        self.lut_strength = lut_strength;
        self.fxaa = fxaa;
    }
}

/// Fragment constants for the composite pass: the authored post-process
/// tunables plus the scene-transition fade the backend owns. Pushed verbatim to
/// the composite fragment shader, so the layout must stay in sync with the
/// `CompositeParams` struct there. 64 bytes.
///
/// The fade is not a `PostProcessParams` field because that struct is resolved
/// from the settings assets and re-pushed whenever a settings slider
/// moves, which would reset an in-flight fade. It is also pushed to the bloom
/// prefilter, which has no use for it.
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct CompositeParams {
    /// The authored post-process tunables.
    pub post: PostProcessParams,
    /// Scene-transition fade to black in `[0, 1]`: 0 renders the composited
    /// image untouched, 1 renders it fully black. The shader scales by
    /// `1 - fade` in linear light on the HDR path and on the display-referred
    /// result on the SDR path.
    pub fade: f32,
    /// The viewport's G-buffer channel view as a `ViewMode` discriminant, or 0
    /// for the normal scene path (Lit / Unlit / Wireframe all composite the
    /// scene). Non-zero switches the fragment to visualize the matching
    /// prepass channel.
    pub view_mode: u32,
    /// Distance the depth channel view shows as black.
    pub depth_near: f32,
    /// Distance the depth channel view shows as white; distances between map
    /// on a log scale.
    pub depth_far: f32,
    /// 1.0 when the reactive mask was written this frame; the reactive view
    /// shows black otherwise.
    pub reactive_valid: f32,
    /// How far, in `[0, 1]`, the scene is shown as night vision sees it:
    /// blended toward the rods' luminance with a blue cast, before the
    /// tonemap. 0 leaves the color untouched.
    pub night_vision: f32,
    /// The luminance, in cd/m2, of a unit of the exposed color's
    /// luminance: what turns each pixel's own brightness into the rods' share
    /// of it, so a pixel too bright for the rods keeps its color.
    pub night_nits: f32,
}

/// The farthest distance the depth channel view tells apart for a camera that
/// sees without limit, in world units.
pub(super) const DEPTH_VIEW_UNLIMITED_SPAN: f32 = 10_000.0;

impl CompositeParams {
    /// The `(depth_near, depth_far)` the depth channel view maps for a camera
    /// with near plane `near`: from the near plane out to the view distance, or
    /// to `DEPTH_VIEW_UNLIMITED_SPAN` when there is none. A log scale gives
    /// every decade of distance the same share of the gray ramp.
    pub fn depth_view_range(near: f32, view_distance: Option<f32>) -> (f32, f32) {
        let near = near.max(1e-3);
        let far = view_distance.unwrap_or(DEPTH_VIEW_UNLIMITED_SPAN);
        (near, far.max(2.0 * near))
    }
}

/// Per-frame uniform for the SSAO (GTAO) passes. Carries the clamped authored
/// tunables, the view-ray scale the kernel needs to rebuild a view-space
/// position from linear depth, and the depth range the occlusion reaches.
/// Pushed verbatim to the SSAO depth copy and kernel fragment shaders, so the
/// layout must stay in sync with the `SsaoParams` struct there. 32 bytes.
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct SsaoParams {
    /// World-space hemisphere radius the horizon search covers.
    pub radius: f32,
    /// Occlusion strength multiplier applied to the integrated visibility.
    pub intensity: f32,
    /// `tan(fov_y / 2)`: the vertical view-ray half-extent at unit depth.
    pub tan_half_fov_y: f32,
    /// Viewport aspect ratio (width / height).
    pub aspect: f32,
    /// View depth past which the search radius spans less than a pixel and
    /// the pass contributes nothing. The depth copy stores depth as a fraction
    /// of it.
    pub reach: f32,
    /// Occlusion strength at view depth `d` is `saturate(d * fade_scale +
    /// fade_bias)`: full where the radius spans two pixels or more, none at
    /// `reach`.
    pub fade_scale: f32,
    /// See `fade_scale`.
    pub fade_bias: f32,
    /// Padding to a 16-byte multiple; the shader declares no field here.
    pub _pad: f32,
}

/// Per-frame uniform for the screen-space reflection (SSR) ray-march. Carries
/// the clamped authored tunables, the view-ray scale the resolve pass uses to
/// project a view-space ray point back to a screen UV, and the data the resolve
/// needs to sample the IBL prefilter cubemap as a fallback. Pushed verbatim to
/// the SSR resolve fragment shader, so the layout must stay in sync with the
/// `SsrParams` struct there. 144 bytes.
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct SsrParams {
    /// Reflection blend strength in `[0, 1]`; scales the Fresnel-weighted mix.
    pub intensity: f32,
    /// Maximum world-space distance a reflection ray marches before giving up.
    pub max_distance: f32,
    /// `tan(fov_y / 2)`: the vertical view-ray half-extent at unit depth.
    pub tan_half_fov_y: f32,
    /// Viewport aspect ratio (width / height).
    pub aspect: f32,
    /// World-space length of one ray-march step.
    pub stride: f32,
    /// View-space depth tolerance for accepting a ray/scene-depth intersection.
    pub thickness: f32,
    /// IBL prefilter cubemap mip count. 0 when no `EnvironmentMap` is bound: the
    /// resolve then skips the cube fallback and keeps the base shading instead.
    pub prefilter_mip_count: f32,
    /// The scale the HDR chain is written at; the probe and sky fallback,
    /// absolute radiance, is multiplied by it to match the scene taps.
    pub pre_exposure: f32,
    /// Camera-to-world transform (column-major, the rigid inverse of the view
    /// matrix): the orthonormal 3x3 turns a view-space reflection ray into the
    /// world-space direction the cubemap is sampled with, and the translation
    /// column (the world camera position) lets the resolve rebuild the world-space
    /// surface position a reflection probe box-projects against. Backends that only
    /// sample the cube use the 3x3 (the `r_world` direction) and ignore translation.
    pub inv_view: [[f32; 4]; 4],
    /// Rows of the rotation taking a world-space direction into the environment
    /// cubemap's baked frame, mirroring `ViewUniforms.sky_rot` so the resolve's
    /// sky fallback turns with the main pass, and reads the cube at its
    /// storage scale.
    pub sky_rot: [[f32; 4]; 3],
}

/// Per-frame uniform for the screen-space global-illumination (SSGI) passes.
/// Carries the clamped authored tunables, the view-ray scale the trace uses to
/// project a view-space ray point back to a screen UV, and the frame state the
/// accumulation reads. Pushed verbatim to every SSGI fragment that declares
/// constants, so the layout must stay in sync with the `SsgiParams` struct
/// there. 48 bytes.
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct SsgiParams {
    /// Indirect-bounce blend strength; scales the accumulated radiance the
    /// composite pass adds on top of the existing shading.
    pub intensity: f32,
    /// Maximum view-space distance a hemisphere ray travels before it misses.
    pub max_distance: f32,
    /// `tan(fov_y / 2)`: the vertical view-ray half-extent at unit depth.
    pub tan_half_fov_y: f32,
    /// Viewport aspect ratio (width / height).
    pub aspect: f32,
    /// View-space depth tolerance for accepting a ray/scene-depth intersection.
    pub thickness: f32,
    /// 1.0 when the previous frame's accumulation can be reprojected, else 0.0.
    pub history_valid: f32,
    /// Hemisphere rays traced per trace pixel this frame.
    pub rays: u32,
    /// The frame counter the ray directions are drawn from.
    pub frame: u32,
    /// Levels in the closest-depth pyramid the trace walks.
    pub levels: u32,
    /// Render-resolution divisor of the trace.
    pub gi_scale: u32,
    /// What last frame's accumulation is multiplied by to read at this
    /// frame's pre-exposure ([`crate::gfx::auto_exposure::PreExposure::history_scale`]).
    pub history_scale: f32,
    /// Tail padding to a 16-byte multiple.
    pub _pad: u32,
}

/// Per-frame uniform for the hardware ray-traced reflection pass. Like
/// [`SsrParams`] it carries the clamped intensity / distance, the view-ray
/// scale used to rebuild a view-space position from the SSR pre-pass G-buffer,
/// and the IBL prefilter mip count for the miss fallback. Unlike SSR it traces
/// a world-space ray against an acceleration structure, so it also carries the
/// camera-to-world transform (to lift the view-space hit point + normal into
/// world space), the world camera position (the ray origin), and the sun
/// direction + color the hit-shading uses. Pushed verbatim to the RT kernel,
/// so the layout must stay in sync with the `RtParams` struct there. 192 bytes,
/// 16-byte aligned (every `vec3` is padded to a `float4`).
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct RtParams {
    /// Reflection blend strength in `[0, 1]`; scales the Fresnel-weighted mix.
    pub intensity: f32,
    /// Maximum world-space distance a reflection ray travels before it misses.
    pub max_distance: f32,
    /// `tan(fov_y / 2)`: the vertical view-ray half-extent at unit depth.
    pub tan_half_fov_y: f32,
    /// Viewport aspect ratio (width / height).
    pub aspect: f32,
    /// IBL prefilter cubemap mip count. 0 when no `EnvironmentMap` is bound: the
    /// kernel then keeps the base shading for missed rays instead of a cube tap.
    pub prefilter_mip_count: f32,
    /// 1.0 when a reflected hit casts a sun-shadow ray, 0.0 when it is lit
    /// unshadowed.
    pub sun_shadows: f32,
    /// Per-axis divisor of the reduced glass reflection pre-pass; 1.0 when
    /// glass traces its reflection in place.
    pub trace_divisor: f32,
    /// The scale the HDR chain is written at; the reflection pass shades in
    /// absolute radiance and multiplies what it writes by it.
    pub pre_exposure: f32,
    /// World-space camera position (`xyz`); the reflection ray origin. `w` unused.
    pub cam_pos: [f32; 4],
    /// World-space unit direction *toward* the sun (`xyz`); the hit-shading N·L
    /// term uses it. `w` unused.
    pub sun_dir: [f32; 4],
    /// Sun radiance (`xyz`); the direct term at a reflected hit. `w` unused.
    pub sun_color: [f32; 4],
    /// Camera-to-world transform (column-major, the rigid inverse of the view
    /// matrix). Lifts the view-space reconstructed position + normal into the
    /// world space the acceleration structure is built in.
    pub inv_view: [[f32; 4]; 4],
    /// Rows of the rotation taking a world-space direction into the environment
    /// cubemap's baked frame, mirroring `ViewUniforms.sky_rot` so a missed ray
    /// lands on the same sky the main pass shows. One `float4` per row.
    pub sky_rot: [[f32; 4]; 3],
}

/// One entry of the ray-tracing geometry table, indexed by the intersector's
/// `instance_id` (the instance's position in the TLAS instance buffer, one
/// entry per instance, in instance order). Lets the RT kernel find the hit
/// triangle's indices in the shared index buffer, transform its local-space
/// vertices into world space for the geometric normal, and pick a base albedo to
/// shade the hit with. `#[repr(C)]`, 128 bytes: the layout must stay in sync
/// with the `RtGeomEntry` struct in the shared RT records (`rt_types.hlsl`), where
/// `tint` and `emissive` are `packed_float3` so the field offsets match; a plain
/// `float3` there would stride the buffer differently and fault the trace. The
/// `_pad` tail rounds the struct to 128 bytes so its array stride is a multiple
/// of the 16-byte GPU alignment of the `float4x4` `model` (MSL/HLSL/GLSL round a
/// matrix-bearing struct up to a 16-byte multiple, so a 116-byte Rust struct
/// would stride 116 while the shader strides 128 and faults the trace). The
/// `size_eq` / `offsets` unit tests below lock the Rust side. The model matrix is
/// stored here (rather than read from the intersector's instance transform) so
/// the kernel's normal math is self-contained.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct RtGeomEntry {
    /// Element offset of this object's first index in the shared index buffer
    /// (`DrawObject::index_offset`).
    pub index_offset: u32,
    /// Value added to each fetched index before the vertex lookup
    /// (`DrawObject::base_vertex`).
    pub base_vertex: u32,
    /// Index into the bindless albedo texture pool (`DrawObject::texture_slot`,
    /// clamped to the albedo count). The textured RT-hit shader samples
    /// `tex_pool[albedo_index]` at the hit UV; the flat fallback ignores it.
    pub albedo_index: u32,
    /// Index into the bindless texture pool for this object's normal map
    /// (`albedo_count + normal_map_slot`). The textured RT-hit shader samples
    /// `tex_pool[normal_index]` to perturb the hit normal via the interpolated
    /// tangent frame; objects with no normal map resolve to the 1x1 flat-normal
    /// fallback at normal slot 0, so the sample is always safe (no perturbation).
    pub normal_index: u32,
    /// Base albedo for hit shading (the material tint), `[r, g, b]`. The
    /// textured shader multiplies the sampled albedo by this; the flat fallback
    /// uses it directly.
    pub tint: [f32; 3],
    /// Surface roughness, used to pick the IBL prefilter mip at the hit.
    pub roughness: f32,
    /// Metallic factor [0, 1] for the hit PBR response: metals drop the diffuse
    /// term and tint the reflected environment specular by their albedo.
    pub metallic: f32,
    /// Self-emission added to the hit color, so glowing surfaces light up in
    /// reflections regardless of incident lighting. Fills the three words that
    /// pad `metallic` out to a 16-byte boundary (the MSL side reads it as a
    /// `packed_float3` at the same offset; the layout test pins the exact size).
    pub emissive: [f32; 3],
    /// Column-major object-to-world transform (`DrawObject::model`), used to
    /// lift the hit triangle's local-space vertices into world space.
    pub model: [[f32; 4]; 4],
    /// Bindless albedo-pool index for the emissive map (0 = none). The textured
    /// RT-hit shader multiplies the self-emission by this map sample, so glowing
    /// textured surfaces (e.g. the bistro string lights) reflect in color rather
    /// than the scalar `emissive`; the flat fallback ignores it. Mirrors
    /// `MaterialUniforms::emissive_map_index` / `GpuObjectData::emissive_map_index`.
    pub emissive_map_index: u32,
    /// Pads the struct to 128 bytes (a multiple of the 16-byte GPU alignment of
    /// the `float4x4` `model`) so the Rust array stride matches the shader stride.
    pub _pad: [u32; 3],
}

/// Per-frame uniform consumed by the volumetric-fog ray-march. Carries the
/// clamped authored tunables plus the view inputs the shader needs to
/// reconstruct world positions from depth and integrate sun-aligned
/// scattering. Pushed verbatim to the fog fragment shader, so the layout must
/// stay in sync with `FogParams` in `metal/shaders/fog.metal`. 176 bytes.
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct FogParams {
    /// Inverse view-projection: reconstructs world position from depth.
    pub inv_vp: [[f32; 4]; 4],
    /// Linear-space fog tint (RGB); alpha unused.
    pub color: [f32; 4],
    /// World-space camera position. The marcher starts here and walks the
    /// view ray to the scene point. `_pad` keeps `cam_pos` 16-byte aligned
    /// inside MSL.
    pub cam_pos: [f32; 3],
    /// Padding so the field layout matches the shader-side struct.
    pub _pad0: f32,
    /// First directional light's world-space direction (toward the light),
    /// used for the Henyey-Greenstein phase and the per-step sun radiance.
    pub sun_dir: [f32; 3],
    /// Padding so the field layout matches the shader-side struct.
    pub _pad1: f32,
    /// First directional light's color pre-multiplied with its intensity.
    /// Drives the per-step sun in-scatter alongside the phase function.
    pub sun_color: [f32; 3],
    /// Padding so the field layout matches the shader-side struct.
    pub _pad2: f32,
    /// Base density at `height_reference` (per world unit).
    pub density: f32,
    /// Exponential height-falloff rate; 0 = homogeneous medium.
    pub height_falloff: f32,
    /// World-space Y at which density equals `density`.
    pub height_reference: f32,
    /// Ray-march cap in world units.
    pub max_distance: f32,
    /// Henyey-Greenstein anisotropy in `(-0.95, 0.95)`.
    pub phase_g: f32,
    /// Ambient (sky-side) scattering term added each step.
    pub ambient: f32,
    /// Width / height of the HDR resolve target in pixels: drives the
    /// screen→NDC conversion in the fragment shader.
    pub viewport: [f32; 2],
    /// Pre-computed reciprocal of `max_distance`; the shader uses it to skip
    /// a per-step `1 / max_distance` divide.
    pub inv_max_distance: f32,
    /// The scale the HDR chain is written at; the fog's in-scattered light is
    /// multiplied by it as it blends over the scene.
    pub pre_exposure: f32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad3: [f32; 2],
}

/// Per-frame uniform consumed by the volumetric-fog froxel compute kernel and
/// the matching fragment-shader sample path. Carries the view matrix (so the
/// compute kernel + the fragment sampler can map between world-space froxel
/// positions and the volume's Z axis) and the discrete volume dimensions. Used
/// by the froxel-volume fog path; only worlds that declare a `VolumetricFog`
/// bind it.
///
/// Bound at the fog fragment shader + the froxel kernel. Layout must stay in
/// sync with `FogFroxelParams` in `shaders/fog.hlsl`, the single source both
/// halves compile from on every backend. 96 bytes.
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct FogFroxelParams {
    /// View matrix (world → view), so the compute kernel can pick a CSM
    /// cascade for each froxel and the fragment sampler can map a scene
    /// depth into the volume's Z slice index.
    pub view: [[f32; 4]; 4],
    /// Number of froxels along screen-x / screen-y / view-z.
    pub froxel_dims: [u32; 3],
    /// Explicit 4-byte pad so the MSL `uint3 froxel_dims` (which the
    /// MSL alignment rules promote to a 16-byte slot) and the Rust
    /// `[u32; 3]` (which `#[repr(C)]` packs to 12 bytes) describe the
    /// same on-wire layout. Without this padding `z_near` lands at
    /// offset 76 in Rust but offset 80 in MSL, swapping `z_near` /
    /// `z_far` and inverting the volume's Z mapping.
    pub _pad_align: u32,
    /// Camera near-plane in view units. The fragment sampler maps a scene
    /// view-z to a normalized volume w via `(z - z_near) / (z_far - z_near)`
    /// for linear Z.
    pub z_near: f32,
    /// Camera far-plane mirror: `FogSettings.max_distance`. The volume
    /// covers `[z_near, max_distance]` along view-space depth. Beyond
    /// `max_distance` the sampler clamps to the last slice (already
    /// fully-integrated).
    pub z_far: f32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: [f32; 2],
}

/// Per-frame uniform consumed by the particle compute + render kernels. Carries
/// the resolved emitter tunables (position, direction, gravity, ...) plus the
/// dynamic per-frame inputs the compute kernel needs to age + integrate +
/// respawn the pool. Pushed at compute buffer(2) and vertex buffer(1) of the
/// Metal particle passes, so the layout must stay in sync with
/// `ParticleParams` in `shaders/particle_types.hlsl`. 128 bytes.
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
pub struct ParticleParams {
    /// World-space spawn origin. `packed_float3` in MSL: alignment 4, with
    /// `spread_cos` packed into the trailing float slot of the float4.
    pub position: [f32; 3],
    /// Cosine of the emission cone's half-angle. `1.0` = straight jet,
    /// `-1.0` = full sphere.
    pub spread_cos: f32,
    /// Unit-length mean emission direction.
    pub direction: [f32; 3],
    /// Lower bound of the initial speed range, in units per second.
    pub speed_min: f32,
    /// Constant per-frame acceleration applied to each particle.
    pub gravity: [f32; 3],
    /// Upper bound of the initial speed range, in units per second.
    pub speed_max: f32,
    /// Linear-space RGBA at `age = 0`.
    pub color_start: [f32; 4],
    /// Linear-space RGBA at `age = lifetime`.
    pub color_end: [f32; 4],
    /// Lower bound of the particle lifetime range, in seconds.
    pub lifetime_min: f32,
    /// Upper bound of the particle lifetime range, in seconds.
    pub lifetime_max: f32,
    /// Quad size at `age = 0`.
    pub size_start: f32,
    /// Quad size at `age = lifetime`.
    pub size_end: f32,
    /// Frame delta time (seconds). Drives the age + integration step.
    pub dt: f32,
    /// Number of fresh particles the compute kernel spawns this frame.
    pub spawn_count: u32,
    /// Pool slot of this frame's first spawn. Spawn `k` lands in slot
    /// `(spawn_first + k) % max_particles`.
    pub spawn_first: u32,
    /// Per-emitter, per-frame seed (see `spawn_seed`) mixed with each spawn's
    /// ordinal by the kernel's cheap RNG.
    pub random_seed: u32,
    /// Pool size in slots; the kernel returns early past it.
    pub max_particles: u32,
    /// Padding to the 16-byte multiple a constant buffer is sized in.
    pub _pad: [u32; 3],
}
