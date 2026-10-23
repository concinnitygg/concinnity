//! What the water's wake reads: the simulation's block, refreshed every frame
//! with the window, the time step and the stirs of the bodies in the water,
//! and the lane a planet sea's draw samples the result through. Mirrors
//! `water_wake.hlsl` and `WaterWake` in `water.hlsl`.

/// Bodies' pieces one frame's wake stirs in at most. Mirrors
/// `WATER_WAKE_MAX_STIRS` in `shaders/water_wake.hlsl`.
pub const MAX_WAKE_STIRS: usize = 64;

/// [`WaterWakeParams::grid`] flag: last frame's field holds a window to
/// carry from.
pub const WAKE_LAST_VALID: u32 = 1;

/// [`WaterWakeParams::grid`] flag: the simulated frame moved since last
/// frame, so the field is read back through [`WaterWakeParams::carry`].
pub const WAKE_CARRIED: u32 = 2;

/// [`WaterWakeParams::grid`] flag: this frame writes the second half of the
/// state and reads the first; without it, the other way round.
pub const WAKE_WRITE_SECOND: u32 = 4;

/// One piece's stir at two moments: this frame's and last frame's, each
/// `[x, z, radius, volume]` in this frame's coordinates. The change between
/// them is what the piece pushes into the water. 32 bytes.
#[derive(Copy, Clone, Default, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct WaterWakeStirGpu {
    /// This frame's `[x, z, radius, volume]`.
    pub now: [f32; 4],
    /// Last frame's `[x, z, radius, volume]`.
    pub prev: [f32; 4],
}

/// The wake simulation's block for one frame. Matches `WaterWakeParams` in
/// `shaders/water_wake.hlsl`. 2160 bytes.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct WaterWakeParams {
    /// The cell at the window's corner this frame (`xy`) and last frame
    /// (`zw`), the latter in last frame's coordinates.
    pub window: [i32; 4],
    /// The rows that take a cell center's `(x, z, 1)` in this frame to its
    /// `x` and `z` in last frame's: identity unless the frame moved.
    pub carry: [[f32; 4]; 2],
    /// `[cells per side, log2 of it, stirs in use, flags]`, the flags
    /// [`WAKE_LAST_VALID`], [`WAKE_CARRIED`] and [`WAKE_WRITE_SECOND`].
    pub grid: [u32; 4],
    /// `[cell size, time step, gravity, sponge width]`, in meters and
    /// seconds.
    pub cell: [f32; 4],
    /// `[damping per second, damping per second per square radian per
    /// meter, sponge damping per second, _]`.
    pub damping: [f32; 4],
    /// `[share of foam kept, foam per meter stirred, steepness that starts
    /// to foam, foam per second per unit of steepness past it]`.
    pub foam: [f32; 4],
    /// The pieces stirring the water; the first `grid.z` are used.
    pub stirs: [WaterWakeStirGpu; MAX_WAKE_STIRS],
}

impl Default for WaterWakeParams {
    fn default() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

/// Where a planet sea's draw finds the wake. Matches `WaterWake` in
/// `shaders/water.hlsl`. 32 bytes; all zero turns the wake off.
#[derive(Copy, Clone, Default, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct WaterWakeGpu {
    /// `[1 / window width, cell size, center x, center z]`: a point's
    /// texture coordinate is its `xz` over the window's width, wrapped.
    pub lookup: [f32; 4],
    /// `[half width of the clear area, 1 / sponge width, max height per
    /// meter of depth, on]`: the field fades out across the sponge from the
    /// clear area to the window's edge; `on` is 1 while the wake runs.
    pub fade: [f32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    #[test]
    fn water_wake_stir_layout_matches_shader() {
        assert_eq!(size_of::<WaterWakeStirGpu>(), 32);
        assert_eq!(offset_of!(WaterWakeStirGpu, now), 0);
        assert_eq!(offset_of!(WaterWakeStirGpu, prev), 16);
    }

    #[test]
    fn water_wake_params_layout_matches_shader() {
        assert_eq!(size_of::<WaterWakeParams>(), 112 + 32 * MAX_WAKE_STIRS);
        assert_eq!(offset_of!(WaterWakeParams, window), 0);
        assert_eq!(offset_of!(WaterWakeParams, carry), 16);
        assert_eq!(offset_of!(WaterWakeParams, grid), 48);
        assert_eq!(offset_of!(WaterWakeParams, cell), 64);
        assert_eq!(offset_of!(WaterWakeParams, damping), 80);
        assert_eq!(offset_of!(WaterWakeParams, foam), 96);
        assert_eq!(offset_of!(WaterWakeParams, stirs), 112);
        assert!(size_of::<WaterWakeParams>() <= 4096);
    }

    #[test]
    fn water_wake_layout_matches_shader() {
        assert_eq!(size_of::<WaterWakeGpu>(), 32);
        assert_eq!(offset_of!(WaterWakeGpu, lookup), 0);
        assert_eq!(offset_of!(WaterWakeGpu, fade), 16);
    }
}
