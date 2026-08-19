// src/uniforms/probe.rs
//
// The reflection-probe set the forward, SSR and ray-traced resolves all read.
// Matches `ProbeUniforms` / `ProbeSet` in `shaders/probe_types.slang`, whose
// `MAX_PROBES` is baked in from the constant below.

// Maximum reflection probes a frame can bind. The shader's `MAX_PROBES` define
// is injected from this value, so the two cannot drift.
pub const MAX_PROBES: usize = 8;

// Every automatically seeded probe has to fit in the bound set.
const _: () = assert!(crate::reflection_probe::AUTO_SEED_BUDGET <= MAX_PROBES);

// One reflection probe's parallax box. The specular IBL term box-projects the
// reflection vector against [box_min, box_max] (the probe's influence volume)
// and re-anchors the cube sample at the box hit relative to `probe_pos` (the
// capture point), so a static captured cube tracks a moving first-person
// camera. Three float4s keep every field 16-byte aligned. `box_min.w` is the
// enabled flag: 0 disables parallax (and signals no baked probe), so the shader
// samples the raw reflection vector.
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct ProbeUniforms {
    // xyz = influence-box min; w = enabled (1.0 = parallax on, 0.0 = off).
    pub box_min: [f32; 4],
    // xyz = influence-box max; w unused.
    pub box_max: [f32; 4],
    // xyz = probe capture position; w unused.
    pub probe_pos: [f32; 4],
}

impl ProbeUniforms {
    // The "no probe" value: parallax disabled, so the shader samples the raw
    // reflection vector (which, with the probe cube slot aliasing the sky until
    // a bake, reproduces the pre-probe reflection exactly).
    pub const DISABLED: ProbeUniforms = ProbeUniforms {
        box_min: [0.0; 4],
        box_max: [0.0; 4],
        probe_pos: [0.0; 4],
    };
}

// The full set of reflection probes. `count` is how many of `probes` are live;
// the fragment shader blends every probe whose influence box covers the surface
// (a partition-of-unity weight by signed box distance), falling back to the
// nearest when the surface is outside all boxes, and samples those slices of
// the probe cube array. Slices beyond `count` hold the sky fallback cube and a
// `DISABLED` box.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct ProbeSet {
    pub count: u32,
    pub _pad: [u32; 3],
    pub probes: [ProbeUniforms; MAX_PROBES],
}

impl ProbeSet {
    pub const EMPTY: ProbeSet = ProbeSet {
        count: 0,
        _pad: [0; 3],
        probes: [ProbeUniforms::DISABLED; MAX_PROBES],
    };
}
