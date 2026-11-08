//! The fullscreen post passes' bindings, which a post program's declaration
//! counts place by position rather than by name.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv};

    /// The pass's constants, as root constants.
    pub const PARAMS: Register<Cbv> = Register::new("params", 0);
    /// A probe-reading pass's live probe count.
    pub const PROBE_SET: Register<Cbv> = Register::new("probe_set", 1);
    /// A probe-reading pass's cluster grid parameters.
    pub const CLUSTER: Register<Cbv> = Register::new("cluster", 2);

    /// Declared source `index`.
    pub const fn source(index: u32) -> Register<Srv> {
        Register::new("source", index)
    }

    /// Declared source `index`'s own sampler.
    pub const fn source_sampler(index: u32) -> Register<Sampler> {
        Register::new("source_sampler", index)
    }

    /// The probe cube array, the next register after a pass's `sources`.
    pub const fn probe_cubes(sources: u32) -> Register<Srv> {
        Register::new("probe_cubes", sources)
    }

    /// The probe cube array's sampler, after the sources' own.
    pub const fn probe_cube_sampler(sources: u32) -> Register<Sampler> {
        Register::new("probe_cube_sampler", sources)
    }

    /// The probe records, after the cube array.
    pub const fn probe_records(sources: u32) -> Register<Srv> {
        Register::new("probe_records", sources + 1)
    }

    /// The per-cluster lists, after the probe records.
    pub const fn cluster_list(sources: u32) -> Register<Srv> {
        Register::new("cluster_list", sources + 2)
    }
}
