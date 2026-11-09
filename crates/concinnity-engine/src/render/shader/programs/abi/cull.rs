//! The draw cull's bindings: the decision kernels, and the scan and scatter
//! that pack the draws they keep.

use super::group::ThreadGroup;
use crate::gfx::render_types::CULL_GROUP_RECORDS;

/// The decision kernels' group.
pub const CULL_GROUP: ThreadGroup = ThreadGroup::new(CULL_GROUP_RECORDS, 1, 1);
/// The scan's group.
pub const SCAN_GROUP: ThreadGroup = ThreadGroup::new(64, 1, 1);
/// The scatter's group.
pub const SCATTER_GROUP: ThreadGroup = ThreadGroup::new(CULL_GROUP_RECORDS, 1, 1);

/// The DirectX root signatures: one the decision kernels share, and one the
/// scan and the scatter share.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Srv, Uav};

    /// The cull's parameters, as root constants.
    pub const CULL: Register<Cbv> = Register::new("cull", 0);
    /// The frame's records.
    pub const OBJECTS: Register<Srv> = Register::new("objects", 0);
    /// Each record's draw arguments.
    pub const DRAW_ARGS: Register<Srv> = Register::new("draw_args", 1);
    /// The Hi-Z pyramid, every mip.
    pub const HIZ: Register<Srv> = Register::new("hiz_tex", 2);
    /// The view's routes and bucket regions.
    pub const VARIANTS: Register<Srv> = Register::new("variants", 3);
    /// The survivors each group counts.
    pub const GROUP_COUNTS: Register<Uav> = Register::new("group_counts", 0);
    /// Each record's decision, which the late pass reads back.
    pub const CULL_STATUS: Register<Uav> = Register::new("cull_status", 1);

    /// The scan and the scatter, which share one root signature.
    pub mod compact {
        use crate::render::shader::programs::abi::register::{Cbv, Register, Srv, Uav};

        /// The packing's parameters, as root constants.
        pub const PARAMS: Register<Cbv> = Register::new("params", 0);
        /// The view's routes and bucket regions.
        pub const VARIANTS: Register<Srv> = Register::new("variants", 0);
        /// Each record's draw arguments.
        pub const DRAW_ARGS: Register<Srv> = Register::new("draw_args", 1);
        /// The survivors each decision group counted.
        pub const GROUP_COUNTS: Register<Uav> = Register::new("group_counts", 0);
        /// Each record's decision.
        pub const CULL_STATUS: Register<Uav> = Register::new("cull_status", 1);
        /// The scan's per-stream list lengths, in the slot the scatter writes
        /// its commands through.
        pub const STREAM_COUNTS: Register<Uav> = Register::new("stream_counts", 2);
        /// The scatter's packed draw commands.
        pub const COMMANDS: Register<Uav> = Register::new("commands", 2);
    }
}

/// The Metal slots, one encoder binding them for the decision and the scan
/// that follows it.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Slot, Texture};

    /// The frame's records.
    pub const OBJECTS: Slot<Buffer> = Slot::new("objects", 0);
    /// Each record's draw arguments.
    pub const DRAW_ARGS: Slot<Buffer> = Slot::new("draw_args", 1);
    /// The cull's parameters.
    pub const CULL: Slot<Buffer> = Slot::new("cull", 2);
    /// Each record's decision.
    pub const CULL_STATUS: Slot<Buffer> = Slot::new("cull_status", 5);
    /// The view's routes and bucket regions.
    pub const VARIANTS: Slot<Buffer> = Slot::new("variants", 8);
    /// The survivors each group counts.
    pub const GROUP_COUNTS: Slot<Buffer> = Slot::new("group_counts", 9);
    /// The scan's per-stream list lengths.
    pub const STREAM_COUNTS: Slot<Buffer> = Slot::new("stream_counts", 10);
    /// The scan's parameters.
    pub const SCAN_PARAMS: Slot<Buffer> = Slot::new("params", 11);
    /// The Hi-Z pyramid.
    pub const HIZ: Slot<Texture> = Slot::new("hiz_tex", 0);
}
