//! Readings of the machine and this process: physical and resident memory,
//! core count and CPU time. Each is best-effort and `None` where the platform
//! cannot answer.

pub mod cpu;
pub mod memory;
