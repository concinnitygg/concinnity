//! When the tables that depend on the air alone need building again.
//!
//! The transmittance and multiple-scattering tables depend on the medium and
//! on the ground radius they are built over. The medium is fixed once the
//! atmosphere is resolved, but with an apparent radius the ground moves as
//! the camera climbs through the fade, so the tables are rebuilt whenever the
//! frame's ground radius is not the one they hold.

use core::sync::atomic::{AtomicU32, Ordering};

// The bits of a NaN, which no ground radius equals.
const UNBUILT: u32 = 0x7fc0_0000;

/// The ground radius the air-only tables were last built over.
#[derive(Debug)]
pub struct AirTables {
    built_for: AtomicU32,
}

impl Default for AirTables {
    fn default() -> Self {
        Self {
            built_for: AtomicU32::new(UNBUILT),
        }
    }
}

impl AirTables {
    /// Whether this frame must build the tables for ground `radius` (km),
    /// noting that it does.
    pub fn take_rebuild(&self, radius: f32) -> bool {
        self.built_for.swap(radius.to_bits(), Ordering::Relaxed) != radius.to_bits()
    }

    /// Forget the tables, so the next frame builds them: after their
    /// kernels were rebuilt.
    pub fn invalidate(&self) {
        self.built_for.store(UNBUILT, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tables_build_once_per_ground() {
        let t = AirTables::default();
        assert!(t.take_rebuild(6360.0));
        assert!(!t.take_rebuild(6360.0));
        assert!(t.take_rebuild(3000.0));
        assert!(!t.take_rebuild(3000.0));
        t.invalidate();
        assert!(t.take_rebuild(3000.0));
    }
}
