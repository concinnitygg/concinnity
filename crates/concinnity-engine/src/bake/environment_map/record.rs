//! What the renderer reads of an environment map beside its baked lighting,
//! as its resource record carries it.

use alloc::vec::Vec;

use crate::codec::{FrameError, decode_exact};
use crate::components::EnvironmentMap;

/// The runtime fields an environment map's resource record carries beside its
/// baked payload. Only what the renderer reads: the authored source and bake
/// settings stay with the cook.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentMapRecord {
    /// Draw the map as the background (see [`EnvironmentMap::background`]).
    pub background: bool,
}

impl Default for EnvironmentMapRecord {
    fn default() -> Self {
        Self::from(&EnvironmentMap::default())
    }
}

impl From<&EnvironmentMap> for EnvironmentMapRecord {
    fn from(map: &EnvironmentMap) -> Self {
        Self {
            background: map.background,
        }
    }
}

impl EnvironmentMapRecord {
    /// The record's bytes, as a resource record's `data_bytes` carries them.
    pub fn to_bytes(&self) -> Vec<u8> {
        postcard::to_allocvec(self).unwrap_or_default()
    }

    /// Decode a resource record's `data_bytes`. A record with no runtime
    /// fields holds the defaults.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, FrameError> {
        match bytes {
            [] => Ok(Self::default()),
            _ => decode_exact(bytes),
        }
    }
}
