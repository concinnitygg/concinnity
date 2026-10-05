//! What the renderer reads of an environment map beside its baked lighting.

use alloc::vec::Vec;

use super::EnvironmentMapTable;
use crate::blob::{FrameError, decode_exact};
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
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FrameError> {
        match bytes {
            [] => Ok(Self::default()),
            _ => decode_exact(bytes),
        }
    }
}

impl EnvironmentMapTable {
    /// The runtime fields of the map at `handle`; the defaults when there is
    /// no such map.
    pub fn record(&self, handle: usize) -> Result<EnvironmentMapRecord, FrameError> {
        self.0
            .get(handle)
            .map_or(Ok(EnvironmentMapRecord::default()), |e| {
                EnvironmentMapRecord::from_bytes(&e.data_bytes)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::ResourceEntry;
    use alloc::vec;

    fn table(data_bytes: Vec<u8>) -> EnvironmentMapTable {
        EnvironmentMapTable(vec![ResourceEntry {
            data_bytes,
            ..Default::default()
        }])
    }

    #[test]
    fn a_record_round_trips_through_its_bytes() {
        for background in [true, false] {
            let record = EnvironmentMapRecord { background };
            assert_eq!(
                EnvironmentMapRecord::from_bytes(&record.to_bytes()),
                Ok(record)
            );
        }
    }

    #[test]
    fn the_default_record_draws_the_background_as_the_asset_does() {
        assert!(EnvironmentMapRecord::default().background);
        assert!(EnvironmentMap::default().background);
    }

    #[test]
    fn a_record_without_runtime_fields_holds_the_defaults() {
        assert_eq!(
            table(Vec::new()).record(0),
            Ok(EnvironmentMapRecord::default())
        );
        assert_eq!(
            EnvironmentMapTable::default().record(0),
            Ok(EnvironmentMapRecord::default())
        );
    }

    // Bytes that are not a record are a broken build, not a default.
    #[test]
    fn undecodable_bytes_are_an_error() {
        assert!(table(vec![7]).record(0).is_err());
        let mut long = EnvironmentMapRecord::default().to_bytes();
        long.push(0);
        assert_eq!(table(long).record(0), Err(FrameError::Trailing(1)));
    }
}
