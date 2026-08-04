// Where an asset's compiled payload lives in the data blob.
//
// Carried on a blob-backed asset as a `#[serde(skip)]` field (filled in at load
// time, not authored) and in the blob defs table's `BlobAssetDef`.

// Points to an asset's compiled binary payload within the data blob files.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PayloadLocator {
    // Index into the blob file list (0 = data/0, 1 = data/1, ...).
    pub blob_index: u32,
    // Byte offset into the payload section of the target blob.
    pub offset: u64,
    // Byte length of the payload.
    pub len: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_locator_round_trips_through_the_blob_defs_table() {
        // The defs table is postcard, so the offset and length have to survive a
        // format that carries no field names and varint-encodes every number.
        let loc = PayloadLocator {
            blob_index: 2,
            offset: 4_294_967_296,
            len: 1_048_576,
        };
        let bytes = postcard::to_allocvec(&loc).unwrap();
        assert_eq!(postcard::from_bytes::<PayloadLocator>(&bytes).unwrap(), loc);
    }

    #[test]
    fn a_locator_parses_from_its_json_form() {
        let loc: PayloadLocator =
            serde_json::from_str(r#"{"blob_index":1,"offset":64,"len":256}"#).unwrap();
        assert_eq!(loc.blob_index, 1);
        assert_eq!(loc.offset, 64);
        assert_eq!(loc.len, 256);
        assert_ne!(
            loc,
            PayloadLocator {
                blob_index: 0,
                offset: 64,
                len: 256,
            }
        );
        assert_eq!(loc.clone(), loc);
        assert!(alloc::format!("{loc:?}").contains("PayloadLocator"));
    }
}
