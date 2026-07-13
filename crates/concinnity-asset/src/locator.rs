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
