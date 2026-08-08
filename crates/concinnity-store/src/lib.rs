// concinnity-store
//
// The project's state tree on disk: where it is anchored (`paths`), how a source
// asset is found inside it (`source`), and how the compiled blob is read out of
// it (`blob`).
//
// This is the engine's only home for filesystem knowledge below the engine and
// cook crates. concinnity-blob owns the pure bytes <-> metadata transforms and
// performs no I/O; concinnity-core owns the runtime blob types and the
// `PayloadStore` seam and knows neither paths nor files. The reads that join the
// two live here.

pub mod blob;
pub mod paths;
pub mod source;
