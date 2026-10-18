//! The decisions a world's graphics load makes before the backend sees it:
//! which texture slots decode, which blobs free, which shader buckets grow a
//! far band, where reflection probes go, and the one-shot effect content each
//! world declares.

pub mod blob_release;
pub mod draw_geometry;
pub mod far_band_shaders;
pub mod prop_draws;
pub mod scene_deferral;
pub mod stream_seed;
pub mod tables;
pub mod texture_payloads;
pub mod world_fx;
pub mod world_scan;
