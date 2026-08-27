//! Level of detail: baking the alternate index lists a mesh is drawn from, and
//! picking between them per draw.
//!
//! [`decimate_by_qem`] collapses LOD0's index list down to a triangle budget
//! while leaving its vertex set untouched, so a level is an extra index list
//! rather than a second mesh. [`pick_lod_slice`] is the draw-time counterpart
//! every backend runs per object, per frame.

mod decimate;
mod select;

pub use decimate::{decimate_by_qem, default_distance_for_level, target_tri_count_for_level};
pub use select::{
    bounds_finite, camera_distance, pick_lod_level, pick_lod_slice, skinned_camera_distance,
};
