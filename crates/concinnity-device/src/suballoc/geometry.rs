//! Placement of one mesh's vertex and index ranges in a pair of range
//! allocators, all or nothing. Every backend's streamed-mesh and chunk upload
//! places through here, so pool exhaustion is classified the same everywhere.

use concinnity_core::render::error::{RenderError, RenderResult};

use super::range_alloc::RangeAllocator;

// Place `v_len` vertex bytes and `i_len` index bytes, returning their offsets.
// A full pool is `OutOfDeviceMemory`. When only the index pool is full the
// vertex range goes back at retire frame 0, since nothing wrote or drew it.
// `what` names the upload in the error message.
pub(crate) fn place_mesh(
    vtx: &mut RangeAllocator,
    idx: &mut RangeAllocator,
    v_len: usize,
    i_len: usize,
    what: impl Fn() -> String,
) -> RenderResult<(usize, usize)> {
    let v_off = vtx.alloc(v_len as u64).ok_or_else(|| {
        RenderError::OutOfDeviceMemory(format!(
            "{}: no free vertex space for {v_len} bytes",
            what()
        ))
    })?;
    let Some(i_off) = idx.alloc(i_len as u64) else {
        vtx.free(v_off, v_len as u64, 0);
        return Err(RenderError::OutOfDeviceMemory(format!(
            "{}: no free index space for {i_len} bytes",
            what()
        )));
    };
    Ok((v_off as usize, i_off as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(size: u64) -> RangeAllocator {
        let mut a = RangeAllocator::new();
        a.free(0, size, 0);
        a.reclaim(0);
        a
    }

    fn what() -> String {
        "upload_mesh: draw 3".to_string()
    }

    #[test]
    fn places_both_ranges() {
        let (mut vtx, mut idx) = (seeded(64), seeded(32));
        assert_eq!(place_mesh(&mut vtx, &mut idx, 16, 8, what), Ok((0, 0)));
        assert_eq!(place_mesh(&mut vtx, &mut idx, 16, 8, what), Ok((16, 8)));
    }

    #[test]
    fn a_full_vertex_pool_is_out_of_device_memory() {
        let (mut vtx, mut idx) = (seeded(8), seeded(32));
        let err = place_mesh(&mut vtx, &mut idx, 16, 8, what).unwrap_err();
        assert!(matches!(err, RenderError::OutOfDeviceMemory(_)), "{err}");
        assert!(err.to_string().contains("upload_mesh: draw 3"), "{err}");
        assert_eq!(idx.free_bytes(), 32);
    }

    #[test]
    fn a_full_index_pool_hands_the_vertex_range_back() {
        let (mut vtx, mut idx) = (seeded(64), seeded(4));
        let err = place_mesh(&mut vtx, &mut idx, 16, 8, what).unwrap_err();
        assert!(matches!(err, RenderError::OutOfDeviceMemory(_)), "{err}");
        vtx.reclaim(0);
        assert_eq!(vtx.free_bytes(), 64);
    }
}
