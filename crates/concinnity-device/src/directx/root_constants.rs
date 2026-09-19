//! Typed root-constant pushes. The DWORD count comes from the value itself, so
//! a push cannot pair a pointer with a hand-written count that disagrees with
//! it, and a root signature declares its constants from the same type.

use bytemuck::NoUninit;
use windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList;

// 32-bit values a root-constant block of type `T` occupies.
pub(in crate::directx) const fn root_dwords<T: NoUninit>() -> u32 {
    const {
        assert!(
            size_of::<T>().is_multiple_of(4),
            "a root-constant block must be whole 32-bit values"
        )
    };
    (size_of::<T>() / 4) as u32
}

// 32-bit values a runtime-assembled root-constant block occupies.
fn byte_dwords(bytes: &[u8]) -> u32 {
    assert!(
        bytes.len().is_multiple_of(4),
        "a root-constant block must be whole 32-bit values, got {} bytes",
        bytes.len()
    );
    (bytes.len() / 4) as u32
}

pub(in crate::directx) trait RootConstants {
    // SAFETY: the caller holds the list in the recording state, with a graphics
    // root signature bound that declares `root_dwords::<T>()` constants at `param`.
    unsafe fn set_graphics_root_constants<T: NoUninit>(&self, param: u32, value: &T);

    // SAFETY: as `set_graphics_root_constants`, for `bytes.len() / 4` constants.
    unsafe fn set_graphics_root_constant_bytes(&self, param: u32, bytes: &[u8]);

    // SAFETY: the caller holds the list in the recording state, with a compute
    // root signature bound that declares `root_dwords::<T>()` constants at `param`.
    unsafe fn set_compute_root_constants<T: NoUninit>(&self, param: u32, value: &T);
}

impl RootConstants for ID3D12GraphicsCommandList {
    unsafe fn set_graphics_root_constants<T: NoUninit>(&self, param: u32, value: &T) {
        // SAFETY: the caller upholds the list and signature contract, and the
        // count is `value`'s own size, which the pointer borrows for the call.
        unsafe {
            self.SetGraphicsRoot32BitConstants(
                param,
                root_dwords::<T>(),
                (value as *const T).cast(),
                0,
            )
        }
    }

    unsafe fn set_graphics_root_constant_bytes(&self, param: u32, bytes: &[u8]) {
        let count = byte_dwords(bytes);
        // SAFETY: the caller upholds the list and signature contract, and the
        // count covers exactly `bytes`, which the pointer borrows for the call.
        unsafe { self.SetGraphicsRoot32BitConstants(param, count, bytes.as_ptr().cast(), 0) }
    }

    unsafe fn set_compute_root_constants<T: NoUninit>(&self, param: u32, value: &T) {
        // SAFETY: the caller upholds the list and signature contract, and the
        // count is `value`'s own size, which the pointer borrows for the call.
        unsafe {
            self.SetComputeRoot32BitConstants(
                param,
                root_dwords::<T>(),
                (value as *const T).cast(),
                0,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_dwords_counts_whole_values() {
        assert_eq!(root_dwords::<u32>(), 1);
        assert_eq!(root_dwords::<[f32; 4]>(), 4);
        assert_eq!(root_dwords::<[[f32; 4]; 4]>(), 16);
    }

    #[test]
    fn byte_dwords_counts_whole_values() {
        assert_eq!(byte_dwords(&[]), 0);
        assert_eq!(byte_dwords(&[0; 144]), 36);
    }

    #[test]
    #[should_panic(expected = "whole 32-bit values")]
    fn byte_dwords_rejects_a_partial_value() {
        byte_dwords(&[0; 6]);
    }
}
