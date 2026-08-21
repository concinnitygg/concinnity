// Safe wrappers over recurring COM idioms: read-only property queries on live
// interfaces, and refcount-free interface borrows for `ManuallyDrop` struct
// fields.

use std::mem::ManuallyDrop;

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::core::Interface;

// The GPU virtual address of a live buffer resource.
pub(super) fn gpu_va(resource: &ID3D12Resource) -> u64 {
    // SAFETY: a property query on a live resource; it only reads.
    unsafe { resource.GetGPUVirtualAddress() }
}

// Borrow an interface into a `ManuallyDrop<Option<I>>` struct field without an
// AddRef. The field must stay `ManuallyDrop` (never released), and the copied
// pointer is only valid while the borrow it came from is live; the D3D12 call
// consuming the struct asserts that in its own SAFETY comment.
pub(super) fn borrowed<I: Interface>(interface: &I) -> ManuallyDrop<Option<I>> {
    // SAFETY: `Option<I>` is a nullable interface pointer with `I`'s layout, so
    // this copies the raw pointer with no refcount change, and `ManuallyDrop`
    // never releases it.
    unsafe { std::mem::transmute_copy(interface) }
}
