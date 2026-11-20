//! RAII guard for Metal command encoders. Wrapping a freshly-created encoder in
//! a `ScopedEncoder` makes two things automatic at end of scope: its debug group
//! is popped and `endEncoding()` is called, even if a `?` returns early
//! mid-encode. Without it, an early return between an encoder's creation and its
//! manual `endEncoding()` leaves the encoder open, which Metal rejects when the
//! command buffer is committed.
//!
//! The guard is generic over the encoder protocol (render / compute / blit /
//! acceleration-structure all conform to `MTLCommandEncoder`) and derefs to the
//! underlying encoder, so existing `enc.setX(...)` calls work unchanged.
//!
//! IMPORTANT: Metal forbids two open encoders on one command buffer at once. A
//! function that opens several encoders in sequence must therefore give each its
//! own inner `{ }` scope so one guard drops (ending its encoder) before the next
//! is created.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::MTLCommandEncoder;

// RAII wrapper that pops its debug group and ends the encoder on drop. Generic
// over the encoder protocol `E` (e.g. `dyn MTLRenderCommandEncoder`).
pub(crate) struct ScopedEncoder<E: ?Sized + MTLCommandEncoder> {
    enc: Retained<ProtocolObject<E>>,
}

impl<E: ?Sized + MTLCommandEncoder> ScopedEncoder<E> {
    // Take `enc` and push the debug group `label`, popped and ended on drop.
    // A static `NSString` label costs no per-frame allocation.
    pub(crate) fn new(enc: Retained<ProtocolObject<E>>, label: &NSString) -> Self {
        enc.pushDebugGroup(label);
        Self { enc }
    }
}

impl<E: ?Sized + MTLCommandEncoder> std::ops::Deref for ScopedEncoder<E> {
    type Target = ProtocolObject<E>;
    fn deref(&self) -> &Self::Target {
        &self.enc
    }
}

impl<E: ?Sized + MTLCommandEncoder> Drop for ScopedEncoder<E> {
    fn drop(&mut self) {
        self.enc.popDebugGroup();
        self.enc.endEncoding();
    }
}
