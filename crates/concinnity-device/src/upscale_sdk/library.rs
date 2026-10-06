//! Resolving a vendor runtime library's C entry points.

use std::ffi::{CStr, c_void};

/// A loaded vendor runtime library. Each backend loads the library its own way;
/// the shared bindings only look its exports up by name.
pub(crate) trait SdkLibrary {
    /// The address of the export `name`, or `None` when the library lacks it.
    fn symbol(&self, name: &CStr) -> Option<*const c_void>;
}

/// The export `name` of `library` as the function pointer type `F`.
///
/// # Safety
///
/// `F` must be the `extern "C"` function pointer type the SDK header declares
/// for `name`, and the pointer is only valid while `library` stays loaded.
pub(crate) unsafe fn entry_point<F: Copy>(library: &impl SdkLibrary, name: &CStr) -> Option<F> {
    const {
        assert!(size_of::<F>() == size_of::<*const c_void>());
    }
    let address = library.symbol(name).filter(|p| !p.is_null())?;
    // SAFETY: `F` is pointer-sized (checked above) and, per the caller's contract, the function
    // pointer type of the non-null export `address` points at.
    Some(unsafe { std::mem::transmute_copy::<*const c_void, F>(&address) })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Exports;

    extern "C" fn answer() -> i32 {
        42
    }

    impl SdkLibrary for Exports {
        fn symbol(&self, name: &CStr) -> Option<*const c_void> {
            match name.to_bytes() {
                b"answer" => Some(answer as *const c_void),
                b"null" => Some(std::ptr::null()),
                _ => None,
            }
        }
    }

    #[test]
    fn resolves_present_exports_and_rejects_missing_or_null_ones() {
        // SAFETY: `answer` is declared as exactly this prototype.
        let f = unsafe { entry_point::<extern "C" fn() -> i32>(&Exports, c"answer") };
        assert_eq!(f.map(|f| f()), Some(42));
        // SAFETY: neither lookup yields a pointer that is called.
        unsafe {
            assert!(entry_point::<extern "C" fn() -> i32>(&Exports, c"missing").is_none());
            assert!(entry_point::<extern "C" fn() -> i32>(&Exports, c"null").is_none());
        }
    }
}
