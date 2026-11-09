// The Windows system clipboard as UTF-16 text (CF_UNICODETEXT), opened on
// behalf of the window: Win32 only lets an owning window replace its contents.

use concinnity_engine::window::clipboard::Clipboard;
use windows::Win32::Foundation::{GlobalFree, HANDLE, HGLOBAL, HWND};
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, GetClipboardData, OpenClipboard, SetClipboardData,
};
use windows::Win32::System::Memory::{
    GMEM_MOVEABLE, GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock,
};
use windows::Win32::System::Ole::CF_UNICODETEXT;

use super::window::WindowState;

// Holds the clipboard open for one read or write; closing on drop keeps an
// early return from leaving it locked against every other application.
struct OpenClipboardGuard;

impl OpenClipboardGuard {
    fn open(hwnd: HWND) -> Option<Self> {
        // SAFETY: `hwnd` is this window's live handle; the call only records it as the owner.
        match unsafe { OpenClipboard(Some(hwnd)) } {
            Ok(()) => Some(Self),
            Err(e) => {
                tracing::warn!("the system clipboard is busy: {e}");
                None
            }
        }
    }
}

impl Drop for OpenClipboardGuard {
    #[expect(
        clippy::let_underscore_must_use,
        reason = "a drop has nowhere to report, and the next open reports a clipboard still held"
    )]
    fn drop(&mut self) {
        // SAFETY: the guard exists only while this thread holds the clipboard open.
        let _ = unsafe { CloseClipboard() };
    }
}

impl Clipboard for WindowState {
    #[expect(
        clippy::let_underscore_must_use,
        reason = "GlobalUnlock reports an error once the block is fully unlocked, the expected outcome here"
    )]
    fn text(&mut self) -> Option<String> {
        let _open = OpenClipboardGuard::open(self.hwnd)?;
        // SAFETY: the clipboard is open, so the handle it returns stays valid until it closes.
        let handle = unsafe { GetClipboardData(u32::from(CF_UNICODETEXT.0)) }.ok()?;
        let mem = HGLOBAL(handle.0);
        // SAFETY: `mem` is the clipboard's live global block while the clipboard stays open.
        let ptr = unsafe { GlobalLock(mem) } as *const u16;
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `mem` is the same live block, now locked by the call above.
        let size = unsafe { GlobalSize(mem) };
        // SAFETY: `ptr` is the locked block of `size` bytes, readable as `size / 2` units until
        // the unlock below.
        let units = unsafe { std::slice::from_raw_parts(ptr, size / 2) };
        let text = decode_units(units);
        // SAFETY: balances the lock above; `units` is not used past this point.
        let _ = unsafe { GlobalUnlock(mem) };
        Some(text)
    }

    #[expect(
        clippy::let_underscore_must_use,
        reason = "GlobalUnlock reports an error once the block is fully unlocked; freeing on a failure path is best effort"
    )]
    fn set_text(&mut self, text: &str) {
        let units = encode_units(text);
        let Some(_open) = OpenClipboardGuard::open(self.hwnd) else {
            return;
        };
        // SAFETY: the clipboard is open and owned by this window.
        if unsafe { EmptyClipboard() }.is_err() {
            return;
        }
        // SAFETY: a plain movable allocation sized for `units`; nothing else holds it yet.
        let alloc = unsafe { GlobalAlloc(GMEM_MOVEABLE, units.len() * 2) };
        let Ok(mem) = alloc else {
            return;
        };
        // SAFETY: `mem` is the block just allocated, owned here and not yet locked.
        let ptr = unsafe { GlobalLock(mem) }.cast::<u16>();
        if ptr.is_null() {
            // SAFETY: the lock failed, so the block is still solely ours to free.
            let _ = unsafe { GlobalFree(Some(mem)) };
            return;
        }
        // SAFETY: the locked block is sized for `units.len()` units and is a separate
        // allocation from `units`.
        unsafe { std::ptr::copy_nonoverlapping(units.as_ptr(), ptr, units.len()) };
        // SAFETY: balances the lock above; `ptr` is not used past this point.
        let _ = unsafe { GlobalUnlock(mem) };
        // SAFETY: the clipboard is open and owned by this window; once this call accepts the
        // unlocked block the system owns it.
        if unsafe { SetClipboardData(u32::from(CF_UNICODETEXT.0), Some(HANDLE(mem.0))) }.is_err() {
            tracing::warn!("the system clipboard refused the copied text");
            // SAFETY: the system refused the block, so it is still ours and is freed once.
            let _ = unsafe { GlobalFree(Some(mem)) };
        }
    }
}

// The clipboard's UTF-16 up to its terminating NUL (the block may be larger).
fn decode_units(units: &[u16]) -> String {
    let len = units.iter().position(|&u| u == 0).unwrap_or(units.len());
    String::from_utf16_lossy(&units[..len])
}

// `text` as NUL-terminated UTF-16 with Windows line endings, the convention
// other Windows applications expect to paste.
fn encode_units(text: &str) -> Vec<u16> {
    let mut out = Vec::with_capacity(text.len() + 1);
    let mut prev = '\0';
    for c in text.chars() {
        if c == '\n' && prev != '\r' {
            out.push(u16::from(b'\r'));
        }
        let mut buf = [0u16; 2];
        out.extend_from_slice(c.encode_utf16(&mut buf));
        prev = c;
    }
    out.push(0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_writes_crlf_and_a_terminator() {
        let units = encode_units("a\nb\r\nc");
        assert_eq!(String::from_utf16(&units).unwrap(), "a\r\nb\r\nc\0");
    }

    #[test]
    fn decode_stops_at_the_terminator() {
        let mut units: Vec<u16> = "héllo".encode_utf16().collect();
        units.extend([0, u16::from(b'x')]);
        assert_eq!(decode_units(&units), "héllo");
        assert_eq!(decode_units(&[]), "");
    }

    #[test]
    fn encode_round_trips_outside_the_basic_plane() {
        let units = encode_units("\u{1F600}");
        assert_eq!(decode_units(&units), "\u{1F600}");
    }
}
