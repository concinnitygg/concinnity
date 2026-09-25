// The Windows system clipboard as UTF-16 text (CF_UNICODETEXT), opened on
// behalf of the window: Win32 only lets an owning window replace its contents.

use concinnity_core::window::clipboard::Clipboard;
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
    fn drop(&mut self) {
        // SAFETY: the guard exists only while this thread holds the clipboard open.
        let _ = unsafe { CloseClipboard() };
    }
}

impl Clipboard for WindowState {
    fn text(&mut self) -> Option<String> {
        let _open = OpenClipboardGuard::open(self.hwnd)?;
        // SAFETY: the clipboard is open, so the handle it returns stays valid until it
        // closes; the global block is locked for the read, which `GlobalSize` bounds.
        unsafe {
            let handle = GetClipboardData(u32::from(CF_UNICODETEXT.0)).ok()?;
            let mem = HGLOBAL(handle.0);
            let ptr = GlobalLock(mem) as *const u16;
            if ptr.is_null() {
                return None;
            }
            let units = std::slice::from_raw_parts(ptr, GlobalSize(mem) / 2);
            let text = decode_units(units);
            let _ = GlobalUnlock(mem);
            Some(text)
        }
    }

    fn set_text(&mut self, text: &str) {
        let units = encode_units(text);
        let Some(_open) = OpenClipboardGuard::open(self.hwnd) else {
            return;
        };
        // SAFETY: the clipboard is open and owned by this window. The block is sized for
        // `units` and locked while they are copied in; once `SetClipboardData` accepts
        // it the system owns it, and it is freed here only when that call fails.
        unsafe {
            if EmptyClipboard().is_err() {
                return;
            }
            let Ok(mem) = GlobalAlloc(GMEM_MOVEABLE, units.len() * 2) else {
                return;
            };
            let ptr = GlobalLock(mem) as *mut u16;
            if ptr.is_null() {
                let _ = GlobalFree(Some(mem));
                return;
            }
            std::ptr::copy_nonoverlapping(units.as_ptr(), ptr, units.len());
            let _ = GlobalUnlock(mem);
            if SetClipboardData(u32::from(CF_UNICODETEXT.0), Some(HANDLE(mem.0))).is_err() {
                tracing::warn!("the system clipboard refused the copied text");
                let _ = GlobalFree(Some(mem));
            }
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
