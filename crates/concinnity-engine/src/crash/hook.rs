// src/crash/hook.rs
//
// The process panic hook: writes a crash report (and a minidump where
// supported), then chains to the previously installed hook so the standard
// stderr message and unwind behavior are unchanged. A reentry guard plus
// catch_unwind keep a fault inside the reporting path from recursing; its
// allocations are bounded by the report caps.

use super::report::{CrashReport, MAX_MESSAGE_BYTES, ReportKind};
use super::write;
use std::panic::PanicHookInfo;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

static INSTALL: Once = Once::new();
static IN_HOOK: AtomicBool = AtomicBool::new(false);

pub(crate) fn install() {
    INSTALL.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !IN_HOOK.swap(true, Ordering::SeqCst) {
                let body = std::panic::AssertUnwindSafe(|| report_panic(info));
                let _ = std::panic::catch_unwind(body);
                IN_HOOK.store(false, Ordering::SeqCst);
            }
            prev(info);
        }));
    });
}

fn report_panic(info: &PanicHookInfo<'_>) {
    let mut report = CrashReport::gather(ReportKind::Panic, payload_message(info));
    report.thread = std::thread::current().name().map(str::to_owned);
    report.location = info.location().map(ToString::to_string);
    report.backtrace = Some(std::backtrace::Backtrace::force_capture().to_string());

    let Some(dir) = concinnity_store::paths::crashes_dir() else {
        return;
    };
    let stem = write::unique_stem(&dir, &report.file_stem());
    if write::write_report_named(&dir, &stem, &report).is_ok() {
        #[cfg(any(target_os = "macos", target_os = "windows"))]
        super::minidump::write_self_dump(&dir, &stem);
        write::prune(&dir, write::RETAINED_REPORTS);
    }
}

// The panic payload as text: the `&str` / `String` payloads every `panic!`
// produces, or a placeholder for a `panic_any` value.
fn payload_message(info: &PanicHookInfo<'_>) -> String {
    let payload = info.payload();
    let text = if let Some(s) = payload.downcast_ref::<&str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.as_str()
    } else {
        "<non-string panic payload>"
    };
    let mut message = String::with_capacity(text.len().min(MAX_MESSAGE_BYTES));
    let mut end = text.len().min(MAX_MESSAGE_BYTES);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    message.push_str(&text[..end]);
    message
}
