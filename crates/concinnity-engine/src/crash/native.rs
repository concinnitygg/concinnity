// src/crash/native.rs
//
// Native fault capture (macOS mach exceptions, Windows SEH) via
// crash-handler. Faults in unsafe or driver code never reach the panic hook;
// this path writes the minidump first (the artifact that matters most, with
// the writer designed for a compromised context) and then attempts a sidecar
// text report with nonblocking snapshots. The fault is reported as unhandled
// so the OS default crash behavior still runs.

use super::report::{CrashReport, ReportKind};
use super::write;
use crash_handler::{CrashContext, CrashEventResult, CrashHandler};
use std::sync::OnceLock;

// Keeps the handler (and its exception thread on macOS) alive for the
// process lifetime.
static HANDLER: OnceLock<CrashHandler> = OnceLock::new();

pub(crate) fn install() {
    if HANDLER.get().is_some() {
        return;
    }
    // SAFETY: the closure runs in a compromised context; it confines itself
    // to minidump-writer (built for that context), try_lock snapshots, and
    // bounded formatting.
    let event = unsafe {
        crash_handler::make_crash_event(|ctx: &CrashContext| {
            report_fault(ctx);
            CrashEventResult::Handled(false)
        })
    };
    if let Ok(handler) = CrashHandler::attach(event) {
        let _ = HANDLER.set(handler);
    }
}

fn report_fault(ctx: &CrashContext) {
    let dir = concinnity_store::paths::crashes_dir();
    let report = CrashReport::gather_nonblocking(ReportKind::NativeFault, fault_message(ctx));
    let stem = write::unique_stem(&dir, &report.file_stem());
    super::minidump::write_fault_dump(&dir, &stem, ctx);
    let _ = write::write_report_named(&dir, &stem, &report);
    write::prune(&dir, write::RETAINED_REPORTS);
}

#[cfg(target_os = "macos")]
fn fault_message(ctx: &CrashContext) -> String {
    match ctx.exception {
        Some(e) => format!(
            "native fault: mach exception kind {:#x}, code {:#x}",
            e.kind, e.code
        ),
        None => "native fault: unknown mach exception".to_string(),
    }
}

#[cfg(target_os = "windows")]
fn fault_message(ctx: &CrashContext) -> String {
    format!(
        "native fault: exception code {:#010x} on thread {}",
        ctx.exception_code, ctx.thread_id
    )
}
