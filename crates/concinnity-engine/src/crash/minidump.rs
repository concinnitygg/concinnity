// src/crash/minidump.rs
//
// Minidump emission via minidump-writer, macOS and Windows only: both have a
// supported in-process dump path. Linux would need an external dumper process
// (ptrace cannot target the running process itself), so it ships text reports
// only. Failures are swallowed after cleaning up the partial file; the text
// report already landed.

use std::path::Path;

// Dump the current (healthy) process, used from the panic hook.
pub(crate) fn write_self_dump(dir: &Path, stem: &str) {
    let Ok((mut file, path)) = super::write::create_dump_file(dir, stem) else {
        return;
    };
    if !dump_self(&mut file) {
        drop(file);
        let _ = std::fs::remove_file(path);
    }
}

// Dump using the native fault handler's crash context, so the dump records
// the faulting thread and exception rather than the handler.
pub(crate) fn write_fault_dump(dir: &Path, stem: &str, ctx: &crash_handler::CrashContext) {
    let Ok((mut file, path)) = super::write::create_dump_file(dir, stem) else {
        return;
    };
    if !dump_fault(&mut file, ctx) {
        drop(file);
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(target_os = "macos")]
fn dump_self(file: &mut std::fs::File) -> bool {
    minidump_writer::minidump_writer::MinidumpWriter::new(None, None)
        .dump(file)
        .is_ok()
}

#[cfg(target_os = "macos")]
fn dump_fault(file: &mut std::fs::File, ctx: &crash_handler::CrashContext) -> bool {
    // The context fields are plain mach ports and Copy exception info; rebuilt
    // here because the writer takes ownership.
    let owned = minidump_writer::crash_context::CrashContext {
        task: ctx.task,
        thread: ctx.thread,
        handler_thread: ctx.handler_thread,
        exception: ctx.exception,
    };
    minidump_writer::minidump_writer::MinidumpWriter::with_crash_context(owned)
        .dump(file)
        .is_ok()
}

#[cfg(target_os = "windows")]
fn dump_self(file: &mut std::fs::File) -> bool {
    minidump_writer::minidump_writer::MinidumpWriter::dump_local_context(None, None, None, file)
        .is_ok()
}

#[cfg(target_os = "windows")]
fn dump_fault(file: &mut std::fs::File, ctx: &crash_handler::CrashContext) -> bool {
    minidump_writer::minidump_writer::MinidumpWriter::dump_crash_context(ctx, None, file).is_ok()
}
