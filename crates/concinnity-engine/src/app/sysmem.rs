// src/app/sysmem.rs
//
// Host-memory queries used to compute the process memory budget (see
// `app::budget`) and to report live usage. Two values, each best-effort: the
// machine's total physical RAM (the budget is a fraction of it) and this
// process's resident set size (what it is actually using now).
//
// Deliberately a small hand-rolled platform shim rather than a dependency like
// `sysinfo`: the engine needs exactly these two numbers, and both are a single
// syscall per platform. Every query returns `None` when the platform call is
// unavailable or fails, and the callers degrade gracefully (the budget falls
// back to the hard ceiling; a usage readout shows "unknown").

// Total physical RAM installed on the machine, in bytes. `None` if the platform
// query is unsupported or fails.
pub fn total_physical_bytes() -> Option<u64> {
    imp::total_physical_bytes()
}

// Resident set size of the current process, in bytes: the physical memory it
// currently occupies. `None` if the platform query is unsupported or fails.
pub fn process_resident_bytes() -> Option<u64> {
    imp::process_resident_bytes()
}

#[cfg(target_os = "macos")]
mod imp {
    // hw.memsize is total physical RAM; task_info(MACH_TASK_BASIC_INFO) carries
    // the process resident size.
    pub(super) fn total_physical_bytes() -> Option<u64> {
        let mut value: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = c"hw.memsize";
        // SAFETY: `name` is a valid NUL-terminated C string; `value`/`len`
        // describe a correctly sized u64 output buffer for this sysctl.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut value as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0 && value > 0).then_some(value)
    }

    // `libc` deprecates its mach bindings in favor of the `mach2` crate;
    // `mach_task_self_` is a stable fundamental symbol, so we keep the direct
    // libc use rather than pull in another dependency for one static.
    #[allow(deprecated)]
    pub(super) fn process_resident_bytes() -> Option<u64> {
        let mut info: libc::mach_task_basic_info = unsafe { std::mem::zeroed() };
        let mut count = (std::mem::size_of::<libc::mach_task_basic_info>()
            / std::mem::size_of::<libc::natural_t>())
            as libc::mach_msg_type_number_t;
        // SAFETY: `info`/`count` are a correctly sized MACH_TASK_BASIC_INFO
        // output buffer; `mach_task_self_` is the current task port (the raw
        // static behind the deprecated `mach_task_self()` wrapper).
        let rc = unsafe {
            libc::task_info(
                libc::mach_task_self_,
                libc::MACH_TASK_BASIC_INFO,
                &mut info as *mut _ as libc::task_info_t,
                &mut count,
            )
        };
        (rc == libc::KERN_SUCCESS).then_some(info.resident_size)
    }
}

#[cfg(target_os = "linux")]
mod imp {
    // MemTotal from /proc/meminfo (kB); resident pages (field 2) from
    // /proc/self/statm times the page size.
    pub(super) fn total_physical_bytes() -> Option<u64> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }

    pub(super) fn process_resident_bytes() -> Option<u64> {
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let resident_pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
        // SAFETY: sysconf with a valid name has no memory effects.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        (page_size > 0).then(|| resident_pages * page_size as u64)
    }
}

#[cfg(target_os = "windows")]
mod imp {
    use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows::Win32::System::Threading::GetCurrentProcess;

    pub(super) fn total_physical_bytes() -> Option<u64> {
        let mut status = MEMORYSTATUSEX {
            dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        // SAFETY: `status.dwLength` is set to the struct size as the API requires.
        unsafe { GlobalMemoryStatusEx(&mut status) }.ok()?;
        (status.ullTotalPhys > 0).then_some(status.ullTotalPhys)
    }

    pub(super) fn process_resident_bytes() -> Option<u64> {
        let mut counters = PROCESS_MEMORY_COUNTERS::default();
        // SAFETY: the counters buffer is sized to its own type, as the API requires.
        let ok = unsafe {
            GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut counters,
                std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            )
        };
        ok.ok().map(|()| counters.WorkingSetSize as u64)
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
mod imp {
    pub(super) fn total_physical_bytes() -> Option<u64> {
        None
    }
    pub(super) fn process_resident_bytes() -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // On a supported platform the machine has some RAM and this process holds
    // some resident memory; both queries return a plausible positive value.
    // (macOS / Linux / Windows are all supported; other targets return None and
    // skip the assertion.)
    #[test]
    fn queries_return_plausible_values() {
        if cfg!(any(
            target_os = "macos",
            target_os = "linux",
            target_os = "windows"
        )) {
            let total = total_physical_bytes().expect("total RAM query works on this platform");
            assert!(
                total >= 256 * 1024 * 1024,
                "implausibly small total RAM: {total}"
            );

            let rss = process_resident_bytes().expect("RSS query works on this platform");
            assert!(rss > 0, "process resident size should be positive");
            assert!(rss <= total, "RSS {rss} exceeds total RAM {total}");
        }
    }
}
