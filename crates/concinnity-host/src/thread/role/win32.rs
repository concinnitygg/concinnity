// Power throttling (EcoQoS): a throttled thread is steered to efficiency cores
// on hybrid parts and run at lower clocks. Frame threads opt out explicitly so
// a power plan cannot throttle them; background threads opt in. A raised frame
// thread also runs above normal priority, so a busy machine's normal-priority
// work cannot hold it off a core.

use windows::Win32::System::Threading::{
    GetCurrentThread, SetThreadInformation, SetThreadPriority,
    THREAD_POWER_THROTTLING_CURRENT_VERSION, THREAD_POWER_THROTTLING_EXECUTION_SPEED,
    THREAD_POWER_THROTTLING_STATE, THREAD_PRIORITY, THREAD_PRIORITY_ABOVE_NORMAL,
    THREAD_PRIORITY_NORMAL, ThreadPowerThrottling,
};

use super::ThreadRole;

pub(super) fn throttling_state(role: ThreadRole) -> THREAD_POWER_THROTTLING_STATE {
    THREAD_POWER_THROTTLING_STATE {
        Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: THREAD_POWER_THROTTLING_EXECUTION_SPEED,
        StateMask: if role.is_frame() {
            0
        } else {
            THREAD_POWER_THROTTLING_EXECUTION_SPEED
        },
    }
}

pub(super) fn priority(role: ThreadRole) -> THREAD_PRIORITY {
    if role.raised() {
        THREAD_PRIORITY_ABOVE_NORMAL
    } else {
        THREAD_PRIORITY_NORMAL
    }
}

pub(super) fn apply(role: ThreadRole) -> std::io::Result<()> {
    // SAFETY: `GetCurrentThread` is a pseudo-handle valid for the calling
    // thread, and the priority is one of the documented relative levels.
    unsafe { SetThreadPriority(GetCurrentThread(), priority(role)) }
        .map_err(|e| std::io::Error::other(format!("SetThreadPriority: {e}")))?;
    let state = throttling_state(role);
    // SAFETY: `GetCurrentThread` is a pseudo-handle valid for the calling
    // thread, and `state` is a live `THREAD_POWER_THROTTLING_STATE` whose size
    // is the one passed, as `ThreadPowerThrottling` requires.
    unsafe {
        SetThreadInformation(
            GetCurrentThread(),
            ThreadPowerThrottling,
            (&raw const state).cast(),
            size_of::<THREAD_POWER_THROTTLING_STATE>() as u32,
        )
    }
    .map_err(|e| std::io::Error::other(format!("SetThreadInformation(ThreadPowerThrottling): {e}")))
}

#[cfg(test)]
mod tests {
    use windows::Win32::System::Threading::GetThreadPriority;

    use super::super::EVERY_ROLE;
    use super::*;

    #[test]
    fn a_role_sets_the_thread_priority() {
        for role in EVERY_ROLE {
            let current = std::thread::spawn(move || {
                apply(role).expect("the role applies");
                // SAFETY: `GetCurrentThread` is a pseudo-handle valid for the
                // calling thread.
                unsafe { GetThreadPriority(GetCurrentThread()) }
            })
            .join()
            .expect("the thread runs");
            assert_eq!(current, priority(role).0, "{role:?}");
        }
    }

    #[test]
    fn only_raised_frame_threads_run_above_normal() {
        let above: Vec<bool> = EVERY_ROLE
            .iter()
            .map(|&role| priority(role) == THREAD_PRIORITY_ABOVE_NORMAL)
            .collect();
        assert_eq!(above, [false, true, true, false, false, true, false]);
    }

    #[test]
    fn frame_threads_opt_out_and_background_opts_in() {
        for role in EVERY_ROLE.into_iter().filter(|r| r.is_frame()) {
            let frame = throttling_state(role);
            assert_eq!(frame.ControlMask, THREAD_POWER_THROTTLING_EXECUTION_SPEED);
            assert_eq!(frame.StateMask, 0, "{role:?}");
        }

        let background = throttling_state(ThreadRole::Background);
        assert_eq!(
            background.ControlMask,
            THREAD_POWER_THROTTLING_EXECUTION_SPEED
        );
        assert_eq!(
            background.StateMask,
            THREAD_POWER_THROTTLING_EXECUTION_SPEED
        );
    }
}
