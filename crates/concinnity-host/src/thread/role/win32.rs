// Power throttling (EcoQoS): a throttled thread is steered to efficiency cores
// on hybrid parts and run at lower clocks. Frame threads opt out explicitly so
// a power plan cannot throttle them; background threads opt in. Frame threads
// also run above normal priority, so a busy machine's normal-priority work
// cannot hold them off a core.

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
        StateMask: match role {
            ThreadRole::Frame => 0,
            ThreadRole::Background => THREAD_POWER_THROTTLING_EXECUTION_SPEED,
        },
    }
}

pub(super) fn priority(role: ThreadRole) -> THREAD_PRIORITY {
    match role {
        ThreadRole::Frame => THREAD_PRIORITY_ABOVE_NORMAL,
        ThreadRole::Background => THREAD_PRIORITY_NORMAL,
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

    use super::*;

    #[test]
    fn a_role_sets_the_thread_priority() {
        for role in [ThreadRole::Frame, ThreadRole::Background] {
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
    fn frame_opts_out_and_background_opts_in() {
        let frame = throttling_state(ThreadRole::Frame);
        assert_eq!(frame.ControlMask, THREAD_POWER_THROTTLING_EXECUTION_SPEED);
        assert_eq!(frame.StateMask, 0);

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
