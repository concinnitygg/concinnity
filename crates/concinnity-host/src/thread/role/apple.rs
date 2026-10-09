// Quality-of-service classes: the scheduler wakes higher classes sooner and
// prefers performance cores for them.

use libc::qos_class_t;

use super::ThreadRole;

pub(super) fn qos_class(role: ThreadRole) -> qos_class_t {
    match role {
        ThreadRole::Frame => qos_class_t::QOS_CLASS_USER_INTERACTIVE,
        ThreadRole::Background => qos_class_t::QOS_CLASS_UTILITY,
    }
}

pub(super) fn apply(role: ThreadRole) -> std::io::Result<()> {
    // SAFETY: `pthread_set_qos_class_self_np` only changes the calling thread's
    // scheduling; it takes a valid class and a relative priority of 0, which is
    // within the documented 0..=QOS_MIN_RELATIVE_PRIORITY range.
    let status = unsafe { libc::pthread_set_qos_class_self_np(qos_class(role), 0) };
    if status == 0 {
        return Ok(());
    }
    let error = std::io::Error::from_raw_os_error(status);
    Err(std::io::Error::other(format!(
        "pthread_set_qos_class_self_np: {error}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn current_qos_class() -> qos_class_t {
        let mut class = qos_class_t::QOS_CLASS_UNSPECIFIED;
        let mut relative = 0;
        // SAFETY: both out-pointers are live locals, and `pthread_self` names
        // the calling thread, which outlives the call.
        let status = unsafe {
            libc::pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut relative)
        };
        assert_eq!(status, 0);
        class
    }

    #[test]
    fn a_role_sets_the_thread_qos_class() {
        for role in [ThreadRole::Frame, ThreadRole::Background] {
            let class = std::thread::spawn(move || {
                apply(role).expect("the role applies");
                current_qos_class()
            })
            .join()
            .expect("the thread runs");
            assert_eq!(class as u32, qos_class(role) as u32, "{role:?}");
        }
    }

    #[test]
    fn frame_outranks_background() {
        assert!(qos_class(ThreadRole::Frame) as u32 > qos_class(ThreadRole::Background) as u32);
    }
}
