// The extern "C" surface cbindgen turns into include/concinnity.h.
//
// A world's lifecycle inside a view the host owns: open, step, close. That is
// the shape a platform whose OS owns the run loop needs, because the engine
// never gets to run a loop of its own there. The host creates the view, opens
// a world into it, and calls `cn_world_step` once per display refresh.
//
// Every cn_* function must be called from one thread, the one that owns the
// view. GPU objects inside the world are not thread-safe; the mutex here stops
// re-entrancy, it does not lift that requirement.

use std::ffi::{CStr, c_void};
use std::os::raw::{c_char, c_int};
use std::ptr::NonNull;
use std::sync::{Mutex, OnceLock};

use concinnity_core::ecs::StepResult;
use concinnity_core::error::WorldError;
use concinnity_core::render::backend_init::EmbeddedSurface;
use concinnity_engine::{Runtime, StartupError};
use concinnity_host::store::paths::StateTree;

/// Why a `cn_` call failed, or [`Ok`](CnError::Ok) when it did not.
///
/// The engine reports a structured error with a cause chain to its Rust
/// callers; a C host cannot hold one, so this is what a failure flattens to
/// here, at the boundary, with the full chain going to the log.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CnError {
    /// The call succeeded.
    Ok = 0,
    /// A pointer argument was null, or a string was not UTF-8.
    InvalidArgument = 1,
    /// `cn_init` has not run, so there is no host state to act on.
    NotInitialized = 2,
    /// The root names nothing a world can be read from.
    NoWorldData = 3,
    /// The world data is present but did not load.
    UnreadableWorldData = 4,
    /// The world loaded but refused to start.
    StartFailed = 5,
}

/// What one step of a world reports.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CnStep {
    /// The world is running; step it again next refresh.
    Continue = 0,
    /// The world reached its own end.
    Done = 1,
    /// The world asked to stop.
    Stopped = 2,
    /// No world is open, so there was nothing to step.
    NoWorld = -1,
}

impl From<StepResult> for CnStep {
    fn from(result: StepResult) -> Self {
        match result {
            StepResult::Continue => CnStep::Continue,
            StepResult::Done => CnStep::Done,
            StepResult::Stop => CnStep::Stopped,
        }
    }
}

struct HostState {
    world: Option<Runtime>,
}

// SAFETY: the only `HostState` lives behind the mutex below, so one thread at a
// time reaches the `Runtime` inside it. The single-thread requirement the module
// documents is what makes that thread always the same one.
unsafe impl Send for HostState {}

static STATE: OnceLock<Mutex<HostState>> = OnceLock::new();

/// Initialize logging and the host state. Call once, from the thread that owns
/// the view, before any other `cn_` function.
///
/// The log level is not a parameter: it follows the same default the player
/// uses and `RUST_LOG` overrides it.
#[unsafe(no_mangle)]
pub extern "C" fn cn_init() {
    concinnity_engine::app::run::init_logging();
    STATE.get_or_init(|| Mutex::new(HostState { world: None }));
}

/// Open the built world under `root` and render it into `view`.
///
/// `root` is a NUL-terminated absolute path to the directory holding the
/// world's `data/`. `view` is the platform's view pointer, an `NSView*` on
/// macOS and a `UIView*` on iOS, and must outlive the world. A non-zero
/// `pump_events` asks the engine to drain the platform's event queue during a
/// step, for a host that does not run an event loop of its own; pass 0 when
/// the host dispatches input itself, which is the usual case.
///
/// Replaces whatever world was open. Returns [`CnError::Ok`] on success, and
/// otherwise the code for why it failed, with the full cause chain logged.
///
/// # Safety
///
/// `root` must be a NUL-terminated C string, and `view` a live view pointer of
/// the platform's type, both valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cn_world_open(
    root: *const c_char,
    view: *mut c_void,
    pump_events: c_int,
) -> CnError {
    // SAFETY: the caller's contract above; `ptr_to_string` handles null.
    let Some(root) = (unsafe { ptr_to_string(root) }) else {
        tracing::error!("cn_world_open: root is null or not UTF-8");
        return CnError::InvalidArgument;
    };
    let Some(view) = NonNull::new(view) else {
        tracing::error!("cn_world_open: view is null");
        return CnError::InvalidArgument;
    };
    let Some(state) = STATE.get() else {
        tracing::error!("cn_world_open: call cn_init first");
        return CnError::NotInitialized;
    };
    let Ok(mut state) = state.lock() else {
        return CnError::NotInitialized;
    };

    // Dropped before the new world is built, so the outgoing world's view is
    // detached before the incoming one attaches.
    state.world = None;

    let surface = EmbeddedSurface {
        view,
        pump_events: pump_events != 0,
    };
    match open_world(&root, surface) {
        Ok(world) => {
            state.world = Some(world);
            CnError::Ok
        }
        Err(e) => {
            tracing::error!("cn_world_open: {}", chain(&e));
            code_for(&e)
        }
    }
}

/// Step the open world once. Returns a [`CnStep`].
#[unsafe(no_mangle)]
pub extern "C" fn cn_world_step() -> CnStep {
    let Some(state) = STATE.get() else {
        return CnStep::NoWorld;
    };
    let Ok(mut state) = state.lock() else {
        return CnStep::NoWorld;
    };
    match state.world.as_mut() {
        Some(world) => world.world_step().into(),
        None => CnStep::NoWorld,
    }
}

/// Close the open world, releasing its GPU resources and detaching it from the
/// host's view. Does nothing when no world is open.
#[unsafe(no_mangle)]
pub extern "C" fn cn_world_close() {
    if let Some(state) = STATE.get()
        && let Ok(mut state) = state.lock()
    {
        state.world = None;
    }
}

// Why a world did not open, in the terms this boundary maps to a code. Rust's
// own errors stay whole underneath: the variants carry them so the log can
// print the chain the code cannot.
#[derive(Debug, thiserror::Error)]
enum OpenError {
    #[error("{0} is not a directory")]
    NotADirectory(String),
    #[error("loading the world failed")]
    Load(#[from] StartupError),
    #[error("starting the world failed")]
    Start(#[from] WorldError),
}

// The flat code a C host reads. Absent data and unusable data stay apart,
// since only the first is the host's to fix.
fn code_for(e: &OpenError) -> CnError {
    match e {
        OpenError::NotADirectory(_) => CnError::InvalidArgument,
        OpenError::Load(StartupError::MissingData { .. } | StartupError::NoStateRoot) => {
            CnError::NoWorldData
        }
        OpenError::Load(_) => CnError::UnreadableWorldData,
        OpenError::Start(_) => CnError::StartFailed,
    }
}

// The whole cause chain on one line, which is what the code leaves out.
fn chain(e: &dyn std::error::Error) -> String {
    let mut line = e.to_string();
    let mut source = e.source();
    while let Some(cause) = source {
        line.push_str(": ");
        line.push_str(&cause.to_string());
        source = cause.source();
    }
    line
}

// Build and start a world rooted at `root` that renders into `surface`. Split
// out so the failure path has one shape.
fn open_world(root: &str, surface: EmbeddedSurface) -> Result<Runtime, OpenError> {
    let root = std::path::Path::new(root);
    if !root.is_dir() {
        return Err(OpenError::NotADirectory(root.display().to_string()));
    }
    let mut world = Runtime::new().in_tree(StateTree::at(root));
    world.load_blob()?;
    world.world_mut().insert_resource(surface);
    world.start()?;
    Ok(world)
}

// A C string as an owned `String`, or `None` when it is null or not UTF-8.
//
// # Safety
//
// `ptr` must be null or a NUL-terminated C string valid for this call.
unsafe fn ptr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the caller's contract; the null case returned above.
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .ok()
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_testing::{ExclusiveAccess, TempTree, exclusive};
    use std::ffi::CString;

    // Every test that reaches the process-global host state holds the one
    // guard over it, so they do not race each other's world.
    fn host() -> ExclusiveAccess {
        let guard = exclusive();
        cn_init();
        cn_world_close();
        guard
    }

    // A view pointer that is merely non-null. Nothing here opens a world, so
    // no backend ever dereferences it.
    fn some_view(byte: &mut u8) -> *mut c_void {
        std::ptr::from_mut(byte).cast()
    }

    #[test]
    fn every_step_result_has_a_code() {
        assert_eq!(CnStep::from(StepResult::Continue), CnStep::Continue);
        assert_eq!(CnStep::from(StepResult::Done), CnStep::Done);
        assert_eq!(CnStep::from(StepResult::Stop), CnStep::Stopped);
    }

    // A C caller compares against these integers, so they are the contract and
    // not an implementation detail.
    #[test]
    fn the_step_codes_are_the_ones_the_header_publishes() {
        assert_eq!(CnStep::Continue as i32, 0);
        assert_eq!(CnStep::Done as i32, 1);
        assert_eq!(CnStep::Stopped as i32, 2);
        assert_eq!(CnStep::NoWorld as i32, -1);
    }

    #[test]
    fn stepping_with_no_world_open_says_so_rather_than_failing() {
        let _guard = host();
        assert_eq!(cn_world_step(), CnStep::NoWorld);
    }

    #[test]
    fn closing_with_nothing_open_is_a_no_op() {
        let _guard = host();
        cn_world_close();
        assert_eq!(cn_world_step(), CnStep::NoWorld);
    }

    #[test]
    fn opening_refuses_a_null_root() {
        let _guard = host();
        let mut byte = 0u8;
        // SAFETY: the root is the null this asks about; the view is a live
        // pointer, and the call returns before either is used.
        let opened = unsafe { cn_world_open(std::ptr::null(), some_view(&mut byte), 0) };
        assert_eq!(opened, CnError::InvalidArgument);
    }

    #[test]
    fn opening_refuses_a_null_view() {
        let _guard = host();
        let tree = TempTree::new();
        let root = CString::new(tree.root_path()).expect("a temp path holds no NUL");
        // SAFETY: `root` is a live NUL-terminated string; the null view is
        // what this asks about.
        let opened = unsafe { cn_world_open(root.as_ptr(), std::ptr::null_mut(), 0) };
        assert_eq!(opened, CnError::InvalidArgument);
    }

    #[test]
    fn opening_refuses_a_root_that_is_not_a_directory() {
        let _guard = host();
        let tree = TempTree::new();
        let file = tree.write("not-a-root", b"");
        let root = CString::new(concinnity_testing::utf8(&file)).expect("no NUL in a temp path");
        let mut byte = 0u8;
        // SAFETY: both pointers are live for the call.
        let opened = unsafe { cn_world_open(root.as_ptr(), some_view(&mut byte), 0) };
        assert_eq!(opened, CnError::InvalidArgument);
        assert_eq!(cn_world_step(), CnStep::NoWorld);
    }

    // A directory with no `data/` in it has no world to load, which fails
    // before anything reaches a GPU.
    #[test]
    fn opening_refuses_a_root_holding_no_world() {
        let _guard = host();
        let tree = TempTree::new();
        let root = CString::new(tree.root_path()).expect("a temp path holds no NUL");
        let mut byte = 0u8;
        // SAFETY: both pointers are live for the call.
        let opened = unsafe { cn_world_open(root.as_ptr(), some_view(&mut byte), 0) };
        assert_eq!(opened, CnError::NoWorldData);
        assert_eq!(cn_world_step(), CnStep::NoWorld);
    }

    // The refusal carries the load failure's own text, so a host's log says
    // what was missing and how to produce it.
    #[test]
    fn opening_a_root_holding_no_world_names_the_missing_blob() {
        let tree = TempTree::new();
        let mut byte = 0u8;
        let surface = EmbeddedSurface {
            view: NonNull::new(some_view(&mut byte)).expect("a live pointer is non-null"),
            pump_events: false,
        };

        let error = open_world(&tree.root_path(), surface).expect_err("no world was built");
        let logged = chain(&error);
        assert!(logged.contains("concinnity build"), "{logged}");
        assert!(logged.contains(&tree.root_path()), "{logged}");
        assert_eq!(code_for(&error), CnError::NoWorldData);
    }

    // A C caller compares against these integers, so they are the contract and
    // not an implementation detail. Zero is success, which is what lets a host
    // write `if (cn_world_open(...))`.
    #[test]
    fn the_error_codes_are_the_ones_the_header_publishes() {
        assert_eq!(CnError::Ok as i32, 0);
        assert_eq!(CnError::InvalidArgument as i32, 1);
        assert_eq!(CnError::NotInitialized as i32, 2);
        assert_eq!(CnError::NoWorldData as i32, 3);
        assert_eq!(CnError::UnreadableWorldData as i32, 4);
        assert_eq!(CnError::StartFailed as i32, 5);
    }

    // Data that is absent and data that is present but unusable are different
    // to a host, and flattening is where that distinction is easiest to lose.
    #[test]
    fn the_load_failures_keep_absent_and_unusable_apart() {
        let blob = std::path::PathBuf::from("/apps/MyGame/data/0");
        let unreadable = StartupError::UnreadableData {
            blob: blob.clone(),
            cause: concinnity_engine::WorldLoadError::Asset(
                concinnity_core::error::AssetError::UnknownComponent { discriminant: 9 },
            ),
        };

        assert_eq!(
            code_for(&OpenError::Load(StartupError::MissingData { blob })),
            CnError::NoWorldData
        );
        assert_eq!(
            code_for(&OpenError::Load(StartupError::NoStateRoot)),
            CnError::NoWorldData
        );
        assert_eq!(
            code_for(&OpenError::Load(unreadable)),
            CnError::UnreadableWorldData
        );
        assert_eq!(
            code_for(&OpenError::Start(WorldError::AlreadyStarted)),
            CnError::StartFailed
        );
    }

    // The chain is what the code leaves out, so a host's log has to carry it.
    #[test]
    fn the_logged_line_walks_the_whole_cause_chain() {
        let error = OpenError::Start(WorldError::DuplicateSystemName("Counter"));
        let logged = chain(&error);
        assert!(logged.contains("starting the world failed"), "{logged}");
        assert!(logged.contains("Counter"), "{logged}");
    }

    #[test]
    fn a_null_c_string_reads_as_absent() {
        // SAFETY: null is the case under test.
        assert_eq!(unsafe { ptr_to_string(std::ptr::null()) }, None);
    }

    #[test]
    fn a_c_string_reads_as_its_text() {
        let text = CString::new("hello").expect("no interior NUL");
        // SAFETY: `text` is a live NUL-terminated string that outlives the call.
        let read = unsafe { ptr_to_string(text.as_ptr()) };
        assert_eq!(read, Some("hello".to_string()));
    }
}
