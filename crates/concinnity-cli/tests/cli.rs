// tests/cli.rs
//
// End-to-end tests that drive the built `concinnity` binary through the
// exit-code paths that never touch the renderer: `--help`, a missing
// subcommand, and the `cn debug` client commands pointed at a dead server. They
// exercise `fn main()` and the command dispatch (which the in-crate unit tests
// cannot, since those never run the binary). Under `cargo llvm-cov` the profile
// data the spawned binary writes on exit is merged, so this coverage counts.
//
// Only the non-engine paths are driven here; `cn run` and a bare `cn debug`
// stand up a renderer + window and are verified by screenshot probes instead.

use std::io::ErrorKind;
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::process::{Command, Output};
use std::time::Duration;

// Path to the freshly built binary, provided by Cargo to integration tests.
const BIN: &str = env!("CARGO_BIN_EXE_concinnity");

// Candidate localhost ports for the "server is not running" tests. They sit
// below every platform's ephemeral range (Linux from 32768, macOS/Windows from
// 49152), which `bind(:0)` never allocates, so no concurrent test can occupy
// one -- avoiding the race in the older bind-an-ephemeral-port-then-drop-it
// trick, where the kernel could hand the freed port to a parallel test that
// then listens on it.
const DEAD_PORT_CANDIDATES: [u16; 4] = [28470, 28471, 28472, 28473];

// The first candidate that promptly refuses a connection (nothing listening),
// or `None` when the host refuses none. Callers skip when `None`: the exit-code
// paths they assert are platform-independent and still run on hosts where the
// connection is refused.
fn find_dead_port() -> Option<String> {
    DEAD_PORT_CANDIDATES
        .into_iter()
        .find(|&port| connect_refused(port))
        .map(|port| port.to_string())
}

// True when connecting to `127.0.0.1:port` is refused, i.e. nothing is
// listening. A connect that instead times out (a dropped SYN with no RST, as
// some firewalled loopback stacks do) returns false: the port is unused but not
// promptly refused, so it is not a usable dead port and the caller skips rather
// than wait out the client's multi-second connect timeout on every test.
fn connect_refused(port: u16) -> bool {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    match TcpStream::connect_timeout(&addr, Duration::from_millis(250)) {
        Ok(_) => false,
        Err(e) => e.kind() == ErrorKind::ConnectionRefused,
    }
}

// Stable Rust has no runtime "skipped" test state, so a test whose host
// precondition is unmet prints this and returns (counting as passed). Visible
// under `cargo test -- --nocapture`.
fn skip_no_dead_port(test: &str) {
    eprintln!("[skip] {test}: no localhost port refuses connections on this host");
}

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("spawn concinnity binary")
}

#[test]
fn help_exits_zero() {
    let out = run(&["--help"]);
    assert!(out.status.success(), "--help should exit 0");
    assert!(String::from_utf8_lossy(&out.stdout).contains("Usage"));
}

#[test]
fn missing_subcommand_is_a_usage_error() {
    let out = run(&[]);
    assert!(
        !out.status.success(),
        "a bare invocation should be a usage error"
    );
}

#[test]
fn debug_send_invalid_json_exits_transport() {
    // Validation fails before any socket work, so the port is never used (any
    // value works; a dead port is not required, so this never skips).
    let port = DEAD_PORT_CANDIDATES[0].to_string();
    let out = run(&["debug", "send", "{not json", "--port", &port]);
    assert_eq!(out.status.code(), Some(3));
}

#[test]
fn debug_send_dead_port_exits_transport() {
    let Some(port) = find_dead_port() else {
        return skip_no_dead_port("debug_send_dead_port_exits_transport");
    };
    let out = run(&["debug", "send", r#"{"cmd":"state"}"#, "--port", &port]);
    assert_eq!(out.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot connect"));
}

#[test]
fn debug_smoke_dead_port_reports_loop_never_started() {
    let Some(port) = find_dead_port() else {
        return skip_no_dead_port("debug_smoke_dead_port_reports_loop_never_started");
    };
    let out = run(&["debug", "smoke", "--wait", "1", "--port", &port]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn debug_watch_dead_port_exits_transport_on_first_poll() {
    let Some(port) = find_dead_port() else {
        return skip_no_dead_port("debug_watch_dead_port_exits_transport_on_first_poll");
    };
    let out = run(&["debug", "watch", "camera", "--port", &port]);
    assert_eq!(out.status.code(), Some(3));
}
