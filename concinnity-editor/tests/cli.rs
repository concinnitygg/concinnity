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

use std::net::TcpListener;
use std::process::{Command, Output};

// Path to the freshly built binary, provided by Cargo to integration tests.
const BIN: &str = env!("CARGO_BIN_EXE_concinnity");

// A localhost port with nothing listening: bind an ephemeral port, learn its
// number, then drop the listener so a connect there is refused immediately.
fn dead_port() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port.to_string()
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
    // Validation fails before any socket work, so the port is irrelevant.
    let out = run(&["debug", "send", "{not json", "--port", &dead_port()]);
    assert_eq!(out.status.code(), Some(3));
}

#[test]
fn debug_send_dead_port_exits_transport() {
    let out = run(&[
        "debug",
        "send",
        r#"{"cmd":"state"}"#,
        "--port",
        &dead_port(),
    ]);
    assert_eq!(out.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot connect"));
}

#[test]
fn debug_smoke_dead_port_reports_loop_never_started() {
    let out = run(&["debug", "smoke", "--wait", "1", "--port", &dead_port()]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn debug_watch_dead_port_exits_transport_on_first_poll() {
    let out = run(&["debug", "watch", "camera", "--port", &dead_port()]);
    assert_eq!(out.status.code(), Some(3));
}
