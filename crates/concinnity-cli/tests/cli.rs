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

// A localhost port that reliably refuses connections. A connect is only refused
// promptly (ECONNREFUSED) when nothing is bound to the port; the older trick of
// binding an ephemeral port and dropping it races, because the kernel can hand
// that just-freed port to a parallel test that then listens on it, so the CLI
// connects instead of being refused. These candidates sit below every platform's
// ephemeral range (Linux from 32768, macOS/Windows from 49152), which `bind(:0)`
// never allocates, so no concurrent test can occupy one. Return the first that is
// currently refused, both to confirm nothing is listening and to skip a port some
// unrelated local service happens to hold.
fn dead_port() -> String {
    const CANDIDATES: [u16; 4] = [28470, 28471, 28472, 28473];
    for port in CANDIDATES {
        if connect_refused(port) {
            return port.to_string();
        }
    }
    panic!("no refusing localhost port among {CANDIDATES:?}");
}

// True when connecting to `127.0.0.1:port` is refused, i.e. nothing is listening.
fn connect_refused(port: u16) -> bool {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    match TcpStream::connect_timeout(&addr, Duration::from_millis(250)) {
        Ok(_) => false,
        Err(e) => e.kind() == ErrorKind::ConnectionRefused,
    }
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
