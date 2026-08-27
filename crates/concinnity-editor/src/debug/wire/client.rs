//! The client half of the runtime debug protocol: a thin localhost WebSocket
//! client the `concinnity debug <subcommand>` commands drive to talk to a
//! running `cn debug` server (see `super::server` for the other end).
//!
//! One TCP connection per request: the server answers each request on its own
//! thread and the requests are tiny, so a fresh connection per command is
//! simpler and more robust than holding a persistent socket. Every connect and
//! read is bounded by a timeout, so a gone or wedged server surfaces a clear
//! error instead of hanging. The socket-free helpers these commands use
//! (payload validation, reply inspection, the watch-target enum) live in
//! `super::super::protocol`, where they are unit-tested directly.
//!
//! Subcommands:
//!   send `<json>`      send one raw JSON command (with its own "cmd" field)
//!                      and print the reply; the escape hatch for any command
//!                      the typed helpers below do not cover
//!   screenshot `<path>`  capture the last presented frame to a PNG
//!   watch `<target>`     poll a read-only snapshot and print it until Ctrl-C

use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::{self, Message};

use crate::debug::protocol::{WatchTarget, reply_ok, validate_payload};

// Bound every connect so a missing server fails fast instead of hanging.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
// Bound every read/write so a wedged server surfaces a timeout, not a hang.
const IO_TIMEOUT: Duration = Duration::from_secs(5);

// Exit code for a transport- or protocol-level failure (cannot reach the
// server, connection dropped, malformed reply). Distinct from a logical
// failure (a well-formed `{"ok":false}` reply) so callers can tell "the
// server was unreachable" apart from "the command was rejected".
const EXIT_TRANSPORT: i32 = 3;

type WsStream = tungstenite::WebSocket<TcpStream>;

// Open a WebSocket connection to the localhost debug server on `port`.
//
// Uses an explicit connect timeout and read/write timeouts on the underlying
// socket so a dead or unresponsive server produces a clear error rather than
// blocking forever. A handshake that cannot complete within the read timeout
// is reported as a timeout (a live server answers the handshake immediately).
fn connect(port: u16) -> Result<WsStream, String> {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let stream = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT).map_err(|e| {
        format!("cannot connect to ws://127.0.0.1:{port}: {e} (is `cn debug` running?)")
    })?;
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .and_then(|()| stream.set_write_timeout(Some(IO_TIMEOUT)))
        .map_err(|e| format!("cannot configure socket timeouts: {e}"))?;

    let url = format!("ws://127.0.0.1:{port}/");
    match tungstenite::client::client(url.as_str(), stream) {
        Ok((ws, _resp)) => Ok(ws),
        Err(tungstenite::HandshakeError::Failure(e)) => {
            Err(format!("websocket handshake failed on port {port}: {e}"))
        }
        // A blocking-with-timeout socket reports a stalled handshake as
        // WouldBlock, which tungstenite surfaces as `Interrupted`. Treat it as
        // a timeout rather than retrying (a retry would just block again).
        Err(tungstenite::HandshakeError::Interrupted(_)) => Err(format!(
            "websocket handshake timed out on port {port} (>{}s)",
            IO_TIMEOUT.as_secs()
        )),
    }
}

// Read WebSocket messages until a text frame arrives, returning its payload.
fn read_text(ws: &mut WsStream) -> Result<String, String> {
    loop {
        match ws.read() {
            Ok(Message::Text(text)) => return Ok(text),
            // The server never pings, but answer one anyway to be well-behaved.
            Ok(Message::Ping(payload)) => {
                let _ = ws.send(Message::Pong(payload));
            }
            Ok(Message::Close(_)) => return Err("server closed the connection".to_string()),
            // Binary / Pong / continuation frames are never sent by the server.
            Ok(_) => {}
            Err(e) => return Err(map_read_error(e)),
        }
    }
}

// Turn a read error into a message that names a timeout as such.
fn map_read_error(e: tungstenite::Error) -> String {
    if let tungstenite::Error::Io(io) = &e
        && matches!(
            io.kind(),
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
        )
    {
        return format!("timed out waiting for reply (>{}s)", IO_TIMEOUT.as_secs());
    }
    format!("read failed: {e}")
}

// Send one JSON request and return the parsed reply.
fn request(port: u16, payload: &str) -> Result<Value, String> {
    let mut ws = connect(port)?;
    ws.send(Message::text(payload))
        .map_err(|e| format!("failed to send request: {e}"))?;
    let reply = read_text(&mut ws)?;
    // Best-effort clean close so the server logs a tidy disconnect.
    let _ = ws.close(None);
    let _ = ws.flush();
    serde_json::from_str(&reply).map_err(|e| format!("malformed reply from server: {e}"))
}

// Send a bare `{"cmd": <name>}` request and return the parsed reply.
fn request_cmd(port: u16, cmd: &str) -> Result<Value, String> {
    request(port, &json!({ "cmd": cmd }).to_string())
}

// Print a JSON value to stdout, pretty-printed when possible.
fn print_reply(reply: &Value) {
    println!(
        "{}",
        serde_json::to_string_pretty(reply).unwrap_or_else(|_| reply.to_string())
    );
}

// Print a transport error to stderr and exit with the transport code.
fn fail_transport(msg: &str) -> ! {
    eprintln!("cn debug: {msg}");
    std::process::exit(EXIT_TRANSPORT);
}

/// `concinnity debug send <json>`: send one raw JSON command and print the
/// reply. Exits 0 only when the server answered `"ok": true`, so it composes in
/// shell `&&` chains; a rejected command exits 1 and a transport failure exits 3.
pub fn send(port: u16, json: &str) -> std::io::Result<()> {
    let payload = match validate_payload(json) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("cn debug send: {msg}");
            std::process::exit(EXIT_TRANSPORT);
        }
    };
    let reply = request(port, &payload).unwrap_or_else(|msg| fail_transport(&msg));
    print_reply(&reply);
    if reply_ok(&reply) {
        Ok(())
    } else {
        std::process::exit(1);
    }
}

/// `concinnity debug screenshot <path>`: capture the last presented frame to a
/// PNG. Resolves `path` to an absolute path so the file lands where the caller
/// expects regardless of the engine's working directory. Exits 0 only on success.
pub fn screenshot(port: u16, path: &str) -> std::io::Result<()> {
    let abs = std::path::absolute(path).unwrap_or_else(|_| std::path::PathBuf::from(path));
    let abs = abs.to_string_lossy().to_string();
    let reply = request(
        port,
        &json!({ "cmd": "screenshot", "path": abs }).to_string(),
    )
    .unwrap_or_else(|msg| fail_transport(&msg));
    if reply_ok(&reply) {
        let saved = reply.get("path").and_then(Value::as_str).unwrap_or(&abs);
        println!("[screenshot] saved: {saved}");
        Ok(())
    } else {
        let err = reply
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        eprintln!("[screenshot] failed: {err}");
        std::process::exit(1);
    }
}

// Sleep for `total_ms`, but in short chunks so a Ctrl-C flag flip is noticed
// promptly instead of after a full interval.
fn sleep_interruptible(total_ms: u64, running: &AtomicBool) {
    let mut left = total_ms;
    while left > 0 && running.load(Ordering::SeqCst) {
        let chunk = left.min(100);
        std::thread::sleep(Duration::from_millis(chunk));
        left -= chunk;
    }
}

/// `concinnity debug watch <target>`: poll a read-only snapshot every
/// `interval_ms` and print each reply until Ctrl-C. A connection failure on the
/// very first poll is fatal (exit 3) -- there is nothing to watch; later
/// failures are printed and retried, so a server restart mid-session recovers.
pub fn watch(port: u16, target: WatchTarget, interval_ms: u64) -> std::io::Result<()> {
    let running = Arc::new(AtomicBool::new(true));
    let flag = Arc::clone(&running);
    // If a handler is already installed the loop still exits on process signal,
    // so an error here is not fatal.
    let _ = ctrlc::set_handler(move || flag.store(false, Ordering::SeqCst));

    let cmd = target.cmd();
    println!(
        "[watch] polling {} on ws://127.0.0.1:{port} every {interval_ms}ms (Ctrl-C to stop)",
        target.label()
    );

    let mut ever_ok = false;
    while running.load(Ordering::SeqCst) {
        match request_cmd(port, cmd) {
            Ok(reply) => {
                ever_ok = true;
                print_reply(&reply);
            }
            Err(msg) => {
                if !ever_ok {
                    fail_transport(&msg);
                }
                eprintln!("[watch] {msg}");
            }
        }
        sleep_interruptible(interval_ms, &running);
    }
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    // Candidate localhost ports for the "server is not running" tests. They sit
    // below every platform's ephemeral range so `bind(:0)` never hands one out,
    // avoiding the race where a dropped ephemeral port is reassigned to a
    // parallel test that then listens on it. Same rationale as concinnity-cli's
    // cli.rs.
    const DEAD_PORT_CANDIDATES: [u16; 4] = [28474, 28475, 28476, 28477];

    // The first candidate that promptly refuses a connection (nothing
    // listening), or `None` when the host refuses none. Callers skip when
    // `None`: the paths they assert are platform-independent and still run on
    // hosts where the connection is refused.
    fn find_dead_port() -> Option<u16> {
        DEAD_PORT_CANDIDATES
            .into_iter()
            .find(|&port| connect_refused(port))
    }

    // True when connecting to `127.0.0.1:port` is refused, i.e. nothing is
    // listening. A connect that instead times out (a dropped SYN with no RST, as
    // some firewalled loopback stacks do) returns false: the port is unused but
    // not promptly refused, so it is not a usable dead port and the caller skips
    // rather than wait out the multi-second connect timeout.
    fn connect_refused(port: u16) -> bool {
        let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
        matches!(
            TcpStream::connect_timeout(&addr, Duration::from_millis(250)),
            Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused
        )
    }

    // Stable Rust has no runtime "skipped" test state, so a test whose host
    // precondition is unmet prints this and returns (counting as passed).
    // Visible under `cargo test -- --nocapture`.
    fn skip_no_dead_port(test: &str) {
        eprintln!("[skip] {test}: no localhost port refuses connections on this host");
    }

    #[test]
    fn connect_to_dead_port_errors_fast() {
        let Some(port) = find_dead_port() else {
            return skip_no_dead_port("connect_to_dead_port_errors_fast");
        };
        // Connecting to a port nothing listens on must fail promptly with a
        // clear message rather than hanging.
        let start = Instant::now();
        let err = connect(port).expect_err("connect to a dead port must fail");
        assert!(
            start.elapsed() < CONNECT_TIMEOUT,
            "a refused connection should fail well before the connect timeout"
        );
        assert!(err.contains("cannot connect"), "unexpected error: {err}");
    }

    #[test]
    fn request_to_dead_port_errors() {
        let Some(port) = find_dead_port() else {
            return skip_no_dead_port("request_to_dead_port_errors");
        };
        assert!(request_cmd(port, "state").is_err());
    }
}
