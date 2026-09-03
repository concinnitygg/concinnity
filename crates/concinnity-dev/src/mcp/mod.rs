//! The Model Context Protocol bridge: `cn mcp`.
//!
//! A stdio child process, not a listener. An MCP client spawns it, speaks
//! JSON-RPC 2.0 over stdin and stdout, and every `tools/call` is forwarded to a
//! running app over the localhost WebSocket the `cn debug` subcommands already
//! use. The engine gains nothing: no component, no setting, no shipped code.
//! The tool surface is the debug protocol's own verb catalog, so this file
//! declares no commands of its own.
//!
//! The split below is what keeps the protocol testable without a socket:
//!   jsonrpc  message parsing and response building, transport-agnostic
//!   tools    the catalog rendered as MCP tools, and the request one call sends
//!   server   the methods answered, over an injected executor
//!   stdio    newline-delimited framing over any byte streams
//!   bridge   the real executor, one WebSocket round trip per call

mod bridge;
mod jsonrpc;
mod server;
mod stdio;
mod tools;

/// Serve MCP over stdin and stdout until the client closes stdin, forwarding
/// tool calls to the debug server on `port`.
pub fn run(port: u16) -> std::io::Result<()> {
    eprintln!("[mcp] serving the debug protocol for ws://127.0.0.1:{port}");
    let server = server::Server::new(|payload: &str| bridge::execute(port, payload));
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    stdio::serve(&server, stdin.lock(), &mut stdout)
}
