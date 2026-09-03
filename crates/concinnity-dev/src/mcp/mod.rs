//! The Model Context Protocol: the one transport the runtime debug surface
//! speaks.
//!
//! A running app serves MCP itself on its debug port (`cn debug`, or
//! `cn editor --debug-port N`) over the Streamable HTTP transport in its
//! stateless form, so any MCP client can post straight to
//! `http://127.0.0.1:8777/mcp`. `cn mcp` is the stdio entry a client that
//! spawns servers as child processes uses instead: it answers `initialize`,
//! `tools/list` and `ping` from the verb catalog, so a client connects with no
//! app running, and forwards each `tools/call` to the app.
//!
//! The tool surface is the debug protocol's own verb catalog, so this module
//! declares no commands of its own. The split below is what keeps the protocol
//! testable without a socket:
//!   jsonrpc  message parsing and response building, transport-agnostic
//!   tools    the catalog rendered as MCP tools, and the body one call carries
//!   server   the methods answered, over an injected call executor
//!   http     the app's transport: one request, one response, one connection
//!   stdio    newline-delimited framing over any byte streams
//!   app      the executor that runs a call against the live world snapshot
//!   bridge   the executor that forwards a call to a running app
//!   remote   the client that posts one JSON-RPC message to an app

mod app;
mod bridge;
mod http;
mod jsonrpc;
mod remote;
mod server;
mod stdio;
mod tools;

pub(crate) use app::AppServer;

/// Serve MCP over stdin and stdout until the client closes stdin, forwarding
/// tool calls to the app on `port`.
pub fn run(port: u16) -> std::io::Result<()> {
    eprintln!("[mcp] forwarding tool calls to {}", remote::endpoint(port));
    let server = server::Server::new(bridge::Forward::new(port));
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    stdio::serve(&server, stdin.lock(), &mut stdout)
}
