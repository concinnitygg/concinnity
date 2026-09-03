// src/debug/wire/mod.rs
//
// The listener half of the runtime debug protocol: the accept / connection
// loop that hands each connection to the MCP server. It is the part that can
// only run against a live socket and a live engine, so it is excluded from
// coverage the same way the per-backend GPU directories are -- there is no way
// to exercise it without a running process. The socket-free logic it wraps
// (the HTTP framing, request dispatch, the snapshot data model) lives in
// `crate::mcp::http` and `super::{dispatch, state}`, where it is unit-tested
// directly.

mod server;

pub(crate) use server::DebugServer;
