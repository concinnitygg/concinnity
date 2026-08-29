// src/debug/wire/mod.rs
//
// The WebSocket transport for the runtime debug protocol: the server accept /
// connection loop and the client that drives it. These modules are the parts
// that can only run against a live socket (and, for the server, a live engine),
// so they are grouped here and excluded from coverage the same way the
// per-backend GPU directories are -- there is no way to exercise them without a
// running process. The socket-free logic they wrap (request dispatch, payload
// validation, the snapshot data model) lives one level up in
// `super::{dispatch, protocol, state}`, where it is unit-tested directly.

pub mod client;
mod server;

pub(crate) use server::DebugServer;
