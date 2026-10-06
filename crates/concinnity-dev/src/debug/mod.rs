//! The localhost runtime debug server of the concinnity-dev library, declared in
//! `lib.rs`.
//!
//! `cn debug` starts a localhost MCP server (see `crate::mcp`). The engine
//! stays debug-agnostic: the only coupling is the `FrameHook` trait, which the
//! run loop invokes once per frame on the main thread (see
//! `crate::frame_hook`). `DebugServer::tick` snapshots the live world into
//! shared state; connection threads answer calls from that snapshot.
//!
//! Every verb is one MCP tool, called by name with its parameters as the tool
//! arguments, and declared once in `self::verbs` beside its handler: its
//! description, whether it reads or mutates, and its typed parameters.
//! `self::catalog` collects them into the table every call goes through. A
//! call is checked against the parameters before the handler runs. A verb that
//! reads answers from the snapshot; one that needs the world, the animation
//! system or the render backend queues a closure the per-frame debug drive runs
//! on the engine thread, and blocks on its result for at most a few frames.
//!
//! A reply is a JSON object: `{"ok":true,...}` with the verb's fields, or
//! `{"ok":false,"error":"..."}`.

// Submodules:
//   wire       the listener and the per-frame drive; coverage-excluded
//   catalog    the verb table and the entry point that answers a call
//   verb       the vocabulary a verb is declared in: params, schema, checks
//   verbs      the verbs themselves, grouped by what they reach
//   call       what a handler is handed: the snapshot and the engine queue
//   queue      the jobs handlers queue for the engine thread
//   state      the shared world-snapshot data model
//   hot_reload asset / shader / world.jsonl reload machinery
//
// The `wire` submodule holds everything that can only run against a live socket
// and a live engine. It is excluded from coverage like the per-backend GPU
// directories; the logic it wraps lives in the modules above and in
// `crate::mcp`, which are unit-tested without a running process.
mod call;
pub(crate) mod catalog;
pub(crate) mod hot_reload;
mod memory;
mod queue;
pub(crate) mod state;
pub(crate) mod verb;
mod verbs;
mod wire;

pub(crate) use wire::DebugServer;
