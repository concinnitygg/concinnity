//! The behavior virtual machine: an authored [`Behavior`](crate::components::Behavior)
//! compiled to slot-resolved form, and the evaluator that runs it.
//!
//! [`compile`] resolves every authored name to a dense slot once, producing a
//! [`Program`]; [`exec`] runs a program's body against a [`View`] of what the
//! body may read this tick. Evaluation mutates nothing: it appends [`Effect`]s
//! the caller applies afterwards, which is what lets bodies run concurrently
//! without observing each other's writes.
//!
//! [`BehaviorSystem`] is what drives the two over a world each tick: it gathers
//! the view, runs the bodies, and applies their effects. Nothing below reads a
//! clock, a file, or a device, which is why the whole of it sits in the
//! vocabulary rather than in a host. Where persisted state is kept
//! ([`BehaviorStore`]) and how a tick's runs fan out ([`EvalScheduler`]) are the
//! host's to supply; a world with neither runs its behaviors serially and
//! persists nothing.

mod compile;
mod program;
mod run;
mod system;
mod value;

pub use compile::compile;
pub use program::{CExpr, CNode, COp, Program, VarTable};
pub use run::{Effect, SpawnEffect, View, exec};
pub use system::{
    BehaviorState, BehaviorStore, BehaviorSystem, EvalBucket, EvalScheduler, def_hash,
};
pub use value::{Arith, Cmp, Val};
