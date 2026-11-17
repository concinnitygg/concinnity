// Variables schema: the world's shared, typed variable table.

use alloc::string::String;
use alloc::vec::Vec;

use crate::components::BehaviorLiteral;

/// The world's shared variables: the state [Behavior](#behavior)s read with
/// `var` and write with `set`, and the state a `save` node persists.
///
/// Declaring this asset makes the table authoritative: every variable a
/// behavior names must appear here, and its declared value fixes both the
/// variable's type and its starting value. A world without a `Variables` asset
/// keeps every variable implicit and integer-typed, so declaring one is how a
/// world opts into typed variables and into catching misspelled names at build
/// time.
///
/// Variables are world-scoped and shared. Per-entity state belongs in a
/// behavior's `locals`, which are typed the same way but private to one entity
/// and never persisted.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct Variables {
    /// Every variable the world declares.
    pub vars: Vec<VariableDecl>,
}

/// One variable declared by the world's [Variables](#variables). The declared
/// value fixes both the variable's type and the value it holds at world start.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct VariableDecl {
    /// The name behaviors read and write the variable by.
    pub name: String,
    /// The variable's type and starting value.
    pub value: BehaviorLiteral,
}

/// A change to a world variable made from outside the behaviors, by a UI
/// action on a `var:` binding. The behavior system applies it before its next
/// tick, so a behavior watching the variable sees the change as its own.
#[derive(Debug, Clone, PartialEq)]
pub struct VariableCommand {
    /// The variable's name.
    pub name: String,
    /// What is done to it.
    pub op: crate::components::ValueOp,
}

/// The world variables' current values by name, republished by the behavior
/// system whenever one changes, for `var:` bindings to read.
#[derive(Debug, Clone, Default)]
pub struct VariableValues {
    /// Each variable's value.
    pub values: alloc::collections::BTreeMap<String, crate::components::UiValue>,
    /// A count that changes with every republish.
    pub revision: u64,
}
