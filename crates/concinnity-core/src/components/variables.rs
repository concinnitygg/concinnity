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
