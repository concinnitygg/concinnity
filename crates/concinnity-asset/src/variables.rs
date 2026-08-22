// Variables schema: the world's shared, typed variable table.

use alloc::string::String;
use alloc::vec::Vec;

use crate::AssetId;
use crate::behavior::BehaviorLiteral;

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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Variables {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Every variable the world declares.
    pub vars: Vec<VariableDecl>,
}

/// One variable declared by the world's [Variables](#variables). The declared
/// value fixes both the variable's type and the value it holds at world start.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct VariableDecl {
    /// The name behaviors read and write the variable by.
    pub name: String,
    /// The variable's type and starting value.
    pub value: BehaviorLiteral,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_declare_nothing() {
        assert!(Variables::default().vars.is_empty());
    }

    #[test]
    fn typed_declarations_parse() {
        let v: Variables = serde_json::from_str(
            r#"{"vars":[{"name":"health","value":{"float":100.0}},
                       {"name":"spawn","value":{"vec3":[0,1,0]}}]}"#,
        )
        .expect("variables parse");
        assert_eq!(v.vars.len(), 2);
        assert_eq!(v.vars[0].name, "health");
        assert_eq!(v.vars[0].value, BehaviorLiteral::Float(100.0));
        assert_eq!(v.vars[1].value, BehaviorLiteral::Vec3([0.0, 1.0, 0.0]));
    }

    #[test]
    fn round_trips_through_postcard() {
        let v: Variables =
            serde_json::from_str(r#"{"vars":[{"name":"n","value":{"int":3}}]}"#).unwrap();
        let bytes = postcard::to_allocvec(&v).expect("encodes");
        let back: Variables = postcard::from_bytes(&bytes).expect("decodes");
        assert_eq!(back.vars[0].name, "n");
        assert_eq!(back.vars[0].value, BehaviorLiteral::Int(3));
    }
}
