// src/editor/behavior/trace.rs
//
// The editor side of execution tracing: converting the engine's cross-boundary
// forms (node paths, values) into the editor's path type and display text. The
// hook's `trace_drive` owns the per-frame exchange; nothing here touches the
// world.

use super::path::{Path, Step};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{TraceStep, TraceVal};

// A traced node's path in the editor's own path type, so it resolves to rows
// and cards through the same helpers checker faults use.
pub(crate) fn to_path(at: &[TraceStep]) -> Path {
    at.iter()
        .map(|s| match s {
            TraceStep::Field(f) => Step::Field((*f).to_string()),
            TraceStep::Index(i) => Step::Index(*i as usize),
        })
        .collect()
}

// The inverse, for publishing a breakpoint the editor holds as a path.
pub(crate) fn matches(at: &[TraceStep], path: &[Step]) -> bool {
    at.len() == path.len()
        && at.iter().zip(path).all(|(a, b)| match (a, b) {
            (TraceStep::Field(f), Step::Field(g)) => *f == g,
            (TraceStep::Index(i), Step::Index(j)) => *i as usize == *j,
            _ => false,
        })
}

// A traced value as the type word and display text the Variables panel's
// columns use, matching how declarations read.
pub(crate) fn text(val: TraceVal) -> (&'static str, String) {
    match val {
        TraceVal::Bool(b) => ("bool", b.to_string()),
        TraceVal::Int(i) => ("int", i.to_string()),
        TraceVal::Float(f) => ("float", format!("{f}")),
        TraceVal::Vec3(v) => ("vec3", format!("{}, {}, {}", v[0], v[1], v[2])),
        TraceVal::Entity(_) => ("entity", "entity".to_string()),
    }
}

// The interned id of `name`, when this build interned it (every declared asset
// is). Resolved fresh each use: ids drift across preview rebuilds, so a stored
// one could silently retarget.
pub(crate) fn id_of(name: &str) -> Option<AssetId> {
    crate::ecs::asset_id::lookup(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_round_trip_between_the_two_forms() {
        let at = [
            TraceStep::Field("do"),
            TraceStep::Index(1),
            TraceStep::Field("if"),
            TraceStep::Field("then"),
            TraceStep::Index(0),
        ];
        let path = to_path(&at);
        assert_eq!(path.len(), 5);
        assert_eq!(path[0], Step::Field("do".to_string()));
        assert_eq!(path[1], Step::Index(1));
        assert!(matches(&at, &path));
        assert!(!matches(&at, &path[..4]), "a prefix is a different node");
        let mut other = path.clone();
        other[1] = Step::Index(2);
        assert!(!matches(&at, &other));
    }

    #[test]
    fn values_read_like_declarations() {
        assert_eq!(text(TraceVal::Int(3)), ("int", "3".to_string()));
        assert_eq!(text(TraceVal::Float(12.5)), ("float", "12.5".to_string()));
        assert_eq!(text(TraceVal::Bool(true)), ("bool", "true".to_string()));
        assert_eq!(
            text(TraceVal::Vec3([0.0, 1.0, 0.0])),
            ("vec3", "0, 1, 0".to_string())
        );
    }
}
