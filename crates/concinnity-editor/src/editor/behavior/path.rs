// src/editor/behavior/path.rs
//
// Addressing one place inside a Behavior's args. The panel edits the authored
// JSON directly (the same value the checker reads), so every outline row
// carries the path of the value it edits and the structural edits below are
// the only thing that mutates it.

use serde_json::Value;

// One hop into a JSON value: an object key or an array index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Step {
    Field(String),
    Index(usize),
}

pub(crate) type Path = Vec<Step>;

pub(crate) fn field(name: &str) -> Step {
    Step::Field(name.to_string())
}

// `base` extended by one hop, leaving `base` untouched (rows are built by
// descending, so each level keeps its own prefix).
pub(crate) fn child(base: &[Step], step: Step) -> Path {
    let mut p = base.to_vec();
    p.push(step);
    p
}

// Whether `path` addresses `prefix` or something inside it.
pub(crate) fn starts_with(path: &[Step], prefix: &[Step]) -> bool {
    path.len() >= prefix.len() && path[..prefix.len()] == *prefix
}

pub(crate) fn get<'a>(root: &'a Value, path: &[Step]) -> Option<&'a Value> {
    let mut cur = root;
    for step in path {
        cur = match step {
            Step::Field(k) => cur.get(k)?,
            Step::Index(i) => cur.get(*i)?,
        };
    }
    Some(cur)
}

pub(crate) fn get_mut<'a>(root: &'a mut Value, path: &[Step]) -> Option<&'a mut Value> {
    let mut cur = root;
    for step in path {
        cur = match step {
            Step::Field(k) => cur.as_object_mut()?.get_mut(k)?,
            Step::Index(i) => cur.as_array_mut()?.get_mut(*i)?,
        };
    }
    Some(cur)
}

// Write `value` at `path`. A missing object key is created (an omitted
// defaulted field, e.g. an `if` that never carried an `else`); a missing array
// index or a missing parent is a no-op.
pub(crate) fn set(root: &mut Value, path: &[Step], value: Value) -> bool {
    let Some((last, parent)) = path.split_last() else {
        *root = value;
        return true;
    };
    let Some(target) = get_mut(root, parent) else {
        return false;
    };
    match last {
        Step::Field(k) => match target.as_object_mut() {
            Some(map) => {
                map.insert(k.clone(), value);
                true
            }
            None => false,
        },
        Step::Index(i) => match target.as_array_mut().and_then(|a| a.get_mut(*i)) {
            Some(slot) => {
                *slot = value;
                true
            }
            None => false,
        },
    }
}

// Append to the array at `path`, materializing it when the key is absent or
// null (a list the authored JSON left out entirely).
pub(crate) fn push(root: &mut Value, path: &[Step], value: Value) -> bool {
    if get(root, path).is_none_or(Value::is_null) && !set(root, path, Value::Array(Vec::new())) {
        return false;
    }
    match get_mut(root, path).and_then(|v| v.as_array_mut()) {
        Some(a) => {
            a.push(value);
            true
        }
        None => false,
    }
}

// Drop the array element at `path`. Only array elements are removed: every
// removable outline row is a list member, and an object field is cleared by
// writing null instead.
pub(crate) fn remove(root: &mut Value, path: &[Step]) -> bool {
    let Some((Step::Index(i), parent)) = path.split_last() else {
        return false;
    };
    match get_mut(root, parent).and_then(|v| v.as_array_mut()) {
        Some(a) if *i < a.len() => {
            a.remove(*i);
            true
        }
        _ => false,
    }
}

// Move the array element at `path` by `delta` places, clamped to the array.
// Returns the new index, or `None` when it could not move.
pub(crate) fn shift(root: &mut Value, path: &[Step], delta: isize) -> Option<usize> {
    let (Step::Index(i), parent) = path.split_last()? else {
        return None;
    };
    let array = get_mut(root, parent)?.as_array_mut()?;
    let to = usize::try_from(*i as isize + delta).ok()?;
    if *i >= array.len() || to >= array.len() {
        return None;
    }
    array.swap(*i, to);
    Some(to)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample() -> Value {
        json!({"do": [{"set": {"var": "a"}}, {"save": null}], "scope": ["Prop"]})
    }

    #[test]
    fn get_walks_fields_and_indices() {
        let v = sample();
        let p = vec![field("do"), Step::Index(0), field("set"), field("var")];
        assert_eq!(get(&v, &p).and_then(Value::as_str), Some("a"));
        assert!(get(&v, &[field("nope")]).is_none());
        assert!(get(&v, &[field("do"), Step::Index(9)]).is_none());
    }

    #[test]
    fn set_writes_a_leaf_and_creates_a_missing_key() {
        let mut v = sample();
        assert!(set(
            &mut v,
            &[field("do"), Step::Index(0), field("set"), field("var")],
            json!("b")
        ));
        assert_eq!(v["do"][0]["set"]["var"], json!("b"));
        // An absent object key is created rather than refused.
        assert!(set(&mut v, &[field("once")], json!(true)));
        assert_eq!(v["once"], json!(true));
        // A missing array index is not.
        assert!(!set(&mut v, &[field("scope"), Step::Index(4)], json!("x")));
    }

    #[test]
    fn push_materializes_a_missing_list() {
        let mut v = json!({});
        assert!(push(&mut v, &[field("do")], json!({"save": null})));
        assert!(push(&mut v, &[field("do")], json!({"save": null})));
        assert_eq!(v["do"].as_array().unwrap().len(), 2);
        // A null-valued key is treated the same as an absent one.
        let mut nulled = json!({"queries": null});
        assert!(push(&mut nulled, &[field("queries")], json!({"name": "q"})));
        assert_eq!(nulled["queries"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn remove_drops_an_array_element_only() {
        let mut v = sample();
        assert!(remove(&mut v, &[field("do"), Step::Index(0)]));
        assert_eq!(v["do"].as_array().unwrap().len(), 1);
        assert!(v["do"][0].get("save").is_some());
        // Out of range, and object fields, are refused.
        assert!(!remove(&mut v, &[field("do"), Step::Index(7)]));
        assert!(!remove(&mut v, &[field("scope")]));
    }

    #[test]
    fn shift_swaps_within_the_array_and_stops_at_its_ends() {
        let mut v = sample();
        assert_eq!(shift(&mut v, &[field("do"), Step::Index(1)], -1), Some(0));
        assert!(v["do"][0].get("save").is_some());
        // Past either end nothing moves.
        assert_eq!(shift(&mut v, &[field("do"), Step::Index(0)], -1), None);
        assert_eq!(shift(&mut v, &[field("do"), Step::Index(1)], 1), None);
        assert_eq!(v["do"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn starts_with_matches_a_path_and_what_is_inside_it() {
        let node = vec![field("do"), Step::Index(0)];
        assert!(starts_with(&node, &node), "a path is inside itself");
        assert!(starts_with(&child(&node, field("cond")), &node));
        assert!(!starts_with(&node, &child(&node, field("cond"))));
        assert!(!starts_with(&[field("do"), Step::Index(1)], &node));
    }

    #[test]
    fn child_extends_without_disturbing_the_prefix() {
        let base = vec![field("do")];
        let a = child(&base, Step::Index(0));
        let b = child(&base, Step::Index(1));
        assert_eq!(base.len(), 1);
        assert_ne!(a, b);
    }
}
