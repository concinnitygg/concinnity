// src/editor/behavior/palette.rs
//
// The closed vocabulary a behavior is built from: its event sources, its
// nodes, and its expressions, each with the JSON a fresh one starts as. The
// schema fixes this set (`concinnity-core/src/components/behavior.rs`), so the panel
// offers a fixed palette rather than free text and the checker never sees an
// invented verb.

use serde_json::{Value, json};

// What an expression's body looks like. Replacing a verb with one of the same
// shape keeps the operands already built, so stepping `lt` to `gt` (or `add`
// to `sub`) does not throw the subtree away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    // A bare string with no body: `self`, `dt`, `elapsed`.
    Unit,
    // A typed constant, whose payload the value field edits.
    Literal,
    // A single name: a variable, local, binding, asset, or query.
    Name,
    // One nested expression.
    Unary,
    // Exactly two nested expressions.
    Binary,
    // Any number of nested expressions.
    List,
}

// One offerable verb: the JSON key, and the hint shown beside it.
pub(crate) struct Entry {
    pub verb: &'static str,
    pub hint: &'static str,
}

// The event sources, in the order the source row cycles through them.
pub(crate) const SOURCES: &[Entry] = &[
    Entry {
        verb: "start",
        hint: "once, when the world starts",
    },
    Entry {
        verb: "tick",
        hint: "every frame",
    },
    Entry {
        verb: "timer",
        hint: "after an interval, once or repeating",
    },
    Entry {
        verb: "variable",
        hint: "whenever a world variable changes",
    },
    Entry {
        verb: "enter",
        hint: "on entering a trigger volume",
    },
    Entry {
        verb: "exit",
        hint: "on leaving a trigger volume",
    },
    Entry {
        verb: "interact",
        hint: "on interacting with an entity",
    },
    Entry {
        verb: "spawned",
        hint: "the tick after an entity spawns",
    },
];

// The nodes, grouped flow / state / lifecycle / world.
pub(crate) const NODES: &[Entry] = &[
    Entry {
        verb: "if",
        hint: "run nodes when a condition holds",
    },
    Entry {
        verb: "for_each",
        hint: "run nodes per entity a query matched",
    },
    Entry {
        verb: "let",
        hint: "bind an expression to a name",
    },
    Entry {
        verb: "set",
        hint: "write a world variable",
    },
    Entry {
        verb: "set_local",
        hint: "write a per-entity local",
    },
    Entry {
        verb: "set_transform",
        hint: "move, rotate, or scale an entity",
    },
    Entry {
        verb: "spawn",
        hint: "copy a placement into the world",
    },
    Entry {
        verb: "despawn",
        hint: "remove an entity and its children",
    },
    Entry {
        verb: "reparent",
        hint: "re-point an entity's parent",
    },
    Entry {
        verb: "show",
        hint: "reveal a hidden entity",
    },
    Entry {
        verb: "hide",
        hint: "hide an entity without removing it",
    },
    Entry {
        verb: "sound",
        hint: "play an audio clip",
    },
    Entry {
        verb: "scene",
        hint: "jump to a scene",
    },
    Entry {
        verb: "screen",
        hint: "show a screen",
    },
    Entry {
        verb: "story",
        hint: "start or continue the story",
    },
    Entry {
        verb: "save",
        hint: "persist world variables and once state",
    },
];

// The expressions, constants first, then reads, then operators.
pub(crate) const EXPRS: &[Entry] = &[
    Entry {
        verb: "bool",
        hint: "a true / false constant",
    },
    Entry {
        verb: "int",
        hint: "a whole-number constant",
    },
    Entry {
        verb: "float",
        hint: "a decimal constant",
    },
    Entry {
        verb: "vec3",
        hint: "a three-component constant",
    },
    Entry {
        verb: "var",
        hint: "read a world variable",
    },
    Entry {
        verb: "local",
        hint: "read a per-entity local",
    },
    Entry {
        verb: "bind",
        hint: "read a bound name",
    },
    Entry {
        verb: "named",
        hint: "an entity declared in the world",
    },
    Entry {
        verb: "self",
        hint: "the entity this behavior runs for",
    },
    Entry {
        verb: "dt",
        hint: "seconds since the previous tick",
    },
    Entry {
        verb: "elapsed",
        hint: "seconds since the world started",
    },
    Entry {
        verb: "position",
        hint: "an entity's world position",
    },
    Entry {
        verb: "distance",
        hint: "distance between two entities",
    },
    Entry {
        verb: "alive",
        hint: "whether an entity still exists",
    },
    Entry {
        verb: "first",
        hint: "the first entity a query matched",
    },
    Entry {
        verb: "count",
        hint: "how many entities a query matched",
    },
    Entry {
        verb: "add",
        hint: "sum",
    },
    Entry {
        verb: "sub",
        hint: "difference",
    },
    Entry {
        verb: "mul",
        hint: "product",
    },
    Entry {
        verb: "div",
        hint: "quotient",
    },
    Entry {
        verb: "normalize",
        hint: "a vector rescaled to unit length",
    },
    Entry {
        verb: "eq",
        hint: "equal",
    },
    Entry {
        verb: "ne",
        hint: "not equal",
    },
    Entry {
        verb: "lt",
        hint: "less than",
    },
    Entry {
        verb: "le",
        hint: "less than or equal",
    },
    Entry {
        verb: "gt",
        hint: "greater than",
    },
    Entry {
        verb: "ge",
        hint: "greater than or equal",
    },
    Entry {
        verb: "all",
        hint: "every condition holds",
    },
    Entry {
        verb: "any",
        hint: "some condition holds",
    },
    Entry {
        verb: "not",
        hint: "logical negation",
    },
];

// The typed constants, offered where only a literal fits (a local's declared
// value).
pub(crate) const LITERALS: &[Entry] = &[
    Entry {
        verb: "bool",
        hint: "true / false",
    },
    Entry {
        verb: "int",
        hint: "a whole number",
    },
    Entry {
        verb: "float",
        hint: "a decimal",
    },
    Entry {
        verb: "vec3",
        hint: "three components",
    },
];

// Whether `verb` is one of the vocabulary's own, so authored JSON naming
// something else is shown as-is instead of being given a subtree it has no
// schema for.
pub(crate) fn known(entries: &[Entry], verb: &str) -> bool {
    entries.iter().any(|e| e.verb == verb)
}

// The single key of a node / expression object, or the whole word when the
// value is a bare string (`"self"`, `"tick"`).
pub(crate) fn verb_of(value: &Value) -> &str {
    if let Some(word) = value.as_str() {
        return word;
    }
    match value.as_object() {
        Some(map) if map.len() == 1 => map.keys().next().map(String::as_str).unwrap_or(""),
        _ => "",
    }
}

pub(crate) fn shape(verb: &str) -> Shape {
    match verb {
        "self" | "dt" | "elapsed" => Shape::Unit,
        "bool" | "int" | "float" | "vec3" => Shape::Literal,
        "var" | "local" | "bind" | "named" | "first" | "count" => Shape::Name,
        "position" | "alive" | "normalize" | "not" => Shape::Unary,
        "all" | "any" => Shape::List,
        _ => Shape::Binary,
    }
}

// A single-key object, the shape every node, expression, and parameterized
// source takes.
pub(crate) fn single(verb: &str, body: Value) -> Value {
    let mut map = serde_json::Map::new();
    map.insert(verb.to_string(), body);
    Value::Object(map)
}

pub(crate) fn source_default(verb: &str) -> Value {
    match verb {
        "timer" => json!({"timer": {"interval": 1.0, "repeat": false}}),
        "variable" => json!({"variable": ""}),
        "enter" | "exit" | "interact" => single(verb, Value::Null),
        other => json!(other),
    }
}

pub(crate) fn node_default(verb: &str) -> Value {
    match verb {
        "if" => json!({"if": {"cond": {"bool": true}, "then": [], "else": []}}),
        "for_each" => json!({"for_each": {"query": "", "bind": "each", "do": []}}),
        "let" => json!({"let": {"name": "value", "value": {"int": 0}}}),
        "set" => json!({"set": {"var": "", "value": {"int": 0}, "add": false}}),
        "set_local" => json!({"set_local": {"local": "", "value": {"int": 0}, "add": false}}),
        "set_transform" => json!({"set_transform": {
            "entity": "self", "position": null, "rotation_deg": null, "scale": null}}),
        "spawn" => json!({"spawn": {
            "template": null, "position": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0], "lifetime": 0.0, "bind": null}}),
        "despawn" | "show" | "hide" => single(verb, json!({"target": "self"})),
        "reparent" => json!({"reparent": {"child": "self", "parent": null}}),
        "sound" => json!({"sound": {"clip": null, "kind": "sound", "volume": 1.0}}),
        "scene" => json!({"scene": {"scene": null, "transition": "FadeBlack"}}),
        "screen" => json!({"screen": {"screen": null}}),
        "story" => json!({"story": "start"}),
        _ => single("save", Value::Null),
    }
}

// A typed constant of `verb` holding that type's zero, for a slot that has to
// start somewhere.
pub(crate) fn literal_default(verb: &str) -> Value {
    single(verb, literal_body(&Value::Null, verb))
}

pub(crate) fn expr_default(verb: &str) -> Value {
    match shape(verb) {
        Shape::Unit => json!(verb),
        Shape::Literal => literal_default(verb),
        Shape::Name => single(verb, json!("")),
        Shape::Unary => single(verb, unary_operand(verb)),
        Shape::List => single(verb, json!([])),
        Shape::Binary => single(verb, binary_operands(verb)),
    }
}

// A fresh unary operand of the type the verb needs, so a newly picked
// expression type-checks instead of arriving broken.
fn unary_operand(verb: &str) -> Value {
    match verb {
        "position" | "alive" => json!("self"),
        "normalize" => json!({"vec3": [0.0, 0.0, 0.0]}),
        _ => json!({"bool": true}),
    }
}

fn binary_operands(verb: &str) -> Value {
    match verb {
        "distance" => json!(["self", "self"]),
        _ => json!([{"float": 0.0}, {"float": 0.0}]),
    }
}

// Replace the verb of the expression `current`, keeping its operands when the
// new verb takes the same shape.
pub(crate) fn swap_expr(current: &Value, verb: &str) -> Value {
    let (from, to) = (shape(verb_of(current)), shape(verb));
    if from != to {
        return expr_default(verb);
    }
    let body = body_of(current);
    match to {
        Shape::Unit => json!(verb),
        Shape::Literal => single(verb, literal_body(body.unwrap_or(&Value::Null), verb)),
        _ => single(verb, body.cloned().unwrap_or(Value::Null)),
    }
}

// Replace the type of the literal `current`, carrying its number across when
// both types hold one.
pub(crate) fn swap_literal(current: &Value, verb: &str) -> Value {
    let body = body_of(current);
    single(verb, literal_body(body.unwrap_or(&Value::Null), verb))
}

// The body under a single-key object; `None` for a bare-string expression or
// anything malformed.
pub(crate) fn body_of(value: &Value) -> Option<&Value> {
    match value.as_object() {
        Some(map) if map.len() == 1 => map.values().next(),
        _ => None,
    }
}

// A literal's payload retyped as `verb`: a number carries across the numeric
// types, and anything else starts from zero.
fn literal_body(from: &Value, verb: &str) -> Value {
    let number = from.as_f64();
    match verb {
        "bool" => json!(from.as_bool().unwrap_or(number.is_some_and(|n| n != 0.0))),
        "int" => json!(number.unwrap_or(0.0) as i64),
        "float" => json!(number.unwrap_or(0.0)),
        _ => match from.as_array() {
            Some(a) if a.len() == 3 => from.clone(),
            _ => json!([0.0, 0.0, 0.0]),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every offered verb produces JSON the schema parses back into the type it
    // names, so nothing the palette inserts can be a shape the runtime rejects.
    #[test]
    fn every_node_default_parses_as_a_node() {
        for entry in NODES {
            let v = node_default(entry.verb);
            assert_eq!(verb_of(&v), entry.verb);
            serde_json::from_value::<crate::components::BehaviorNode>(v)
                .unwrap_or_else(|e| panic!("node `{}` does not parse: {e}", entry.verb));
        }
    }

    #[test]
    fn every_expr_default_parses_as_an_expression() {
        for entry in EXPRS {
            let v = expr_default(entry.verb);
            assert_eq!(verb_of(&v), entry.verb);
            serde_json::from_value::<crate::components::BehaviorExpr>(v)
                .unwrap_or_else(|e| panic!("expression `{}` does not parse: {e}", entry.verb));
        }
    }

    #[test]
    fn every_source_default_parses_as_a_source() {
        for entry in SOURCES {
            let v = source_default(entry.verb);
            serde_json::from_value::<crate::components::BehaviorSource>(v)
                .unwrap_or_else(|e| panic!("source `{}` does not parse: {e}", entry.verb));
        }
    }

    // A freshly picked expression is well typed on its own terms, so the
    // status line reports what the user still has to fill in, not the palette's
    // own leftovers.
    #[test]
    fn fresh_operands_match_the_operator() {
        assert_eq!(
            expr_default("position"),
            serde_json::json!({"position": "self"})
        );
        assert_eq!(
            expr_default("normalize"),
            serde_json::json!({"normalize": {"vec3": [0.0, 0.0, 0.0]}})
        );
        assert_eq!(
            expr_default("distance"),
            serde_json::json!({"distance": ["self", "self"]})
        );
        assert_eq!(
            expr_default("not"),
            serde_json::json!({"not": {"bool": true}})
        );
    }

    #[test]
    fn swapping_within_a_shape_keeps_the_operands() {
        let lt = serde_json::json!({"lt": [{"float": 1.0}, {"float": 2.0}]});
        assert_eq!(
            swap_expr(&lt, "ge"),
            serde_json::json!({"ge": [{"float": 1.0}, {"float": 2.0}]})
        );
        let var = serde_json::json!({"var": "health"});
        assert_eq!(
            swap_expr(&var, "local"),
            serde_json::json!({"local": "health"})
        );
    }

    #[test]
    fn swapping_across_shapes_starts_from_the_default() {
        let lt = serde_json::json!({"lt": [{"float": 1.0}, {"float": 2.0}]});
        assert_eq!(swap_expr(&lt, "not"), expr_default("not"));
        assert_eq!(swap_expr(&lt, "self"), serde_json::json!("self"));
    }

    #[test]
    fn literals_carry_their_number_across_the_numeric_types() {
        let f = serde_json::json!({"float": 2.75});
        assert_eq!(swap_literal(&f, "int"), serde_json::json!({"int": 2}));
        assert_eq!(swap_literal(&f, "bool"), serde_json::json!({"bool": true}));
        let v = serde_json::json!({"vec3": [1.0, 2.0, 3.0]});
        assert_eq!(swap_literal(&v, "vec3"), v);
        assert_eq!(swap_literal(&v, "float"), serde_json::json!({"float": 0.0}));
    }

    #[test]
    fn verb_of_reads_bare_words_and_single_key_objects() {
        assert_eq!(verb_of(&serde_json::json!("self")), "self");
        assert_eq!(verb_of(&serde_json::json!({"int": 3})), "int");
        assert_eq!(verb_of(&serde_json::json!({"a": 1, "b": 2})), "");
        assert_eq!(verb_of(&Value::Null), "");
    }

    // Every entry carries a hint, so no palette row draws a bare verb.
    #[test]
    fn every_entry_has_a_hint() {
        for entries in [SOURCES, NODES, EXPRS, LITERALS] {
            for e in entries {
                assert!(!e.hint.is_empty(), "`{}` has no hint", e.verb);
            }
        }
    }
}
