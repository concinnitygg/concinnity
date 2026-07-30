// src/editor/behavior/edit.rs
//
// The edits an outline row answers: picking a verb from the palette, typing a
// value, and removing or reordering a list member. Every one rewrites the
// authored args in place, so what the checker reads afterwards is exactly what
// the panel shows.
//
// Selecting a row never changes it. Every field, down to a boolean, offers its
// options through the palette instead, so a value only ever moves when the user
// picks one or types one.

use serde_json::{Map, Value, json};

use super::outline::{self, Kind, List, Row, Text};
use super::palette::{self, Entry, Shape};
use super::path;

// The palette entry that clears an optional expression slot back to unset.
pub(crate) const UNSET: &str = "";

// One offerable palette option: the verb inserted, and the hint drawn beside it.
pub(crate) struct Pick {
    pub verb: &'static str,
    pub hint: String,
}

// The two options a boolean field offers, worded as the outline draws its value.
const FLAG_PICKS: [&str; 2] = ["true", "false"];

// What the selected row's Pick button offers. An empty list means the row has
// no palette (a plain text field, whose value is typed). `components` is the
// registered component vocabulary, supplied by the caller so this module stays
// clear of the registry.
pub(crate) fn picks(kind: &Kind, components: &[&'static str]) -> Vec<Pick> {
    match kind {
        Kind::Source => from(palette::SOURCES),
        Kind::Flag => words(&FLAG_PICKS),
        Kind::Choice(options) => words(options),
        Kind::Node | Kind::List(List::Nodes) => from(palette::NODES),
        Kind::List(List::Operands) => from(palette::EXPRS),
        Kind::Expr { optional } => {
            let mut picks = from(palette::EXPRS);
            if *optional {
                picks.insert(
                    0,
                    Pick {
                        verb: UNSET,
                        hint: "leave this field unchanged".to_string(),
                    },
                );
            }
            picks
        }
        Kind::Literal | Kind::List(List::Locals) => from(palette::LITERALS),
        Kind::List(List::Scope) | Kind::List(List::Components) => components
            .iter()
            .map(|c| Pick {
                verb: c,
                hint: String::new(),
            })
            .collect(),
        Kind::List(List::Queries) => vec![Pick {
            verb: "query",
            hint: "a new world read".to_string(),
        }],
        Kind::Text(_) => Vec::new(),
    }
}

// The caption a pick draws: the verb, or the word for the clear-to-unset entry.
pub(crate) fn pick_caption(pick: &Pick) -> &str {
    if pick.verb == UNSET {
        "unset"
    } else {
        pick.verb
    }
}

fn from(entries: &'static [Entry]) -> Vec<Pick> {
    entries
        .iter()
        .map(|e| Pick {
            verb: e.verb,
            hint: e.hint.to_string(),
        })
        .collect()
}

// A fixed word set offered as itself: the word is both the option and the value.
fn words(options: &[&'static str]) -> Vec<Pick> {
    options
        .iter()
        .map(|w| Pick {
            verb: w,
            hint: String::new(),
        })
        .collect()
}

// Insert (or replace with) the palette verb the user picked.
pub(crate) fn apply_pick(args: &mut Value, row: &Row, verb: &str) -> bool {
    match &row.kind {
        // Replacing a source drops the old one's parameters for the new one's
        // defaults, because no two source shapes share a field.
        Kind::Source => path::set(args, &row.path, palette::source_default(verb)),
        Kind::Flag => path::set(args, &row.path, Value::Bool(verb == FLAG_PICKS[0])),
        Kind::Choice(_) => path::set(args, &row.path, Value::String(verb.to_string())),
        Kind::Node => path::set(args, &row.path, palette::node_default(verb)),
        Kind::Expr { .. } if verb == UNSET => path::set(args, &row.path, Value::Null),
        Kind::Expr { .. } => {
            let current = path::get(args, &row.path).cloned().unwrap_or(Value::Null);
            path::set(args, &row.path, palette::swap_expr(&current, verb))
        }
        Kind::Literal => {
            let current = path::get(args, &row.path).cloned().unwrap_or(Value::Null);
            path::set(args, &row.path, palette::swap_literal(&current, verb))
        }
        Kind::List(List::Nodes) => path::push(args, &row.path, palette::node_default(verb)),
        Kind::List(List::Operands) => path::push(args, &row.path, palette::expr_default(verb)),
        Kind::List(List::Scope) | Kind::List(List::Components) => {
            path::push(args, &row.path, Value::String(verb.to_string()))
        }
        Kind::List(List::Locals) => {
            let name = unique_decl_name(args, "locals", "local");
            let mut decl = Map::new();
            decl.insert("name".to_string(), Value::String(name));
            decl.insert("value".to_string(), palette::expr_default(verb));
            path::push(args, &row.path, Value::Object(decl))
        }
        Kind::List(List::Queries) => {
            let name = unique_decl_name(args, "queries", "query");
            path::push(args, &row.path, json!({"name": name, "has": []}))
        }
        Kind::Text(_) => false,
    }
}

// The text the value field seeds with for `row`, or `None` when the row has no
// typed value (a flag, a fixed word, or an operator with only operands).
pub(crate) fn text_value(args: &Value, row: &Row) -> Option<String> {
    match &row.kind {
        Kind::Text(_) => Some(row.value.clone()),
        Kind::Literal => Some(outline::literal_text(payload(args, row))),
        Kind::Expr { .. } => {
            let value = path::get(args, &row.path)?;
            match palette::shape(palette::verb_of(value)) {
                Shape::Literal => Some(outline::literal_text(palette::body_of(value))),
                Shape::Name => Some(name_of(palette::body_of(value))),
                _ => None,
            }
        }
        Kind::Node | Kind::Source | Kind::Flag | Kind::Choice(_) | Kind::List(_) => None,
    }
}

// Commit the value field's text into `row`. The error is the message the panel
// shows on its status line, phrased the way the checker's are.
pub(crate) fn apply_text(args: &mut Value, row: &Row, text: &str) -> Result<(), String> {
    let text = text.trim();
    let value = match &row.kind {
        Kind::Text(Text::Str) => Value::String(text.to_string()),
        Kind::Text(Text::OptStr) => optional_string(text),
        Kind::Text(Text::Num) => Value::from(number(text)?),
        Kind::Text(Text::Vec3) => vec3(text)?,
        Kind::Literal | Kind::Expr { .. } => {
            let current = path::get(args, &row.path).cloned().unwrap_or(Value::Null);
            let verb = palette::verb_of(&current);
            match palette::shape(verb) {
                Shape::Literal => palette::single(verb, literal_payload(verb, text)?),
                Shape::Name => palette::single(verb, Value::String(text.to_string())),
                _ => return Err("this row has no value to type".to_string()),
            }
        }
        Kind::Node | Kind::Source | Kind::Flag | Kind::Choice(_) | Kind::List(_) => {
            return Err("this row has no value to type".to_string());
        }
    };
    if path::set(args, &row.path, value) {
        Ok(())
    } else {
        Err("that field is no longer there".to_string())
    }
}

// A typed constant built from what was typed for it: the same parse the value
// field does for a `Literal` row, for callers editing a literal that is not part
// of a behavior body (the world's variable table).
pub(crate) fn literal(verb: &str, text: &str) -> Result<Value, String> {
    Ok(palette::single(verb, literal_payload(verb, text.trim())?))
}

pub(crate) fn remove(args: &mut Value, row: &Row) -> bool {
    row.element
        .as_ref()
        .is_some_and(|element| path::remove(args, element))
}

// Move the selected member by `delta` places, returning where it landed so the
// caller can keep the selection on it.
pub(crate) fn shift(args: &mut Value, row: &Row, delta: isize) -> Option<path::Path> {
    let element = row.element.as_ref()?;
    let at = path::shift(args, element, delta)?;
    let mut moved = element.clone();
    *moved.last_mut()? = path::Step::Index(at);
    Some(moved)
}

// `base`, `base_2`, ... until it does not collide with a sibling declaration.
fn unique_decl_name(args: &Value, key: &str, base: &str) -> String {
    let taken: Vec<&str> = args
        .get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|d| d.get("name").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();
    if !taken.contains(&base) {
        return base.to_string();
    }
    let mut i = 2;
    loop {
        let candidate = format!("{base}_{i}");
        if !taken.contains(&candidate.as_str()) {
            return candidate;
        }
        i += 1;
    }
}

fn payload<'a>(args: &'a Value, row: &Row) -> Option<&'a Value> {
    palette::body_of(path::get(args, &row.path)?)
}

fn name_of(body: Option<&Value>) -> String {
    body.and_then(Value::as_str).unwrap_or("").to_string()
}

fn optional_string(text: &str) -> Value {
    if text.is_empty() {
        Value::Null
    } else {
        Value::String(text.to_string())
    }
}

fn number(text: &str) -> Result<f64, String> {
    text.parse::<f64>()
        .map_err(|_| format!("'{text}' is not a number"))
}

fn vec3(text: &str) -> Result<Value, String> {
    let parts: Vec<&str> = text
        .split([',', ' '])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() != 3 {
        return Err(format!("'{text}' needs three numbers, e.g. 0, 1, 0"));
    }
    let mut out = Vec::with_capacity(3);
    for part in parts {
        out.push(Value::from(number(part)?));
    }
    Ok(Value::Array(out))
}

fn literal_payload(verb: &str, text: &str) -> Result<Value, String> {
    match verb {
        "bool" => match text {
            "true" | "1" => Ok(Value::Bool(true)),
            "false" | "0" => Ok(Value::Bool(false)),
            _ => Err(format!("'{text}' is not true or false")),
        },
        "int" => text
            .parse::<i64>()
            .map(Value::from)
            .map_err(|_| format!("'{text}' is not a whole number")),
        "float" => number(text).map(Value::from),
        _ => vec3(text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::behavior::outline::rows;

    fn row_named<'a>(rows: &'a [Row], label: &str) -> &'a Row {
        rows.iter().find(|r| r.label == label).expect("row exists")
    }

    fn edited(mut args: Value, label: &str, edit: impl Fn(&mut Value, &Row)) -> Value {
        let row = row_named(&rows(&args), label).clone();
        edit(&mut args, &row);
        args
    }

    // The fields that used to change on the click that selected them: each now
    // offers its options, and only a pick moves the value.
    #[test]
    fn every_fixed_field_offers_its_options() {
        let offered = |kind| -> Vec<&'static str> {
            picks(&kind, &["Prop"]).iter().map(|p| p.verb).collect()
        };
        let sources: Vec<&str> = palette::SOURCES.iter().map(|e| e.verb).collect();
        assert_eq!(
            offered(Kind::Source),
            sources,
            "the whole source vocabulary"
        );
        assert_eq!(offered(Kind::Flag), ["true", "false"]);
        assert_eq!(
            offered(Kind::Choice(outline::CUE_KINDS)),
            ["sound", "music"]
        );
        assert!(
            offered(Kind::Text(Text::Str)).is_empty(),
            "a typed field has no palette"
        );
    }

    #[test]
    fn picking_a_source_brings_its_parameters() {
        let picked = edited(json!({"on": "start"}), "on", |a, r| {
            apply_pick(a, r, "timer");
        });
        assert_eq!(
            picked["on"],
            json!({"timer": {"interval": 1.0, "repeat": false}})
        );
    }

    #[test]
    fn picking_sets_a_flag_and_a_fixed_word() {
        let set = edited(json!({"once": false}), "once", |a, r| {
            apply_pick(a, r, "true");
        });
        assert_eq!(set["once"], json!(true));
        let cleared = edited(set, "once", |a, r| {
            apply_pick(a, r, "false");
        });
        assert_eq!(
            cleared["once"],
            json!(false),
            "picking again is not a toggle"
        );

        let args = json!({"do": [{"sound": {"clip": "c", "kind": "sound", "volume": 1.0}}]});
        let picked = edited(args, "kind", |a, r| {
            apply_pick(a, r, "music");
        });
        assert_eq!(picked["do"][0]["sound"]["kind"], json!("music"));
    }

    // Picking from a list appends; picking on a slot replaces in place.
    #[test]
    fn picking_appends_to_a_list_and_replaces_a_slot() {
        let appended = edited(json!({}), "do", |a, r| {
            assert!(apply_pick(a, r, "save"));
        });
        assert_eq!(appended["do"], json!([{"save": null}]));

        let args = json!({"do": [{"despawn": {"target": "self"}}]});
        let replaced = edited(args, "despawn", |a, r| {
            assert!(apply_pick(a, r, "hide"));
        });
        assert_eq!(replaced["do"], json!([{"hide": {"target": "self"}}]));
    }

    #[test]
    fn picking_a_declaration_names_it_uniquely() {
        let mut args = json!({});
        for _ in 0..3 {
            let row = row_named(&rows(&args), "locals").clone();
            assert!(apply_pick(&mut args, &row, "float"));
        }
        let names: Vec<&str> = args["locals"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["local", "local_2", "local_3"]);
        assert_eq!(args["locals"][0]["value"], json!({"float": 0.0}));
    }

    #[test]
    fn an_optional_slot_offers_unset_and_a_required_one_does_not() {
        let args =
            json!({"do": [{"set_transform": {"entity": "self", "scale": {"vec3": [1, 1, 1]}}}]});
        let all = rows(&args);
        let optional = picks(&row_named(&all, "scale").kind, &[]);
        assert_eq!(optional.first().map(|p| p.verb), Some(UNSET));
        let required = picks(&row_named(&all, "entity").kind, &[]);
        assert_ne!(required.first().map(|p| p.verb), Some(UNSET));

        let cleared = edited(args, "scale", |a, r| {
            assert!(apply_pick(a, r, UNSET));
        });
        assert_eq!(cleared["do"][0]["set_transform"]["scale"], Value::Null);
    }

    #[test]
    fn typed_text_lands_in_the_right_json_type() {
        let args = json!({"do": [{"spawn": {"template": null, "position": [0, 0, 0],
            "rotation_deg": [0, 0, 0], "scale": [1, 1, 1], "lifetime": 0.0, "bind": null}}]});
        let named = edited(args.clone(), "template", |a, r| {
            apply_text(a, r, "crate").expect("a name commits");
        });
        assert_eq!(named["do"][0]["spawn"]["template"], json!("crate"));
        // Emptying an optional name clears it rather than storing "".
        let cleared = edited(named, "template", |a, r| {
            apply_text(a, r, "  ").expect("an empty name clears");
        });
        assert_eq!(cleared["do"][0]["spawn"]["template"], Value::Null);

        let moved = edited(args.clone(), "position", |a, r| {
            apply_text(a, r, "1, 2.5, -3").expect("a vector commits");
        });
        assert_eq!(moved["do"][0]["spawn"]["position"], json!([1.0, 2.5, -3.0]));

        let timed = edited(args, "lifetime", |a, r| {
            apply_text(a, r, "2.5").expect("a number commits");
        });
        assert_eq!(timed["do"][0]["spawn"]["lifetime"], json!(2.5));
    }

    #[test]
    fn a_mistyped_value_is_reported_and_changes_nothing() {
        let mut args = json!({"delay": 1.0});
        let row = row_named(&rows(&args), "delay").clone();
        let e = apply_text(&mut args, &row, "soon").expect_err("not a number");
        assert!(e.contains("'soon' is not a number"), "{e}");
        assert_eq!(args["delay"], json!(1.0), "the old value stands");

        let mut args = json!({"do": [{"spawn": {"position": [0, 0, 0]}}]});
        let row = row_named(&rows(&args), "position").clone();
        let e = apply_text(&mut args, &row, "1, 2").expect_err("needs three");
        assert!(e.contains("three numbers"), "{e}");
    }

    // A literal expression takes its payload from the value field, and its type
    // from the palette; a nested operator has no text of its own.
    #[test]
    fn literal_and_named_expressions_carry_a_typed_payload() {
        let args = json!({"do": [{"let": {"name": "n", "value": {"float": 1.5}}}]});
        let all = rows(&args);
        let value = row_named(&all, "value").clone();
        assert_eq!(text_value(&args, &value).as_deref(), Some("1.5"));

        let mut retyped = args.clone();
        assert!(apply_pick(&mut retyped, &value, "int"));
        assert_eq!(retyped["do"][0]["let"]["value"], json!({"int": 1}));
        apply_text(&mut retyped, &value, "7").expect("an int commits");
        assert_eq!(retyped["do"][0]["let"]["value"], json!({"int": 7}));

        let mut named = args;
        assert!(apply_pick(&mut named, &value, "var"));
        let value = row_named(&rows(&named), "value").clone();
        apply_text(&mut named, &value, "health").expect("a name commits");
        assert_eq!(named["do"][0]["let"]["value"], json!({"var": "health"}));

        let ops =
            json!({"do": [{"let": {"name": "n", "value": {"add": [{"int": 1}, {"int": 2}]}}}]});
        let row = row_named(&rows(&ops), "value").clone();
        assert_eq!(text_value(&ops, &row), None, "an operator has no payload");
    }

    #[test]
    fn members_delete_and_reorder_while_fixed_operands_do_not() {
        let args = json!({"do": [{"save": null}, {"hide": {"target": "self"}}]});
        let all = rows(&args);
        let hide = row_named(&all, "hide").clone();

        let mut moved = args.clone();
        assert_eq!(
            shift(&mut moved, &hide, -1),
            Some(vec![path::field("do"), path::Step::Index(0)]),
            "the selection follows the member it moved"
        );
        assert!(moved["do"][0].get("hide").is_some());

        let mut removed = args.clone();
        assert!(remove(&mut removed, &hide));
        assert_eq!(removed["do"].as_array().unwrap().len(), 1);

        // A binary operand is part of its operator's fixed arity, so it has no
        // element to delete.
        let mut ops = json!({"do": [{"if": {"cond": {"lt": [{"int": 1}, {"int": 2}]}}}]});
        let operand = row_named(&rows(&ops), "a").clone();
        assert!(operand.element.is_none());
        assert!(!remove(&mut ops, &operand));
        assert!(shift(&mut ops, &operand, 1).is_none());
    }

    #[test]
    fn component_picks_come_from_the_supplied_vocabulary() {
        let args = json!({});
        let scope = row_named(&rows(&args), "scope").clone();
        let picks = picks(&scope.kind, &["Prop", "Camera3D"]);
        assert_eq!(
            picks.iter().map(|p| p.verb).collect::<Vec<_>>(),
            ["Prop", "Camera3D"]
        );
        let mut args = args;
        assert!(apply_pick(&mut args, &scope, "Prop"));
        assert_eq!(args["scope"], json!(["Prop"]));
    }

    #[test]
    fn pick_caption_words_the_clear_entry() {
        assert_eq!(
            pick_caption(&Pick {
                verb: UNSET,
                hint: String::new()
            }),
            "unset"
        );
        assert_eq!(
            pick_caption(&Pick {
                verb: "if",
                hint: String::new()
            }),
            "if"
        );
    }
}
