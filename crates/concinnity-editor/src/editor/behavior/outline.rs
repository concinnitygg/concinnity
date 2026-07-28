// src/editor/behavior/outline.rs
//
// Flattening a Behavior's args into the panel's row list. Everything the asset
// declares -- its source, its scope, locals, queries, and the whole node body
// with its expression trees -- becomes one indented outline, so a node graph
// and its operands are browsed and edited the same way. Each row carries the
// path of the value it edits and, when it is a list member, the path the
// delete / reorder buttons act on.

use serde_json::Value;

use super::palette::{self, Shape};
use super::path::{Path, Step, child, field};

// How a row's text is typed back into the args when the value field commits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Text {
    // A plain string: a variable, local, binding, or component name.
    Str,
    // A string that clears to null when emptied: an optional asset name.
    OptStr,
    // A number.
    Num,
    // Three comma-separated numbers.
    Vec3,
}

// The list a row's palette appends to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum List {
    Nodes,
    Scope,
    Locals,
    Queries,
    // The component names one query matches on.
    Components,
    // The operands of an `all` / `any` expression.
    Operands,
}

// What a row offers: which clicks it answers, and which palette (if any) its
// Pick button opens.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Kind {
    // The `on` source; clicking steps to the next one.
    Source,
    // A boolean; clicking flips it.
    Flag,
    // A fixed word set; clicking steps to the next option.
    Choice(&'static [&'static str]),
    // Edited through the value field.
    Text(Text),
    // A typed constant: the palette picks the type, the value field the payload.
    Literal,
    // A slot holding one node; the palette replaces it.
    Node,
    // A slot holding one expression; the palette replaces it, and a literal or
    // named expression also takes its payload from the value field. An optional
    // slot can be cleared back to unset.
    Expr { optional: bool },
    // A list the palette appends to.
    List(List),
}

// The fixed word sets a `Choice` row steps through.
pub(crate) const CUE_KINDS: &[&str] = &["sound", "music"];
pub(crate) const TRANSITIONS: &[&str] = &["FadeBlack", "Cut"];
pub(crate) const PLAYBACKS: &[&str] = &["start", "continue"];

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Row {
    pub depth: usize,
    pub label: String,
    pub value: String,
    pub kind: Kind,
    pub path: Path,
    // The list element the delete / reorder buttons act on, when this row is
    // one. A node's is its own path; a local's name row points at the whole
    // local, so deleting from the name deletes the declaration.
    pub element: Option<Path>,
}

impl Row {
    fn new(depth: usize, label: &str, value: impl Into<String>, kind: Kind, path: Path) -> Row {
        Row {
            depth,
            label: label.to_string(),
            value: value.into(),
            kind,
            path,
            element: None,
        }
    }

    fn member(mut self, element: Path) -> Row {
        self.element = Some(element);
        self
    }
}

// The whole behavior as one outline.
pub(crate) fn rows(args: &Value) -> Vec<Row> {
    let mut out = Vec::new();
    source_rows(args, &mut out);
    out.push(Row::new(
        0,
        "once",
        bool_text(args.get("once")),
        Kind::Flag,
        vec![field("once")],
    ));
    for key in ["delay", "cooldown"] {
        out.push(Row::new(
            0,
            key,
            num_text(args.get(key)),
            Kind::Text(Text::Num),
            vec![field(key)],
        ));
    }
    scope_rows(args, &mut out);
    local_rows(args, &mut out);
    query_rows(args, &mut out);
    node_list(args.get("do"), 0, vec![field("do")], "do", &mut out);
    out
}

fn source_rows(args: &Value, out: &mut Vec<Row>) {
    let on = args.get("on");
    let verb = on.map_or("start", palette::verb_of);
    let base = vec![field("on")];
    out.push(Row::new(
        0,
        "on",
        source_summary(verb, on),
        Kind::Source,
        base.clone(),
    ));
    let body = on.and_then(palette::body_of);
    match verb {
        "timer" => {
            let timer = child(&base, field("timer"));
            out.push(Row::new(
                1,
                "interval",
                num_text(body.and_then(|b| b.get("interval"))),
                Kind::Text(Text::Num),
                child(&timer, field("interval")),
            ));
            out.push(Row::new(
                1,
                "repeat",
                bool_text(body.and_then(|b| b.get("repeat"))),
                Kind::Flag,
                child(&timer, field("repeat")),
            ));
        }
        "variable" => out.push(Row::new(
            1,
            "name",
            str_text(body),
            Kind::Text(Text::Str),
            child(&base, field("variable")),
        )),
        "enter" | "exit" => out.push(Row::new(
            1,
            "volume",
            str_text(body),
            Kind::Text(Text::OptStr),
            child(&base, field(verb)),
        )),
        "interact" => out.push(Row::new(
            1,
            "entity",
            str_text(body),
            Kind::Text(Text::OptStr),
            child(&base, field("interact")),
        )),
        _ => {}
    }
}

fn scope_rows(args: &Value, out: &mut Vec<Row>) {
    let base = vec![field("scope")];
    let items = array(args.get("scope"));
    out.push(Row::new(
        0,
        "scope",
        count_text(items.len()),
        Kind::List(List::Scope),
        base.clone(),
    ));
    for (i, item) in items.iter().enumerate() {
        let p = child(&base, Step::Index(i));
        out.push(
            Row::new(
                1,
                "component",
                str_text(Some(item)),
                Kind::Text(Text::Str),
                p.clone(),
            )
            .member(p),
        );
    }
}

fn local_rows(args: &Value, out: &mut Vec<Row>) {
    let base = vec![field("locals")];
    let items = array(args.get("locals"));
    out.push(Row::new(
        0,
        "locals",
        count_text(items.len()),
        Kind::List(List::Locals),
        base.clone(),
    ));
    for (i, item) in items.iter().enumerate() {
        let element = child(&base, Step::Index(i));
        out.push(
            Row::new(
                1,
                "local",
                str_text(item.get("name")),
                Kind::Text(Text::Str),
                child(&element, field("name")),
            )
            .member(element.clone()),
        );
        let value = item.get("value");
        out.push(Row::new(
            2,
            "value",
            value.map_or_else(|| "unset".to_string(), expr_summary),
            Kind::Literal,
            child(&element, field("value")),
        ));
    }
}

fn query_rows(args: &Value, out: &mut Vec<Row>) {
    let base = vec![field("queries")];
    let items = array(args.get("queries"));
    out.push(Row::new(
        0,
        "queries",
        count_text(items.len()),
        Kind::List(List::Queries),
        base.clone(),
    ));
    for (i, item) in items.iter().enumerate() {
        let element = child(&base, Step::Index(i));
        out.push(
            Row::new(
                1,
                "query",
                str_text(item.get("name")),
                Kind::Text(Text::Str),
                child(&element, field("name")),
            )
            .member(element.clone()),
        );
        let has = child(&element, field("has"));
        let components = array(item.get("has"));
        out.push(Row::new(
            2,
            "has",
            count_text(components.len()),
            Kind::List(List::Components),
            has.clone(),
        ));
        for (j, component) in components.iter().enumerate() {
            let p = child(&has, Step::Index(j));
            out.push(
                Row::new(
                    3,
                    "component",
                    str_text(Some(component)),
                    Kind::Text(Text::Str),
                    p.clone(),
                )
                .member(p),
            );
        }
    }
}

// A node list (`do`, an `if` branch, a `for_each` body): its heading row and
// every node under it.
fn node_list(value: Option<&Value>, depth: usize, path: Path, label: &str, out: &mut Vec<Row>) {
    let items = array(value);
    out.push(Row::new(
        depth,
        label,
        count_text(items.len()),
        Kind::List(List::Nodes),
        path.clone(),
    ));
    for (i, node) in items.iter().enumerate() {
        node_rows(node, depth + 1, child(&path, Step::Index(i)), out);
    }
}

// The fields of one node: its body, the path that body sits at, and the depth
// its rows indent to. Every field is keyed by the name it carries in the schema,
// which is also the label its row draws.
struct Fields<'a> {
    body: Option<&'a Value>,
    base: Path,
    depth: usize,
}

impl<'a> Fields<'a> {
    fn at(&self, key: &str) -> Option<&'a Value> {
        self.body.and_then(|v| v.get(key))
    }

    fn path(&self, key: &str) -> Path {
        child(&self.base, field(key))
    }

    fn text(&self, out: &mut Vec<Row>, key: &str, t: Text) {
        let value = self.at(key);
        let text = match t {
            Text::Num => num_text(value),
            Text::Vec3 => vec3_text(value),
            Text::Str | Text::OptStr => str_text(value),
        };
        out.push(Row::new(
            self.depth,
            key,
            text,
            Kind::Text(t),
            self.path(key),
        ));
    }

    fn expr(&self, out: &mut Vec<Row>, key: &str, optional: bool) {
        expr_rows(self.at(key), key, self.depth, self.path(key), optional, out);
    }

    fn nodes(&self, out: &mut Vec<Row>, key: &str) {
        node_list(self.at(key), self.depth, self.path(key), key, out);
    }

    fn flag(&self, out: &mut Vec<Row>, key: &str) {
        let text = bool_text(self.at(key));
        out.push(Row::new(self.depth, key, text, Kind::Flag, self.path(key)));
    }

    fn choice(&self, out: &mut Vec<Row>, key: &str, options: &'static [&'static str]) {
        let text = word_text(self.at(key), options);
        let kind = Kind::Choice(options);
        out.push(Row::new(self.depth, key, text, kind, self.path(key)));
    }
}

fn node_rows(node: &Value, depth: usize, path: Path, out: &mut Vec<Row>) {
    let verb = palette::verb_of(node);
    if !palette::known(palette::NODES, verb) {
        out.push(Row::new(depth, "node", "(unknown)", Kind::Node, path.clone()).member(path));
        return;
    }
    let body = palette::body_of(node);
    let summary = node_summary(verb, body);
    out.push(Row::new(depth, verb, summary, Kind::Node, path.clone()).member(path.clone()));
    let f = Fields {
        body,
        base: child(&path, field(verb)),
        depth: depth + 1,
    };
    match verb {
        "if" => {
            f.expr(out, "cond", false);
            f.nodes(out, "then");
            f.nodes(out, "else");
        }
        "for_each" => {
            f.text(out, "query", Text::Str);
            f.text(out, "bind", Text::Str);
            f.nodes(out, "do");
        }
        "let" => {
            f.text(out, "name", Text::Str);
            f.expr(out, "value", false);
        }
        "set" | "set_local" => {
            f.text(out, if verb == "set" { "var" } else { "local" }, Text::Str);
            f.expr(out, "value", false);
            f.flag(out, "add");
        }
        "set_transform" => {
            f.expr(out, "entity", false);
            for key in ["position", "rotation_deg", "scale"] {
                f.expr(out, key, true);
            }
        }
        "spawn" => {
            f.text(out, "template", Text::OptStr);
            for key in ["position", "rotation_deg", "scale"] {
                f.text(out, key, Text::Vec3);
            }
            f.text(out, "lifetime", Text::Num);
            f.text(out, "bind", Text::OptStr);
        }
        "despawn" | "show" | "hide" => f.expr(out, "target", false),
        "reparent" => {
            f.expr(out, "child", false);
            f.expr(out, "parent", true);
        }
        "sound" => {
            f.text(out, "clip", Text::OptStr);
            f.choice(out, "kind", CUE_KINDS);
            f.text(out, "volume", Text::Num);
        }
        "scene" => {
            f.text(out, "scene", Text::OptStr);
            f.choice(out, "transition", TRANSITIONS);
        }
        "screen" => f.text(out, "screen", Text::OptStr),
        // The playback command is the node's whole body, not a field of it.
        "story" => {
            let text = word_text(body, PLAYBACKS);
            let kind = Kind::Choice(PLAYBACKS);
            out.push(Row::new(f.depth, "playback", text, kind, f.base));
        }
        _ => {}
    }
}

// One expression slot and, when it nests, the operands under it.
fn expr_rows(
    value: Option<&Value>,
    label: &str,
    depth: usize,
    path: Path,
    optional: bool,
    out: &mut Vec<Row>,
) {
    let kind = Kind::Expr { optional };
    let Some(value) = value.filter(|v| !v.is_null()) else {
        out.push(Row::new(depth, label, "unset", kind, path));
        return;
    };
    let verb = palette::verb_of(value);
    out.push(Row::new(
        depth,
        label,
        expr_summary(value),
        kind,
        path.clone(),
    ));
    if !palette::known(palette::EXPRS, verb) {
        return;
    }
    let body = palette::body_of(value);
    let b = child(&path, field(verb));
    let d = depth + 1;
    match palette::shape(verb) {
        Shape::Unary => expr_rows(body, "a", d, b, false, out),
        Shape::Binary => {
            for (i, name) in ["a", "b"].iter().enumerate() {
                let operand = body.and_then(|v| v.get(i));
                expr_rows(operand, name, d, child(&b, Step::Index(i)), false, out);
            }
        }
        Shape::List => {
            let items = array(body);
            out.push(Row::new(
                d,
                "of",
                count_text(items.len()),
                Kind::List(List::Operands),
                b.clone(),
            ));
            for (i, operand) in items.iter().enumerate() {
                let p = child(&b, Step::Index(i));
                let before = out.len();
                expr_rows(Some(operand), &i.to_string(), d + 1, p.clone(), false, out);
                out[before].element = Some(p);
            }
        }
        Shape::Unit | Shape::Literal | Shape::Name => {}
    }
}

// The right-column detail of a source row: the verb plus whatever parameter
// makes one source distinguishable from another of the same kind.
pub(crate) fn source_summary(verb: &str, on: Option<&Value>) -> String {
    let body = on.and_then(palette::body_of);
    match verb {
        "timer" => {
            let interval = num_text(body.and_then(|b| b.get("interval")));
            let repeat = body
                .and_then(|b| b.get("repeat"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            format!(
                "timer {interval}s{}",
                if repeat { " repeating" } else { "" }
            )
        }
        "variable" | "enter" | "exit" | "interact" => {
            let name = str_text(body);
            if name.is_empty() {
                format!("{verb} (unnamed)")
            } else {
                format!("{verb} {name}")
            }
        }
        other if palette::known(palette::SOURCES, other) => other.to_string(),
        _ => "(unknown)".to_string(),
    }
}

// The right-column detail of a node row: the one field that identifies it at a
// glance, so a collapsed read of the body still says what each node touches.
fn node_summary(verb: &str, body: Option<&Value>) -> String {
    let at = |key: &str| body.and_then(|v| v.get(key));
    let name = |key: &str| str_text(at(key));
    let target = |key: &str| at(key).map(expr_summary).unwrap_or_default();
    match verb {
        "for_each" => format!("{} -> {}", name("query"), name("bind")),
        "let" | "set_local" | "spawn" | "scene" | "screen" | "sound" => name(first_field(verb)),
        "set" => name("var"),
        "set_transform" => target("entity"),
        "despawn" | "show" | "hide" => target("target"),
        "reparent" => target("child"),
        "story" => body.and_then(Value::as_str).unwrap_or("start").to_string(),
        _ => String::new(),
    }
}

// The field a node is summarized by, for the verbs whose summary is one name.
fn first_field(verb: &str) -> &'static str {
    match verb {
        "let" => "name",
        "set_local" => "local",
        "spawn" => "template",
        "scene" => "scene",
        "screen" => "screen",
        _ => "clip",
    }
}

// An expression as one line: the verb, plus its payload for the leaf shapes
// that carry one.
pub(crate) fn expr_summary(value: &Value) -> String {
    let verb = palette::verb_of(value);
    if !palette::known(palette::EXPRS, verb) {
        return "(unknown)".to_string();
    }
    match palette::shape(verb) {
        Shape::Literal => format!("{verb} {}", literal_text(palette::body_of(value))),
        Shape::Name => {
            let name = str_text(palette::body_of(value));
            if name.is_empty() {
                format!("{verb} (unnamed)")
            } else {
                format!("{verb} {name}")
            }
        }
        _ => verb.to_string(),
    }
}

// A literal's payload as text: what the value field seeds with and commits.
pub(crate) fn literal_text(body: Option<&Value>) -> String {
    match body {
        Some(Value::Bool(b)) => b.to_string(),
        Some(Value::Array(_)) => vec3_text(body),
        Some(Value::Number(_)) => num_text(body),
        _ => String::new(),
    }
}

fn array(value: Option<&Value>) -> &[Value] {
    value
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

fn count_text(n: usize) -> String {
    format!("({n})")
}

fn bool_text(value: Option<&Value>) -> String {
    value.and_then(Value::as_bool).unwrap_or(false).to_string()
}

fn str_text(value: Option<&Value>) -> String {
    value.and_then(Value::as_str).unwrap_or("").to_string()
}

// A fixed-word field's current value, falling back to the first option so a
// missing or malformed word still shows something the row can step from.
fn word_text(value: Option<&Value>, options: &[&str]) -> String {
    let current = str_text(value);
    if options.contains(&current.as_str()) {
        current
    } else {
        options.first().unwrap_or(&"").to_string()
    }
}

fn num_text(value: Option<&Value>) -> String {
    fmt_num(value.and_then(Value::as_f64).unwrap_or(0.0))
}

fn vec3_text(value: Option<&Value>) -> String {
    let a = array(value);
    let at = |i: usize| fmt_num(a.get(i).and_then(Value::as_f64).unwrap_or(0.0));
    format!("{}, {}, {}", at(0), at(1), at(2))
}

// Numbers read back the way they were typed: a whole value keeps no trailing
// `.0`, so re-committing a row does not churn the authored JSON.
pub(crate) fn fmt_num(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1.0e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn find<'a>(rows: &'a [Row], label: &str) -> &'a Row {
        rows.iter()
            .find(|r| r.label == label)
            .unwrap_or_else(|| panic!("no `{label}` row in {:?}", labels(rows)))
    }

    fn labels(rows: &[Row]) -> Vec<&str> {
        rows.iter().map(|r| r.label.as_str()).collect()
    }

    // Every behavior, however empty, shows the same declaration skeleton, so
    // there is always somewhere to start authoring from.
    #[test]
    fn a_blank_behavior_still_offers_every_section() {
        let rows = rows(&json!({}));
        for label in [
            "on", "once", "delay", "cooldown", "scope", "locals", "queries", "do",
        ] {
            let r = find(&rows, label);
            assert_eq!(r.depth, 0, "`{label}` is a top-level row");
        }
        assert_eq!(find(&rows, "on").value, "start");
        assert_eq!(find(&rows, "do").value, "(0)");
        assert_eq!(find(&rows, "do").kind, Kind::List(List::Nodes));
    }

    #[test]
    fn a_timer_source_discloses_its_interval_and_repeat() {
        let rows = rows(&json!({"on": {"timer": {"interval": 5.0, "repeat": true}}}));
        assert_eq!(find(&rows, "on").value, "timer 5s repeating");
        let interval = find(&rows, "interval");
        assert_eq!(interval.value, "5");
        assert_eq!(interval.kind, Kind::Text(Text::Num));
        assert_eq!(
            interval.path,
            vec![field("on"), field("timer"), field("interval")]
        );
        assert_eq!(find(&rows, "repeat").kind, Kind::Flag);
        // A source with no parameters discloses nothing.
        let tick = super::rows(&json!({"on": "tick"}));
        assert!(!labels(&tick).contains(&"interval"));
    }

    #[test]
    fn declarations_nest_their_parts_under_a_deletable_member() {
        let rows = rows(&json!({
            "scope": ["Prop"],
            "locals": [{"name": "speed", "value": {"float": 3.0}}],
            "queries": [{"name": "player", "has": ["Camera3D"]}],
        }));
        let component = find(&rows, "component");
        assert_eq!(component.value, "Prop");
        assert_eq!(
            component.element,
            Some(vec![field("scope"), Step::Index(0)])
        );

        let local = find(&rows, "local");
        assert_eq!(local.value, "speed");
        assert_eq!(
            local.path,
            vec![field("locals"), Step::Index(0), field("name")]
        );
        assert_eq!(
            local.element,
            Some(vec![field("locals"), Step::Index(0)]),
            "deleting from the name row deletes the whole declaration"
        );
        let value = find(&rows, "value");
        assert_eq!(value.kind, Kind::Literal);
        assert_eq!(value.value, "float 3");

        let query = find(&rows, "query");
        assert_eq!(query.value, "player");
        assert_eq!(find(&rows, "has").kind, Kind::List(List::Components));
    }

    // The body flattens into an indented tree: a node's fields sit under it and
    // an expression's operands under those.
    #[test]
    fn the_body_flattens_into_an_indented_tree() {
        let rows = rows(&json!({"scope": ["Prop"], "do": [
            {"if": {"cond": {"lt": [{"distance": ["self", "self"]}, {"float": 20.0}]},
                    "then": [{"save": null}], "else": []}}]}));
        let body = find(&rows, "do");
        assert_eq!(body.value, "(1)");
        let branch = find(&rows, "if");
        assert_eq!(branch.depth, body.depth + 1);
        assert_eq!(branch.kind, Kind::Node);
        assert_eq!(branch.element, Some(vec![field("do"), Step::Index(0)]));

        let cond = find(&rows, "cond");
        assert_eq!(cond.depth, branch.depth + 1);
        assert_eq!(cond.value, "lt");
        // Both operands are addressed by index under the verb, and the nested
        // comparison's own operands sit a level deeper again.
        let under_cond = |steps: &[Step]| {
            let mut p = vec![field("do"), Step::Index(0), field("if"), field("cond")];
            p.extend_from_slice(steps);
            rows.iter().find(|r| r.path == p).expect("row at that path")
        };
        let lhs = under_cond(&[field("lt"), Step::Index(0)]);
        assert_eq!((lhs.label.as_str(), lhs.value.as_str()), ("a", "distance"));
        assert_eq!(lhs.depth, cond.depth + 1);
        let rhs = under_cond(&[field("lt"), Step::Index(1)]);
        assert_eq!((rhs.label.as_str(), rhs.value.as_str()), ("b", "float 20"));
        let inner = under_cond(&[
            field("lt"),
            Step::Index(0),
            field("distance"),
            Step::Index(1),
        ]);
        assert_eq!(inner.value, "self");
        assert_eq!(inner.depth, lhs.depth + 1);
        // Both branches are listed, the empty one included, so a node can be
        // added to either.
        let branches: Vec<&Row> = rows
            .iter()
            .filter(|r| r.label == "then" || r.label == "else")
            .collect();
        assert_eq!(branches.len(), 2);
        assert_eq!(branches[0].value, "(1)");
        assert_eq!(branches[1].value, "(0)");
    }

    #[test]
    fn optional_transform_fields_read_as_unset() {
        let rows = rows(&json!({"do": [
            {"set_transform": {"entity": "self", "position": {"vec3": [0, 1, 0]}}}]}));
        assert_eq!(find(&rows, "position").value, "vec3 0, 1, 0");
        assert_eq!(find(&rows, "scale").value, "unset");
        assert_eq!(
            find(&rows, "scale").kind,
            Kind::Expr { optional: true },
            "an omitted transform field can still be cleared back to unset"
        );
        assert_eq!(
            find(&rows, "entity").kind,
            Kind::Expr { optional: false },
            "the entity is required"
        );
    }

    #[test]
    fn an_operand_list_is_its_own_appendable_row() {
        let rows = rows(&json!({"do": [
            {"if": {"cond": {"all": [{"bool": true}, {"bool": false}]}}}]}));
        let of = find(&rows, "of");
        assert_eq!(of.kind, Kind::List(List::Operands));
        assert_eq!(of.value, "(2)");
        let first = find(&rows, "0");
        assert_eq!(first.value, "bool true");
        assert_eq!(
            first.element,
            Some(vec![
                field("do"),
                Step::Index(0),
                field("if"),
                field("cond"),
                field("all"),
                Step::Index(0)
            ]),
            "operands are removable"
        );
    }

    #[test]
    fn node_rows_summarize_the_field_that_identifies_them() {
        let rows = rows(&json!({"do": [
            {"set": {"var": "visits", "value": {"int": 1}, "add": true}},
            {"for_each": {"query": "props", "bind": "e", "do": []}},
            {"hide": {"target": {"named": "door"}}},
            {"sound": {"clip": "ping", "kind": "music", "volume": 0.5}},
            {"story": "continue"}]}));
        assert_eq!(find(&rows, "set").value, "visits");
        assert_eq!(find(&rows, "for_each").value, "props -> e");
        assert_eq!(find(&rows, "hide").value, "named door");
        assert_eq!(find(&rows, "sound").value, "ping");
        assert_eq!(find(&rows, "kind").kind, Kind::Choice(CUE_KINDS));
        assert_eq!(find(&rows, "kind").value, "music");
        assert_eq!(find(&rows, "playback").value, "continue");
        assert_eq!(find(&rows, "add").kind, Kind::Flag);
        assert_eq!(find(&rows, "add").value, "true");
    }

    // Malformed authored JSON is browsable rather than fatal: an unknown verb
    // shows as one row with no children invented for it.
    #[test]
    fn unknown_verbs_do_not_grow_a_subtree() {
        let rows = rows(&json!({"do": [{"teleport": {"to": "x"}}]}));
        let node = find(&rows, "node");
        assert_eq!(node.value, "(unknown)");
        assert!(node.element.is_some(), "it can still be deleted");
        assert_eq!(rows.iter().filter(|r| r.depth > node.depth).count(), 0);
    }

    #[test]
    fn numbers_read_back_without_a_trailing_zero() {
        assert_eq!(fmt_num(3.0), "3");
        assert_eq!(fmt_num(-0.5), "-0.5");
        assert_eq!(fmt_num(0.0), "0");
    }

    #[test]
    fn a_missing_choice_word_falls_back_to_the_first_option() {
        assert_eq!(word_text(None, CUE_KINDS), "sound");
        assert_eq!(
            word_text(Some(&json!("nonsense")), TRANSITIONS),
            "FadeBlack"
        );
        assert_eq!(word_text(Some(&json!("Cut")), TRANSITIONS), "Cut");
    }
}
