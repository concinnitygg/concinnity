// src/editor/behavior/clip.rs
//
// Carrying part of a body from one place to another. The unit is a list member,
// which is the same thing the toolbar's delete and reorder already act on
// (`Row::element`), so a node comes with its whole subtree -- branches, their
// nodes, and the expressions inside them -- for free.
//
// A clip remembers which kind of list it came out of, and a paste only lands in
// a list of that kind. Nothing here consults the checker: refusing the pastes
// that could never type-check is about not offering a move that visibly does
// nothing, while whether the result is valid stays the checker's answer.

use serde_json::Value;

use super::outline::{Kind, List, Row};
use super::path::{self, Path, Step};

// A member of a list, and the kind of list it belongs in.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Clip {
    pub list: List,
    pub value: Value,
}

// The member `row` stands for, or `None` when the row is not one.
pub(crate) fn of(args: &Value, rows: &[Row], row: &Row) -> Option<Clip> {
    let element = row.element.as_ref()?;
    Some(Clip {
        list: holding_list(rows, element)?,
        value: path::get(args, element)?.clone(),
    })
}

// Put `clip` into the list a selection on `row` addresses, and say where it
// landed so the caller can follow it. A row that is itself a member takes the
// copy directly after it, which is what puts a duplicate beside its original;
// the list's own row appends.
pub(crate) fn paste(args: &mut Value, rows: &[Row], row: &Row, clip: &Clip) -> Option<Path> {
    let (list, index) = target(rows, row, clip.list)?;
    let at = path::insert(args, &list, index, clip.value.clone())?;
    Some(path::child(&list, Step::Index(at)))
}

// The list a paste goes into and the index it takes there, `usize::MAX` meaning
// the end.
fn target(rows: &[Row], row: &Row, want: List) -> Option<(Path, usize)> {
    if row.kind == Kind::List(want) {
        return Some((row.path.clone(), usize::MAX));
    }
    let element = row.element.as_ref()?;
    if holding_list(rows, element)? != want {
        return None;
    }
    let (Step::Index(i), list) = element.split_last()? else {
        return None;
    };
    Some((list.to_vec(), i + 1))
}

// The kind of list `element` is a member of, read off the row addressing that
// list. A member's own row says what it is, not what it is in.
fn holding_list(rows: &[Row], element: &[Step]) -> Option<List> {
    let (Step::Index(_), list) = element.split_last()? else {
        return None;
    };
    rows.iter()
        .find(|r| r.path == list)
        .and_then(|r| match r.kind {
            Kind::List(l) => Some(l),
            _ => None,
        })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::outline;
    use super::*;

    fn body() -> Value {
        json!({
            "on": "tick",
            "scope": ["Prop"],
            "do": [
                {"save": {}},
                {"if": {
                    "cond": {"bool": true},
                    "then": [{"hide": {"target": "self"}}],
                }},
            ],
        })
    }

    fn row_of(rows: &[Row], label: &str) -> Row {
        rows.iter()
            .find(|r| r.label == label)
            .unwrap_or_else(|| panic!("no `{label}` row"))
            .clone()
    }

    // A node comes with everything under it, which is the point: a branch is
    // worth duplicating precisely because of what it carries.
    #[test]
    fn a_clip_of_a_node_carries_its_whole_subtree() {
        let args = body();
        let rows = outline::rows(&args);
        let clip = of(&args, &rows, &row_of(&rows, "if")).expect("the `if` node is a member");
        assert_eq!(clip.list, List::Nodes);
        assert_eq!(clip.value["if"]["then"][0]["hide"]["target"], json!("self"));
    }

    #[test]
    fn a_row_that_is_not_a_member_has_nothing_to_clip() {
        let args = body();
        let rows = outline::rows(&args);
        assert!(of(&args, &rows, &row_of(&rows, "on")).is_none());
        assert!(of(&args, &rows, &row_of(&rows, "do")).is_none());
        assert!(of(&args, &rows, &row_of(&rows, "cond")).is_none());
    }

    // A paste beside a member is what makes duplicating read as duplicating.
    #[test]
    fn a_paste_onto_a_member_lands_directly_after_it() {
        let mut args = body();
        let rows = outline::rows(&args);
        let clip = of(&args, &rows, &row_of(&rows, "save")).expect("a clip");
        let at = paste(&mut args, &rows, &row_of(&rows, "save"), &clip).expect("it landed");

        assert_eq!(at, vec![path::field("do"), Step::Index(1)]);
        let list = args["do"].as_array().unwrap();
        assert_eq!(list.len(), 3);
        assert!(list[0].get("save").is_some());
        assert!(list[1].get("save").is_some(), "the copy sits beside it");
        assert!(list[2].get("if").is_some(), "and the rest moved along");
    }

    #[test]
    fn a_paste_onto_a_list_appends_to_it() {
        let mut args = body();
        let rows = outline::rows(&args);
        let clip = of(&args, &rows, &row_of(&rows, "save")).expect("a clip");
        let at = paste(&mut args, &rows, &row_of(&rows, "do"), &clip).expect("it landed");

        assert_eq!(at, vec![path::field("do"), Step::Index(2)]);
        assert!(args["do"][2].get("save").is_some());
    }

    // A branch list takes nodes too, so a node clipped from the body pastes into
    // a `then` without any of it being about where it came from.
    #[test]
    fn a_node_pastes_into_any_list_of_nodes() {
        let mut args = body();
        let rows = outline::rows(&args);
        let clip = of(&args, &rows, &row_of(&rows, "save")).expect("a clip");
        let then = rows
            .iter()
            .find(|r| {
                r.path
                    == vec![
                        path::field("do"),
                        Step::Index(1),
                        path::field("if"),
                        path::field("then"),
                    ]
            })
            .expect("the `then` branch has a row")
            .clone();

        paste(&mut args, &rows, &then, &clip).expect("it landed");
        assert_eq!(args["do"][1]["if"]["then"].as_array().unwrap().len(), 2);
    }

    // A list of component names is not a list of nodes, so nothing is offered.
    #[test]
    fn a_clip_is_refused_by_a_list_of_another_kind() {
        let mut args = body();
        let rows = outline::rows(&args);
        let clip = of(&args, &rows, &row_of(&rows, "save")).expect("a clip");
        let scope = row_of(&rows, "scope");

        assert!(paste(&mut args, &rows, &scope, &clip).is_none());
        assert_eq!(args["scope"], json!(["Prop"]), "and nothing was written");
    }

    // A scope entry is a member of its own kind of list, so it travels the same
    // way a node does.
    #[test]
    fn a_member_of_any_list_kind_can_be_clipped() {
        let mut args = json!({"on": "start", "scope": ["Prop", "Camera3D"]});
        let rows = outline::rows(&args);
        let entry = rows
            .iter()
            .find(|r| r.element == Some(vec![path::field("scope"), Step::Index(0)]))
            .expect("a scope entry")
            .clone();
        let clip = of(&args, &rows, &entry).expect("a clip");
        assert_eq!(clip.list, List::Scope);

        paste(&mut args, &rows, &entry, &clip).expect("it landed");
        assert_eq!(args["scope"], json!(["Prop", "Prop", "Camera3D"]));
    }
}
