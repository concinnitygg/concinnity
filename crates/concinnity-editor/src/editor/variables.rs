// src/editor/variables.rs
//
// The world's variable table as the panel's rows. A row is one variable: its
// name, its type, and the value it starts the world with.
//
// Declaring the table is what makes it authoritative, so a name a behavior uses
// and the table does not declare is a build error waiting to happen. Those names
// get rows of their own -- undeclared, so the panel can say what is missing and
// declare it in one press -- because a table that leaves one out is worse than no
// table at all.

use serde_json::Value;

use super::behavior::outline;
use super::behavior::palette;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Row {
    pub name: String,
    // Where this variable sits in the table's `vars`, or `None` for a name the
    // behaviors use that the table does not declare.
    pub at: Option<usize>,
    // The declared type and starting value; both empty when undeclared.
    pub ty: String,
    pub value: String,
    // A live session's per-entity behavior local, shown for inspection only:
    // never part of the table, so the toolbar's declare / retype / remove and
    // the text fields all stand down on it.
    pub local: bool,
}

impl Row {
    pub(crate) fn declared(&self) -> bool {
        self.at.is_some()
    }
}

// The table's own declarations, then every name the behaviors use that it leaves
// out. Declared rows keep the order the asset lists them in, so editing one
// never reorders the rest.
pub(crate) fn rows(args: Option<&Value>, used: &[String]) -> Vec<Row> {
    let declared: Vec<Row> = decls(args)
        .iter()
        .enumerate()
        .map(|(i, decl)| {
            let value = decl.get("value");
            Row {
                name: name_of(decl).to_string(),
                at: Some(i),
                ty: value.map(palette::verb_of).unwrap_or_default().to_string(),
                value: outline::literal_text(value.and_then(palette::body_of)),
                local: false,
            }
        })
        .collect();
    let missing = used
        .iter()
        .filter(|name| !name.is_empty() && !declared.iter().any(|r| &&r.name == name));
    declared
        .iter()
        .cloned()
        .chain(missing.map(|name| Row {
            name: name.clone(),
            at: None,
            ty: String::new(),
            value: String::new(),
            local: false,
        }))
        .collect()
}

// Whether the world declares a table at all. Without one every variable is
// implicit and integer-typed, so nothing a behavior names can be wrong yet.
pub(crate) fn authoritative(args: Option<&Value>) -> bool {
    args.is_some()
}

// `base`, `base_2`, ... until no declaration already holds it.
pub(crate) fn unique_name(args: Option<&Value>, base: &str) -> String {
    let taken: Vec<&str> = decls(args).iter().map(name_of).collect();
    if !taken.contains(&base) {
        return base.to_string();
    }
    (2..)
        .map(|n| format!("{base}_{n}"))
        .find(|name| !taken.iter().any(|t| *t == name))
        .unwrap_or_else(|| base.to_string())
}

fn decls(args: Option<&Value>) -> &[Value] {
    args.and_then(|a| a.get("vars"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

fn name_of(decl: &Value) -> &str {
    decl.get("name").and_then(Value::as_str).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn table() -> Value {
        json!({"vars": [
            {"name": "visits", "value": {"int": 3}},
            {"name": "health", "value": {"float": 12.5}},
            {"name": "spawn", "value": {"vec3": [0, 1, 0]}},
        ]})
    }

    #[test]
    fn a_declaration_reads_as_its_name_type_and_starting_value() {
        let rows = rows(Some(&table()), &[]);
        assert_eq!(rows.len(), 3);
        assert_eq!(
            (
                rows[0].name.as_str(),
                rows[0].ty.as_str(),
                rows[0].value.as_str()
            ),
            ("visits", "int", "3"),
        );
        assert_eq!(rows[1].value, "12.5");
        assert_eq!(rows[2].ty, "vec3");
        assert_eq!(rows[2].value, "0, 1, 0");
        assert!(rows.iter().all(Row::declared));
    }

    // A name the behaviors use that the table leaves out is the whole reason the
    // panel exists: declaring the table without it breaks that behavior.
    #[test]
    fn a_used_name_the_table_leaves_out_gets_an_undeclared_row() {
        let used = ["visits".to_string(), "score".to_string()];
        let rows = rows(Some(&table()), &used);
        assert_eq!(
            rows.len(),
            4,
            "one row per declaration plus the missing name"
        );
        let missing: Vec<&Row> = rows.iter().filter(|r| !r.declared()).collect();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].name, "score");
        assert!(missing[0].ty.is_empty(), "it has no declared type yet");
    }

    // A world with no table at all: every name the behaviors use is undeclared,
    // and nothing is wrong with that until a table exists.
    #[test]
    fn a_world_with_no_table_lists_what_its_behaviors_use() {
        let used = ["score".to_string(), "visits".to_string()];
        let rows = rows(None, &used);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| !r.declared()));
        assert!(!authoritative(None));
        assert!(authoritative(Some(&table())));
    }

    #[test]
    fn declarations_keep_the_order_the_asset_lists_them_in() {
        let names: Vec<String> = rows(Some(&table()), &["aaa".to_string()])
            .into_iter()
            .map(|r| r.name)
            .collect();
        assert_eq!(names, ["visits", "health", "spawn", "aaa"]);
    }

    #[test]
    fn a_malformed_declaration_still_gets_a_row() {
        let rows = rows(Some(&json!({"vars": [{}, {"name": "ok"}]})), &[]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].name, "");
        assert_eq!(rows[1].ty, "", "no `value` means no declared type");
    }

    #[test]
    fn a_new_name_steps_past_the_ones_already_taken() {
        let args = table();
        assert_eq!(unique_name(Some(&args), "score"), "score");
        assert_eq!(unique_name(Some(&args), "visits"), "visits_2");
        assert_eq!(unique_name(None, "visits"), "visits");
    }
}
