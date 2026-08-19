// src/check/behavior.rs
//
// Semantic validation of Behavior args: declaration shape, name binding, and
// expression types. Cross-asset name lookups (spawn templates, clips, scenes,
// trigger volumes) are handled by the Behavior `CrossReferenced` impl.
//
// World variables take their type from the world's `Variables` asset. A world
// that declares none keeps them implicit and integer-typed, so `check` is given
// the declared table (empty when there is none) and `check_world` enforces that
// a declared table accounts for every name a behavior uses.
//
// Every complaint carries where it was found (`check::fault`): the walk attaches
// the hop it descended through as the error unwinds, so a caller holding the
// authored JSON can address the value at fault. A build reports the message
// alone, which is why the string-returning entry points are unchanged.

use serde_json::Value;

use crate::check::fault::{Fault, Locate, Step, field};
use crate::registry::ComponentType;

// What an expression produces. Entity values are opaque handles: they compare
// for identity and feed entity-taking expressions, and nothing else.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    Bool,
    Int,
    Float,
    Vec3,
    Entity,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Ty::Bool => "bool",
            Ty::Int => "int",
            Ty::Float => "float",
            Ty::Vec3 => "vec3",
            Ty::Entity => "entity",
        }
    }

    fn is_numeric(self) -> bool {
        matches!(self, Ty::Int | Ty::Float | Ty::Vec3)
    }

    // Only scalars have an ordering; comparing vectors or entities with `lt`
    // and friends is a type error rather than a component-wise operation.
    fn is_ordered(self) -> bool {
        matches!(self, Ty::Int | Ty::Float)
    }
}

// The world's declared variables, by name. Empty when the world declares no
// `Variables` asset, which leaves every variable implicit and integer-typed.
#[derive(Default)]
pub struct DeclaredVars {
    declared: bool,
    types: Vec<(String, Ty)>,
}

impl DeclaredVars {
    // Read the world's `Variables` asset args. Malformed entries are skipped;
    // the asset's own check reports them.
    pub(crate) fn from_args(args: &Value) -> DeclaredVars {
        DeclaredVars {
            declared: true,
            types: array(args, "vars")
                .iter()
                .filter_map(|d| {
                    let name = d.get("name")?.as_str()?;
                    Some((name.to_string(), d.get("value").and_then(literal_ty)?))
                })
                .collect(),
        }
    }

    fn ty(&self, name: &str) -> Option<Ty> {
        self.types
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| *t)
            // Undeclared variables are integers unless a table is authoritative.
            .or(if self.declared { None } else { Some(Ty::Int) })
    }
}

// Names visible while checking one behavior body.
struct Scope<'a> {
    entity_scoped: bool,
    locals: Vec<(&'a str, Ty)>,
    queries: Vec<&'a str>,
    // `let`, `for_each`, and `spawn` bindings, innermost last.
    bindings: Vec<(&'a str, Ty)>,
    vars: &'a DeclaredVars,
}

impl<'a> Scope<'a> {
    fn local(&self, name: &str) -> Option<Ty> {
        self.locals
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, t)| *t)
    }

    fn binding(&self, name: &str) -> Option<Ty> {
        self.bindings
            .iter()
            .rev()
            .find(|(n, _)| *n == name)
            .map(|(_, t)| *t)
    }

    fn has_query(&self, name: &str) -> bool {
        self.queries.contains(&name)
    }
}

pub(crate) fn check(name: &str, args: &Value) -> Result<(), String> {
    check_with_vars(name, args, &DeclaredVars::default())
}

/// Validate one behavior's args against the world's declared variables, for
/// callers holding a single asset rather than an expanded world (the editor's
/// Behavior panel). `variables` is the world `Variables` asset's args, or
/// `None` when the world declares none, which leaves every variable implicit
/// and integer-typed. Cross-asset name resolution is not covered here; that is
/// the `CrossReferenced` pass over the whole world.
///
/// The [Fault] says where in `args` the problem is,
/// for callers that can show the author the spot; its `message` is the same text
/// a build reports.
pub fn check_with_variables(
    name: &str,
    args: &Value,
    variables: Option<&Value>,
) -> Result<(), Fault> {
    let declared = variables.map(DeclaredVars::from_args).unwrap_or_default();
    locate(name, args, &declared)
}

/// Validate the world's `Variables` asset: every declaration needs a name and a
/// typed starting value, and no name may repeat. For callers holding the asset
/// alone rather than an expanded world (the editor's Variables panel).
pub fn check_variables(name: &str, args: &Value) -> Result<(), String> {
    let err = |detail: String| format!("Variables '{name}': {detail}");
    let mut seen: Vec<&str> = Vec::new();
    for (i, decl) in array(args, "vars").iter().enumerate() {
        let var = decl.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if var.is_empty() {
            return Err(err(format!("variable #{i} has no `name`")));
        }
        if seen.contains(&var) {
            return Err(err(format!("duplicate variable '{var}'")));
        }
        seen.push(var);
        if decl.get("value").and_then(literal_ty).is_none() {
            return Err(err(format!(
                "variable '{var}' needs a typed `value` (bool, int, float, or vec3)"
            )));
        }
    }
    Ok(())
}

pub(crate) fn check_with_vars(name: &str, args: &Value, vars: &DeclaredVars) -> Result<(), String> {
    locate(name, args, vars).map_err(|f| f.message)
}

// Every complaint reads "Behavior '<name>': ..." whoever asked, so the asset's
// own label is attached once here rather than at each of the returns below.
fn locate(name: &str, args: &Value, vars: &DeclaredVars) -> Result<(), Fault> {
    body(args, vars).map_err(|f| f.about(&format!("Behavior '{name}':")))
}

fn body(args: &Value, vars: &DeclaredVars) -> Result<(), Fault> {
    let scope_names = str_array(args, "scope");
    for component in &scope_names {
        if ComponentType::parse(component).is_none() {
            return Err(
                Fault::new(format!("`scope` names unknown component '{component}'"))
                    .within(field("scope")),
            );
        }
    }
    let entity_scoped = !scope_names.is_empty();

    let mut locals: Vec<(&str, Ty)> = Vec::new();
    for (i, decl) in array(args, "locals").iter().enumerate() {
        let at = |f: Fault| f.within(Step::Index(i)).within(field("locals"));
        let local_name = decl.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if local_name.is_empty() {
            return Err(at(Fault::new(format!("local #{i} has no `name`"))));
        }
        if locals.iter().any(|(n, _)| *n == local_name) {
            return Err(at(Fault::new(format!("duplicate local '{local_name}'"))));
        }
        let Some(ty) = decl.get("value").and_then(literal_ty) else {
            return Err(at(Fault::new(format!(
                "local '{local_name}' needs a typed `value` (bool, int, float, or vec3)"
            ))
            .within(field("value"))));
        };
        locals.push((local_name, ty));
    }
    if !entity_scoped && !locals.is_empty() {
        return Err(Fault::new(
            "`locals` need a `scope`: a world-scoped behavior has no entity to hold them",
        )
        .within(field("locals")));
    }

    let mut queries: Vec<&str> = Vec::new();
    for (i, decl) in array(args, "queries").iter().enumerate() {
        let at = |f: Fault| f.within(Step::Index(i)).within(field("queries"));
        let query_name = decl.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if query_name.is_empty() {
            return Err(at(Fault::new(format!("query #{i} has no `name`"))));
        }
        if queries.contains(&query_name) {
            return Err(at(Fault::new(format!("duplicate query '{query_name}'"))));
        }
        let has = str_array(decl, "has");
        if has.is_empty() {
            return Err(at(Fault::new(format!(
                "query '{query_name}' needs at least one component in `has`"
            ))
            .within(field("has"))));
        }
        for component in &has {
            if ComponentType::parse(component).is_none() {
                return Err(at(Fault::new(format!(
                    "query '{query_name}' names unknown component '{component}'"
                ))
                .within(field("has"))));
            }
        }
        queries.push(query_name);
    }

    // A variable source must name a variable that exists.
    if let Some(watched) = args.get("on").and_then(|v| v.get("variable")) {
        let watched = watched.as_str().unwrap_or("");
        if scope_ty(vars, watched).is_none() {
            return Err(Fault::new(format!(
                "`variable` source watches undeclared variable '{watched}'; add it to the \
                 world's Variables"
            ))
            .within(field("variable"))
            .within(field("on")));
        }
    }

    if source_is(args, "spawned") && !entity_scoped {
        return Err(Fault::new(
            "`spawned` source needs a `scope`: it fires on the entity that spawned",
        )
        .within(field("on")));
    }

    let mut scope = Scope {
        entity_scoped,
        locals,
        queries,
        bindings: Vec::new(),
        vars,
    };
    check_nodes(args.get("do"), &mut scope).at_field("do")
}

fn check_nodes<'a>(nodes: Option<&'a Value>, scope: &mut Scope<'a>) -> Result<(), Fault> {
    let nodes = nodes.and_then(|v| v.as_array()).map(|a| a.as_slice());
    let depth = scope.bindings.len();
    for (i, node) in nodes.unwrap_or(&[]).iter().enumerate() {
        check_node(node, scope).at_index(i)?;
    }
    // Bindings introduced by this list fall out of scope with it.
    scope.bindings.truncate(depth);
    Ok(())
}

fn check_node<'a>(node: &'a Value, scope: &mut Scope<'a>) -> Result<(), Fault> {
    let Some((verb, body)) = single_key(node) else {
        return Err(Fault::new("a node must be a single-key object"));
    };
    // A node's settings live under its verb, so that key is the hop into it.
    // An unrecognized verb is located under itself too, which addresses nothing
    // in particular -- resolving a location falls back to the nearest place that
    // does, which is the node.
    check_verb(verb, body, scope).at_field(verb)
}

fn check_verb<'a>(verb: &str, body: &'a Value, scope: &mut Scope<'a>) -> Result<(), Fault> {
    match verb {
        "if" => {
            expect(body.get("cond"), Ty::Bool, "`if` condition", scope).at_field("cond")?;
            check_nodes(body.get("then"), scope).at_field("then")?;
            check_nodes(body.get("else"), scope).at_field("else")?;
        }
        "for_each" => {
            let query = body.get("query").and_then(|v| v.as_str()).unwrap_or("");
            if !scope.has_query(query) {
                return Err(
                    Fault::new(format!("`for_each` names undeclared query '{query}'"))
                        .within(field("query")),
                );
            }
            let bind = body.get("bind").and_then(|v| v.as_str()).unwrap_or("");
            if bind.is_empty() {
                return Err(Fault::new("`for_each` requires a `bind` name").within(field("bind")));
            }
            let depth = scope.bindings.len();
            scope.bindings.push((bind, Ty::Entity));
            check_nodes(body.get("do"), scope).at_field("do")?;
            scope.bindings.truncate(depth);
        }
        "let" => {
            let bind = body.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if bind.is_empty() {
                return Err(Fault::new("`let` requires a `name`").within(field("name")));
            }
            let ty = expr_ty(body.get("value"), scope)
                .map_err(|f| f.about("`let`"))
                .at_field("value")?;
            scope.bindings.push((bind, ty));
        }
        "set" => {
            let var = body.get("var").and_then(|v| v.as_str()).unwrap_or("");
            if var.is_empty() {
                return Err(Fault::new("`set` requires a variable `var`").within(field("var")));
            }
            let Some(ty) = scope.vars.ty(var) else {
                return Err(Fault::new(format!(
                    "`set` writes undeclared variable '{var}'; add it to the world's Variables"
                ))
                .within(field("var")));
            };
            expect(body.get("value"), ty, "`set` value", scope).at_field("value")?;
        }
        "set_local" => {
            let local = body.get("local").and_then(|v| v.as_str()).unwrap_or("");
            let Some(ty) = scope.local(local) else {
                return Err(
                    Fault::new(format!("`set_local` names undeclared local '{local}'"))
                        .within(field("local")),
                );
            };
            expect(body.get("value"), ty, "`set_local` value", scope).at_field("value")?;
        }
        "set_transform" => {
            expect(
                body.get("entity"),
                Ty::Entity,
                "`set_transform` entity",
                scope,
            )
            .at_field("entity")?;
            for part in ["position", "rotation_deg", "scale"] {
                if body.get(part).is_some_and(|v| !v.is_null()) {
                    expect(
                        body.get(part),
                        Ty::Vec3,
                        &format!("`set_transform` {part}"),
                        scope,
                    )
                    .at_field(part)?;
                }
            }
        }
        "spawn" => {
            if let Some(bind) = body.get("bind").and_then(|v| v.as_str()) {
                if bind.is_empty() {
                    return Err(Fault::new("`spawn` `bind` cannot be empty").within(field("bind")));
                }
                scope.bindings.push((bind, Ty::Entity));
            }
        }
        "despawn" | "show" | "hide" => {
            expect(
                body.get("target"),
                Ty::Entity,
                &format!("`{verb}` target"),
                scope,
            )
            .at_field("target")?;
        }
        "reparent" => {
            expect(body.get("child"), Ty::Entity, "`reparent` child", scope).at_field("child")?;
            if body.get("parent").is_some_and(|v| !v.is_null()) {
                expect(body.get("parent"), Ty::Entity, "`reparent` parent", scope)
                    .at_field("parent")?;
            }
        }
        "sound" | "scene" | "screen" | "story" | "save" => {}
        other => return Err(Fault::new(format!("unknown node `{other}`"))),
    }
    Ok(())
}

// A variable's declared type, for checks made before the scope exists.
fn scope_ty(vars: &DeclaredVars, name: &str) -> Option<Ty> {
    if name.is_empty() {
        return None;
    }
    vars.ty(name)
}

// Check an expression against an expected type.
fn expect(expr: Option<&Value>, want: Ty, what: &str, scope: &Scope<'_>) -> Result<(), Fault> {
    let got = expr_ty(expr, scope).map_err(|f| f.about(what))?;
    if got != want {
        return Err(Fault::new(format!(
            "{what} must be {}, found {}",
            want.name(),
            got.name()
        )));
    }
    Ok(())
}

fn expr_ty(expr: Option<&Value>, scope: &Scope<'_>) -> Result<Ty, Fault> {
    let Some(expr) = expr else {
        return Err(Fault::new("is missing"));
    };
    // Unit variants serialize as bare strings.
    if let Some(word) = expr.as_str() {
        return match word {
            "self" if scope.entity_scoped => Ok(Ty::Entity),
            "self" => Err(Fault::new(
                "`self` needs a `scope`: a world-scoped behavior has no entity",
            )),
            "dt" | "elapsed" => Ok(Ty::Float),
            other => Err(Fault::new(format!("names unknown expression `{other}`"))),
        };
    }
    let Some((verb, body)) = single_key(expr) else {
        return Err(Fault::new(
            "must be a single-key object or one of `self`, `dt`, `elapsed`",
        ));
    };
    match verb {
        "bool" => Ok(Ty::Bool),
        "int" => Ok(Ty::Int),
        "float" => Ok(Ty::Float),
        "vec3" => Ok(Ty::Vec3),
        "var" => match body.as_str().unwrap_or("") {
            "" => Err(Fault::new("`var` requires a variable name")),
            var => scope.vars.ty(var).ok_or_else(|| {
                Fault::new(format!(
                    "reads undeclared variable '{var}'; add it to the world's Variables"
                ))
            }),
        },
        "local" => {
            let local = body.as_str().unwrap_or("");
            scope
                .local(local)
                .ok_or_else(|| Fault::new(format!("reads undeclared local '{local}'")))
        }
        "bind" => {
            let bind = body.as_str().unwrap_or("");
            scope
                .binding(bind)
                .ok_or_else(|| Fault::new(format!("reads unbound name '{bind}'")))
        }
        // Resolution of the asset name itself is a cross-reference check.
        "named" => Ok(Ty::Entity),
        "position" => {
            expect(Some(body), Ty::Entity, "`position` operand", scope)?;
            Ok(Ty::Vec3)
        }
        "alive" => {
            expect(Some(body), Ty::Entity, "`alive` operand", scope)?;
            Ok(Ty::Bool)
        }
        "distance" => {
            let (a, b) = pair(body, "distance")?;
            expect(Some(a), Ty::Entity, "`distance` operand", scope)?;
            expect(Some(b), Ty::Entity, "`distance` operand", scope)?;
            Ok(Ty::Float)
        }
        "first" | "count" => {
            let query = body.as_str().unwrap_or("");
            if !scope.has_query(query) {
                return Err(Fault::new(format!(
                    "`{verb}` names undeclared query '{query}'"
                )));
            }
            Ok(if verb == "first" { Ty::Entity } else { Ty::Int })
        }
        "normalize" => {
            expect(Some(body), Ty::Vec3, "`normalize` operand", scope)?;
            Ok(Ty::Vec3)
        }
        "not" => {
            expect(Some(body), Ty::Bool, "`not` operand", scope)?;
            Ok(Ty::Bool)
        }
        "all" | "any" => {
            let items = body
                .as_array()
                .ok_or_else(|| Fault::new(format!("`{verb}` takes a list of conditions")))?;
            for item in items {
                expect(Some(item), Ty::Bool, &format!("`{verb}` operand"), scope)?;
            }
            Ok(Ty::Bool)
        }
        "add" | "sub" | "mul" | "div" => {
            let (a, b) = pair(body, verb)?;
            let lhs = expr_ty(Some(a), scope)?;
            let rhs = expr_ty(Some(b), scope)?;
            if !lhs.is_numeric() || !rhs.is_numeric() {
                return Err(Fault::new(format!(
                    "`{verb}` needs numbers, found {} and {}",
                    lhs.name(),
                    rhs.name()
                )));
            }
            // Scaling a vector by a scalar is the one mixed form allowed, and
            // only for the operators where it means something.
            match (lhs, rhs) {
                (a, b) if a == b => Ok(a),
                (Ty::Vec3, Ty::Float) if verb == "mul" || verb == "div" => Ok(Ty::Vec3),
                (Ty::Float, Ty::Vec3) if verb == "mul" => Ok(Ty::Vec3),
                _ => Err(Fault::new(format!(
                    "`{verb}` cannot mix {} and {}",
                    lhs.name(),
                    rhs.name()
                ))),
            }
        }
        "eq" | "ne" => {
            let (a, b) = pair(body, verb)?;
            let lhs = expr_ty(Some(a), scope)?;
            let rhs = expr_ty(Some(b), scope)?;
            if lhs != rhs {
                return Err(Fault::new(format!(
                    "`{verb}` cannot compare {} with {}",
                    lhs.name(),
                    rhs.name()
                )));
            }
            Ok(Ty::Bool)
        }
        "lt" | "le" | "gt" | "ge" => {
            let (a, b) = pair(body, verb)?;
            let lhs = expr_ty(Some(a), scope)?;
            let rhs = expr_ty(Some(b), scope)?;
            if lhs != rhs || !lhs.is_ordered() {
                return Err(Fault::new(format!(
                    "`{verb}` needs two ints or two floats, found {} and {}",
                    lhs.name(),
                    rhs.name()
                )));
            }
            Ok(Ty::Bool)
        }
        other => Err(Fault::new(format!("names unknown expression `{other}`"))),
    }
}

fn literal_ty(value: &Value) -> Option<Ty> {
    match single_key(value)?.0 {
        "bool" => Some(Ty::Bool),
        "int" => Some(Ty::Int),
        "float" => Some(Ty::Float),
        "vec3" => Some(Ty::Vec3),
        _ => None,
    }
}

fn pair<'a>(body: &'a Value, verb: &str) -> Result<(&'a Value, &'a Value), Fault> {
    match body.as_array() {
        Some(items) if items.len() == 2 => Ok((&items[0], &items[1])),
        _ => Err(Fault::new(format!("`{verb}` takes exactly two operands"))),
    }
}

fn single_key(value: &Value) -> Option<(&str, &Value)> {
    let map = value.as_object()?;
    if map.len() != 1 {
        return None;
    }
    map.iter().next().map(|(k, v)| (k.as_str(), v))
}

fn array<'a>(args: &'a Value, key: &str) -> &'a [Value] {
    args.get(key)
        .and_then(|v| v.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[])
}

fn str_array<'a>(args: &'a Value, key: &str) -> Vec<&'a str> {
    array(args, key).iter().filter_map(|v| v.as_str()).collect()
}

// Whether `on` is the named unit source.
fn source_is(args: &Value, word: &str) -> bool {
    args.get("on").and_then(|v| v.as_str()) == Some(word)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_json(json: &str) -> Result<(), String> {
        check("b", &serde_json::from_str(json).expect("args parse"))
    }

    fn expect_err(json: &str, needle: &str) {
        let e = check_json(json).expect_err("expected a validation error");
        assert!(
            e.contains(needle),
            "error {e:?} does not mention {needle:?}"
        );
    }

    // Where a complaint points, for the callers that can show the author the
    // spot rather than only the sentence.
    fn fault_at(json: &str) -> Vec<Step> {
        check_with_variables("b", &serde_json::from_str(json).expect("args parse"), None)
            .expect_err("expected a validation error")
            .at
    }

    fn at(steps: &[&str]) -> Vec<Step> {
        steps
            .iter()
            .map(|s| match s.parse::<usize>() {
                Ok(i) => Step::Index(i),
                Err(_) => field(s),
            })
            .collect()
    }

    #[test]
    fn empty_behavior_passes() {
        check_json("{}").expect("empty behavior is valid");
    }

    #[test]
    fn world_scoped_set_passes() {
        check_json(r#"{"on":"start","do":[{"set":{"var":"visits","value":{"int":1}}}]}"#)
            .expect("world-scoped set is valid");
    }

    #[test]
    fn unknown_scope_component_is_rejected() {
        expect_err(r#"{"scope":["Nonesuch"]}"#, "unknown component 'Nonesuch'");
    }

    #[test]
    fn unknown_query_component_is_rejected() {
        expect_err(
            r#"{"queries":[{"name":"q","has":["Nonesuch"]}]}"#,
            "unknown component 'Nonesuch'",
        );
    }

    #[test]
    fn locals_need_a_scope() {
        expect_err(
            r#"{"locals":[{"name":"speed","value":{"float":1.0}}]}"#,
            "`locals` need a `scope`",
        );
    }

    #[test]
    fn duplicate_local_is_rejected() {
        expect_err(
            r#"{"scope":["Prop"],"locals":[{"name":"a","value":{"int":0}},{"name":"a","value":{"int":1}}]}"#,
            "duplicate local 'a'",
        );
    }

    #[test]
    fn self_needs_a_scope() {
        expect_err(
            r#"{"do":[{"despawn":{"target":"self"}}]}"#,
            "`self` needs a `scope`",
        );
    }

    #[test]
    fn spawned_source_needs_a_scope() {
        expect_err(r#"{"on":"spawned"}"#, "`spawned` source needs a `scope`");
    }

    #[test]
    fn undeclared_query_is_rejected() {
        expect_err(
            r#"{"do":[{"for_each":{"query":"ghosts","bind":"e","do":[]}}]}"#,
            "undeclared query 'ghosts'",
        );
    }

    #[test]
    fn unbound_name_is_rejected() {
        expect_err(
            r#"{"do":[{"despawn":{"target":{"bind":"nope"}}}]}"#,
            "unbound name 'nope'",
        );
    }

    #[test]
    fn binding_falls_out_of_scope_after_its_list() {
        expect_err(
            r#"{"queries":[{"name":"q","has":["Prop"]}],
                "do":[{"for_each":{"query":"q","bind":"e","do":[]}},
                      {"despawn":{"target":{"bind":"e"}}}]}"#,
            "unbound name 'e'",
        );
    }

    #[test]
    fn spawn_binding_is_usable_afterwards() {
        check_json(r#"{"do":[{"spawn":{"bind":"made"}},{"despawn":{"target":{"bind":"made"}}}]}"#)
            .expect("a spawn binding is usable later in the same list");
    }

    #[test]
    fn type_mismatch_in_distance_is_rejected() {
        expect_err(
            r#"{"scope":["Prop"],"do":[{"let":{"name":"d","value":{"distance":["self",{"float":3.0}]}}}]}"#,
            "must be entity, found float",
        );
    }

    #[test]
    fn mixing_vec3_and_float_in_add_is_rejected() {
        expect_err(
            r#"{"do":[{"let":{"name":"x","value":{"add":[{"vec3":[0,0,0]},{"float":1.0}]}}}]}"#,
            "cannot mix vec3 and float",
        );
    }

    #[test]
    fn scaling_a_vector_by_a_scalar_is_allowed() {
        check_json(
            r#"{"do":[{"let":{"name":"x","value":{"mul":[{"vec3":[0,1,0]},{"float":2.0}]}}}]}"#,
        )
        .expect("vec3 * float scales");
    }

    #[test]
    fn ordering_a_vector_is_rejected() {
        expect_err(
            r#"{"do":[{"if":{"cond":{"lt":[{"vec3":[0,0,0]},{"vec3":[1,1,1]}]}}}]}"#,
            "needs two ints or two floats",
        );
    }

    #[test]
    fn set_local_type_must_match_declaration() {
        expect_err(
            r#"{"scope":["Prop"],"locals":[{"name":"speed","value":{"float":1.0}}],
                "do":[{"set_local":{"local":"speed","value":{"int":2}}}]}"#,
            "must be float, found int",
        );
    }

    #[test]
    fn undeclared_world_variables_are_integers() {
        expect_err(
            r#"{"do":[{"set":{"var":"v","value":{"float":1.0}}}]}"#,
            "must be int, found float",
        );
    }

    fn declared(json: &str) -> DeclaredVars {
        DeclaredVars::from_args(&serde_json::from_str(json).expect("vars parse"))
    }

    #[test]
    fn a_declared_variable_carries_its_type() {
        let vars = declared(r#"{"vars":[{"name":"health","value":{"float":100.0}}]}"#);
        check_with_vars(
            "b",
            &serde_json::from_str(r#"{"do":[{"set":{"var":"health","value":{"float":50.0}}}]}"#)
                .unwrap(),
            &vars,
        )
        .expect("a float variable takes a float");

        let e = check_with_vars(
            "b",
            &serde_json::from_str(r#"{"do":[{"set":{"var":"health","value":{"int":50}}}]}"#)
                .unwrap(),
            &vars,
        )
        .expect_err("an int does not fit a float variable");
        assert!(e.contains("must be float, found int"), "{e}");
    }

    #[test]
    fn a_declared_table_rejects_undeclared_names() {
        let vars = declared(r#"{"vars":[{"name":"health","value":{"float":1.0}}]}"#);
        let e = check_with_vars(
            "b",
            &serde_json::from_str(r#"{"do":[{"set":{"var":"helth","value":{"float":1.0}}}]}"#)
                .unwrap(),
            &vars,
        )
        .expect_err("a misspelled name is caught");
        assert!(e.contains("undeclared variable 'helth'"), "{e}");
    }

    #[test]
    fn a_declared_vec3_variable_reads_as_a_vector() {
        let vars = declared(r#"{"vars":[{"name":"spawn","value":{"vec3":[0,1,0]}}]}"#);
        check_with_vars(
            "b",
            &serde_json::from_str(
                r#"{"scope":["Prop"],"do":[{"set_transform":{"entity":"self","position":{"var":"spawn"}}}]}"#,
            )
            .unwrap(),
            &vars,
        )
        .expect("a vec3 variable feeds a transform");
    }

    #[test]
    fn a_variable_source_must_name_a_declared_variable() {
        let vars = declared(r#"{"vars":[{"name":"health","value":{"int":0}}]}"#);
        let e = check_with_vars(
            "b",
            &serde_json::from_str(r#"{"on":{"variable":"ghost"}}"#).unwrap(),
            &vars,
        )
        .expect_err("an unknown watched variable is caught");
        assert!(e.contains("undeclared variable 'ghost'"), "{e}");
    }

    #[test]
    fn duplicate_declarations_are_rejected() {
        let args = serde_json::from_str(
            r#"{"vars":[{"name":"a","value":{"int":0}},{"name":"a","value":{"int":1}}]}"#,
        )
        .unwrap();
        let e = check_variables("v", &args).expect_err("duplicates are caught");
        assert!(e.contains("duplicate variable 'a'"), "{e}");
    }

    #[test]
    fn an_untyped_declaration_is_rejected() {
        let args = serde_json::from_str(r#"{"vars":[{"name":"a"}]}"#).unwrap();
        let e = check_variables("v", &args).expect_err("an untyped declaration is caught");
        assert!(e.contains("needs a typed `value`"), "{e}");
    }

    #[test]
    fn unknown_node_is_rejected() {
        expect_err(r#"{"do":[{"teleport":{}}]}"#, "unknown node `teleport`");
    }

    #[test]
    fn chase_body_type_checks() {
        check_json(
            r#"{"on":"tick","scope":["Prop"],
                "locals":[{"name":"speed","value":{"float":3.0}}],
                "queries":[{"name":"player","has":["Camera3D"]}],
                "do":[
                  {"let":{"name":"target","value":{"first":"player"}}},
                  {"if":{"cond":{"lt":[{"distance":["self",{"bind":"target"}]},{"float":20.0}]},
                         "then":[{"set_transform":{"entity":"self","position":
                           {"add":[{"position":"self"},
                                   {"mul":[{"normalize":{"sub":[{"position":{"bind":"target"}},
                                                               {"position":"self"}]}},
                                           {"mul":[{"local":"speed"},"dt"]}]}]}}}]}}]}"#,
        )
        .expect("the chase example type checks");
    }

    // A node's complaint addresses the node, through however many branch lists
    // it took to reach it: the walk attaches each hop as the error unwinds.
    #[test]
    fn a_node_fault_addresses_the_node_that_carries_it() {
        assert_eq!(
            fault_at(r#"{"on":"start","do":[{"save":{}},{"teleport":{}}]}"#),
            at(&["do", "1", "teleport"]),
        );
        assert_eq!(
            fault_at(
                r#"{"on":"start","do":[{"if":{"cond":{"bool":true},
                     "then":[{"save":{}}],"else":[{"save":{}},{"teleport":{}}]}}]}"#
            ),
            at(&["do", "0", "if", "else", "1", "teleport"]),
        );
    }

    // A field's complaint addresses the field, which is the same path the
    // authored JSON reaches it by.
    #[test]
    fn a_field_fault_addresses_the_field_at_fault() {
        assert_eq!(
            fault_at(r#"{"on":"start","do":[{"if":{"cond":{"int":1}}}]}"#),
            at(&["do", "0", "if", "cond"]),
        );
        assert_eq!(
            fault_at(r#"{"on":"start","do":[{"hide":{"target":{"int":1}}}]}"#),
            at(&["do", "0", "hide", "target"]),
        );
        assert_eq!(
            fault_at(r#"{"on":"start","do":[{"for_each":{"query":"nope","bind":"e"}}]}"#),
            at(&["do", "0", "for_each", "query"]),
        );
    }

    // A declaration's complaint addresses that declaration, not the whole list,
    // so a table with one bad row says which row.
    #[test]
    fn a_declaration_fault_addresses_the_entry_at_fault() {
        assert_eq!(
            fault_at(
                r#"{"scope":["Prop"],"locals":[{"name":"a","value":{"int":0}},{"name":"b"}]}"#
            ),
            at(&["locals", "1", "value"]),
        );
        assert_eq!(
            fault_at(r#"{"queries":[{"name":"q","has":["Nonesuch"]}]}"#),
            at(&["queries", "0", "has"]),
        );
        assert_eq!(fault_at(r#"{"scope":["Nonesuch"]}"#), at(&["scope"]));
        // A watched name is only wrong once a declared table makes it so, so
        // this one needs a world that declares its variables.
        let declared = serde_json::json!({"vars": [{"name": "score", "value": {"int": 0}}]});
        let f = check_with_variables(
            "b",
            &serde_json::from_str(r#"{"on":{"variable":"ghost"},"do":[]}"#).expect("args parse"),
            Some(&declared),
        )
        .expect_err("a watched variable must be declared");
        assert_eq!(f.at, at(&["on", "variable"]));
    }

    // A rule about the asset as a whole has nothing narrower to blame.
    #[test]
    fn a_whole_asset_rule_carries_no_location() {
        assert_eq!(
            fault_at(r#"{"locals":[{"name":"speed","value":{"float":1.0}}]}"#),
            at(&["locals"]),
        );
        let f = check_with_variables(
            "b",
            &serde_json::from_str(r#"{"on":"spawned"}"#).expect("args parse"),
            None,
        )
        .expect_err("a spawned source needs a scope");
        assert_eq!(f.at, at(&["on"]));
    }

    // A build reports the message and nothing else, so locating a fault must not
    // have changed a single character of it.
    #[test]
    fn the_message_reads_the_same_whichever_entry_point_asked() {
        let json = r#"{"on":"start","do":[{"teleport":{}}]}"#;
        let args: Value = serde_json::from_str(json).expect("args parse");
        let text = check("chase", &args).expect_err("expected a validation error");
        assert_eq!(text, "Behavior 'chase': unknown node `teleport`");
        assert_eq!(
            check_with_variables("chase", &args, None)
                .expect_err("expected a validation error")
                .to_string(),
            text,
        );
    }
}
