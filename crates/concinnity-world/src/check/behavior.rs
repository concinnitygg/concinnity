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

use serde_json::Value;

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

// The world's `Variables` asset: every declaration needs a name and a typed
// starting value, and no name may repeat.
pub(crate) fn check_variables(name: &str, args: &Value) -> Result<(), String> {
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
    let err = |detail: String| format!("Behavior '{name}': {detail}");

    let scope_names = str_array(args, "scope");
    for component in &scope_names {
        if ComponentType::parse(component).is_none() {
            return Err(err(format!(
                "`scope` names unknown component '{component}'"
            )));
        }
    }
    let entity_scoped = !scope_names.is_empty();

    let mut locals: Vec<(&str, Ty)> = Vec::new();
    for (i, decl) in array(args, "locals").iter().enumerate() {
        let local_name = decl.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if local_name.is_empty() {
            return Err(err(format!("local #{i} has no `name`")));
        }
        if locals.iter().any(|(n, _)| *n == local_name) {
            return Err(err(format!("duplicate local '{local_name}'")));
        }
        let Some(ty) = decl.get("value").and_then(literal_ty) else {
            return Err(err(format!(
                "local '{local_name}' needs a typed `value` (bool, int, float, or vec3)"
            )));
        };
        locals.push((local_name, ty));
    }
    if !entity_scoped && !locals.is_empty() {
        return Err(err(
            "`locals` need a `scope`: a world-scoped behavior has no entity to hold them".into(),
        ));
    }

    let mut queries: Vec<&str> = Vec::new();
    for (i, decl) in array(args, "queries").iter().enumerate() {
        let query_name = decl.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if query_name.is_empty() {
            return Err(err(format!("query #{i} has no `name`")));
        }
        if queries.contains(&query_name) {
            return Err(err(format!("duplicate query '{query_name}'")));
        }
        let has = str_array(decl, "has");
        if has.is_empty() {
            return Err(err(format!(
                "query '{query_name}' needs at least one component in `has`"
            )));
        }
        for component in &has {
            if ComponentType::parse(component).is_none() {
                return Err(err(format!(
                    "query '{query_name}' names unknown component '{component}'"
                )));
            }
        }
        queries.push(query_name);
    }

    // A variable source must name a variable that exists.
    if let Some(watched) = args.get("on").and_then(|v| v.get("variable")) {
        let watched = watched.as_str().unwrap_or("");
        if scope_ty(vars, watched).is_none() {
            return Err(err(format!(
                "`variable` source watches undeclared variable '{watched}'; add it to the \
                 world's Variables"
            )));
        }
    }

    if source_is(args, "spawned") && !entity_scoped {
        return Err(err(
            "`spawned` source needs a `scope`: it fires on the entity that spawned".into(),
        ));
    }

    let mut scope = Scope {
        entity_scoped,
        locals,
        queries,
        bindings: Vec::new(),
        vars,
    };
    check_nodes(args.get("do"), &mut scope).map_err(err)
}

fn check_nodes<'a>(nodes: Option<&'a Value>, scope: &mut Scope<'a>) -> Result<(), String> {
    let nodes = nodes.and_then(|v| v.as_array()).map(|a| a.as_slice());
    let depth = scope.bindings.len();
    for node in nodes.unwrap_or(&[]) {
        check_node(node, scope)?;
    }
    // Bindings introduced by this list fall out of scope with it.
    scope.bindings.truncate(depth);
    Ok(())
}

fn check_node<'a>(node: &'a Value, scope: &mut Scope<'a>) -> Result<(), String> {
    let Some((verb, body)) = single_key(node) else {
        return Err("a node must be a single-key object".into());
    };
    match verb {
        "if" => {
            expect(body.get("cond"), Ty::Bool, "`if` condition", scope)?;
            check_nodes(body.get("then"), scope)?;
            check_nodes(body.get("else"), scope)?;
        }
        "for_each" => {
            let query = body.get("query").and_then(|v| v.as_str()).unwrap_or("");
            if !scope.has_query(query) {
                return Err(format!("`for_each` names undeclared query '{query}'"));
            }
            let bind = body.get("bind").and_then(|v| v.as_str()).unwrap_or("");
            if bind.is_empty() {
                return Err("`for_each` requires a `bind` name".into());
            }
            let depth = scope.bindings.len();
            scope.bindings.push((bind, Ty::Entity));
            check_nodes(body.get("do"), scope)?;
            scope.bindings.truncate(depth);
        }
        "let" => {
            let bind = body.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if bind.is_empty() {
                return Err("`let` requires a `name`".into());
            }
            let ty = expr_ty(body.get("value"), scope).map_err(|e| format!("`let` {e}"))?;
            scope.bindings.push((bind, ty));
        }
        "set" => {
            let var = body.get("var").and_then(|v| v.as_str()).unwrap_or("");
            if var.is_empty() {
                return Err("`set` requires a variable `var`".into());
            }
            let Some(ty) = scope.vars.ty(var) else {
                return Err(format!(
                    "`set` writes undeclared variable '{var}'; add it to the world's Variables"
                ));
            };
            expect(body.get("value"), ty, "`set` value", scope)?;
        }
        "set_local" => {
            let local = body.get("local").and_then(|v| v.as_str()).unwrap_or("");
            let Some(ty) = scope.local(local) else {
                return Err(format!("`set_local` names undeclared local '{local}'"));
            };
            expect(body.get("value"), ty, "`set_local` value", scope)?;
        }
        "set_transform" => {
            expect(
                body.get("entity"),
                Ty::Entity,
                "`set_transform` entity",
                scope,
            )?;
            for field in ["position", "rotation_deg", "scale"] {
                if body.get(field).is_some_and(|v| !v.is_null()) {
                    expect(
                        body.get(field),
                        Ty::Vec3,
                        &format!("`set_transform` {field}"),
                        scope,
                    )?;
                }
            }
        }
        "spawn" => {
            if let Some(bind) = body.get("bind").and_then(|v| v.as_str()) {
                if bind.is_empty() {
                    return Err("`spawn` `bind` cannot be empty".into());
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
            )?;
        }
        "reparent" => {
            expect(body.get("child"), Ty::Entity, "`reparent` child", scope)?;
            if body.get("parent").is_some_and(|v| !v.is_null()) {
                expect(body.get("parent"), Ty::Entity, "`reparent` parent", scope)?;
            }
        }
        "sound" | "scene" | "screen" | "story" | "save" => {}
        other => return Err(format!("unknown node `{other}`")),
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
fn expect(expr: Option<&Value>, want: Ty, what: &str, scope: &Scope<'_>) -> Result<(), String> {
    let got = expr_ty(expr, scope).map_err(|e| format!("{what} {e}"))?;
    if got != want {
        return Err(format!(
            "{what} must be {}, found {}",
            want.name(),
            got.name()
        ));
    }
    Ok(())
}

fn expr_ty(expr: Option<&Value>, scope: &Scope<'_>) -> Result<Ty, String> {
    let Some(expr) = expr else {
        return Err("is missing".into());
    };
    // Unit variants serialize as bare strings.
    if let Some(word) = expr.as_str() {
        return match word {
            "self" if scope.entity_scoped => Ok(Ty::Entity),
            "self" => Err("`self` needs a `scope`: a world-scoped behavior has no entity".into()),
            "dt" | "elapsed" => Ok(Ty::Float),
            other => Err(format!("names unknown expression `{other}`")),
        };
    }
    let Some((verb, body)) = single_key(expr) else {
        return Err("must be a single-key object or one of `self`, `dt`, `elapsed`".into());
    };
    match verb {
        "bool" => Ok(Ty::Bool),
        "int" => Ok(Ty::Int),
        "float" => Ok(Ty::Float),
        "vec3" => Ok(Ty::Vec3),
        "var" => match body.as_str().unwrap_or("") {
            "" => Err("`var` requires a variable name".into()),
            var => scope.vars.ty(var).ok_or_else(|| {
                format!("reads undeclared variable '{var}'; add it to the world's Variables")
            }),
        },
        "local" => {
            let local = body.as_str().unwrap_or("");
            scope
                .local(local)
                .ok_or_else(|| format!("reads undeclared local '{local}'"))
        }
        "bind" => {
            let bind = body.as_str().unwrap_or("");
            scope
                .binding(bind)
                .ok_or_else(|| format!("reads unbound name '{bind}'"))
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
                return Err(format!("`{verb}` names undeclared query '{query}'"));
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
                .ok_or_else(|| format!("`{verb}` takes a list of conditions"))?;
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
                return Err(format!(
                    "`{verb}` needs numbers, found {} and {}",
                    lhs.name(),
                    rhs.name()
                ));
            }
            // Scaling a vector by a scalar is the one mixed form allowed, and
            // only for the operators where it means something.
            match (lhs, rhs) {
                (a, b) if a == b => Ok(a),
                (Ty::Vec3, Ty::Float) if verb == "mul" || verb == "div" => Ok(Ty::Vec3),
                (Ty::Float, Ty::Vec3) if verb == "mul" => Ok(Ty::Vec3),
                _ => Err(format!(
                    "`{verb}` cannot mix {} and {}",
                    lhs.name(),
                    rhs.name()
                )),
            }
        }
        "eq" | "ne" => {
            let (a, b) = pair(body, verb)?;
            let lhs = expr_ty(Some(a), scope)?;
            let rhs = expr_ty(Some(b), scope)?;
            if lhs != rhs {
                return Err(format!(
                    "`{verb}` cannot compare {} with {}",
                    lhs.name(),
                    rhs.name()
                ));
            }
            Ok(Ty::Bool)
        }
        "lt" | "le" | "gt" | "ge" => {
            let (a, b) = pair(body, verb)?;
            let lhs = expr_ty(Some(a), scope)?;
            let rhs = expr_ty(Some(b), scope)?;
            if lhs != rhs || !lhs.is_ordered() {
                return Err(format!(
                    "`{verb}` needs two ints or two floats, found {} and {}",
                    lhs.name(),
                    rhs.name()
                ));
            }
            Ok(Ty::Bool)
        }
        other => Err(format!("names unknown expression `{other}`")),
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

fn pair<'a>(body: &'a Value, verb: &str) -> Result<(&'a Value, &'a Value), String> {
    match body.as_array() {
        Some(items) if items.len() == 2 => Ok((&items[0], &items[1])),
        _ => Err(format!("`{verb}` takes exactly two operands")),
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
}
