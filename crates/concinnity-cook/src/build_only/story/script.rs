use super::helpers::{parse_int, parse_var_name};
use super::model::{Condition, Gate, ScriptLine, VarOp};

// One script line: `set <var>`, `clear <var>`, `set <var> = <int>`,
// `add <var> <int>`, or a conditional jump `if <condition> -> #anchor`.
pub(super) fn parse_script_line(line: &str) -> Result<ScriptLine, String> {
    if let Some(rest) = line.strip_prefix("set ") {
        let rest = rest.trim();
        return Ok(ScriptLine::Op(match rest.split_once('=') {
            Some((name, value)) => VarOp {
                name: parse_var_name(name.trim())?,
                value: parse_int(value.trim())?,
                add: false,
            },
            None => VarOp {
                name: parse_var_name(rest)?,
                value: 1,
                add: false,
            },
        }));
    }
    if let Some(rest) = line.strip_prefix("clear ") {
        return Ok(ScriptLine::Op(VarOp {
            name: parse_var_name(rest.trim())?,
            value: 0,
            add: false,
        }));
    }
    if let Some(rest) = line.strip_prefix("add ") {
        let rest = rest.trim();
        let (name, amount) = rest
            .split_once(char::is_whitespace)
            .ok_or_else(|| format!("script line 'add {}' is missing the amount to add", rest))?;
        return Ok(ScriptLine::Op(VarOp {
            name: parse_var_name(name.trim())?,
            value: parse_int(amount.trim())?,
            add: true,
        }));
    }
    if let Some(rest) = line.strip_prefix("if ") {
        let (condition, target) = rest.split_once("->").ok_or_else(|| {
            format!(
                "script line '{}' is missing the `-> #anchor` jump target",
                line
            )
        })?;
        let target = target.trim();
        let Some(anchor) = target.strip_prefix('#') else {
            return Err(format!(
                "script jump target '{}' must be a `#heading` anchor",
                target
            ));
        };
        return Ok(ScriptLine::Gate(Gate {
            condition: parse_bare_condition(condition.trim())?,
            target: anchor.to_string(),
        }));
    }
    Err(format!(
        "script line '{}' is not `set <var> [= <int>]`, `clear <var>`, \
         `add <var> <int>`, or `if <condition> -> #anchor`",
        line
    ))
}

// The condition body after `if `: `<var>`, `not <var>`, or
// `<var> <op> <int>` with op one of == != < <= > >=. Compiled to a
// comparison against an integer (an unset variable reads as 0, so a bare
// variable test is `!= 0` and its negation `== 0`).
fn parse_bare_condition(condition: &str) -> Result<Condition, String> {
    if let Some(rest) = condition.strip_prefix("not ") {
        return Ok(Condition {
            name: parse_var_name(rest.trim())?,
            op: "eq",
            value: 0,
        });
    }
    for (symbol, op) in [
        ("==", "eq"),
        ("!=", "ne"),
        ("<=", "le"),
        (">=", "ge"),
        ("<", "lt"),
        (">", "gt"),
    ] {
        if let Some((name, value)) = condition.split_once(symbol) {
            return Ok(Condition {
                name: parse_var_name(name.trim())?,
                op,
                value: parse_int(value.trim())?,
            });
        }
    }
    Ok(Condition {
        name: parse_var_name(condition)?,
        op: "ne",
        value: 0,
    })
}

// A choice condition from a link title: `if <condition>`.
pub(super) fn parse_condition(title: &str) -> Result<Condition, String> {
    let Some(rest) = title.strip_prefix("if ") else {
        return Err(format!(
            "choice condition '{}' must be `if <var>`, `if not <var>`, or \
             `if <var> <op> <int>`",
            title
        ));
    };
    parse_bare_condition(rest.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_if_line_needs_a_jump_target() {
        // `.err()` avoids requiring the Ok type (ScriptLine) to be Debug.
        let err = parse_script_line("if asked").err().unwrap();
        assert!(err.contains("jump target"), "{err}");
    }

    #[test]
    fn script_jump_target_must_be_an_anchor() {
        let err = parse_script_line("if asked -> b").err().unwrap();
        assert!(err.contains("anchor"), "{err}");
    }

    #[test]
    fn bare_condition_rejects_a_bad_variable_name() {
        assert!(
            parse_bare_condition("not BAD")
                .unwrap_err()
                .contains("not a variable name")
        );
        assert!(
            parse_bare_condition("BAD")
                .unwrap_err()
                .contains("not a variable name")
        );
    }

    #[test]
    fn choice_condition_requires_an_if_prefix() {
        let err = parse_condition("gold >= 3").unwrap_err();
        assert!(err.contains("choice condition"), "{err}");
        let ok = parse_condition("if gold >= 3").unwrap();
        assert_eq!((ok.name.as_str(), ok.op, ok.value), ("gold", "ge", 3));
    }
}
