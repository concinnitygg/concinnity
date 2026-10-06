//! The debug server's verb table: every verb, in the order `tools/list`
//! reports them, and the one entry point that answers a call to any of them.

use serde_json::{Map, Value};
use std::sync::Mutex;

use super::call::Call;
use super::state::DebugState;
use super::verb::{Reply, Verb};
use super::verbs::{anim, camera, render, scene, settings, snapshot};

const GROUPS: [&[Verb]; 6] = [
    snapshot::VERBS,
    camera::VERBS,
    render::VERBS,
    anim::VERBS,
    settings::VERBS,
    scene::VERBS,
];

/// Every verb.
pub(crate) fn all() -> impl Iterator<Item = &'static Verb> {
    GROUPS.iter().flat_map(|group| group.iter())
}

/// Look one verb up by name.
pub(crate) fn find(name: &str) -> Option<&'static Verb> {
    all().find(|verb| verb.name == name)
}

/// Answer one call against the shared snapshot: check the arguments against
/// the verb's parameters, then run its handler.
pub(crate) fn run(name: &str, arguments: Map<String, Value>, shared: &Mutex<DebugState>) -> Reply {
    let Some(verb) = find(name) else {
        let known: Vec<_> = all().map(|verb| verb.name).collect();
        return Err(format!(
            "unknown verb '{name}' (known: {})",
            known.join(", ")
        ));
    };
    let args = verb.arguments(arguments)?;
    (verb.run)(&Call::new(verb.name, shared), args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::verb::Kind;
    use crate::debug::verbs::testing::Engine;
    use crate::test_support;
    use concinnity_core::ecs::World;
    use serde_json::json;
    use std::collections::BTreeSet;

    // A value every parameter of `kind` accepts.
    fn sample(kind: Kind) -> Value {
        match kind {
            Kind::Text | Kind::Name | Kind::TextOrNull => json!("x"),
            Kind::Number | Kind::NumberOrNull => json!(0.5),
            Kind::Count => json!(1),
            Kind::Vec3 => json!([0.0, 0.0, 0.0]),
            Kind::Vec4 => json!([0.0, 0.0, 0.0, 0.0]),
            Kind::NumberList => json!([0.0, 1.0]),
            Kind::Choice(values) => json!(values[0]),
        }
    }

    // A call filling every required parameter of `verb` with a sample.
    fn required_only(verb: &Verb) -> Map<String, Value> {
        verb.params
            .iter()
            .filter(|p| p.required)
            .map(|p| (p.name.to_string(), sample(p.kind)))
            .collect()
    }

    // Run a call that must be refused before anything reaches the engine.
    fn refused(name: &str, arguments: Map<String, Value>) -> String {
        let shared = Mutex::new(DebugState::default());
        let queue = shared.lock().unwrap().queue.clone();
        let error = run(name, arguments, &shared).expect_err(name);
        let jobs = queue.take();
        assert!(
            jobs.world.is_empty() && jobs.backend.is_empty(),
            "{name} queued work for a refused call"
        );
        error
    }

    #[test]
    fn an_unknown_verb_names_the_known_ones() {
        let error = refused("bogus", Map::new());
        assert!(error.starts_with("unknown verb 'bogus'"), "{error}");
        for verb in all() {
            assert!(error.contains(verb.name), "{} is unlisted", verb.name);
        }
    }

    #[test]
    fn every_verb_refuses_a_call_missing_a_required_parameter() {
        for verb in all() {
            for param in verb.params.iter().filter(|p| p.required) {
                let mut call = required_only(verb);
                call.remove(param.name);
                assert_eq!(
                    refused(verb.name, call),
                    format!("{}: missing '{}'", verb.name, param.name)
                );
            }
        }
    }

    #[test]
    fn every_verb_refuses_a_parameter_of_the_wrong_type() {
        for verb in all() {
            for param in verb.params {
                let mut call = required_only(verb);
                call.insert(param.name.to_string(), json!({ "wrong": "shape" }));
                let error = refused(verb.name, call);
                assert!(
                    error.starts_with(&format!("{}: '{}' must be", verb.name, param.name)),
                    "{error}"
                );
            }
        }
    }

    #[test]
    fn every_verb_refuses_a_blank_name_as_missing() {
        let mut checked = 0;
        for verb in all() {
            for param in verb.params.iter().filter(|p| p.kind == Kind::Name) {
                for blank in ["", "  "] {
                    let mut call = required_only(verb);
                    call.insert(param.name.to_string(), json!(blank));
                    assert_eq!(
                        refused(verb.name, call),
                        format!("{}: missing '{}'", verb.name, param.name)
                    );
                }
                checked += 1;
            }
        }
        assert!(checked >= 10, "only {checked} name parameters were checked");
    }

    #[test]
    fn every_verb_refuses_an_undeclared_parameter() {
        for verb in all() {
            let mut call = required_only(verb);
            call.insert("positon".to_string(), json!([1.0, 2.0, 3.0]));
            let error = refused(verb.name, call);
            assert!(error.contains("unknown field `positon`"), "{error}");
        }
    }

    // A call filling every declared parameter reaches the verb's handler, which
    // reads it into its own request shape; whatever the engine then answers, it
    // is not a refusal of the arguments.
    #[test]
    fn every_verb_reads_a_call_that_fills_each_of_its_parameters() {
        let _guard = test_support::lock();
        let mut engine = Engine::new(World::new());
        for verb in all() {
            let call: Map<String, Value> = verb
                .params
                .iter()
                .map(|p| (p.name.to_string(), sample(p.kind)))
                .collect();
            if let Err(error) = engine.call(verb.name, Value::Object(call)) {
                assert!(!error.contains("invalid arguments"), "{error}");
                assert!(!error.contains("must be"), "{error}");
            }
        }
    }

    #[test]
    fn verb_names_are_unique_and_kebab_case() {
        let mut seen = BTreeSet::new();
        for verb in all() {
            assert!(seen.insert(verb.name), "duplicate verb {}", verb.name);
            let words: Vec<_> = verb.name.split('-').collect();
            assert!(
                words
                    .iter()
                    .all(|w| !w.is_empty() && w.chars().all(|c| c.is_ascii_lowercase())),
                "{} is not kebab-case",
                verb.name
            );
        }
        assert!(find("ping").is_some());
        assert!(find("nope").is_none());
    }

    #[test]
    fn descriptions_are_one_sentence_lines() {
        for verb in all() {
            let lines =
                std::iter::once(verb.description).chain(verb.params.iter().map(|p| p.description));
            for line in lines {
                assert!(
                    line.ends_with('.') && !line.contains('\n'),
                    "{}: {line}",
                    verb.name
                );
            }
        }
    }

    #[test]
    fn every_schema_declares_exactly_the_verbs_parameters() {
        for verb in all() {
            let schema = verb.schema();
            let properties = schema["properties"].as_object().expect("properties");
            let declared: BTreeSet<_> = verb.params.iter().map(|p| p.name).collect();
            assert_eq!(
                declared.len(),
                verb.params.len(),
                "{} repeats a parameter",
                verb.name
            );
            assert_eq!(
                properties
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>(),
                declared,
                "{}",
                verb.name
            );
            let required: Vec<_> = verb
                .params
                .iter()
                .filter(|p| p.required)
                .map(|p| p.name)
                .collect();
            match schema.get("required") {
                Some(listed) => assert_eq!(listed, &json!(required), "{}", verb.name),
                None => assert!(
                    required.is_empty(),
                    "{} dropped a required parameter",
                    verb.name
                ),
            }
        }
    }

    #[test]
    fn a_parameterless_verb_declares_an_empty_closed_schema() {
        let ping = find("ping").expect("ping");
        assert_eq!(
            ping.schema(),
            json!({ "type": "object", "properties": {}, "additionalProperties": false })
        );
    }
}
