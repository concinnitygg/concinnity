// src/debug/catalog.rs
//
// The single source of truth for the debug protocol's verb surface: every
// command `super::dispatch::handle_request` answers, with a one-line
// description, whether it only reads the world snapshot or mutates the running
// world, and a JSON Schema for its parameters.
//
// Schemas describe the request body's parameters only. The transport adds the
// `"cmd"` field, so it is not a property here and `additionalProperties` stays
// closed. A parameter is `required` when the server rejects the request without
// it; every other parameter carries its default in its description.
//
// A drift test below scrapes the dispatcher's own match arms, so a new verb
// without a catalog entry (or an entry without a verb) fails to build green.

use serde_json::{Value, json};

/// Whether a command reads the world snapshot or mutates the running world.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Access {
    /// Answers from the per-frame snapshot and changes nothing.
    ReadOnly,
    /// Queues a change the engine applies on a later frame.
    Mutating,
}

impl Access {
    pub(crate) fn is_read_only(self) -> bool {
        matches!(self, Access::ReadOnly)
    }
}

/// The JSON type of one command parameter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    Text,
    TextOrNull,
    Number,
    NumberOrNull,
    /// A non-negative whole number.
    Count,
    Vec3,
    Vec4,
    NumberList,
}

impl Kind {
    fn json_type(self) -> Value {
        let numbers = |len: u64| {
            json!({
                "type": "array",
                "items": { "type": "number" },
                "minItems": len,
                "maxItems": len,
            })
        };
        match self {
            Kind::Text => json!({ "type": "string" }),
            Kind::TextOrNull => json!({ "type": ["string", "null"] }),
            Kind::Number => json!({ "type": "number" }),
            Kind::NumberOrNull => json!({ "type": ["number", "null"] }),
            Kind::Count => json!({ "type": "integer", "minimum": 0 }),
            Kind::Vec3 => numbers(3),
            Kind::Vec4 => numbers(4),
            Kind::NumberList => json!({ "type": "array", "items": { "type": "number" } }),
        }
    }
}

/// One parameter of a command's request body.
pub(crate) struct Param {
    pub(crate) name: &'static str,
    pub(crate) kind: Kind,
    /// True when the server rejects a request that omits this parameter.
    pub(crate) required: bool,
    pub(crate) description: &'static str,
}

const fn required(name: &'static str, kind: Kind, description: &'static str) -> Param {
    Param {
        name,
        kind,
        required: true,
        description,
    }
}

const fn optional(name: &'static str, kind: Kind, description: &'static str) -> Param {
    Param {
        name,
        kind,
        required: false,
        description,
    }
}

/// One command the debug server answers.
pub(crate) struct Command {
    pub(crate) name: &'static str,
    pub(crate) description: &'static str,
    pub(crate) access: Access,
    pub(crate) params: &'static [Param],
}

impl Command {
    /// The command's parameters as a draft 2020-12 object schema.
    pub(crate) fn schema(&self) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();
        for param in self.params {
            let mut property = param.kind.json_type();
            property["description"] = Value::String(param.description.to_string());
            properties.insert(param.name.to_string(), property);
            if param.required {
                required.push(Value::String(param.name.to_string()));
            }
        }
        let mut schema = json!({
            "type": "object",
            "properties": Value::Object(properties),
            "additionalProperties": false,
        });
        if !required.is_empty() {
            schema["required"] = Value::Array(required);
        }
        schema
    }
}

/// Every command, in the order the dispatcher matches them.
pub(crate) fn all() -> &'static [Command] {
    COMMANDS
}

/// Look one command up by its wire verb.
pub(crate) fn find(name: &str) -> Option<&'static Command> {
    COMMANDS.iter().find(|c| c.name == name)
}

/// Every verb name, comma separated, for an unknown-verb reply.
pub(crate) fn verb_list() -> String {
    COMMANDS
        .iter()
        .map(|c| c.name)
        .collect::<Vec<_>>()
        .join(", ")
}

const COMMANDS: &[Command] = &[
    Command {
        name: "ping",
        description: "Check the debug server is alive; replies with a pong.",
        access: Access::ReadOnly,
        params: &[],
    },
    Command {
        name: "state",
        description: "Report the current frame, the system and component counts, and the running system names.",
        access: Access::ReadOnly,
        params: &[],
    },
    Command {
        name: "assets",
        description: "Report how many instances of each component the running world holds, keyed by discriminant.",
        access: Access::ReadOnly,
        params: &[],
    },
    Command {
        name: "names",
        description: "Report the build interner's asset id to name table, indexed by id.",
        access: Access::ReadOnly,
        params: &[],
    },
    Command {
        name: "streaming",
        description: "Report streaming residency for the texture, mesh, and chunk pools, plus the memory back-off reading.",
        access: Access::ReadOnly,
        params: &[],
    },
    Command {
        name: "memory",
        description: "Report the allocation layer's heap counters, per-tag ledger, and busiest size class.",
        access: Access::ReadOnly,
        params: &[],
    },
    Command {
        name: "budget",
        description: "Report the resolved thread and memory budgets alongside the current resident set size.",
        access: Access::ReadOnly,
        params: &[],
    },
    Command {
        name: "profile",
        description: "Report last-frame CPU time per system plus render draw-call, object, and per-pass GPU timings.",
        access: Access::ReadOnly,
        params: &[],
    },
    Command {
        name: "camera-get",
        description: "Report the active camera's position, yaw, pitch, vertical field of view, and clip planes.",
        access: Access::ReadOnly,
        params: &[],
    },
    Command {
        name: "shutdown",
        description: "Cancel the run loop's shutdown token so the engine exits cleanly on its next iteration.",
        access: Access::Mutating,
        params: &[],
    },
    Command {
        name: "reload-shaders",
        description: "Queue a rebuild of every built-in render pipeline from disk-resident shader source.",
        access: Access::Mutating,
        params: &[],
    },
    Command {
        name: "reload-assets",
        description: "Queue a re-decode of file-backed textures and a reload of world, animation, and shader stage sources.",
        access: Access::Mutating,
        params: &[],
    },
    Command {
        name: "decal-add",
        description: "Place a decal in the running world and return the slot id that removes it.",
        access: Access::Mutating,
        params: &[
            optional(
                "texture",
                Kind::TextOrNull,
                "Texture asset name; omit for the built-in decal texture.",
            ),
            optional(
                "position",
                Kind::Vec3,
                "World-space position. Defaults to the origin.",
            ),
            optional(
                "rotation_deg",
                Kind::Vec3,
                "Euler rotation in degrees. Defaults to no rotation.",
            ),
            optional(
                "size",
                Kind::Vec3,
                "Projection box extents. Defaults to one unit on each axis.",
            ),
            optional(
                "tint",
                Kind::Vec4,
                "Linear RGBA tint. Defaults to opaque white.",
            ),
        ],
    },
    Command {
        name: "decal-remove",
        description: "Remove a decal previously placed by decal-add.",
        access: Access::Mutating,
        params: &[required(
            "id",
            Kind::Count,
            "Slot id returned by decal-add.",
        )],
    },
    Command {
        name: "emitter-add",
        description: "Place a particle emitter in the running world and return the slot id that removes it.",
        access: Access::Mutating,
        params: &[
            optional(
                "texture",
                Kind::TextOrNull,
                "Particle texture asset name; omit for the built-in texture.",
            ),
            optional(
                "position",
                Kind::Vec3,
                "World-space position. Defaults to the origin.",
            ),
            optional(
                "direction",
                Kind::Vec3,
                "Emission direction. Defaults to the emitter's own default.",
            ),
            optional(
                "spread_deg",
                Kind::Number,
                "Cone half-angle around the direction, in degrees.",
            ),
            optional(
                "speed_min",
                Kind::Number,
                "Lower bound of the initial particle speed.",
            ),
            optional(
                "speed_max",
                Kind::Number,
                "Upper bound of the initial particle speed.",
            ),
            optional(
                "lifetime_min",
                Kind::Number,
                "Lower bound of the particle lifetime, in seconds.",
            ),
            optional(
                "lifetime_max",
                Kind::Number,
                "Upper bound of the particle lifetime, in seconds.",
            ),
            optional(
                "gravity",
                Kind::Vec3,
                "Constant acceleration applied to every particle.",
            ),
            optional("spawn_rate", Kind::Number, "Particles emitted per second."),
            optional(
                "max_particles",
                Kind::Count,
                "Ceiling on live particles for this emitter.",
            ),
            optional("size_start", Kind::Number, "Particle size at birth."),
            optional("size_end", Kind::Number, "Particle size at death."),
            optional("color_start", Kind::Vec4, "Linear RGBA colour at birth."),
            optional("color_end", Kind::Vec4, "Linear RGBA colour at death."),
        ],
    },
    Command {
        name: "emitter-remove",
        description: "Remove a particle emitter previously placed by emitter-add.",
        access: Access::Mutating,
        params: &[required(
            "id",
            Kind::Count,
            "Slot id returned by emitter-add.",
        )],
    },
    Command {
        name: "anim-crossfade",
        description: "Ramp a skinned mesh's clip blend weights toward new values over a duration.",
        access: Access::Mutating,
        params: &[
            required(
                "target",
                Kind::Text,
                "Skinned mesh asset name, as reported by the names command.",
            ),
            optional(
                "weights",
                Kind::NumberList,
                "Per-clip blend weights; the length must match the target's clip count.",
            ),
            optional(
                "duration_secs",
                Kind::Number,
                "Ramp duration in seconds. Zero snaps to the new weights.",
            ),
        ],
    },
    Command {
        name: "anim-param",
        description: "Set one animation graph parameter on a skinned mesh.",
        access: Access::Mutating,
        params: &[
            required(
                "target",
                Kind::Text,
                "Skinned mesh asset name, as reported by the names command.",
            ),
            required("name", Kind::Text, "Graph parameter name."),
            optional(
                "value",
                Kind::Number,
                "New parameter value. Defaults to zero.",
            ),
        ],
    },
    Command {
        name: "anim-state",
        description: "Report a skinned mesh's animation state, clock, fade progress, blend weights, and parameters.",
        access: Access::ReadOnly,
        params: &[required(
            "target",
            Kind::Text,
            "Skinned mesh asset name, as reported by the names command.",
        )],
    },
    Command {
        name: "screenshot",
        description: "Capture the last presented frame to a PNG file.",
        access: Access::Mutating,
        params: &[required("path", Kind::Text, "Destination PNG path.")],
    },
    Command {
        name: "camera-set",
        description: "Teleport the active camera to a pose, optionally changing its field of view.",
        access: Access::Mutating,
        params: &[
            optional(
                "position",
                Kind::Vec3,
                "World-space position. Defaults to the origin.",
            ),
            optional("yaw", Kind::Number, "Yaw in radians. Defaults to zero."),
            optional("pitch", Kind::Number, "Pitch in radians. Defaults to zero."),
            optional(
                "fov_y_degrees",
                Kind::NumberOrNull,
                "Vertical field of view in degrees; omit to leave it untouched.",
            ),
        ],
    },
    Command {
        name: "camera-move",
        description: "Apply a per-frame camera pose delta over a span of frames so the renderer sees sustained motion.",
        access: Access::Mutating,
        params: &[
            optional(
                "forward",
                Kind::Number,
                "Per-frame offset along the look direction, in world units.",
            ),
            optional(
                "right",
                Kind::Number,
                "Per-frame offset along the right vector, in world units.",
            ),
            optional(
                "up",
                Kind::Number,
                "Per-frame offset along the up vector, in world units.",
            ),
            optional("yaw", Kind::Number, "Per-frame yaw delta in radians."),
            optional("pitch", Kind::Number, "Per-frame pitch delta in radians."),
            optional(
                "frames",
                Kind::Count,
                "How many frames to apply the delta for. Zero holds until camera-stop.",
            ),
        ],
    },
    Command {
        name: "camera-stop",
        description: "Clear any camera motion left running by camera-move.",
        access: Access::Mutating,
        params: &[],
    },
    Command {
        name: "quality-set",
        description: "Cycle one quality graphics setting live, the way the settings menu does.",
        access: Access::Mutating,
        params: &[
            required(
                "setting",
                Kind::Text,
                "Setting key, one of taa, ssao, ssr, ssgi, auto_exposure.",
            ),
            optional(
                "op",
                Kind::Text,
                "Cycle direction, next or prev. Defaults to next.",
            ),
        ],
    },
    Command {
        name: "rebind",
        description: "Bind one movement action to a different key, the way a settings menu capture does.",
        access: Access::Mutating,
        params: &[
            required(
                "setting",
                Kind::Text,
                "Action key, such as key_forward or key_jump.",
            ),
            required(
                "key",
                Kind::Text,
                "Input key variant name, such as W, Space, Shift, Num1, or Up.",
            ),
        ],
    },
    Command {
        name: "despawn",
        description: "Remove an authored placement and its descendants from the running world.",
        access: Access::Mutating,
        params: &[required("name", Kind::Text, "Placement name to remove.")],
    },
    Command {
        name: "reparent",
        description: "Move an authored placement under a new parent, or detach it to a root.",
        access: Access::Mutating,
        params: &[
            required("child", Kind::Text, "Placement name to move."),
            optional(
                "parent",
                Kind::TextOrNull,
                "New parent placement name; omit to detach the child to a root.",
            ),
        ],
    },
    Command {
        name: "spawn",
        description: "Instantiate a runtime copy of an authored placement at a given pose.",
        access: Access::Mutating,
        params: &[
            required("template", Kind::Text, "Existing placement name to copy."),
            required("name", Kind::Text, "Name for the new instance."),
            optional(
                "position",
                Kind::Vec3,
                "World-space position. Defaults to the origin.",
            ),
            optional(
                "rotation_deg",
                Kind::Vec3,
                "Euler rotation in degrees. Defaults to no rotation.",
            ),
            optional(
                "scale",
                Kind::Vec3,
                "Scale on each axis. Defaults to unit scale.",
            ),
            optional(
                "lifetime",
                Kind::NumberOrNull,
                "Seconds before the instance despawns itself; omit to keep it alive.",
            ),
        ],
    },
    Command {
        name: "story",
        description: "Drive the story system with one control action, the way a stage click or key press does.",
        access: Access::Mutating,
        params: &[
            required(
                "action",
                Kind::Text,
                "One of start, continue, advance, choose, slot, auto, skip, log, save, load, pause, settings, settings_back.",
            ),
            optional(
                "option",
                Kind::Count,
                "Index for the choose and slot actions. Defaults to zero.",
            ),
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    // Scrape the verbs `handle_request` matches on straight out of its source,
    // so the catalog cannot drift from the dispatcher it describes.
    fn dispatch_verbs() -> Vec<String> {
        const SOURCE: &str = include_str!("dispatch.rs");
        const OPEN: &str = "let body = match cmd.as_str() {";
        const CLOSE: &str = "other => {";

        let body = SOURCE
            .split_once(OPEN)
            .expect("dispatch still opens its command match on cmd.as_str()")
            .1;
        let body = body
            .split_once(CLOSE)
            .expect("dispatch still closes its command match with an `other` arm")
            .0;

        let mut verbs = Vec::new();
        for line in body.lines() {
            verbs.extend(arm_verbs(line));
        }
        verbs
    }

    // The verb literals of one match arm, or nothing when the line is not an
    // arm whose pattern is made of string literals.
    fn arm_verbs(line: &str) -> Vec<String> {
        let Some(arrow) = line.find("=>") else {
            return Vec::new();
        };
        let mut verbs = Vec::new();
        for piece in line[..arrow].split('|') {
            let piece = piece.trim();
            let Some(verb) = piece
                .strip_prefix('"')
                .and_then(|rest| rest.strip_suffix('"'))
            else {
                return Vec::new();
            };
            verbs.push(verb.to_string());
        }
        verbs
    }

    // A payload every property of the schema fills at its declared type, so the
    // request struct in `super::commands` sees each field.
    fn sample_payload(command: &Command) -> String {
        let mut body = serde_json::Map::new();
        body.insert("cmd".to_string(), json!(command.name));
        for param in command.params {
            let value = match param.kind {
                Kind::Text | Kind::TextOrNull => json!("x"),
                Kind::Number | Kind::NumberOrNull => json!(0.5),
                Kind::Count => json!(1),
                Kind::Vec3 => json!([0.0, 0.0, 0.0]),
                Kind::Vec4 => json!([0.0, 0.0, 0.0, 0.0]),
                Kind::NumberList => json!([0.0, 1.0]),
            };
            body.insert(param.name.to_string(), value);
        }
        Value::Object(body).to_string()
    }

    #[test]
    fn catalog_covers_every_dispatch_arm() {
        let dispatched: BTreeSet<String> = dispatch_verbs().into_iter().collect();
        // A scan that stopped matching arms would otherwise pass vacuously.
        assert!(
            dispatched.len() > 25,
            "the dispatch scan found only {} verbs: {dispatched:?}",
            dispatched.len()
        );
        let catalogued: BTreeSet<String> = all().iter().map(|c| c.name.to_string()).collect();
        let missing: Vec<_> = dispatched.difference(&catalogued).collect();
        assert!(
            missing.is_empty(),
            "dispatched but not in the catalog: {missing:?}"
        );
        let extra: Vec<_> = catalogued.difference(&dispatched).collect();
        assert!(
            extra.is_empty(),
            "in the catalog but never dispatched: {extra:?}"
        );
    }

    #[test]
    fn every_schema_is_a_closed_object_schema() {
        for command in all() {
            let schema = command.schema();
            assert_eq!(schema["type"], "object", "{}", command.name);
            assert!(
                schema["properties"].is_object(),
                "{} has no properties object",
                command.name
            );
            assert_eq!(schema["additionalProperties"], false, "{}", command.name);
            let properties = schema["properties"].as_object().unwrap();
            assert_eq!(properties.len(), command.params.len(), "{}", command.name);
            for (name, property) in properties {
                assert!(
                    property.get("type").is_some(),
                    "{}.{name} has no type",
                    command.name
                );
                assert!(
                    property["description"]
                        .as_str()
                        .is_some_and(|d| !d.is_empty()),
                    "{}.{name} has no description",
                    command.name
                );
            }
        }
    }

    #[test]
    fn every_required_key_is_a_declared_property() {
        for command in all() {
            let schema = command.schema();
            let Some(required) = schema.get("required") else {
                assert!(
                    command.params.iter().all(|p| !p.required),
                    "{} dropped a required parameter",
                    command.name
                );
                continue;
            };
            let properties = schema["properties"].as_object().unwrap();
            for key in required.as_array().expect("required is an array") {
                let key = key.as_str().expect("required keys are strings");
                assert!(
                    properties.contains_key(key),
                    "{} requires undeclared '{key}'",
                    command.name
                );
            }
        }
    }

    #[test]
    fn verb_names_are_unique_and_kebab_case() {
        let mut seen = BTreeSet::new();
        for command in all() {
            assert!(seen.insert(command.name), "duplicate verb {}", command.name);
            assert!(!command.name.is_empty());
            assert!(
                !command.name.starts_with('-') && !command.name.ends_with('-'),
                "{} is not kebab-case",
                command.name
            );
            assert!(
                command
                    .name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c == '-'),
                "{} is not kebab-case",
                command.name
            );
            assert!(
                !command.name.contains("--"),
                "{} is not kebab-case",
                command.name
            );
        }
    }

    #[test]
    fn descriptions_are_one_sentence_lines() {
        for command in all() {
            assert!(
                !command.description.is_empty() && command.description.ends_with('.'),
                "{} needs a sentence description",
                command.name
            );
            assert!(
                !command.description.contains('\n'),
                "{} description spans lines",
                command.name
            );
        }
    }

    #[test]
    fn parameter_names_are_unique_within_a_command() {
        for command in all() {
            let mut seen = BTreeSet::new();
            for param in command.params {
                assert!(
                    seen.insert(param.name),
                    "{} declares {} twice",
                    command.name,
                    param.name
                );
                assert!(
                    param.description.ends_with('.'),
                    "{}.{} needs a sentence description",
                    command.name,
                    param.name
                );
            }
        }
    }

    // Every declared parameter must deserialize into the request struct the live
    // handler parses, so a schema that drifts from `super::commands` fails here.
    #[test]
    fn schemas_deserialize_into_the_request_structs() {
        for command in all().iter().filter(|c| !c.params.is_empty()) {
            let payload = sample_payload(command);
            super::super::commands::parse_probe(command.name, &payload)
                .unwrap_or_else(|e| panic!("{} rejects its own schema: {e}", command.name));
        }
    }

    #[test]
    fn lookup_and_listing_cover_the_table() {
        assert!(find("ping").is_some());
        assert!(find("nope").is_none());
        let list = verb_list();
        for command in all() {
            assert!(list.contains(command.name), "{} is unlisted", command.name);
        }
    }

    #[test]
    fn parameterless_commands_declare_empty_schemas() {
        let ping = find("ping").expect("ping is catalogued");
        assert_eq!(
            ping.schema(),
            json!({ "type": "object", "properties": {}, "additionalProperties": false })
        );
    }
}
