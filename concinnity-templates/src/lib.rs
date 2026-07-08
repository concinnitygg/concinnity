// concinnity-templates
//
// Engine-owned content templates: named bundles of world.jsonl entries the
// editor and `cn add --template` layer onto a world. Pure static data with zero
// runtime dependencies (`#![no_std]`), so it can never drift into logic and is
// consumable from std and no_std alike. Each template's payload is a
// `&'static str` of JSONL (one asset object per line) in exactly the format the
// engine's `parse_world_jsonl` already speaks; consumers parse it themselves.
//
// To add a template: write its JSONL as a `const &str` below and add a
// `Template` entry to `TEMPLATES`. Keep every asset standalone (no required
// cross-references) and free of source-file dependencies (e.g. a procedural sky,
// not an `.hdr`), so a template applies cleanly to any fresh world.

#![no_std]

// The test harness (and serde_json, the dev-only validator) link std; name it so
// the `#[cfg(test)]` module can use `std::vec::Vec` / `String` for its checks.
// The library target itself pulls in nothing beyond `core`.
#[cfg(test)]
extern crate std;

/// A named bundle of world.jsonl entries.
pub struct Template {
    /// Stable machine name used on the command line and in the editor
    /// (`cn add --template <name>`).
    pub name: &'static str,
    /// Human-facing label shown in the editor's templates dropdown.
    pub title: &'static str,
    /// One-line description of what the template layers onto a world.
    pub description: &'static str,
    /// The template's entries as JSONL: one asset object per line, in the format
    /// `parse_world_jsonl` accepts. Consumers parse this into asset values.
    pub jsonl: &'static str,
}

// Aesthetic-defaults template: layers visual polish so a plain scene lands in a
// richer setting -- a warm key light, bloom + tonemap, an IBL environment from
// the procedural sky (no .hdr needed), and volumetric fog.
const SHOWCASE: &str = r#"{"name":"ambient_light","type":"DirectionalLight","args":{"color":[1.0,0.95,0.8],"direction":[-0.3,0.85,0.4],"intensity":1.1}}
{"name":"showcase_post","type":"PostProcessConfig","args":{"bloom_intensity":0.7,"bloom_threshold":1.1,"exposure_ev":0.0}}
{"name":"showcase_env","type":"EnvironmentMap","args":{"generator":"sky"}}
{"name":"showcase_fog","type":"VolumetricFog","args":{"density":0.02,"color":[0.75,0.82,0.95],"height_falloff":0.18,"max_distance":180.0,"phase_g":0.5}}"#;

// A neutral three-point-ish lighting rig: a key directional plus two fill point
// lights, so an unlit scene reads with shape and depth without any post stack.
const LIT_STAGE: &str = r#"{"name":"stage_key","type":"DirectionalLight","args":{"color":[1.0,0.97,0.92],"direction":[-0.4,0.8,0.3],"intensity":2.2}}
{"name":"stage_fill_l","type":"PointLight","args":{"color":[0.6,0.7,1.0],"position":[-4.0,3.0,2.0],"intensity":8.0,"range":20.0}}
{"name":"stage_fill_r","type":"PointLight","args":{"color":[1.0,0.8,0.6],"position":[4.0,2.5,-2.0],"intensity":6.0,"range":20.0}}"#;

// An atmosphere pass: volumetric fog plus a procedural-sky environment, for a
// hazy outdoor mood without touching the lighting.
const ATMOSPHERE: &str = r#"{"name":"atmo_env","type":"EnvironmentMap","args":{"generator":"sky"}}
{"name":"atmo_fog","type":"VolumetricFog","args":{"density":0.035,"color":[0.7,0.78,0.9],"height_falloff":0.12,"max_distance":140.0,"phase_g":0.6}}"#;

/// Every engine-owned template, in display order.
pub const TEMPLATES: &[Template] = &[
    Template {
        name: "showcase",
        title: "Showcase polish",
        description: "Warm key light, bloom + tonemap, sky IBL, and volumetric fog.",
        jsonl: SHOWCASE,
    },
    Template {
        name: "lit-stage",
        title: "Lit stage",
        description: "A key directional plus two fill point lights.",
        jsonl: LIT_STAGE,
    },
    Template {
        name: "atmosphere",
        title: "Atmosphere",
        description: "Procedural-sky environment and volumetric fog.",
        jsonl: ATMOSPHERE,
    },
];

/// Look up a template by its machine `name`.
pub fn by_name(name: &str) -> Option<&'static Template> {
    TEMPLATES.iter().find(|t| t.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::string::String;
    use std::vec::Vec;

    // Every template's JSONL parses as one JSON object per line, each a
    // well-formed asset (a `name`, a `type`, and an `args` object). This is the
    // contract the engine's `parse_world_jsonl` + asset construction rely on.
    #[test]
    fn every_template_jsonl_is_well_formed() {
        for t in TEMPLATES {
            let mut lines = 0;
            for line in t.jsonl.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                lines += 1;
                let v: serde_json::Value = serde_json::from_str(line)
                    .unwrap_or_else(|e| panic!("template '{}' line is not JSON: {e}", t.name));
                assert!(
                    v.get("name").and_then(|n| n.as_str()).is_some(),
                    "template '{}' entry missing a string name",
                    t.name
                );
                assert!(
                    v.get("type").and_then(|n| n.as_str()).is_some(),
                    "template '{}' entry missing a string type",
                    t.name
                );
                assert!(
                    v.get("args").map(|a| a.is_object()).unwrap_or(false),
                    "template '{}' entry args must be an object",
                    t.name
                );
            }
            assert!(lines > 0, "template '{}' has no entries", t.name);
        }
    }

    // Template names are unique (they are the lookup key) and resolvable.
    #[test]
    fn names_are_unique_and_resolvable() {
        for (i, t) in TEMPLATES.iter().enumerate() {
            assert_eq!(
                by_name(t.name).map(|r| r.name),
                Some(t.name),
                "by_name resolves '{}'",
                t.name
            );
            for other in &TEMPLATES[i + 1..] {
                assert_ne!(t.name, other.name, "duplicate template name '{}'", t.name);
            }
        }
        assert!(by_name("does-not-exist").is_none());
    }

    // Asset names are unique across each template so applying it never collides
    // an entry with itself.
    #[test]
    fn asset_names_within_a_template_are_unique() {
        for t in TEMPLATES {
            let mut names = alloc_names(t.jsonl);
            names.sort();
            let before = names.len();
            names.dedup();
            assert_eq!(
                before,
                names.len(),
                "template '{}' has duplicate asset names",
                t.name
            );
        }
    }

    fn alloc_names(jsonl: &str) -> Vec<String> {
        jsonl
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .filter_map(|l| {
                serde_json::from_str::<serde_json::Value>(l)
                    .ok()
                    .and_then(|v| v.get("name").and_then(|n| n.as_str()).map(String::from))
            })
            .collect()
    }
}
