//! World templates: named bundles of asset specs the editor and `cn add --template`
//! layer onto a world. Each template's assets are built from the typed
//! `crate::authoring::spec::asset` builders, so a consumer gets structured specs (never a JSON
//! string to parse).
//! Every asset stays standalone (no required cross-references) and free of source-file
//! dependencies, so a template applies cleanly to any fresh world.

mod minimal_world;

use crate::authoring::spec::AssetSpec;

/// A named bundle of asset specs.
pub struct WorldTemplate {
    /// Stable machine name used on the command line and in the editor
    /// (`cn add --template <name>`).
    pub name: &'static str,
    /// Human-facing label shown in the editor's templates dropdown.
    pub title: &'static str,
    /// One-line description of what the template layers onto a world.
    pub description: &'static str,
    // Builds the template's assets. A function (not a stored slice) because an
    // `AssetSpec` owns heap data and cannot be a `const`.
    build: fn() -> Vec<AssetSpec>,
}

impl WorldTemplate {
    /// The template's assets as typed specs, in application order.
    pub fn assets(&self) -> Vec<AssetSpec> {
        (self.build)()
    }
}

/// Every engine-owned world template, in display order.
pub const TEMPLATES: &[WorldTemplate] = &[WorldTemplate {
    name: "minimal-3d-world",
    title: "Minimal 3D World",
    description: "A lit room with a camera and sky.",
    build: minimal_world::assets,
}];

/// Look up a world template by its machine `name`.
pub fn by_name(name: &str) -> Option<&'static WorldTemplate> {
    TEMPLATES.iter().find(|t| t.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every template builds at least one asset, each a well-formed spec (a
    // non-empty name and a non-empty args object). This is the structural contract
    // the engine's asset construction relies on; type-level validity against the
    // real asset schemas is checked where a std consumer can reach the registry
    // (the `cn add` template test).
    #[test]
    fn every_template_builds_well_formed_specs() {
        for t in TEMPLATES {
            let specs = t.assets();
            assert!(!specs.is_empty(), "template '{}' has no assets", t.name);
            for s in &specs {
                assert!(
                    !s.name.is_empty(),
                    "template '{}' has an unnamed asset",
                    t.name
                );
                assert!(
                    !s.fields.is_empty(),
                    "template '{}' asset '{}' sets no args",
                    t.name,
                    s.name
                );
            }
        }
    }

    // Template names are unique (the lookup key) and resolvable.
    #[test]
    fn names_are_unique_and_resolvable() {
        for (i, t) in TEMPLATES.iter().enumerate() {
            assert_eq!(by_name(t.name).map(|r| r.name), Some(t.name));
            for other in &TEMPLATES[i + 1..] {
                assert_ne!(t.name, other.name, "duplicate template name '{}'", t.name);
            }
        }
        assert!(by_name("does-not-exist").is_none());
    }

    // Asset names are unique within each template, so applying it never collides an
    // entry with itself.
    #[test]
    fn asset_names_within_a_template_are_unique() {
        for t in TEMPLATES {
            let mut names: Vec<String> = t.assets().into_iter().map(|s| s.name).collect();
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
}
