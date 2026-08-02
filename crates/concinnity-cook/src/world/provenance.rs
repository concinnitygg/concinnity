// src/world/provenance.rs
//
// Where one asset of an expanded world came from. `prepare_world` records what
// each pass injected, generated, and skipped; this reads those records back
// against a name. Both `cn list --expanded` / `cn explain` and the editor's
// Expanded tab classify rows through here, so the two cannot drift.

use super::LoadedWorld;

// The origin of one expanded-world asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provenance {
    // A world.jsonl line.
    Authored,
    // A world.jsonl line that replaces what an expansion would have produced;
    // the named source no longer drives this asset.
    AuthoredShadowing { generated_by: String },
    // Added by an injection pass (a companion, an engine default).
    Injected { by: String },
    // Produced by the expansion of the named authored asset.
    Generated { by: String },
    // Produced by a build-time macro expansion that does not record its output
    // (menus, stories, prefabs, and the other primitive-emitting passes).
    Expanded,
}

impl Provenance {
    // The authored asset or pass this came from, for grouping a listing by
    // source. `None` for a plain authored line and the unattributed expansions.
    pub fn source(&self) -> Option<&str> {
        match self {
            Provenance::AuthoredShadowing { generated_by } => Some(generated_by),
            Provenance::Injected { by } => Some(by),
            Provenance::Generated { by } => Some(by),
            Provenance::Authored | Provenance::Expanded => None,
        }
    }

    // Whether the asset has a world.jsonl line of its own.
    pub fn is_authored(&self) -> bool {
        matches!(
            self,
            Provenance::Authored | Provenance::AuthoredShadowing { .. }
        )
    }

    // Whether copying this asset's entry into world.jsonl overrides it rather
    // than duplicating it: only the passes that skip a name the world claims
    // (scene imports, injections) can be overridden this way. The macro
    // expansions emit their primitives unconditionally, so a copy of one would
    // land beside the generated asset, not replace it.
    pub fn is_overridable(&self) -> bool {
        matches!(
            self,
            Provenance::Injected { .. } | Provenance::Generated { .. }
        )
    }
}

impl std::fmt::Display for Provenance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provenance::Authored => write!(f, "authored"),
            Provenance::AuthoredShadowing { generated_by } => {
                write!(f, "authored (shadows {})", generated_by)
            }
            Provenance::Injected { by } => write!(f, "injected:{}", by),
            Provenance::Generated { by } => write!(f, "generated:{}", by),
            Provenance::Expanded => write!(f, "expanded"),
        }
    }
}

impl LoadedWorld {
    // Where the asset called `name` in this expanded world came from.
    pub fn provenance(&self, name: &str) -> Provenance {
        if self.authored.iter().any(|n| n == name) {
            return match self.shadowed.iter().find(|s| s.name == name) {
                Some(s) => Provenance::AuthoredShadowing {
                    generated_by: s.generated_by.clone(),
                },
                None => Provenance::Authored,
            };
        }
        if let Some(i) = self.injected.iter().find(|i| i.name == name) {
            return Provenance::Injected {
                by: i.injected_by.to_string(),
            };
        }
        if let Some(g) = self.generated.iter().find(|g| g.name == name) {
            return Provenance::Generated {
                by: g.generated_by.clone(),
            };
        }
        Provenance::Expanded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{GeneratedAsset, InjectedAsset, ShadowedAsset};

    fn world() -> LoadedWorld {
        LoadedWorld {
            assets: Vec::new(),
            injected: vec![InjectedAsset {
                name: "hud_font".to_string(),
                asset_type: "Font".to_string(),
                args: serde_json::json!({}),
                injected_by: "debug_hud",
            }],
            generated: vec![GeneratedAsset {
                name: "bistro_mat_wood".to_string(),
                asset_type: "Material".to_string(),
                generated_by: "bistro".to_string(),
            }],
            shadowed: vec![ShadowedAsset {
                name: "bistro_mat_glass".to_string(),
                asset_type: "Material".to_string(),
                generated_by: "bistro".to_string(),
                args: serde_json::json!({}),
            }],
            authored: vec!["cam".to_string(), "bistro_mat_glass".to_string()],
        }
    }

    #[test]
    fn each_record_classifies_its_names() {
        let w = world();
        assert_eq!(w.provenance("cam"), Provenance::Authored);
        assert_eq!(
            w.provenance("hud_font"),
            Provenance::Injected {
                by: "debug_hud".to_string()
            }
        );
        assert_eq!(
            w.provenance("bistro_mat_wood"),
            Provenance::Generated {
                by: "bistro".to_string()
            }
        );
        assert_eq!(
            w.provenance("bistro_mat_glass"),
            Provenance::AuthoredShadowing {
                generated_by: "bistro".to_string()
            }
        );
        // A macro expansion's primitives record nothing, so they fall back.
        assert_eq!(w.provenance("main_menu_tab_0"), Provenance::Expanded);
    }

    // The display strings are what `cn list --expanded` prints.
    #[test]
    fn display_matches_the_listing_vocabulary() {
        let w = world();
        assert_eq!(w.provenance("cam").to_string(), "authored");
        assert_eq!(w.provenance("hud_font").to_string(), "injected:debug_hud");
        assert_eq!(
            w.provenance("bistro_mat_wood").to_string(),
            "generated:bistro"
        );
        assert_eq!(
            w.provenance("bistro_mat_glass").to_string(),
            "authored (shadows bistro)"
        );
        assert_eq!(w.provenance("nothing").to_string(), "expanded");
    }

    #[test]
    fn source_names_the_producing_asset_or_pass() {
        let w = world();
        assert_eq!(w.provenance("bistro_mat_wood").source(), Some("bistro"));
        assert_eq!(w.provenance("hud_font").source(), Some("debug_hud"));
        assert_eq!(w.provenance("bistro_mat_glass").source(), Some("bistro"));
        assert_eq!(w.provenance("cam").source(), None);
        assert_eq!(w.provenance("nothing").source(), None);
    }

    // Only the passes that skip a claimed name can be overridden by a copy; the
    // unattributed macro expansions would duplicate instead.
    #[test]
    fn only_skipping_passes_are_overridable() {
        let w = world();
        assert!(w.provenance("bistro_mat_wood").is_overridable());
        assert!(w.provenance("hud_font").is_overridable());
        assert!(!w.provenance("nothing").is_overridable());
        assert!(!w.provenance("cam").is_overridable());
        // An asset already overridden is authored, so it is not offered again.
        assert!(!w.provenance("bistro_mat_glass").is_overridable());
    }

    #[test]
    fn is_authored_covers_a_plain_line_and_an_overriding_copy() {
        let w = world();
        assert!(w.provenance("cam").is_authored());
        assert!(w.provenance("bistro_mat_glass").is_authored());
        assert!(!w.provenance("bistro_mat_wood").is_authored());
        assert!(!w.provenance("nothing").is_authored());
    }
}
