// src/editor/lighting.rs
//
// The data half of the Lighting panel: a curated view over the lighting-related
// fields of several existing world assets (the sun, fog, shadows, and ambient /
// GI settings), each section bound to one asset type. The panel does not invent
// its own field machinery -- every binding resolves to a `form::FormField`
// derived from the type's registered args (`form::fields_for_with`), and edits
// commit through the same `form::assemble` + `form::validate` path the add /
// edit form uses, so the panel can never write an invalid entry. Pure and
// world-free; the hook owns the entries and the panel module owns the layout.

use super::form::{self, FormField};
use serde_json::{Map, Value};

// One themed section: a heading, the asset type whose first world entry it
// edits, vector paths to disclose into per-element leaves, and the curated
// `(arg path, row caption)` list shown under the heading.
pub(crate) struct Section {
    pub title: &'static str,
    pub ty: &'static str,
    pub expand: &'static [&'static str],
    pub fields: &'static [(&'static str, &'static str)],
}

// The curated lighting view. Each section binds to the FIRST entry of its type
// (the lighting assets are effectively singletons); a world without one shows an
// add row instead of the fields.
pub(crate) const SECTIONS: &[Section] = &[
    Section {
        title: "Sun",
        ty: "DirectionalLight",
        expand: &["direction"],
        fields: &[
            ("direction.0", "direction x"),
            ("direction.1", "direction y"),
            ("direction.2", "direction z"),
            ("color", "color"),
            ("intensity", "intensity"),
        ],
    },
    Section {
        title: "Fog",
        ty: "VolumetricFog",
        expand: &[],
        fields: &[
            ("enabled", "enabled"),
            ("color", "color"),
            ("density", "density"),
        ],
    },
    Section {
        title: "Shadows",
        ty: "GraphicsConfig",
        expand: &[],
        fields: &[
            ("shadow_map_size", "map size"),
            ("shadow_distance", "distance"),
            ("shadow_cascades", "cascades"),
        ],
    },
    Section {
        title: "Ambient / GI",
        ty: "PostProcessConfig",
        expand: &[],
        fields: &[
            ("ambient_intensity", "ambient"),
            ("ssgi_intensity", "SSGI"),
            ("exposure_ev", "exposure EV"),
        ],
    },
];

// The total binding count across every section: the size of the panel's control
// pools, and the global index space (`binding(i)`).
pub(crate) fn binding_count() -> usize {
    SECTIONS.iter().map(|s| s.fields.len()).sum()
}

// Binding `i` as `(section index, arg path, row caption)`.
pub(crate) fn binding(i: usize) -> (usize, &'static str, &'static str) {
    let mut base = 0;
    for (s, section) in SECTIONS.iter().enumerate() {
        if i < base + section.fields.len() {
            let (path, caption) = section.fields[i - base];
            return (s, path, caption);
        }
        base += section.fields.len();
    }
    panic!("lighting binding {i} out of range");
}

// The global binding index of section `s`'s first field.
pub(crate) fn section_base(s: usize) -> usize {
    SECTIONS[..s].iter().map(|sec| sec.fields.len()).sum()
}

// The `FormField`s backing section `s` for an entry's args (the type defaults
// when adding), in the section's declared order. Derived through the same
// `fields_for_with` the edit form uses, so kinds / current values / colour
// detection all match; a curated path missing from the derivation (a schema
// change) is simply dropped from the panel rather than erroring.
pub(crate) fn section_fields(s: &Section, args: Option<&Map<String, Value>>) -> Vec<FormField> {
    let expanded: std::collections::HashSet<String> =
        s.expand.iter().map(|p| p.to_string()).collect();
    let all = form::fields_for_with(s.ty, args, &expanded);
    s.fields
        .iter()
        .filter_map(|(path, _)| all.iter().find(|f| f.key == *path).cloned())
        .collect()
}

// One panel row: a section heading, a bound field (global binding index), or the
// add row shown when the section's asset type has no world entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Row {
    Header(usize),
    Field(usize),
    Add(usize),
}

// The panel's rows for the given per-section presence flags.
pub(crate) fn rows(present: &[bool]) -> Vec<Row> {
    let mut out = Vec::new();
    for (s, section) in SECTIONS.iter().enumerate() {
        out.push(Row::Header(s));
        if present.get(s).copied().unwrap_or(false) {
            let base = section_base(s);
            out.extend((0..section.fields.len()).map(|j| Row::Field(base + j)));
        } else {
            out.push(Row::Add(s));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every curated binding resolves against the type's real registered args:
    // a schema rename breaks this test instead of silently dropping the row.
    #[test]
    fn every_binding_resolves_to_a_registered_field() {
        for section in SECTIONS {
            let fields = section_fields(section, None);
            assert_eq!(
                fields.len(),
                section.fields.len(),
                "{}: every curated path derives a field",
                section.ty
            );
            for (field, (path, _)) in fields.iter().zip(section.fields) {
                assert_eq!(&field.key, path, "{}: declared order kept", section.ty);
            }
        }
    }

    // The derived kinds are what the panel renders: text for numbers, a checkbox
    // for the fog toggle, colour fields flagged for a swatch.
    #[test]
    fn derived_kinds_match_the_panel_controls() {
        use super::super::form::FieldKind;
        let sun = section_fields(&SECTIONS[0], None);
        assert!(matches!(sun[0].kind, FieldKind::Float), "direction leaf");
        assert!(
            matches!(sun[3].kind, FieldKind::Vec { color: true, .. }),
            "sun color draws a swatch"
        );
        let fog = section_fields(&SECTIONS[1], None);
        assert!(
            matches!(fog[0].kind, FieldKind::Bool),
            "fog enabled toggles"
        );
        let shadows = section_fields(&SECTIONS[2], None);
        assert!(matches!(shadows[0].kind, FieldKind::Int));
    }

    // Seeded args flow into the derived fields' current values.
    #[test]
    fn section_fields_read_the_entry_values() {
        let args: Map<String, Value> = serde_json::from_value(serde_json::json!({
            "intensity": 3.5, "direction": [0.1, 0.9, 0.2]
        }))
        .unwrap();
        let sun = section_fields(&SECTIONS[0], Some(&args));
        assert_eq!(sun[4].initial, "3.5");
        assert_eq!(sun[1].initial, "0.9", "expanded leaf reads its element");
    }

    #[test]
    fn binding_index_round_trips() {
        assert!(binding_count() >= 12);
        let (s, path, _) = binding(0);
        assert_eq!((s, path), (0, "direction.0"));
        let last = binding_count() - 1;
        let (s, path, _) = binding(last);
        assert_eq!((s, path), (SECTIONS.len() - 1, "exposure_ev"));
        assert_eq!(section_base(1), SECTIONS[0].fields.len());
    }

    // A missing section collapses to its add row; a present one lists fields.
    #[test]
    fn rows_swap_fields_for_an_add_row_when_absent() {
        let rows = rows(&[true, false, true, true]);
        assert_eq!(rows[0], Row::Header(0));
        assert_eq!(rows[1], Row::Field(0));
        let fog_header = rows
            .iter()
            .position(|r| *r == Row::Header(1))
            .expect("fog section present");
        assert_eq!(rows[fog_header + 1], Row::Add(1));
        assert_eq!(rows[fog_header + 2], Row::Header(2));
        assert!(!rows.contains(&Row::Field(section_base(1))));
    }
}
