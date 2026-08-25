// src/editor/character_shape.rs
//
// The data half of the CharacterShape panel: the slider rows a schema lays
// out over what the target mesh exposes (its morph-target and joint names),
// the panel sections, and the value mapping between a row and the
// `ShapeSlider` / `JointProportion` entries it reads and writes. Pure and
// world-free; the hook owns the entries and `character_shape_panel.rs` the
// layout.

use crate::components::{CharacterSchema, JointProportion, ShapePreset, ShapeSlider};
use rand::{Rng, SeedableRng};

// The trailing section for keys and groups the schema does not place.
pub(crate) const OTHER_SECTION: &str = "Other";

// What a slider row drives.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RowKind {
    // A `name+` / `name-` target pair: one slider in [-1, 1].
    Bipolar,
    // A bare target: one slider in [0, 1].
    Unipolar,
    // A proportion group: one slider in [-1, 1] written as `scale` and / or
    // `length` at full deflection onto every listed joint the skeleton has.
    Proportion {
        joints: Vec<String>,
        scale: f32,
        length: f32,
    },
}

impl RowKind {
    // The slider's value range.
    pub(crate) fn range(&self) -> (f32, f32) {
        match self {
            RowKind::Unipolar => (0.0, 1.0),
            _ => (-1.0, 1.0),
        }
    }
}

// One slider row: the name it reads / writes (the slider name for a morph
// row, the group name for a proportion row), its caption, the section it
// shows under, and what it drives.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SliderRow {
    pub name: String,
    pub caption: String,
    pub section: usize,
    pub kind: RowKind,
}

// The rows a schema lays out for a mesh: section captions in display order
// and the sliders, each naming its section.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Rows {
    pub sections: Vec<String>,
    pub sliders: Vec<SliderRow>,
}

// The base name and polarity a morph-target name implies.
fn split_target(target: &str) -> (&str, bool) {
    match target
        .strip_suffix('+')
        .or_else(|| target.strip_suffix('-'))
    {
        Some(base) if !base.is_empty() => (base, true),
        _ => (target, false),
    }
}

// Every base name on the mesh with whether it came as a pair, in target order.
fn mesh_keys(morph_names: &[String]) -> Vec<(String, bool)> {
    let mut out: Vec<(String, bool)> = Vec::new();
    for target in morph_names {
        let (base, polar) = split_target(target);
        if let Some(existing) = out.iter_mut().find(|(n, _)| n == base) {
            existing.1 |= polar;
        } else {
            out.push((base.to_string(), polar));
        }
    }
    out
}

fn morph_kind(polar: bool) -> RowKind {
    if polar {
        RowKind::Bipolar
    } else {
        RowKind::Unipolar
    }
}

// The rows `schema` lays out over a mesh with `morph_names` and
// `joint_names`: each panel section in order with the keys and proportion
// groups of its regions (keys the mesh lacks and groups with no joint in the
// skeleton are skipped), then an "Other" section for the mesh's targets no
// schema key names and for groups whose region no section lists.
pub(crate) fn derive_rows(
    schema: &CharacterSchema,
    morph_names: &[String],
    joint_names: &[String],
) -> Rows {
    let on_mesh = mesh_keys(morph_names);
    let keys = schema.all_keys();
    let mut rows = Rows::default();
    let mut placed_keys: Vec<&str> = Vec::new();
    let mut placed_groups: Vec<&str> = Vec::new();
    for section in &schema.panel {
        let index = rows.sections.len();
        rows.sections.push(section.caption.clone());
        for region in &section.regions {
            for key in keys.iter().filter(|k| k.region == *region) {
                let Some((_, polar)) = on_mesh.iter().find(|(n, _)| *n == key.name) else {
                    continue;
                };
                placed_keys.push(&key.name);
                rows.sliders.push(SliderRow {
                    name: key.name.clone(),
                    caption: key.caption().to_string(),
                    section: index,
                    kind: morph_kind(*polar),
                });
            }
            for group in schema
                .proportion_groups
                .iter()
                .filter(|g| g.region == *region)
            {
                if !group.joints.iter().any(|j| joint_names.contains(j)) {
                    continue;
                }
                placed_groups.push(&group.name);
                rows.sliders.push(SliderRow {
                    name: group.name.clone(),
                    caption: group.caption().to_string(),
                    section: index,
                    kind: RowKind::Proportion {
                        joints: group.joints.clone(),
                        scale: group.scale,
                        length: group.length,
                    },
                });
            }
        }
    }
    let other = rows.sections.len();
    for (name, polar) in &on_mesh {
        if placed_keys.contains(&name.as_str()) || keys.iter().any(|k| k.name == *name) {
            continue;
        }
        rows.sliders.push(SliderRow {
            name: name.clone(),
            caption: name.clone(),
            section: other,
            kind: morph_kind(*polar),
        });
    }
    for group in &schema.proportion_groups {
        if placed_groups.contains(&group.name.as_str())
            || !group.joints.iter().any(|j| joint_names.contains(j))
        {
            continue;
        }
        rows.sliders.push(SliderRow {
            name: group.name.clone(),
            caption: group.caption().to_string(),
            section: other,
            kind: RowKind::Proportion {
                joints: group.joints.clone(),
                scale: group.scale,
                length: group.length,
            },
        });
    }
    if rows.sliders.iter().any(|r| r.section == other) {
        rows.sections.push(OTHER_SECTION.to_string());
    }
    rows
}

// The edited values of one shape: its slider and proportion entries, as
// authored (entries naming things the mesh lacks ride along untouched).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ShapeValues {
    pub sliders: Vec<ShapeSlider>,
    pub proportions: Vec<JointProportion>,
}

impl ShapeValues {
    // The slider value a row shows. A proportion row reads the mean of its
    // present joints' deflection; a missing entry reads as 0.
    pub(crate) fn get(&self, row: &SliderRow) -> f32 {
        match &row.kind {
            RowKind::Bipolar | RowKind::Unipolar => self
                .sliders
                .iter()
                .find(|s| s.name == row.name)
                .map_or(0.0, |s| s.value),
            RowKind::Proportion {
                joints,
                scale,
                length,
            } => {
                let values: Vec<f32> = self
                    .proportions
                    .iter()
                    .filter(|p| joints.contains(&p.joint))
                    .map(|p| {
                        if *scale != 0.0 {
                            (p.scale - 1.0) / scale
                        } else if *length != 0.0 {
                            p.length / length
                        } else {
                            0.0
                        }
                    })
                    .collect();
                if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f32>() / values.len() as f32
                }
            }
        }
    }

    // Write `value` (clamped to the row's range) into the entries the row
    // drives. A zero slider or identity proportion drops its entry, so a
    // reset shape serializes empty. `joints` limits a proportion row to the
    // joints the skeleton has.
    pub(crate) fn set(&mut self, row: &SliderRow, value: f32, joints: &[String]) {
        let (lo, hi) = row.kind.range();
        let value = round2(value.clamp(lo, hi));
        match &row.kind {
            RowKind::Bipolar | RowKind::Unipolar => {
                self.sliders.retain(|s| s.name != row.name);
                if value != 0.0 {
                    self.sliders.push(ShapeSlider {
                        name: row.name.clone(),
                        value,
                    });
                }
            }
            RowKind::Proportion {
                joints: group,
                scale,
                length,
            } => {
                for joint in group {
                    if !joints.contains(joint) {
                        continue;
                    }
                    let entry = self.proportions.iter().find(|p| p.joint == *joint);
                    let mut p = entry.cloned().unwrap_or_else(|| JointProportion {
                        joint: joint.clone(),
                        ..Default::default()
                    });
                    if *scale != 0.0 {
                        p.scale = round3(1.0 + value * scale);
                    }
                    if *length != 0.0 {
                        p.length = round3(value * length);
                    }
                    self.proportions.retain(|q| q.joint != *joint);
                    if p.scale != 1.0 || p.length != 0.0 {
                        self.proportions.push(p);
                    }
                }
            }
        }
    }

    // Every row to its neutral value.
    pub(crate) fn reset(&mut self, rows: &[SliderRow], joints: &[String]) {
        for row in rows {
            self.set(row, 0.0, joints);
        }
    }

    // Every row to a seeded random value inside `RANDOM_BAND`, so the body
    // stays plausible.
    pub(crate) fn randomize(&mut self, rows: &[SliderRow], joints: &[String], seed: u64) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for row in rows {
            let (lo, hi) = random_band(&row.kind);
            self.set(row, rng.gen_range(lo..=hi), joints);
        }
    }

    // The preset's slider vector, replacing every value.
    pub(crate) fn apply_preset(&mut self, preset: &ShapePreset) {
        self.sliders = preset.sliders.clone();
        self.proportions = preset.proportions.clone();
    }
}

// How far Randomize may push a row: well inside the slider range, so a
// random body never hits the extremes that read as caricature.
pub(crate) const RANDOM_BAND: f32 = 0.6;

// The value band Randomize draws a row from.
pub(crate) fn random_band(kind: &RowKind) -> (f32, f32) {
    let (lo, hi) = kind.range();
    (lo * RANDOM_BAND, hi * RANDOM_BAND)
}

fn round2(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}

fn round3(v: f32) -> f32 {
    (v * 1000.0).round() / 1000.0
}

// One panel row: a section heading (index into `Rows::sections`, or the
// presets heading), a preset button, a slider (index into the derived
// sliders), or the add row shown when the selected mesh has no shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Row {
    Header(usize),
    PresetHeader,
    Preset(usize),
    Slider(usize),
    Add,
}

// The panel's rows: the presets (when the schema has any), then each
// section heading followed by its sliders. An absent shape shows the add
// row alone.
pub(crate) fn rows(derived: &Rows, presets: usize, has_shape: bool) -> Vec<Row> {
    if !has_shape {
        return vec![Row::Add];
    }
    let mut out = Vec::new();
    if presets > 0 {
        out.push(Row::PresetHeader);
        out.extend((0..presets).map(Row::Preset));
    }
    for (section, _) in derived.sections.iter().enumerate() {
        let members: Vec<usize> = derived
            .sliders
            .iter()
            .enumerate()
            .filter(|(_, r)| r.section == section)
            .map(|(i, _)| i)
            .collect();
        if members.is_empty() {
            continue;
        }
        out.push(Row::Header(section));
        out.extend(members.into_iter().map(Row::Slider));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_cook::character::builtin_schema;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    // The bundled humanoid's bones and keys (phase 2).
    fn humanoid_joints() -> Vec<String> {
        builtin_schema::humanoid()
            .joints
            .iter()
            .map(|j| j.name.clone())
            .collect()
    }

    fn humanoid_keys() -> Vec<String> {
        builtin_schema::humanoid().required_target_names()
    }

    fn by_name<'a>(rows: &'a Rows, name: &str) -> &'a SliderRow {
        rows.sliders.iter().find(|r| r.name == name).expect(name)
    }

    #[test]
    fn pairs_collapse_and_bare_names_stay_unipolar() {
        let schema = builtin_schema::humanoid();
        let rows = derive_rows(
            schema,
            &names(&["weight+", "weight-", "muscle", "tail+"]),
            &[],
        );
        assert_eq!(by_name(&rows, "weight").kind, RowKind::Bipolar);
        assert_eq!(by_name(&rows, "muscle").kind, RowKind::Unipolar);
        // A lone `+` target still reads as a bipolar row, under Other.
        let tail = by_name(&rows, "tail");
        assert_eq!(tail.kind, RowKind::Bipolar);
        assert_eq!(rows.sections[tail.section], OTHER_SECTION);
        assert_eq!(rows.sliders.len(), 3, "one row per base name");
        // A bare `+` or `-` is a name of its own, not an empty pair.
        let odd = derive_rows(schema, &names(&["+"]), &[]);
        assert_eq!(odd.sliders[0].name, "+");
        assert_eq!(odd.sliders[0].kind, RowKind::Unipolar);
    }

    #[test]
    fn rows_follow_the_schema_sections_with_unknowns_last() {
        let schema = builtin_schema::humanoid();
        let rows = derive_rows(
            schema,
            &names(&["tail", "jaw+", "jaw-", "weight+", "weight-"]),
            &[],
        );
        let sections: Vec<&str> = rows
            .sliders
            .iter()
            .map(|r| rows.sections[r.section].as_str())
            .collect();
        assert_eq!(sections, ["Face", "Torso", "Other"]);
        assert_eq!(rows.sliders[2].name, "tail");
        assert_eq!(by_name(&rows, "jaw").caption, "jaw");
        // Captions come from the schema.
        let rows = derive_rows(schema, &names(&["biceps", "belly_lower"]), &[]);
        assert_eq!(by_name(&rows, "biceps").caption, "biceps");
        assert_eq!(by_name(&rows, "belly_lower").caption, "lower belly");
        // A schema with no sections puts everything under Other.
        let blank = CharacterSchema::default();
        let rows = derive_rows(&blank, &names(&["jaw+", "jaw-"]), &[]);
        assert_eq!(rows.sections, [OTHER_SECTION]);
    }

    #[test]
    fn humanoid_derives_every_key_and_group() {
        let schema = builtin_schema::humanoid();
        let rows = derive_rows(schema, &humanoid_keys(), &humanoid_joints());
        let morph = rows
            .sliders
            .iter()
            .filter(|r| !matches!(r.kind, RowKind::Proportion { .. }))
            .count();
        assert_eq!(morph, 13, "22 keys collapse into 13 rows");
        let groups = rows
            .sliders
            .iter()
            .filter(|r| matches!(r.kind, RowKind::Proportion { .. }))
            .count();
        assert_eq!(
            groups,
            schema.proportion_groups.len(),
            "every group has a joint"
        );
        assert!(!rows.sections.iter().any(|s| s == OTHER_SECTION));
        // A skeleton without arms loses only the arm groups.
        let legs_only = derive_rows(schema, &[], &names(&["thigh_l", "thigh_r"]));
        assert_eq!(legs_only.sliders.len(), 1);
        assert_eq!(legs_only.sliders[0].name, "leg_length");
        assert_eq!(legs_only.sliders[0].caption, "leg length");
    }

    #[test]
    fn proportion_rows_map_to_their_joints_and_back() {
        let schema = builtin_schema::humanoid();
        let joints = humanoid_joints();
        let rows = derive_rows(schema, &[], &joints);
        let legs = by_name(&rows, "leg_length");
        let height = by_name(&rows, "height");
        let mut v = ShapeValues::default();
        v.set(legs, 0.5, &joints);
        let mut written: Vec<(&str, f32)> = v
            .proportions
            .iter()
            .map(|p| (p.joint.as_str(), p.length))
            .collect();
        written.sort_by(|a, b| a.0.cmp(b.0));
        assert_eq!(written, [("thigh_l", 0.03), ("thigh_r", 0.03)]);
        assert!(
            (v.get(legs) - 0.5).abs() < 1e-4,
            "reads back: {}",
            v.get(legs)
        );
        v.set(height, -1.0, &joints);
        let spine = v.proportions.iter().find(|p| p.joint == "spine").unwrap();
        assert!((spine.scale - 0.92).abs() < 1e-4);
        assert!((v.get(height) + 1.0).abs() < 1e-4);
        // Identity drops the entry; a joint the skeleton lacks is never written.
        v.set(legs, 0.0, &joints);
        assert!(v.proportions.iter().all(|p| !p.joint.starts_with("thigh")));
        v.set(legs, 1.0, &names(&["thigh_l"]));
        assert_eq!(
            v.proportions
                .iter()
                .filter(|p| p.joint.starts_with("thigh"))
                .count(),
            1
        );
    }

    #[test]
    fn slider_rows_write_clamped_values_and_drop_zero() {
        let schema = builtin_schema::humanoid();
        let rows = derive_rows(schema, &names(&["muscle", "jaw+", "jaw-"]), &[]);
        let mut v = ShapeValues::default();
        let muscle = by_name(&rows, "muscle");
        let jaw = by_name(&rows, "jaw");
        v.set(muscle, -0.4, &[]);
        assert!(v.sliders.is_empty(), "a unipolar row clamps at 0 and drops");
        v.set(muscle, 1.7, &[]);
        assert_eq!(v.get(muscle), 1.0);
        v.set(jaw, -0.333, &[]);
        assert_eq!(v.get(jaw), -0.33, "rounded to two decimals");
        // Entries the mesh lacks ride along.
        v.sliders.push(ShapeSlider {
            name: "tail".into(),
            value: 0.2,
        });
        v.reset(&rows.sliders, &[]);
        assert_eq!(v.sliders.len(), 1);
        assert_eq!(v.sliders[0].name, "tail");
    }

    #[test]
    fn randomize_is_seeded_and_stays_in_band() {
        let schema = builtin_schema::humanoid();
        let joints = humanoid_joints();
        let rows = derive_rows(schema, &humanoid_keys(), &joints);
        let mut a = ShapeValues::default();
        let mut b = ShapeValues::default();
        a.randomize(&rows.sliders, &joints, 7);
        b.randomize(&rows.sliders, &joints, 7);
        assert_eq!(a, b, "the same seed gives the same body");
        let mut c = ShapeValues::default();
        c.randomize(&rows.sliders, &joints, 8);
        assert_ne!(a, c, "a different seed gives a different body");
        for row in &rows.sliders {
            let (lo, hi) = random_band(&row.kind);
            let v = a.get(row);
            assert!(v >= lo - 1e-3 && v <= hi + 1e-3, "{}: {v}", row.name);
        }
        assert!(
            a.sliders
                .iter()
                .all(|s| s.value.abs() <= RANDOM_BAND + 1e-3)
        );
    }

    #[test]
    fn a_preset_replaces_every_value() {
        let schema = builtin_schema::humanoid();
        let mut v = ShapeValues::default();
        v.sliders.push(ShapeSlider {
            name: "nose".into(),
            value: 0.9,
        });
        let heavy = schema.presets.iter().find(|p| p.name == "heavy").unwrap();
        v.apply_preset(heavy);
        assert_eq!(v.sliders, heavy.sliders);
        assert!(v.sliders.iter().all(|s| s.name != "nose"));
        let tall = schema.presets.iter().find(|p| p.name == "tall").unwrap();
        v.apply_preset(tall);
        assert_eq!(v.proportions, tall.proportions);
    }

    #[test]
    fn rows_show_presets_headings_per_section_or_the_add_row() {
        let schema = builtin_schema::humanoid();
        let derived = derive_rows(schema, &names(&["jaw+", "jaw-", "tail"]), &names(&["head"]));
        let rows = rows(&derived, 2, true);
        assert_eq!(rows[0], Row::PresetHeader);
        assert_eq!(rows[1], Row::Preset(0));
        assert_eq!(rows[2], Row::Preset(1));
        assert_eq!(rows[3], Row::Header(0));
        assert_eq!(derived.sections[0], "Face");
        assert_eq!(rows[4], Row::Slider(0));
        assert_eq!(rows[5], Row::Slider(1), "head size joins the face");
        let other = derived.sections.len() - 1;
        assert_eq!(rows[6], Row::Header(other));
        assert_eq!(rows[7], Row::Slider(2));
        assert_eq!(rows.len(), 8, "empty sections have no heading");
        assert_eq!(super::rows(&derived, 0, true)[0], Row::Header(0));
        assert_eq!(super::rows(&derived, 2, false), [Row::Add]);
    }
}
