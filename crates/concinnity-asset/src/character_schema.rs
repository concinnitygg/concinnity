// Character-schema: the declarative contract a body conforms to, read by the
// cook (validation, synthesized targets) and the editor (panel layout).

use crate::{JointProportion, ShapeSlider};
use alloc::string::String;
use alloc::vec::Vec;

/// Whether a shape key is one target or a `+` / `-` pair.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KeyPolarity {
    /// One target named exactly `name`; the slider runs `[0, 1]`.
    #[default]
    Unipolar,
    /// Two targets `name+` / `name-`; the slider runs `[-1, 1]`.
    Bipolar,
}

impl KeyPolarity {
    /// The slider range the polarity implies.
    pub fn range(self) -> [f32; 2] {
        match self {
            KeyPolarity::Unipolar => [0.0, 1.0],
            KeyPolarity::Bipolar => [-1.0, 1.0],
        }
    }
}

/// One joint the schema expects in a conforming skeleton.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SchemaJoint {
    /// Joint name.
    pub name: String,
    /// Parent joint name; empty for a root.
    pub parent: String,
    /// A source may omit this joint.
    pub optional: bool,
}

/// One shape key the schema knows, authored on the source or synthesized.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SchemaKey {
    /// Slider name; the target is `name` or the `name+` / `name-` pair.
    pub name: String,
    /// One target or a pair.
    pub polarity: KeyPolarity,
    /// Panel caption; the name when empty.
    pub caption: String,
    /// The region the key belongs to (panel grouping).
    pub region: String,
}

impl Default for SchemaKey {
    fn default() -> Self {
        Self {
            name: String::new(),
            polarity: KeyPolarity::Unipolar,
            caption: String::new(),
            region: String::new(),
        }
    }
}

impl SchemaKey {
    /// The caption, falling back to the name.
    pub fn caption(&self) -> &str {
        if self.caption.is_empty() {
            &self.name
        } else {
            &self.caption
        }
    }
}

/// A named group of joints. A vertex belongs to a region by the skin weight
/// it gives the region's joints.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SchemaRegion {
    /// Region name.
    pub name: String,
    /// Member joints.
    pub joints: Vec<String>,
}

/// A proportion slider: one value in `[-1, 1]` written as a scale and / or
/// length change on every listed joint.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ProportionGroup {
    /// Group name (the panel row).
    pub name: String,
    /// Panel caption; the name when empty.
    pub caption: String,
    /// The region the row belongs to (panel grouping).
    pub region: String,
    /// Joints the row writes; only those the skeleton has are written.
    pub joints: Vec<String>,
    /// Scale change at full deflection (`0` leaves scale alone).
    pub scale: f32,
    /// Length change at full deflection, in model units (`0` leaves it alone).
    pub length: f32,
}

impl ProportionGroup {
    /// The caption, falling back to the name.
    pub fn caption(&self) -> &str {
        if self.caption.is_empty() {
            &self.name
        } else {
            &self.caption
        }
    }
}

/// Generator parameters for a synthesized target. Each generator reads the
/// fields it needs and ignores the rest.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SynthParams {
    /// Displacement at full weight, in model units.
    pub amplitude: f32,
    /// `bulge`: centre along the bone, as a fraction of its length.
    pub along: f32,
    /// `bulge`: width of the lobe along the bone, as a fraction of its length.
    pub sigma: f32,
    /// `bulge`: model-space direction of the lobe; zero means radially away
    /// from the bone.
    pub direction: [f32; 3],
    /// `taper`: ramp from the distal end toward the proximal end instead.
    pub reverse: bool,
    /// `mirror` / `blend_mask`: the authored target to derive from.
    pub source: String,
    /// `surface_offset`: the window along the region's first bone, as
    /// fractions of its length, outside which the offset fades to nothing.
    pub span: [f32; 2],
    /// `surface_offset`: width of the fade at each end of `span`.
    pub falloff: f32,
}

impl Default for SynthParams {
    fn default() -> Self {
        Self {
            amplitude: 0.02,
            along: 0.5,
            sigma: 0.15,
            direction: [0.0, 0.0, 0.0],
            reverse: false,
            source: String::new(),
            span: [0.0, 1.0],
            falloff: 0.1,
        }
    }
}

/// A morph target the build generates from the mesh instead of reading from
/// the source.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SynthesizedTarget {
    /// Slider name. A bipolar target emits `name+` and its negation `name-`.
    pub name: String,
    /// Generator: `girth`, `taper`, `bulge`, `mirror`, `blend_mask`, or
    /// `surface_offset`.
    pub generator: String,
    /// The region the generator works in and the key is grouped under.
    pub region: String,
    /// One target or a pair.
    pub polarity: KeyPolarity,
    /// Panel caption; the name when empty.
    pub caption: String,
    /// Generator parameters.
    pub params: SynthParams,
}

impl SynthesizedTarget {
    /// The key entry this target presents to the panel.
    pub fn key(&self) -> SchemaKey {
        SchemaKey {
            name: self.name.clone(),
            polarity: self.polarity,
            caption: self.caption.clone(),
            region: self.region.clone(),
        }
    }
}

/// One panel section: a caption over the rows of the listed regions.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PanelSection {
    /// Section caption.
    pub caption: String,
    /// Regions whose keys and proportion groups the section shows, in order.
    pub regions: Vec<String>,
}

/// A named slider vector the panel offers as a button.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ShapePreset {
    /// Preset name (the button caption).
    pub name: String,
    /// Slider values the preset sets; every other slider resets to 0.
    pub sliders: Vec<ShapeSlider>,
    /// Proportions the preset sets; every other joint resets to identity.
    pub proportions: Vec<JointProportion>,
}

/// The contract between a character body and everything that uses it.
///
/// A schema names the joints a conforming skeleton must have (with their
/// parents), the shape keys a conforming mesh carries, the regions those keys
/// and the editor group by, the proportion rows the editor offers, the morph
/// targets the build synthesizes from the mesh, the panel's section order,
/// and the presets it offers. A [CharacterModel](#charactermodel) names one
/// schema and is validated against it at build time, so any conforming body
/// gets the same sliders, panel, and animations.
///
/// **Regions** are joint groups. A vertex belongs to a region by the skin
/// weight it gives the region's joints, which needs no authoring and holds
/// at any vertex count. Regions scope every synthesized target and group the
/// panel.
///
/// **Synthesized targets** are ordinary morph targets the build generates:
/// `girth` pushes a region's vertices away from its bone axes, `taper` ramps
/// that push along each bone, `bulge` raises a gaussian lobe at a point along
/// a bone, `mirror` reflects an authored target across X, `blend_mask`
/// restricts an authored whole-body target to a region, and `surface_offset`
/// pushes along the vertex normal. Normals are recomputed from the displaced
/// mesh. At runtime they are indistinguishable from sculpted keys.
///
/// The reserved name `builtin:humanoid` is the schema of the humanoid body
/// the `customize_character` example ships (`base_humanoid.glb`), bundled
/// with the build so any body with the same 25 joints and 21 shape keys
/// conforms to it.
///
/// ```rust
/// # use concinnity_asset::{CharacterSchema, SchemaJoint, SchemaRegion};
/// CharacterSchema {
///     joints: vec![
///         SchemaJoint { name: "root".into(), ..Default::default() },
///         SchemaJoint { name: "spine".into(), parent: "root".into(), optional: false },
///     ],
///     regions: vec![SchemaRegion { name: "torso".into(), joints: vec!["spine".into()] }],
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CharacterSchema {
    /// Required (and optional) joints with their parents.
    pub joints: Vec<SchemaJoint>,
    /// Shape keys a conforming source carries.
    pub keys: Vec<SchemaKey>,
    /// Named joint groups.
    pub regions: Vec<SchemaRegion>,
    /// Proportion rows.
    pub proportion_groups: Vec<ProportionGroup>,
    /// Targets the build generates from the mesh.
    pub synthesized: Vec<SynthesizedTarget>,
    /// Panel sections in display order. Regions no section lists, and keys
    /// the schema does not know, show under a trailing "Other" section.
    pub panel: Vec<PanelSection>,
    /// Named slider vectors offered as buttons.
    pub presets: Vec<ShapePreset>,
}

impl CharacterSchema {
    /// The region a joint belongs to, if any region lists it.
    pub fn region_of_joint(&self, joint: &str) -> Option<&str> {
        self.regions
            .iter()
            .find(|r| r.joints.iter().any(|j| j == joint))
            .map(|r| r.name.as_str())
    }

    /// The region named `name`.
    pub fn region(&self, name: &str) -> Option<&SchemaRegion> {
        self.regions.iter().find(|r| r.name == name)
    }

    /// Every key the panel shows: the authored keys followed by the
    /// synthesized ones.
    pub fn all_keys(&self) -> Vec<SchemaKey> {
        self.keys
            .iter()
            .cloned()
            .chain(self.synthesized.iter().map(SynthesizedTarget::key))
            .collect()
    }

    /// The morph-target names a conforming source must carry: one per
    /// unipolar key, `name+` / `name-` per bipolar key.
    pub fn required_target_names(&self) -> Vec<String> {
        let mut out = Vec::new();
        for key in &self.keys {
            match key.polarity {
                KeyPolarity::Unipolar => out.push(key.name.clone()),
                KeyPolarity::Bipolar => {
                    out.push(alloc::format!("{}+", key.name));
                    out.push(alloc::format!("{}-", key.name));
                }
            }
        }
        out
    }

    /// The names of the joints a source must have.
    pub fn required_joints(&self) -> impl Iterator<Item = &SchemaJoint> {
        self.joints.iter().filter(|j| !j.optional)
    }

    /// Problems in the schema itself: regions naming unknown joints, keys
    /// and groups naming unknown regions, generators naming unknown
    /// sources, duplicate names. Empty when the schema is consistent.
    pub fn consistency_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let joint_known = |name: &str| self.joints.iter().any(|j| j.name == name);
        let region_known = |name: &str| self.regions.iter().any(|r| r.name == name);
        for joint in &self.joints {
            if !joint.parent.is_empty() && !joint_known(&joint.parent) {
                errors.push(alloc::format!(
                    "joint '{}' names unknown parent '{}'",
                    joint.name,
                    joint.parent
                ));
            }
        }
        for region in &self.regions {
            for joint in &region.joints {
                if !joint_known(joint) {
                    errors.push(alloc::format!(
                        "region '{}' lists unknown joint '{}'",
                        region.name,
                        joint
                    ));
                }
            }
        }
        let mut seen: Vec<String> = Vec::new();
        for key in self.all_keys() {
            if !key.region.is_empty() && !region_known(&key.region) {
                errors.push(alloc::format!(
                    "key '{}' names unknown region '{}'",
                    key.name,
                    key.region
                ));
            }
            if seen.contains(&key.name) {
                errors.push(alloc::format!("key '{}' is declared twice", key.name));
            }
            seen.push(key.name.clone());
        }
        for group in &self.proportion_groups {
            if !group.region.is_empty() && !region_known(&group.region) {
                errors.push(alloc::format!(
                    "proportion group '{}' names unknown region '{}'",
                    group.name,
                    group.region
                ));
            }
            for joint in &group.joints {
                if !joint_known(joint) {
                    errors.push(alloc::format!(
                        "proportion group '{}' lists unknown joint '{}'",
                        group.name,
                        joint
                    ));
                }
            }
        }
        for target in &self.synthesized {
            if !region_known(&target.region) {
                errors.push(alloc::format!(
                    "synthesized '{}' names unknown region '{}'",
                    target.name,
                    target.region
                ));
            }
            let needs_source = matches!(target.generator.as_str(), "mirror" | "blend_mask");
            if needs_source && target.params.source.is_empty() {
                errors.push(alloc::format!(
                    "synthesized '{}': generator '{}' needs a source key",
                    target.name,
                    target.generator
                ));
            }
        }
        for section in &self.panel {
            for region in &section.regions {
                if !region_known(region) {
                    errors.push(alloc::format!(
                        "panel section '{}' lists unknown region '{}'",
                        section.caption,
                        region
                    ));
                }
            }
        }
        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn schema() -> CharacterSchema {
        serde_json::from_str(
            r#"{
            "joints": [{"name": "root"}, {"name": "spine", "parent": "root"},
                       {"name": "head", "parent": "spine"}, {"name": "tail", "parent": "root", "optional": true}],
            "keys": [{"name": "weight", "polarity": "bipolar", "region": "torso"},
                     {"name": "brow", "caption": "Brow ridge", "region": "face"}],
            "regions": [{"name": "torso", "joints": ["spine"]}, {"name": "face", "joints": ["head"]}],
            "proportion_groups": [{"name": "height", "region": "torso", "joints": ["spine"], "scale": 0.08}],
            "synthesized": [{"name": "neck_girth", "generator": "girth", "region": "torso",
                             "polarity": "bipolar", "params": {"amplitude": 0.03}}],
            "panel": [{"caption": "Face", "regions": ["face"]}, {"caption": "Body", "regions": ["torso"]}],
            "presets": [{"name": "heavy", "sliders": [{"name": "weight", "value": 0.8}]}]
        }"#,
        )
        .unwrap()
    }

    #[test]
    fn polarity_sets_the_range_and_the_required_targets() {
        assert_eq!(KeyPolarity::Unipolar.range(), [0.0, 1.0]);
        assert_eq!(KeyPolarity::Bipolar.range(), [-1.0, 1.0]);
        let s = schema();
        assert_eq!(s.required_target_names(), ["weight+", "weight-", "brow"]);
        let required: Vec<&str> = s.required_joints().map(|j| j.name.as_str()).collect();
        assert_eq!(
            required,
            ["root", "spine", "head"],
            "optional joints are not required"
        );
    }

    #[test]
    fn regions_resolve_by_joint_and_captions_fall_back_to_names() {
        let s = schema();
        assert_eq!(s.region_of_joint("head"), Some("face"));
        assert_eq!(s.region_of_joint("root"), None);
        assert_eq!(s.region("torso").unwrap().joints, ["spine"]);
        let keys = s.all_keys();
        assert_eq!(keys.len(), 3, "authored keys then synthesized");
        assert_eq!(keys[0].caption(), "weight");
        assert_eq!(keys[1].caption(), "Brow ridge");
        assert_eq!(keys[2].name, "neck_girth");
        assert_eq!(keys[2].polarity, KeyPolarity::Bipolar);
        assert_eq!(s.proportion_groups[0].caption(), "height");
        assert_eq!(s.synthesized[0].params.amplitude, 0.03);
        assert_eq!(
            s.synthesized[0].params.sigma, 0.15,
            "unset params keep their defaults"
        );
    }

    #[test]
    fn a_consistent_schema_reports_nothing() {
        assert!(schema().consistency_errors().is_empty());
    }

    #[test]
    fn inconsistencies_are_all_reported() {
        let mut s = schema();
        s.joints[1].parent = "pelvis".into();
        s.regions[0].joints.push("wing".into());
        s.keys[0].region = "arms".into();
        s.keys.push(s.keys[1].clone());
        s.proportion_groups[0].joints.push("wing".into());
        s.synthesized.push(SynthesizedTarget {
            name: "brow_r".into(),
            generator: "mirror".into(),
            region: "face".into(),
            ..Default::default()
        });
        s.panel[0].regions.push("hair".into());
        let errors = s.consistency_errors();
        let has = |needle: &str| errors.iter().any(|e| e.contains(needle));
        assert!(has("unknown parent 'pelvis'"), "{errors:?}");
        assert!(
            has("region 'torso' lists unknown joint 'wing'"),
            "{errors:?}"
        );
        assert!(
            has("key 'weight' names unknown region 'arms'"),
            "{errors:?}"
        );
        assert!(has("key 'brow' is declared twice"), "{errors:?}");
        assert!(
            has("proportion group 'height' lists unknown joint 'wing'"),
            "{errors:?}"
        );
        assert!(has("generator 'mirror' needs a source key"), "{errors:?}");
        assert!(
            has("panel section 'Face' lists unknown region 'hair'"),
            "{errors:?}"
        );
    }

    #[test]
    fn a_schema_round_trips_through_postcard() {
        let s = schema();
        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: CharacterSchema = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back, s);
        assert_eq!(back.presets[0].sliders[0].value, 0.8);
        let blank = CharacterSchema::default();
        assert!(blank.joints.is_empty() && blank.panel.is_empty());
        assert_eq!(SynthParams::default().span, [0.0, 1.0]);
        assert_eq!(vec![SchemaKey::default().polarity], [KeyPolarity::Unipolar]);
    }
}
