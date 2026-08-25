// Conformance of one imported source to a schema: every required joint is
// present under the right parent and every bipolar key has both targets,
// plus the model's own args. Every problem is reported, in one pass.

use crate::components::{CharacterModel, CharacterSchema, SkeletonJoint};

// Problems with `skeleton` and `target_names` against `schema`; empty when
// the source conforms.
pub(crate) fn source_errors(
    schema: &CharacterSchema,
    skeleton: &[SkeletonJoint],
    target_names: &[String],
) -> Vec<String> {
    let mut errors = Vec::new();
    let parent_of = |name: &str| -> Option<String> {
        let j = skeleton.iter().find(|j| j.name == name)?;
        Some(if j.parent >= 0 {
            skeleton
                .get(j.parent as usize)
                .map(|p| p.name.clone())
                .unwrap_or_default()
        } else {
            String::new()
        })
    };
    for joint in &schema.joints {
        match parent_of(&joint.name) {
            None if joint.optional => {}
            None => errors.push(format!("missing joint '{}'", joint.name)),
            Some(parent) if parent != joint.parent => errors.push(format!(
                "joint '{}' has parent '{}', the schema expects '{}'",
                joint.name,
                if parent.is_empty() { "<root>" } else { &parent },
                if joint.parent.is_empty() {
                    "<root>"
                } else {
                    &joint.parent
                }
            )),
            Some(_) => {}
        }
    }
    let has = |name: &str| target_names.iter().any(|t| t == name);
    for key in &schema.keys {
        match key.polarity {
            crate::components::KeyPolarity::Unipolar => {
                if !has(&key.name) {
                    errors.push(format!("missing shape key '{}'", key.name));
                }
            }
            crate::components::KeyPolarity::Bipolar => {
                let (plus, minus) = (format!("{}+", key.name), format!("{}-", key.name));
                match (has(&plus), has(&minus)) {
                    (true, true) => {}
                    (false, false) => {
                        errors.push(format!("missing shape key pair '{plus}' / '{minus}'"))
                    }
                    (true, false) => {
                        errors.push(format!("shape key '{plus}' has no '{minus}' half"))
                    }
                    (false, true) => {
                        errors.push(format!("shape key '{minus}' has no '{plus}' half"))
                    }
                }
            }
        }
    }
    errors
}

// Problems with the model's own args.
pub(crate) fn model_errors(model: &CharacterModel) -> Vec<String> {
    let mut errors = Vec::new();
    if model.source.is_empty() {
        errors.push("no source".to_string());
    }
    let levels = model.lod_levels.clamp(1, 8) as usize;
    if !model.lod_distances.is_empty() && model.lod_distances.len() != levels - 1 {
        errors.push(format!(
            "lod_distances has {} entries; lod_levels = {} needs {}",
            model.lod_distances.len(),
            model.lod_levels,
            levels - 1
        ));
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{KeyPolarity, SchemaJoint, SchemaKey};

    fn joint(name: &str, parent: i32) -> SkeletonJoint {
        SkeletonJoint {
            name: name.into(),
            parent,
            ..Default::default()
        }
    }

    fn schema() -> CharacterSchema {
        let sj = |name: &str, parent: &str, optional: bool| SchemaJoint {
            name: name.into(),
            parent: parent.into(),
            optional,
        };
        CharacterSchema {
            joints: vec![
                sj("root", "", false),
                sj("spine", "root", false),
                sj("head", "spine", false),
                sj("tail", "root", true),
            ],
            keys: vec![
                SchemaKey {
                    name: "weight".into(),
                    polarity: KeyPolarity::Bipolar,
                    ..Default::default()
                },
                SchemaKey {
                    name: "muscle".into(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_conforming_source_has_no_errors() {
        let sk = vec![joint("root", -1), joint("spine", 0), joint("head", 1)];
        let errs = source_errors(
            &schema(),
            &sk,
            &names(&["weight+", "weight-", "muscle", "extra"]),
        );
        assert!(errs.is_empty(), "{errs:?}");
    }

    #[test]
    fn a_missing_joint_is_an_error_unless_optional() {
        let sk = vec![joint("root", -1), joint("spine", 0)];
        let errs = source_errors(&schema(), &sk, &names(&["weight+", "weight-", "muscle"]));
        assert_eq!(errs, ["missing joint 'head'"]);
    }

    #[test]
    fn a_wrong_parent_is_an_error() {
        let sk = vec![joint("root", -1), joint("spine", 0), joint("head", 0)];
        let errs = source_errors(&schema(), &sk, &names(&["weight+", "weight-", "muscle"]));
        assert_eq!(
            errs,
            ["joint 'head' has parent 'root', the schema expects 'spine'"]
        );
        let sk = vec![joint("spine", -1), joint("root", 0), joint("head", 0)];
        let errs = source_errors(&schema(), &sk, &names(&["weight+", "weight-", "muscle"]));
        assert!(
            errs.contains(
                &"joint 'root' has parent 'spine', the schema expects '<root>'".to_string()
            ),
            "{errs:?}"
        );
    }

    #[test]
    fn an_incomplete_bipolar_pair_is_an_error() {
        let sk = vec![joint("root", -1), joint("spine", 0), joint("head", 1)];
        let errs = source_errors(&schema(), &sk, &names(&["weight+", "muscle"]));
        assert_eq!(errs, ["shape key 'weight+' has no 'weight-' half"]);
        let errs = source_errors(&schema(), &sk, &names(&["weight-"]));
        assert_eq!(
            errs,
            [
                "shape key 'weight-' has no 'weight+' half",
                "missing shape key 'muscle'"
            ]
        );
        let errs = source_errors(&schema(), &sk, &names(&["muscle"]));
        assert_eq!(errs, ["missing shape key pair 'weight+' / 'weight-'"]);
    }

    #[test]
    fn a_missing_source_and_a_bad_distance_count_are_errors() {
        assert_eq!(model_errors(&CharacterModel::default()), ["no source"]);
        let model = CharacterModel {
            source: "a.glb".into(),
            lod_levels: 2,
            lod_distances: vec![5.0, 9.0],
            ..Default::default()
        };
        assert_eq!(
            model_errors(&model),
            ["lod_distances has 2 entries; lod_levels = 2 needs 1"]
        );
        let model = CharacterModel {
            source: "a.glb".into(),
            lod_levels: 3,
            lod_distances: vec![5.0, 9.0],
            ..Default::default()
        };
        assert!(model_errors(&model).is_empty());
    }
}
