use super::super::scan::{defines_helper, identifiers, strip_comments, struct_fields};
use super::*;
use crate::platform::Platform;
use crate::render::shader_programs::raymarch::{self, Family};
use crate::render::shader_source::SourceFile;
use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

#[test]
fn names_are_unique_within_their_struct() {
    let mut seen = BTreeSet::new();
    for e in ENTRIES {
        assert!(
            seen.insert((e.kind.declared_in(), e.name)),
            "{} twice",
            e.name
        );
    }
}

#[test]
fn usage_writes_a_hook_whole_calls_a_helper_and_names_the_rest() {
    let by_name = |n: &str| ENTRIES.iter().find(|e| e.name == n).unwrap().usage();
    assert_eq!(
        by_name("map"),
        "float map(float3 p, SdfParams params, float time)"
    );
    assert_eq!(by_name("sdSphere"), "sdSphere(p, r)");
    assert_eq!(by_name("volume_center"), "volume_center()");
    assert_eq!(by_name("SdfParams"), "SdfParams");
    assert_eq!(by_name("density"), "density");
}

#[test]
fn every_signature_names_its_entry() {
    for e in ENTRIES {
        let names: Vec<&str> = identifiers(e.signature).collect();
        assert!(names.contains(&e.name), "{}: {}", e.name, e.signature);
    }
}

// The template a field compiles inside, in every family on every host, with
// an empty field spliced in.
fn templates() -> impl Iterator<Item = (Platform, Family, String)> {
    let field = SourceFile {
        path: "field.hlsl",
        text: "",
    };
    Platform::ALL.into_iter().flat_map(move |p| {
        [Family::Surface, Family::Volumetric, Family::Shadow]
            .into_iter()
            .map(move |f| (p, f, strip_comments(&raymarch::source(f, p, field))))
    })
}

fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn every_helper_is_defined_in_the_template() {
    for (platform, family, code) in templates() {
        for e in ENTRIES.iter().filter(|e| e.kind == Kind::Helper) {
            assert!(
                defines_helper(&code, e.name),
                "`{}` is not defined in the {platform:?} {family:?} template",
                e.name
            );
        }
    }
}

// The template forward-declares every function a field defines, with the
// signature the table gives, so a field matching the table links.
#[test]
fn every_hook_is_declared_by_the_template_as_listed() {
    for (platform, family, code) in templates() {
        let code = collapse(&code);
        for e in ENTRIES.iter().filter(|e| e.kind == Kind::Hook) {
            let prototype = alloc::format!("{};", collapse(e.signature));
            assert!(
                code.contains(&prototype),
                "{platform:?} {family:?}: no `{prototype}`"
            );
        }
    }
}

#[test]
fn every_type_and_returned_field_is_declared() {
    for (platform, family, code) in templates() {
        for e in ENTRIES.iter().filter(|e| e.kind == Kind::Type) {
            assert!(
                !struct_fields(&code, e.name).is_empty(),
                "{platform:?} {family:?}: `struct {}` declares nothing",
                e.name
            );
        }
        for e in ENTRIES {
            let Some(owner) = e.kind.declared_in() else {
                continue;
            };
            assert!(
                struct_fields(&code, owner).contains(e.name),
                "{platform:?} {family:?}: `{owner}` declares no `{}`",
                e.name
            );
        }
    }
}

// A field the returned structs gain is one a file sets, so the table lists
// every one of them.
#[test]
fn every_field_of_a_returned_struct_is_listed() {
    let (_, _, code) = templates().next().unwrap();
    for owner in [SURFACE_STRUCT, SAMPLE_STRUCT] {
        for name in struct_fields(&code, owner) {
            assert!(
                ENTRIES
                    .iter()
                    .any(|e| e.kind == Kind::ReturnField(owner) && e.name == name),
                "`{owner}.{name}` has no entry"
            );
        }
    }
}

#[cfg(feature = "schema")]
mod rustdoc {
    use super::*;
    use crate::components::SdfVolume;
    use crate::ecs::schema::Schema;

    fn doc() -> &'static str {
        SdfVolume::SCHEMA.doc
    }

    #[test]
    fn every_entry_is_named_in_the_sdf_volume_rustdoc() {
        let named: BTreeSet<&str> = doc()
            .split('`')
            .skip(1)
            .step_by(2)
            .flat_map(identifiers)
            .collect();
        for e in ENTRIES {
            assert!(
                named.contains(e.name),
                "the SdfVolume rustdoc never names `{}`",
                e.name
            );
        }
    }

    // The rustdoc's list of what a field can call starts each item with the
    // name it describes, so a name added there without an entry is caught.
    #[test]
    fn every_name_the_rustdoc_lists_is_an_entry() {
        let known: BTreeSet<&str> = ENTRIES.iter().map(|e| e.name).collect();
        let mut count = 0;
        for line in doc().lines().filter_map(|l| l.strip_prefix("- `")) {
            count += 1;
            let name = identifiers(line).next().unwrap_or_default();
            assert!(known.contains(name), "`{name}` is listed but has no entry");
        }
        assert!(count >= 6, "the rustdoc's list moved; found {count} items");
    }
}
