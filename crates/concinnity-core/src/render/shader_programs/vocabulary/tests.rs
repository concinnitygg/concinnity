use super::*;
use crate::platform::Platform;
use crate::render::shader_programs::surface;
use crate::render::shader_source;
use alloc::collections::BTreeSet;
use alloc::string::ToString;
use alloc::vec::Vec;

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

// Every identifier in `text`.
fn identifiers(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !is_ident(c))
        .filter(|w| w.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_'))
}

#[test]
fn names_are_unique_within_their_owner() {
    let mut seen = BTreeSet::new();
    for e in ENTRIES {
        assert!(seen.insert((e.kind.owner(), e.name)), "{} twice", e.name);
    }
}

#[test]
fn usage_reaches_a_field_through_its_owner_and_calls_a_helper() {
    let by_name = |n: &str| ENTRIES.iter().find(|e| e.name == n).unwrap().usage();
    assert_eq!(by_name("elapsed"), "VIEW.elapsed");
    assert_eq!(by_name("num_pt"), "LIGHTS.num_pt");
    assert_eq!(by_name("albedo_index"), "od.albedo_index");
    assert_eq!(by_name("world_pos"), "v.world_pos");
    assert_eq!(by_name("shade_surface"), "shade_surface(v, od)");
    assert_eq!(by_name("probe_mask_all"), "probe_mask_all()");
    assert_eq!(
        by_name("environment_specular"),
        "environment_specular(probes, world_pos, reflected, roughness, radiance)",
        "an `out` parameter is named, not qualified"
    );
}

#[test]
fn every_signature_names_its_entry() {
    for e in ENTRIES {
        let names: Vec<&str> = identifiers(e.signature).collect();
        assert!(names.contains(&e.name), "{}: {}", e.name, e.signature);
    }
}

// The template a world Shader compiles inside, as each backend assembles it.
fn templates() -> impl Iterator<Item = (Platform, String)> {
    let file = surface::ALL[0].file;
    Platform::ALL
        .into_iter()
        .map(move |p| (p, shader_source::assemble(file, p, &[])))
}

// `text` without its `//` comments.
fn strip_comments(text: &str) -> String {
    text.lines()
        .map(|l| l.split_once("//").map_or(l, |(code, _)| code))
        .collect::<Vec<_>>()
        .join("\n")
}

// Whether `name` is defined in `source`: a macro taking arguments, or a
// function whose parameter list is followed by a body rather than a `;`.
fn defines_helper(source: &str, name: &str) -> bool {
    if source.contains(&alloc::format!("#define {name}(")) {
        return true;
    }
    let call = alloc::format!("{name}(");
    source.match_indices(&call).any(|(at, _)| {
        let line_start = source[..at].rfind('\n').map_or(0, |i| i + 1);
        let before = source[line_start..at].trim();
        if before.is_empty() || !before.chars().all(is_ident) {
            return false;
        }
        let Some(close) = source[at..].find(')') else {
            return false;
        };
        source[at + close + 1..].trim_start().starts_with('{')
    })
}

// The field names `struct name { ... }` declares in `source`.
fn struct_fields(source: &str, name: &str) -> BTreeSet<String> {
    let head = alloc::format!("struct {name}");
    let at = source
        .match_indices(&head)
        .map(|(i, _)| i + head.len())
        .find(|&i| !source[i..].starts_with(is_ident))
        .unwrap_or_else(|| panic!("no `struct {name}`"));
    let body = &source[at..];
    let open = body.find('{').unwrap() + 1;
    let close = body.find("};").unwrap();
    body[open..close]
        .split(';')
        .filter_map(|decl| {
            // Drop attributes, a semantic, and an array extent; the name is
            // the last word left.
            let decl = decl.rsplit("]]").next().unwrap();
            let decl = decl.split(':').next().unwrap();
            let decl = decl.split('[').next().unwrap();
            identifiers(decl).last().map(ToString::to_string)
        })
        .collect()
}

#[test]
fn every_helper_is_defined_in_the_template() {
    for (platform, template) in templates() {
        let code = strip_comments(&template);
        for e in ENTRIES.iter().filter(|e| e.kind == Kind::Helper) {
            assert!(
                defines_helper(&code, e.name),
                "`{}` is not defined in the {platform:?} template",
                e.name
            );
        }
    }
}

#[test]
fn every_field_is_declared_in_its_struct() {
    for (platform, template) in templates() {
        let code = strip_comments(&template);
        for e in ENTRIES {
            let Some(owner) = e.kind.declared_in() else {
                continue;
            };
            assert!(
                struct_fields(&code, owner).contains(e.name),
                "`{owner}` declares no `{}` in the {platform:?} template",
                e.name
            );
        }
    }
}

#[test]
fn every_block_is_bound_under_its_name() {
    for (platform, template) in templates() {
        let code = strip_comments(&template);
        for block in Block::ALL {
            let bound: Vec<&str> = code
                .lines()
                .filter_map(|l| {
                    l.trim()
                        .strip_prefix(&alloc::format!("#define {} ", block.name()))
                })
                .map(str::trim)
                .collect();
            assert_eq!(
                bound.len(),
                1,
                "{platform:?}: `{}` defined once",
                block.name()
            );
            let declared = alloc::format!("ConstantBuffer<{}> {}", block.declared_as(), bound[0]);
            assert!(
                code.contains(&declared),
                "{platform:?}: `{}` is not a `{}`",
                block.name(),
                block.declared_as()
            );
        }
    }
}

#[test]
fn the_hooks_take_the_structs_under_the_names_fields_are_read_through() {
    for (_, template) in templates() {
        assert!(
            template.contains("float4 shade(VertexOut v, GpuObjectData od);"),
            "the `shade` prototype changed; update the owners in Kind::owner"
        );
    }
}

#[cfg(feature = "schema")]
mod rustdoc {
    use super::*;
    use crate::components::Shader;
    use crate::ecs::schema::Schema;

    // Every identifier inside the `Shader` rustdoc's code: its inline spans and
    // its fenced blocks.
    fn code_identifiers() -> BTreeSet<&'static str> {
        Shader::SCHEMA
            .doc
            .split('`')
            .skip(1)
            .step_by(2)
            .flat_map(identifiers)
            .collect()
    }

    #[test]
    fn every_entry_is_named_in_the_shader_rustdoc() {
        let named = code_identifiers();
        for e in ENTRIES {
            assert!(
                named.contains(e.name),
                "the Shader rustdoc never names `{}`",
                e.name
            );
            if let Some(owner) = e.kind.owner() {
                assert!(
                    named.contains(owner),
                    "the Shader rustdoc never names `{owner}`"
                );
            }
        }
    }

    // The rustdoc's list of what the files see starts each item with the name
    // it describes, so a name added there without an entry is caught too.
    #[test]
    fn every_name_the_rustdoc_lists_is_an_entry() {
        let listed = Shader::SCHEMA
            .doc
            .lines()
            .filter_map(|l| l.strip_prefix("- `"))
            .filter_map(|l| identifiers(l).next());
        let known: BTreeSet<&str> = ENTRIES
            .iter()
            .map(|e| e.name)
            .chain(Block::ALL.map(Block::name))
            .collect();
        let mut count = 0;
        for name in listed {
            count += 1;
            assert!(known.contains(name), "`{name}` is listed but has no entry");
        }
        assert!(count >= 8, "the rustdoc's list moved; found {count} items");
    }
}
