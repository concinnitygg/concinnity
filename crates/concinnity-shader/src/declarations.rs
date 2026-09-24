//! The resource declarations in an HLSL source, read off the annotations they
//! carry.
//!
//! An engine `.hlsl` shader annotates every resource twice: `[[vk::binding]]`
//! (or `[[vk::push_constant]]`) is the Vulkan descriptor layout, and
//! `register()` is the D3D slot. This reads both back out, which is what lets
//! the Metal binding table be derived rather than written -- see
//! `metal_bindings`.
//!
//! A third annotation, `[[cn::metal_argument_buffer(n)]]`, names the one thing
//! the other two cannot: Metal binds a whole descriptor SET as a single
//! argument buffer, and no per-resource register can say which set or at which
//! buffer index. It is the engine's own annotation rather than a `vk::` one
//! because Vulkan binds that set as ordinary descriptors. The `cn::` namespace
//! is this module's alone: `strip_engine_attributes` rejects a name it does
//! not know and hands dxc the source without any, so dxc can keep warning about
//! every attribute it does not know.
//!
//! Every texture and every sampler is a declaration of its own with a
//! `vk::binding` of its own, as D3D and Metal bind them: a Vulkan combined
//! image sampler is not part of the engine's binding model, and
//! `strip_engine_attributes` refuses `[[vk::combinedImageSampler]]`. dxc
//! honors neither `-fspv-preserve-interface` nor `-fspv-preserve-bindings` for
//! the entry interface of a source that declares one.
//!
//! It reads preprocessed text: a declaration behind an inactive `#if` is not a
//! declaration, and scanning around one is how a converter picks up a slot the
//! compile never sees.

use std::borrow::Cow;
use std::ops::Range;

/// The `cn::` attributes the engine reads.
const ENGINE_ATTRIBUTES: &[&str] = &["metal_argument_buffer"];

/// The `vk::` attribute that pairs a texture and a sampler into one Vulkan
/// descriptor.
const COMBINED_IMAGE_SAMPLER: &str = "combinedImageSampler";

/// `source` with every `cn::` attribute blanked to spaces, which is the text
/// dxc compiles.
///
/// Newlines survive and every other character becomes one space, so each line
/// and column a dxc diagnostic names is where the author wrote it. An attribute
/// specifier holding nothing else goes whole; one shared with other attributes
/// loses only the `cn::` entry and its comma. Comments are left alone.
///
/// Errs naming the attribute when a `cn::` one is not the engine's, since dxc
/// never sees it to warn, and on a `vk::combinedImageSampler`.
pub(crate) fn strip_engine_attributes(source: &str) -> Result<Cow<'_, str>, String> {
    let mut blank: Vec<Range<usize>> = Vec::new();
    for specifier in attribute_specifiers(source) {
        let items = attribute_items(source, specifier.clone());
        if let Some(item) = items
            .iter()
            .find(|item| is_combined_image_sampler(&source[item.text.clone()]))
        {
            let line = source[..item.text.start].matches('\n').count() + 1;
            return Err(format!(
                "line {line}: `vk::{COMBINED_IMAGE_SAMPLER}` is not supported: declare the \
                 texture and its sampler with a `vk::binding` each"
            ));
        }
        let engine: Vec<usize> = items
            .iter()
            .enumerate()
            .filter(|(_, item)| is_engine_attribute(&source[item.text.clone()]))
            .map(|(i, _)| i)
            .collect();
        for &i in &engine {
            let text = source[items[i].text.clone()].trim();
            let name = attribute_name(text);
            if !ENGINE_ATTRIBUTES.contains(&name) {
                let line = source[..items[i].text.start].matches('\n').count() + 1;
                return Err(format!(
                    "line {line}: unknown engine attribute `cn::{name}` (the engine reads {})",
                    ENGINE_ATTRIBUTES
                        .iter()
                        .map(|a| format!("`cn::{a}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
        if engine.is_empty() {
            continue;
        }
        if engine.len() == items.len() {
            blank.push(specifier);
        } else {
            blank.extend(engine.iter().map(|&i| items[i].with_comma.clone()));
        }
    }
    if blank.is_empty() {
        return Ok(Cow::Borrowed(source));
    }
    let mut bytes = source.as_bytes().to_vec();
    for range in blank {
        for b in &mut bytes[range] {
            if *b != b'\n' && *b != b'\r' {
                *b = b' ';
            }
        }
    }
    String::from_utf8(bytes)
        .map(Cow::Owned)
        .map_err(|e| format!("blanking the engine attributes: {e}"))
}

// One attribute inside a `[[...]]` specifier: its own text, and that text with
// the comma separating it from a neighbor.
struct AttributeItem {
    text: Range<usize>,
    with_comma: Range<usize>,
}

// Every `[[...]]` specifier outside a comment or a string, `[[` to `]]`
// inclusive.
fn attribute_specifiers(source: &str) -> Vec<Range<usize>> {
    let bytes = source.as_bytes();
    let mut found = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                i = source[i..].find('\n').map_or(bytes.len(), |n| i + n);
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                i = source[i + 2..]
                    .find("*/")
                    .map_or(bytes.len(), |n| i + 2 + n + 2);
            }
            b'"' => {
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    i += if bytes[i] == b'\\' { 2 } else { 1 };
                }
                i += 1;
            }
            b'[' if bytes.get(i + 1) == Some(&b'[') => {
                let Some(close) = source[i + 2..].find("]]") else {
                    break;
                };
                let end = i + 2 + close + 2;
                found.push(i..end);
                i = end;
            }
            _ => i += 1,
        }
    }
    found
}

// The comma-separated attributes of one specifier, split outside parentheses.
fn attribute_items(source: &str, specifier: Range<usize>) -> Vec<AttributeItem> {
    let inner = specifier.start + 2..specifier.end - 2;
    let mut commas = Vec::new();
    let mut depth = 0i32;
    for (at, c) in source[inner.clone()].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => commas.push(inner.start + at),
            _ => {}
        }
    }
    let mut bounds = vec![inner.start];
    bounds.extend(commas.iter().map(|c| c + 1));
    let mut ends: Vec<usize> = commas.clone();
    ends.push(inner.end);
    bounds
        .into_iter()
        .zip(ends)
        .enumerate()
        .map(|(i, (start, end))| {
            // The comma before the attribute, or after it for the first one.
            let with_comma = if i > 0 {
                start - 1..end
            } else if i < commas.len() {
                start..end + 1
            } else {
                start..end
            };
            AttributeItem {
                text: start..end,
                with_comma,
            }
        })
        .collect()
}

fn is_combined_image_sampler(item: &str) -> bool {
    let item = item.trim_start();
    item.strip_prefix("vk")
        .is_some_and(|rest| rest.trim_start().starts_with("::"))
        && attribute_name(item) == COMBINED_IMAGE_SAMPLER
}

fn is_engine_attribute(item: &str) -> bool {
    let item = item.trim_start();
    item.strip_prefix("cn")
        .is_some_and(|rest| rest.trim_start().starts_with("::"))
}

// `metal_argument_buffer` out of `cn::metal_argument_buffer(3)`.
fn attribute_name(item: &str) -> &str {
    let after = item
        .find("::")
        .map_or(item, |at| &item[at + 2..])
        .trim_start();
    let end = after
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(after.len());
    &after[..end]
}

/// Where a declaration sits in the Vulkan descriptor layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// `[[vk::binding(binding, set)]]`.
    Qualified {
        /// The descriptor set.
        set: u32,
        /// The binding within it.
        binding: u32,
    },
    /// `[[vk::push_constant]]`.
    PushConstant,
}

/// A D3D register annotation: the class letter, the slot number and the space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Register {
    /// `b`, `t`, `u` or `s`.
    pub class: char,
    /// The slot number within that class.
    pub index: u32,
    /// The D3D register space, 0 when the annotation names none. A D3D slot is
    /// the triple, so `t0` and `t0, space1` are different bindings; neither
    /// Metal nor Vulkan reads it.
    pub space: u32,
}

/// One annotated resource declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Declaration {
    /// The declared name.
    pub name: String,
    /// Where Vulkan binds it, or `None` when it carries no `vk::` annotation.
    pub slot: Option<Slot>,
    /// The D3D register it declares.
    pub register: Register,
    /// Elements, for a sized resource array.
    pub count: Option<u32>,
    /// An unsized resource array (`name[]`), whose length the host decides.
    pub runtime_sized: bool,
    /// Metal buffer index for this declaration's whole descriptor set, when it
    /// carries `[[cn::metal_argument_buffer(n)]]`.
    pub metal_argument_buffer: Option<u32>,
}

/// Every `register()`-annotated resource declaration in `source`.
///
/// A declaration is a statement carrying a `register()` annotation, which
/// nothing inside a struct or a function body does -- so the annotation alone
/// selects them and a struct member or a local cannot be mistaken for one.
#[must_use]
pub(crate) fn resource_declarations(source: &str) -> Vec<Declaration> {
    source.split(';').filter_map(declaration).collect()
}

fn declaration(chunk: &str) -> Option<Declaration> {
    let at = chunk.find("register(")?;
    let register = parse_register(&chunk[at + "register(".len()..])?;
    // A resource declared after a function closes shares its statement with
    // that body, because the brace is not a statement terminator: read the
    // declaration out of what follows the last one.
    let head = &chunk[..at];
    let head = match head.rfind(['{', '}']) {
        Some(brace) => &head[brace + 1..],
        None => head,
    };
    let head = head.trim_end().trim_end_matches(':').trim_end();
    let (head, length) = match head.strip_suffix(']') {
        Some(rest) => {
            let open = rest.rfind('[')?;
            (rest[..open].trim_end(), Some(rest[open + 1..].trim()))
        }
        None => (head, None),
    };
    Some(Declaration {
        name: last_identifier(head)?.to_string(),
        slot: parse_slot(chunk),
        register,
        count: length.and_then(|l| l.parse().ok()),
        runtime_sized: length == Some(""),
        metal_argument_buffer: parse_argument_buffer(chunk),
    })
}

fn parse_argument_buffer(chunk: &str) -> Option<u32> {
    let at = chunk.find("cn::metal_argument_buffer(")? + "cn::metal_argument_buffer(".len();
    chunk[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

// `b0`, `t4, space1`. `rest` is the text after the opening parenthesis.
fn parse_register(rest: &str) -> Option<Register> {
    let args = &rest[..rest.find(')')?];
    let mut chars = args.chars();
    let class = chars.next().filter(|c| "btus".contains(*c))?;
    let digits: String = chars.take_while(char::is_ascii_digit).collect();
    Some(Register {
        class,
        index: digits.parse().ok()?,
        space: parse_space(args),
    })
}

fn parse_space(args: &str) -> u32 {
    let Some(at) = args.find("space") else {
        return 0;
    };
    args[at + "space".len()..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap_or(0)
}

fn parse_slot(chunk: &str) -> Option<Slot> {
    if chunk.contains("vk::push_constant") {
        return Some(Slot::PushConstant);
    }
    let at = chunk.find("vk::binding(")? + "vk::binding(".len();
    let args = &chunk[at..chunk[at..].find(')')? + at];
    let mut parts = args.split(',').map(|p| p.trim().parse::<u32>());
    let binding = parts.next()?.ok()?;
    let set = parts.next().unwrap_or(Ok(0)).ok()?;
    Some(Slot::Qualified { set, binding })
}

fn last_identifier(text: &str) -> Option<&str> {
    let end = text.rfind(|c: char| c.is_alphanumeric() || c == '_')? + 1;
    let start = text[..end]
        .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
        .map_or(0, |at| at + 1);
    Some(&text[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "\
struct P { float a; float b; };\n\
[[vk::push_constant]] ConstantBuffer<P> post : register(b0);\n\
[[vk::binding(0, 0)]] Texture2D<float4> src : register(t0);\n\
[[vk::binding(1, 0)]] SamplerState src_sampler : register(s0);\n\
[[vk::binding(4, 1)]] TextureCube<float4> probe_cubes[8] : register(t4, space1);\n\
float4 shade(float2 uv : TEXCOORD0) : SV_Target { return src.Sample(src_sampler, uv); }\n";

    fn named<'a>(found: &'a [Declaration], name: &str) -> &'a Declaration {
        found.iter().find(|d| d.name == name).expect(name)
    }

    #[test]
    fn every_annotated_resource_is_read_and_nothing_else_is() {
        let found = resource_declarations(SOURCE);
        let names: Vec<&str> = found.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, ["post", "src", "src_sampler", "probe_cubes"]);
    }

    // The cull declares its command buffer after `make_command` closes, so the
    // declaration shares a `;`-delimited statement with that function's body.
    // The DirectX ABI check is what caught this: the resource read as absent,
    // not as mis-registered.
    #[test]
    fn a_declaration_after_a_function_body_is_still_read() {
        let source = "\
DrawCommand make_command(uint i)\n{\n    DrawCommand c;\n    return c;\n}\n\n\
[[vk::binding(2, 0)]] RWStructuredBuffer<DrawCommand> commands : register(u0);\n";
        let found = resource_declarations(source);
        let names: Vec<&str> = found.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, ["commands"]);
        assert_eq!(
            found[0].register,
            Register {
                class: 'u',
                index: 0,
                space: 0
            }
        );
        assert_eq!(found[0].slot, Some(Slot::Qualified { set: 0, binding: 2 }));
    }

    #[test]
    fn a_push_constant_carries_a_slot_and_a_register() {
        let found = resource_declarations(SOURCE);
        let post = named(&found, "post");
        assert_eq!(post.slot, Some(Slot::PushConstant));
        assert_eq!(
            post.register,
            Register {
                class: 'b',
                index: 0,
                space: 0
            }
        );
        assert_eq!(post.count, None);
    }

    // A texture and its sampler are two declarations, each at its own binding
    // and in its own register class.
    #[test]
    fn a_texture_and_its_sampler_bind_apart() {
        let found = resource_declarations(SOURCE);
        assert_eq!(
            named(&found, "src").slot,
            Some(Slot::Qualified { set: 0, binding: 0 })
        );
        assert_eq!(
            named(&found, "src_sampler").slot,
            Some(Slot::Qualified { set: 0, binding: 1 })
        );
        assert_eq!(named(&found, "src").register.class, 't');
        assert_eq!(named(&found, "src_sampler").register.class, 's');
    }

    // `[[vk::binding(binding, set)]]` is binding-first, the reverse of how a
    // descriptor set is usually spoken about.
    #[test]
    fn a_qualified_slot_reads_the_binding_before_the_set() {
        let found = resource_declarations(SOURCE);
        let cubes = named(&found, "probe_cubes");
        assert_eq!(cubes.slot, Some(Slot::Qualified { set: 1, binding: 4 }));
        assert_eq!(cubes.register.index, 4);
        assert_eq!(cubes.count, Some(8));
    }

    // A register space is a D3D concept alone -- no Metal or Vulkan binding
    // reads it -- but it is part of the D3D slot, so it is read back and it
    // must not defeat the class or the number.
    #[test]
    fn a_register_space_is_read_without_disturbing_the_class_or_the_number() {
        let found = resource_declarations("Texture2D<float4> pool[] : register(t0, space1);");
        assert_eq!(found[0].register.class, 't');
        assert_eq!(found[0].register.index, 0);
        assert_eq!(found[0].register.space, 1);
        assert_eq!(found[0].count, None);
    }

    // The common case: no space named is space 0, which is what a root
    // signature's default register space is.
    #[test]
    fn an_unspelled_space_is_zero() {
        let found = resource_declarations("Texture2D<float4> t : register(t7);");
        assert_eq!(found[0].register.space, 0);
    }

    // The scan reads preprocessed text, so a `#line` directive rides along in
    // front of a declaration and must not be taken for its name.
    #[test]
    fn a_line_directive_ahead_of_a_declaration_is_not_its_name() {
        let found = resource_declarations(
            "#line 12 \"bloom.hlsl\"\n[[vk::binding(0, 0)]] SamplerState s : register(s3);",
        );
        assert_eq!(found[0].name, "s");
        assert_eq!(found[0].register.index, 3);
    }

    #[test]
    fn a_declaration_without_a_vk_annotation_reads_as_unbound() {
        let found = resource_declarations("Texture2D<float4> t : register(t7);");
        assert_eq!(found[0].slot, None);
    }

    // The argument-buffer annotation and an unsized array are read off the
    // declaration itself; a sized array keeps its count.
    #[test]
    fn an_argument_buffer_and_an_unsized_array_are_read() {
        let found = resource_declarations(
            "[[vk::binding(0, 1)]] [[cn::metal_argument_buffer(7)]] \
             Texture2DArray<float> shadow : register(t0, space1);\
             [[vk::binding(1, 1)]] Texture2D<float4> pool[] : register(t1, space1);\
             [[vk::binding(0, 2)]] Texture2D<float4> fixed[4] : register(t0, space2);",
        );
        assert_eq!(named(&found, "shadow").metal_argument_buffer, Some(7));
        assert!(!named(&found, "shadow").runtime_sized);
        let pool = named(&found, "pool");
        assert!(pool.runtime_sized);
        assert_eq!(pool.count, None);
        assert_eq!(pool.metal_argument_buffer, None);
        assert!(!named(&found, "fixed").runtime_sized);
        assert_eq!(named(&found, "fixed").count, Some(4));
        let plain = resource_declarations(SOURCE);
        assert!(plain.iter().all(|d| d.metal_argument_buffer.is_none()));
    }

    // dxc never sees a `cn::` attribute, so a misspelled one would otherwise be
    // silent; the error names what was written.
    #[test]
    fn a_misspelled_engine_attribute_is_rejected_by_name() {
        let source = "Texture2D<float4> t : register(t0);\n\
             [[vk::binding(0, 2)]] [[cn::metal_argument_bufer(11)]]\n\
             TextureCube<float4> cubes[8] : register(t4, space2);";
        let err = strip_engine_attributes(source).unwrap_err();
        assert!(err.contains("`cn::metal_argument_bufer`"), "{err}");
        assert!(err.contains("line 2"), "{err}");
    }

    // Blanking keeps every byte offset, so a dxc diagnostic past the attribute
    // still names the line and column the author wrote.
    #[test]
    fn stripping_keeps_every_line_and_column() {
        let source = "[[vk::binding(0, 2)]] [[cn::metal_argument_buffer(11)]]\n\
             TextureCube<float4> cubes[8] : register(t4, space2);\n\
             [[cn::metal_argument_buffer(\n11)]] Texture2D<float4> x : register(t5, space2);\n";
        let stripped = strip_engine_attributes(source).unwrap();
        assert!(!stripped.contains("cn::"), "{stripped}");
        assert_eq!(stripped.len(), source.len());
        let lines = |text: &str| -> Vec<usize> { text.lines().map(str::len).collect() };
        assert_eq!(lines(&stripped), lines(source));
        assert_eq!(stripped.find("cubes"), source.find("cubes"));
        assert_eq!(stripped.find("Texture2D"), source.find("Texture2D"));
        assert!(stripped.starts_with("[[vk::binding(0, 2)]] "));
    }

    // A specifier shared with a `vk::` attribute keeps it, and loses the comma
    // that separated the two.
    #[test]
    fn a_shared_specifier_keeps_its_other_attributes() {
        let stripped = strip_engine_attributes(
            "[[cn::metal_argument_buffer(3), vk::binding(0, 1)]] Texture2D<float4> a[4] : \
             register(t0);\n[[vk::binding(1, 1), cn::metal_argument_buffer(3)]] \
             Texture2D<float4> b[4] : register(t4);",
        )
        .unwrap();
        let squeezed: String = stripped.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(
            squeezed,
            "[[ vk::binding(0, 1)]] Texture2D<float4> a[4] : register(t0); \
             [[vk::binding(1, 1) ]] Texture2D<float4> b[4] : register(t4);"
        );
    }

    // A comment may name any attribute; only code is read.
    #[test]
    fn an_attribute_in_a_comment_is_left_alone() {
        let source = "// see [[cn::anything(1)]]\n/* and [[cn::other]] */\nfloat x;\n";
        let stripped = strip_engine_attributes(source).unwrap();
        assert!(matches!(stripped, Cow::Borrowed(_)));
    }

    #[test]
    fn a_source_without_engine_attributes_is_borrowed_unchanged() {
        let stripped = strip_engine_attributes(SOURCE).unwrap();
        assert!(matches!(stripped, Cow::Borrowed(text) if text == SOURCE));
    }

    // dxc drops unread inputs and unused resources from the entry interface of a
    // source declaring a combined image sampler, whatever it is asked to
    // preserve, so the engine refuses the attribute wherever it is spelled.
    #[test]
    fn a_combined_image_sampler_is_refused_by_line() {
        for spelling in [
            "[[vk::combinedImageSampler]]",
            "[[vk::binding(0, 0), vk::combinedImageSampler]]",
            "[[ vk :: combinedImageSampler ]]",
        ] {
            let source = format!("float x;\n{spelling} Texture2D<float4> t : register(t0);");
            let err = strip_engine_attributes(&source).unwrap_err();
            assert!(
                err.contains("line 2") && err.contains("combinedImageSampler"),
                "{spelling}: {err}"
            );
        }
        let commented = "// [[vk::combinedImageSampler]]\nTexture2D<float4> t : register(t0);";
        assert!(strip_engine_attributes(commented).is_ok());
    }
}
