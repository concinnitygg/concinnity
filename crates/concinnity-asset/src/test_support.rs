// Shared fixtures for the crate's unit tests.
//
// The resolver seams are process-global and install-once, so every test module
// installs the same stand-ins here rather than its own: the result a named
// reference deserializes to is then the same whichever order the harness runs
// tests in.

use serde::de::{self, Deserializer, Visitor};

// A name resolves to its byte length: a deterministic stand-in for the build's
// declaration-ordered resource tables. Names prefixed `unknown_` resolve to
// nothing, standing in for a reference no resource of that kind declares.
pub(crate) fn len_handle_resolver(name: &str) -> Option<u32> {
    if name.starts_with("unknown_") {
        None
    } else {
        Some(name.len() as u32)
    }
}

// A name resolves to its byte length, standing in for the build-time interner
// the handle resolvers fall back to.
pub(crate) fn len_name_resolver(name: &str) -> u32 {
    name.len() as u32
}

// Installs every seam with the stand-ins above.
pub(crate) fn install_resolvers() {
    crate::set_name_resolver(len_name_resolver);
    crate::set_texture_handle_resolver(len_handle_resolver);
    crate::set_audio_clip_handle_resolver(len_handle_resolver);
    crate::set_font_handle_resolver(len_handle_resolver);
    crate::set_mesh_handle_resolver(len_handle_resolver);
    crate::set_material_handle_resolver(len_handle_resolver);
    crate::set_skinned_mesh_handle_resolver(len_handle_resolver);
    crate::set_shader_handle_resolver(len_handle_resolver);
}

// Reports `None` from `deserialize_any`, the way an option-aware self-describing
// format does. serde_json only ever reports a `null` unit, so the optional
// reference helpers' `visit_none` arm needs this stand-in.
pub(crate) struct NoneDeserializer;

impl<'de> Deserializer<'de> for NoneDeserializer {
    type Error = de::value::Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_none()
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map struct enum identifier ignored_any
    }
}
