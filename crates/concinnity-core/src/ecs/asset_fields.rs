//! The authoring metadata an asset's field types carry: which fields reference
//! other assets, and which accept a closed vocabulary.
//!
//! `#[derive(AssetFields)]` generates the [`AssetFields`] impl from the struct's
//! field types, so the tables cannot drift from the schema. A field counts as
//! a reference when its type is a [`ReferenceField`] ([`Ref<T>`](super::Ref),
//! a per-kind resource handle, or an `Option` / `Vec` of one), as a vocabulary
//! when its type is a [`Vocabulary`], and as a nested schema when its type
//! derives `AssetFields` itself, whose fields then appear under dotted paths
//! (`collider.shape`). Every other field is plain data.
//!
//! A path field marked `#[asset(owned_file)]` is also recorded as a file the
//! asset owns: its authored source, as opposed to content other assets share.
//!
//! Paths are the authored keys: a field's serde `rename` replaces its name, a
//! `flatten` field contributes its fields at the parent's level, and a `skip`
//! field is not authored at all.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::components::Vocabulary;
use crate::ecs::reference::{Ref, RefTarget};

/// One reference field: its dotted path and the asset types it may name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefField {
    /// The field's authored key, dotted through nested objects.
    pub path: String,
    /// The registry names of the asset types the field may name; empty for
    /// any declared asset.
    pub targets: &'static [&'static str],
}

/// One closed-vocabulary field: its dotted path and the names it accepts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumField {
    /// The field's authored key, dotted through nested objects.
    pub path: String,
    /// Every name the field accepts, in picker order.
    pub variants: &'static [&'static str],
}

/// The reference, vocabulary and owned-file fields one schema declares.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FieldTable {
    /// Every reference field, in declaration order.
    pub refs: Vec<RefField>,
    /// Every closed-vocabulary field, in declaration order.
    pub enums: Vec<EnumField>,
    /// The dotted path of every field holding a file the asset owns, in
    /// declaration order.
    pub owned_files: Vec<String>,
}

/// An authored schema's reference and vocabulary fields, derived from its
/// field types with `#[derive(AssetFields)]`.
pub trait AssetFields {
    /// Append this schema's fields to `out`, each path under `prefix` (empty
    /// at the top level).
    fn collect_fields(prefix: &str, out: &mut FieldTable);

    /// Both tables at once.
    fn field_table() -> FieldTable
    where
        Self: Sized,
    {
        let mut table = FieldTable::default();
        Self::collect_fields("", &mut table);
        table
    }

    /// Every field that references another asset.
    fn ref_fields() -> Vec<RefField>
    where
        Self: Sized,
    {
        Self::field_table().refs
    }

    /// Every field typed with a [`Vocabulary`].
    fn enum_fields() -> Vec<EnumField>
    where
        Self: Sized,
    {
        Self::field_table().enums
    }
}

// A list or an optional of a nested schema is addressed by the same dotted
// path: authoring tools walk through the array or the null.
impl<T: AssetFields> AssetFields for Option<T> {
    fn collect_fields(prefix: &str, out: &mut FieldTable) {
        T::collect_fields(prefix, out);
    }
}

impl<T: AssetFields> AssetFields for Vec<T> {
    fn collect_fields(prefix: &str, out: &mut FieldTable) {
        T::collect_fields(prefix, out);
    }
}

/// A field type that names other assets.
pub trait ReferenceField {
    /// The registry names of the asset types the field may name; empty for
    /// any declared asset.
    const TARGETS: &'static [&'static str];
}

impl<T: RefTarget> ReferenceField for Ref<T> {
    const TARGETS: &'static [&'static str] = T::TYPES;
}

impl<T: ReferenceField> ReferenceField for Option<T> {
    const TARGETS: &'static [&'static str] = T::TARGETS;
}

impl<T: ReferenceField> ReferenceField for Vec<T> {
    const TARGETS: &'static [&'static str] = T::TARGETS;
}

fn join(prefix: &str, key: Option<&str>) -> String {
    match key {
        None => String::from(prefix),
        Some(key) if prefix.is_empty() => String::from(key),
        Some(key) => format!("{prefix}.{key}"),
    }
}

/// The field classifier the derive expands to. Not an API: the generated code
/// calls `collect` on `&&&&Probe::<FieldType>::NEW`, and method resolution
/// picks the most specific trait the field type satisfies, trying a reference
/// first, then a nested schema, then a vocabulary, and settling on plain data.
#[doc(hidden)]
pub mod probe {
    use core::marker::PhantomData;

    use super::{AssetFields, EnumField, FieldTable, RefField, ReferenceField, Vocabulary, join};

    /// A zero-sized stand-in for a field of type `T`.
    pub struct Probe<T: ?Sized>(PhantomData<fn() -> *const T>);

    impl<T: ?Sized> Probe<T> {
        /// The probe value the generated code borrows.
        pub const NEW: Self = Probe(PhantomData);
    }

    /// A field that references other assets.
    pub trait ByReference {
        /// Record the field.
        fn collect(&self, prefix: &str, key: Option<&str>, out: &mut FieldTable);
    }

    impl<T: ReferenceField> ByReference for &&&Probe<T> {
        fn collect(&self, prefix: &str, key: Option<&str>, out: &mut FieldTable) {
            out.refs.push(RefField {
                path: join(prefix, key),
                targets: T::TARGETS,
            });
        }
    }

    /// A field holding a nested schema.
    pub trait ByNested {
        /// Record the nested schema's fields.
        fn collect(&self, prefix: &str, key: Option<&str>, out: &mut FieldTable);
    }

    impl<T: AssetFields> ByNested for &&Probe<T> {
        fn collect(&self, prefix: &str, key: Option<&str>, out: &mut FieldTable) {
            T::collect_fields(&join(prefix, key), out);
        }
    }

    /// A closed-vocabulary field.
    pub trait ByVocabulary {
        /// Record the field.
        fn collect(&self, prefix: &str, key: Option<&str>, out: &mut FieldTable);
    }

    impl<T: Vocabulary> ByVocabulary for &Probe<T> {
        fn collect(&self, prefix: &str, key: Option<&str>, out: &mut FieldTable) {
            out.enums.push(EnumField {
                path: join(prefix, key),
                variants: T::VARIANTS,
            });
        }
    }

    /// Record the field at `key` as a file the asset owns.
    pub fn owned_file(prefix: &str, key: Option<&str>, out: &mut FieldTable) {
        out.owned_files.push(join(prefix, key));
    }

    /// Plain data: nothing to record.
    pub trait ByNothing {
        /// Record nothing.
        fn collect(&self, prefix: &str, key: Option<&str>, out: &mut FieldTable);
    }

    impl<T: ?Sized> ByNothing for Probe<T> {
        fn collect(&self, _: &str, _: Option<&str>, _: &mut FieldTable) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{PropColliderShape, SpriteFit, TextLabel};
    use crate::ecs::asset_id::AssetId;
    use crate::ecs::reference::AnyAsset;
    use crate::ecs::{AssetFields, TextureHandle};

    struct Either;

    impl RefTarget for Either {
        const TYPES: &'static [&'static str] = &["Prop", "SkyRotation"];
    }

    #[derive(AssetFields)]
    #[expect(dead_code, reason = "only the field types are read")]
    struct Leaf {
        label: Option<Ref<TextLabel>>,
        fit: SpriteFit,
        weight: f32,
        #[asset(owned_file)]
        script: alloc::string::String,
    }

    #[derive(AssetFields)]
    #[expect(dead_code, reason = "only the field types are read")]
    struct Middle {
        leaf: Leaf,
        #[serde(rename = "shape")]
        collider_shape: PropColliderShape,
        many: Vec<Leaf>,
        maybe: Option<Leaf>,
    }

    #[derive(AssetFields)]
    #[expect(dead_code, reason = "only the field types are read")]
    struct Top {
        #[serde(skip)]
        asset_id: AssetId,
        parent: Option<Ref<Either>>,
        list: Vec<Ref<TextLabel>>,
        grid: Vec<Vec<Ref<AnyAsset>>>,
        texture: Option<TextureHandle>,
        #[serde(rename = "inner")]
        middle: Middle,
        #[serde(flatten)]
        flat: Leaf,
        name: alloc::string::String,
        #[asset(owned_file)]
        #[serde(rename = "src")]
        source: Option<alloc::string::String>,
    }

    fn paths<F>(fields: &[F], path: impl Fn(&F) -> &str) -> Vec<&str> {
        fields.iter().map(path).collect()
    }

    #[test]
    fn references_are_found_by_type_through_every_wrapper() {
        let refs = Top::ref_fields();
        assert_eq!(
            paths(&refs, |f| &f.path),
            [
                "parent",
                "list",
                "grid",
                "texture",
                "inner.leaf.label",
                "inner.many.label",
                "inner.maybe.label",
                "label",
            ]
        );
        assert_eq!(refs[0].targets, ["Prop", "SkyRotation"]);
        assert_eq!(refs[1].targets, ["TextLabel"]);
        assert!(refs[2].targets.is_empty());
        assert_eq!(refs[3].targets, ["Texture"]);
    }

    #[test]
    fn vocabulary_fields_are_found_by_type_under_their_authored_keys() {
        let enums = Top::enum_fields();
        assert_eq!(
            paths(&enums, |f| &f.path),
            [
                "inner.leaf.fit",
                "inner.shape",
                "inner.many.fit",
                "inner.maybe.fit",
                "fit",
            ]
        );
        assert_eq!(enums[0].variants, SpriteFit::NAMES);
        assert_eq!(enums[1].variants, PropColliderShape::NAMES);
    }

    #[test]
    fn owned_files_are_found_by_mark_under_their_authored_keys() {
        assert_eq!(
            Top::field_table().owned_files,
            [
                "inner.leaf.script",
                "inner.many.script",
                "inner.maybe.script",
                "script",
                "src",
            ]
        );
    }

    #[test]
    fn a_schema_with_neither_reports_empty_tables() {
        #[derive(AssetFields)]
        struct Empty {}
        assert_eq!(Empty::field_table(), FieldTable::default());
    }
}
