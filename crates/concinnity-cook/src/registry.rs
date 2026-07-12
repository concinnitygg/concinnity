// src/registry.rs
//
// The authoring metadata registry: `ComponentType`, the enum of every asset
// type paired with its on-disk discriminant, plus the authoring-only operations
// over it (name parsing, arg reserialization, enum-field probing, and
// reference-field listing). Consumed by the build pipeline and the in-engine
// editor; never by the runtime ECS, which loads components straight from their
// blob discriminants (`concinnity_core::ecs::ComponentAsset::from_def`).
//
// The component list itself is the single source of truth in
// `concinnity_core::ecs::registry` (the `for_each_component!` macro); this
// module instantiates the authoring half of it.

use crate::ecs::{Component, Registration};
use crate::result::CnResult;

// Extract the allowed enum variants from a serde "unknown variant" error
// message, coping with the count-dependent phrasing: "expected one of `a`, `b`,
// `c`" (3+), "expected `a` or `b`" (2), and "expected `a`" (1). Collects every
// backtick-quoted token that appears after the `expected` keyword (so the
// offending value, quoted before it, is skipped). Returns `None` for any other
// error (a type mismatch, a non-enum field), so callers fall back to treating
// the field as free text.
pub(crate) fn parse_expected_variants(msg: &str) -> Option<Vec<String>> {
    let after = msg.split_once("expected")?.1;
    let mut out = Vec::new();
    let mut rest = after;
    while let Some(open) = rest.find('`') {
        let tail = &rest[open + 1..];
        let close = tail.find('`')?;
        out.push(tail[..close].to_string());
        rest = &tail[close + 1..];
    }
    (!out.is_empty()).then_some(out)
}

// Generate `ComponentType` and its authoring methods from the shared component
// list. Invoked once, below, via `concinnity_core::for_each_component!`.
macro_rules! define_component_type {
    // Each entry carries a `{ ... }` metadata block used by `cn_impl_components!`
    // to generate the trivial `Component` impls; the authoring registry only
    // needs the `Variant => Type` pair and ignores the block.
    ( $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum ComponentType {
            $( $variant ),+
        }

        impl ComponentType {
            pub fn as_str(self) -> &'static str {
                match self { $( Self::$variant => <$ty as Component>::NAME ),+ }
            }
            // The on-disk blob tag / in-memory `ComponentId`, derived from the
            // shared `ComponentTag` enum (list position) so it matches the
            // runtime loader exactly.
            pub fn discriminant(self) -> u8 {
                match self {
                    $( Self::$variant => crate::ecs::ComponentTag::$variant as u8 ),+
                }
            }
            #[allow(dead_code)]
            pub fn from_discriminant(val: u8) -> Option<Self> {
                $( if val == crate::ecs::ComponentTag::$variant as u8 { return Some(Self::$variant); } )+
                None
            }
            // Whether cook emits this type on the baked record path
            // (`RecordKind::Baked`). Reads the type's `Component::BAKED` flag.
            pub fn baked(self) -> bool {
                match self { $( Self::$variant => <$ty as Component>::BAKED ),+ }
            }
            // A name either matches a known type or it does not; callers that
            // want a message supply their own via `ok_or`/`ok_or_else`.
            pub fn parse(s: &str) -> Option<Self> {
                $(
                    if s == <$ty as Component>::NAME { return Some(Self::$variant); }
                )+
                None
            }
            pub fn registration(self) -> Registration {
                match self { $( Self::$variant => <$ty as Component>::registration() ),+ }
            }
            // Re-serialize a JSON args value through the typed `Args` struct.
            // With the build interner active, name-string cross-references in
            // the args are interned to `AssetId` integers; the returned bytes
            // are the JSON `args_bytes` stored in the blob.
            pub fn reserialize_args(self, args: &serde_json::Value) -> Result<Vec<u8>, CnResult> {
                // Deserializing the args interns any name-string cross-reference,
                // which needs the name resolver installed. The build pipeline
                // resets the interner before it gets here; installing it again is
                // a cheap no-op and lets standalone callers (e.g. `cn check`
                // validation) deserialize without doing their own setup.
                crate::ecs::asset_id::ensure_name_resolver();
                match self {
                    $(
                        Self::$variant => {
                            let typed = serde_json::from_value::<<$ty as Component>::Args>(
                                args.clone(),
                            )?;
                            Ok(serde_json::to_vec(&typed)?)
                        }
                    ),+
                }
            }
            // The allowed values of a string-enum `Args` field (in declaration
            // order), or `None` if `field` is a free-form string / absent / not a
            // string-enum. Probes the typed `Args` by deserializing the defaults
            // with `field` set to a sentinel: a string-enum yields serde's
            // "unknown variant ..., expected ..." which `parse_expected_variants`
            // reads; a free string accepts the sentinel and yields `None`. Used by
            // authoring tools to offer a picker instead of a free text box; it
            // degrades to `None` (free text) if serde's phrasing ever changes.
            pub fn field_enum_variants(self, field: &str) -> Option<Vec<String>> {
                const SENTINEL: &str = "\u{0}__cn_enum_probe_sentinel__";
                match self {
                    $(
                        Self::$variant => {
                            let mut probe = match serde_json::to_value(
                                <<$ty as Component>::Args as Default>::default(),
                            ) {
                                Ok(serde_json::Value::Object(m)) => m,
                                _ => return None,
                            };
                            probe.get(field)?;
                            probe.insert(
                                field.to_string(),
                                serde_json::Value::String(SENTINEL.to_string()),
                            );
                            match serde_json::from_value::<<$ty as Component>::Args>(
                                serde_json::Value::Object(probe),
                            ) {
                                Ok(_) => None,
                                Err(e) => parse_expected_variants(&e.to_string()),
                            }
                        }
                    ),+
                }
            }
            // The asset-reference fields of this type, as (field, target_type).
            // See `Component::ref_fields`.
            pub fn ref_fields(self) -> &'static [(&'static str, &'static str)] {
                match self {
                    $(
                        Self::$variant => <$ty as Component>::ref_fields()
                    ),+
                }
            }
            #[allow(dead_code)]
            pub fn addable(self) -> bool {
                self.registration().addable()
            }
            pub fn all() -> &'static [(ComponentType, fn() -> Registration)] {
                &[
                    $( (Self::$variant, <$ty as Component>::registration as fn() -> Registration) ),+
                ]
            }
            pub fn addable_types() -> impl Iterator<Item = (ComponentType, Registration)> {
                Self::all()
                    .iter()
                    .map(|(t, reg_fn)| (*t, reg_fn()))
                    .filter(|(_, reg)| reg.addable())
            }
        }
    };
}

concinnity_core::for_each_component!(define_component_type);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_types_round_trip_name_and_discriminant() {
        for &(ty, _) in ComponentType::all() {
            assert_eq!(ComponentType::parse(ty.as_str()), Some(ty));
            assert_eq!(
                ComponentType::from_discriminant(ty.discriminant()),
                Some(ty)
            );
        }
        assert_eq!(ComponentType::parse("NotARealComponent"), None);
        assert_eq!(ComponentType::from_discriminant(255), None);
    }

    // On-disk discriminants must stay unique and inside the component range; the
    // only iterator over the full list is `ComponentType::all`, so the invariant
    // is checked here even though the discriminants are a runtime/blob concern.
    #[test]
    fn component_discriminants_are_unique_and_in_range() {
        let mut seen = std::collections::HashSet::new();
        for &(ty, _) in ComponentType::all() {
            let d = ty.discriminant();
            assert!(
                d < 128,
                "{} discriminant {} outside the component range",
                ty.as_str(),
                d
            );
            assert!(seen.insert(d), "duplicate discriminant {d}");
        }
    }

    #[test]
    fn reserialize_args_round_trips_and_rejects_bad_types() {
        let ty = ComponentType::parse("Mesh").unwrap();
        let bytes = ty
            .reserialize_args(&serde_json::json!({ "source": "a.glb" }))
            .unwrap();
        let back: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back["source"], "a.glb");
        assert_eq!(
            ty.reserialize_args(&serde_json::json!({ "source": 42 }))
                .unwrap_err(),
            CnResult::InvalidArgument
        );
    }

    // Convention guard for the asset-reference contract: a user-declarable
    // asset's `args` is its public JSON schema: always a JSON object of common
    // types, never a bare scalar or enum. `Component::Args` must therefore
    // serialize to a JSON object, and its `Default` must construct and serialize
    // cleanly. Internal/runtime-only assets (e.g. command enums) are exempt.
    #[test]
    fn declarable_assets_have_object_args_schemas() {
        for &(ty, reg_fn) in ComponentType::all() {
            let reg = reg_fn();
            if !reg.addable() {
                continue;
            }
            let default_args = reg.default_args.as_ref().unwrap_or_else(|| {
                panic!(
                    "{}: Args::default() failed to serialize to JSON",
                    ty.as_str()
                )
            });
            assert!(
                default_args.is_object(),
                "{}: args schema is not a JSON object (got {default_args}). A declarable \
                 asset's args must be a JSON object of common types.",
                ty.as_str()
            );
        }
    }

    // `field_enum_variants` returns a string-enum field's allowed values (in
    // declaration order) and `None` for a free-form string or a non-enum field.
    #[test]
    fn field_enum_variants_reports_string_enum_values() {
        assert_eq!(
            ComponentType::Sprite.field_enum_variants("fit"),
            Some(vec!["fit".into(), "cover".into(), "bottom".into()])
        );
        assert_eq!(
            ComponentType::TextLabel.field_enum_variants("align"),
            Some(vec!["left".into(), "center".into(), "right".into()])
        );
        // A two-variant enum uses serde's "expected `a` or `b`" phrasing.
        assert_eq!(
            ComponentType::AudioCue.field_enum_variants("kind"),
            Some(vec!["music".into(), "sound".into()])
        );
        // A free-form string field is not an enum.
        assert_eq!(ComponentType::HitRegion.field_enum_variants("action"), None);
        assert_eq!(ComponentType::KeyBinding.field_enum_variants("key"), None);
        // An absent field yields None (not a panic).
        assert_eq!(ComponentType::Sprite.field_enum_variants("nope"), None);
        // A non-string field (probing it errors on type, not "unknown variant").
        assert_eq!(ComponentType::Sprite.field_enum_variants("x"), None);
    }

    // `ref_fields` reports each type's asset-reference fields and their targets;
    // every referenced target must itself be a real component type.
    #[test]
    fn ref_fields_name_real_target_types() {
        assert_eq!(ComponentType::Decal.ref_fields(), &[("texture", "Texture")]);
        assert_eq!(
            ComponentType::AudioEmitter.ref_fields(),
            &[("clip", "AudioClip"), ("prop", "Prop")]
        );
        // A type without references reports none.
        assert!(ComponentType::PointLight.ref_fields().is_empty());
        // Every declared ref field names an existing arg key and a real target
        // type -- either a component or a resource-only asset (e.g. AudioClip,
        // which has left the component registry).
        for &(ty, reg_fn) in ComponentType::all() {
            let default_args = reg_fn().default_args;
            for &(field, target) in ty.ref_fields() {
                assert!(
                    ComponentType::parse(target).is_some()
                        || crate::resource_handles::ResourceAssetType::parse(target).is_some(),
                    "{}.{field} targets unknown type {target}",
                    ty.as_str()
                );
                if let Some(serde_json::Value::Object(m)) = &default_args {
                    assert!(
                        m.contains_key(field),
                        "{}.{field} is not an arg of {}",
                        ty.as_str(),
                        ty.as_str()
                    );
                }
            }
        }
    }

    #[test]
    fn parse_expected_variants_handles_serde_phrasings() {
        assert_eq!(
            parse_expected_variants("unknown variant `z`, expected one of `a`, `b`, `c`"),
            Some(vec!["a".into(), "b".into(), "c".into()])
        );
        assert_eq!(
            parse_expected_variants("unknown variant `z`, expected `a` or `b`"),
            Some(vec!["a".into(), "b".into()])
        );
        assert_eq!(
            parse_expected_variants("unknown variant `z`, expected `only`"),
            Some(vec!["only".into()])
        );
        // A type-mismatch error has no backtick list after `expected`.
        assert_eq!(
            parse_expected_variants("invalid type: string \"z\", expected u32"),
            None
        );
    }

    // The per-instance components an entity is composed from are RuntimeOnly:
    // never authored in a world, never in the asset reference, and exempt from
    // the declarable-args contract above. Guard that they stay that way so a
    // stray `External` origin can't leak one into the authoring surface.
    #[test]
    fn per_instance_components_are_runtime_only() {
        for ty in [
            ComponentType::Transform,
            ComponentType::MeshRenderer,
            ComponentType::ModelRenderer,
            ComponentType::Collider,
            ComponentType::Interactable,
            ComponentType::Pickup,
            ComponentType::Parent,
            ComponentType::Children,
            ComponentType::SceneMember,
            ComponentType::GlobalTransform,
            ComponentType::RenderHandle,
            ComponentType::Held,
        ] {
            assert!(
                !ty.registration().addable(),
                "{} must be RuntimeOnly (not declarable)",
                ty.as_str()
            );
        }
    }
}
