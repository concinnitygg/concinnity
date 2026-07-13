// src/registry.rs
//
// The authoring metadata registry: `ComponentType`, the enum of every asset
// type paired with its on-disk discriminant, plus the authoring-only operations
// over it (name parsing, arg reserialization, enum-field probing, and
// reference-field listing). Consumed by the build pipeline and the in-engine
// editor; never by the runtime ECS, which loads components straight from their
// blob discriminants (`concinnity_core::ecs::ComponentAsset::from_baked`).
//
// The component list itself is the single source of truth in
// `concinnity_core::ecs::registry` (the `for_each_component!` macro); this
// module instantiates the authoring half of it.

use crate::ecs::{AssetOrigin, AssetPayload};
use crate::result::CnResult;

// Static authoring metadata for an asset type: how it is declared, whether it
// compiles a payload, and its default args JSON. Derived from the registry
// entry's metadata block -- the runtime `Component` trait carries none of it
// (blob records carry everything a shipped game loads).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Registration {
    pub type_name: &'static str,
    pub origin: AssetOrigin,
    pub payload: AssetPayload,
    pub default_args: Option<serde_json::Value>,
}

impl Registration {
    pub fn addable(&self) -> bool {
        self.origin == AssetOrigin::External
    }

    #[allow(dead_code)]
    pub fn serializable(&self) -> bool {
        self.origin != AssetOrigin::RuntimeOnly
    }

    #[allow(dead_code)]
    pub fn runtime_present(&self) -> bool {
        self.origin != AssetOrigin::BuildOnly
    }

    pub fn needs_compilation(&self) -> bool {
        self.payload == AssetPayload::Compiled
    }
}

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

// The empty args schema of a runtime-only component: never authored, so its
// registration carries an empty default and its reserialize accepts `{}`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct NoArgs {}

// Metadata scanners over a registry entry's `{ ... }` flag tokens. Each walks
// the token stream for its key and falls back to a default when absent; the
// generic `$t:tt` arm skips unrecognized tokens (other flags, `,`, `:`, and
// bracketed lists are each one token tree).

// The authoring origin: `external` / `build_only` / `runtime` (RuntimeOnly is
// also the fallback for entries with no origin flag).
macro_rules! __meta_origin {
    () => { AssetOrigin::RuntimeOnly };
    (external $($r:tt)*) => { AssetOrigin::External };
    (build_only $($r:tt)*) => { AssetOrigin::BuildOnly };
    (runtime $($r:tt)*) => { AssetOrigin::RuntimeOnly };
    ($t:tt $($r:tt)*) => { __meta_origin!($($r)*) };
}

// Whether the type compiles a blob payload (`compiled`).
macro_rules! __meta_payload {
    () => { AssetPayload::None };
    (compiled $($r:tt)*) => { AssetPayload::Compiled };
    ($t:tt $($r:tt)*) => { __meta_payload!($($r)*) };
}

// The authored args schema TYPE: the component itself by default, `NoArgs` for
// runtime-only entries, or the `args: <Type>` override for the types whose
// authored shape diverges from the runtime component.
macro_rules! __meta_args_ty {
    ($default:path;) => { $default };
    ($default:path; runtime $($r:tt)*) => { NoArgs };
    ($default:path; args: $a:ident $($r:tt)*) => { crate::assets::$a };
    ($default:path; $t:tt $($r:tt)*) => { __meta_args_ty!($default; $($r)*) };
}

// The args schema's NAME, for the docs pipeline (which renders the args
// struct's fields).
macro_rules! __meta_args_name {
    ($default:ident;) => { stringify!($default) };
    ($default:ident; args: $a:ident $($r:tt)*) => { stringify!($a) };
    ($default:ident; $t:tt $($r:tt)*) => { __meta_args_name!($default; $($r)*) };
}

// Apply the entry's bake-time validator (`validate: <fn>`, from
// `crate::validate`) to a typed value; identity when the entry declares none.
macro_rules! __meta_validate {
    ($val:expr;) => { $val };
    ($val:expr; validate: $f:ident $($r:tt)*) => { crate::validate::$f($val) };
    ($val:expr; $t:tt $($r:tt)*) => { __meta_validate!($val; $($r)*) };
}

// The `refs: [ ... ]` reference-field list; empty when absent.
macro_rules! __meta_refs {
    () => { &[] };
    (refs: [ $( ($fld:literal, $tgt:literal) ),+ $(,)? ] $($r:tt)*) => { &[ $( ($fld, $tgt) ),+ ] };
    ($t:tt $($r:tt)*) => { __meta_refs!($($r)*) };
}

// Generate `ComponentType` and its authoring methods from the shared component
// list. Invoked once, below, via `concinnity_core::for_each_component!`. All
// authoring metadata (origin, payload, args schema, validators, reference
// fields) derives from each entry's `{ ... }` metadata block; the runtime
// `Component` trait carries none of it.
macro_rules! define_component_type {
    ( $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum ComponentType {
            $( $variant ),+
        }

        impl ComponentType {
            pub fn as_str(self) -> &'static str {
                match self { $( Self::$variant => stringify!($variant) ),+ }
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
            // A name either matches a known type or it does not; callers that
            // want a message supply their own via `ok_or`/`ok_or_else`.
            pub fn parse(s: &str) -> Option<Self> {
                $(
                    if s == stringify!($variant) { return Some(Self::$variant); }
                )+
                None
            }
            // The name of this type's authored args schema struct: the
            // component itself for pass-through types, the `args:` override
            // for the divergent ones. The docs pipeline renders that struct's
            // fields as the asset's parameters.
            #[allow(dead_code)]
            pub fn args_struct_name(self) -> &'static str {
                match self {
                    $( Self::$variant => __meta_args_name!($variant; $($meta)*) ),+
                }
            }
            pub fn registration(self) -> Registration {
                match self {
                    $(
                        Self::$variant => Registration {
                            type_name: stringify!($variant),
                            origin: __meta_origin!($($meta)*),
                            payload: __meta_payload!($($meta)*),
                            default_args: serde_json::to_value(
                                <__meta_args_ty!($ty; $($meta)*) as Default>::default(),
                            )
                            .ok(),
                        }
                    ),+
                }
            }
            // Bake a JSON args value into the blob record's component bytes:
            // deserialize through the typed args schema (interning name-string
            // cross-references), apply the type's bake-time validator, and
            // serialize the runtime component. For a pass-through type the
            // args ARE the component; a divergent type (`args:` metadata)
            // routes through its `bake` translation in `bake_divergent`.
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
                            let typed = serde_json::from_value::<__meta_args_ty!($ty; $($meta)*)>(
                                args.clone(),
                            )?;
                            Ok(serde_json::to_vec(&__meta_validate!(typed; $($meta)*))?)
                        }
                    ),+
                }
            }
            // The allowed values of a string-enum args field (in declaration
            // order), or `None` if `field` is a free-form string / absent / not a
            // string-enum. Probes the typed args by deserializing the defaults
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
                                <__meta_args_ty!($ty; $($meta)*) as Default>::default(),
                            ) {
                                Ok(serde_json::Value::Object(m)) => m,
                                _ => return None,
                            };
                            probe.get(field)?;
                            probe.insert(
                                field.to_string(),
                                serde_json::Value::String(SENTINEL.to_string()),
                            );
                            match serde_json::from_value::<__meta_args_ty!($ty; $($meta)*)>(
                                serde_json::Value::Object(probe),
                            ) {
                                Ok(_) => None,
                                Err(e) => parse_expected_variants(&e.to_string()),
                            }
                        }
                    ),+
                }
            }
            // The asset-reference fields of this type, as (field, target_type),
            // from the entry's `refs:` metadata.
            pub fn ref_fields(self) -> &'static [(&'static str, &'static str)] {
                match self {
                    $(
                        Self::$variant => __meta_refs!($($meta)*)
                    ),+
                }
            }
            #[allow(dead_code)]
            pub fn addable(self) -> bool {
                self.registration().addable()
            }
            pub fn all() -> &'static [ComponentType] {
                &[ $( Self::$variant ),+ ]
            }
            pub fn addable_types() -> impl Iterator<Item = (ComponentType, Registration)> {
                Self::all()
                    .iter()
                    .map(|t| (*t, t.registration()))
                    .filter(|(_, reg)| reg.addable())
            }
        }
    };
}

concinnity_core::for_each_component!(define_component_type);

// Bake the runtime component for the asset types whose baked form diverges
// from their authored args (the entries with `args:` metadata): run the type's
// `bake` translation at build time and serialize the component itself, which
// the type's `from_baked` deserializes at load. Pass-through types return
// `None`; cook reserializes their args (which ARE the component).
pub fn bake_divergent(
    ct: ComponentType,
    args: &serde_json::Value,
) -> Result<Option<Vec<u8>>, CnResult> {
    // Deserializing the args interns name-string cross-references, exactly as
    // `reserialize_args` does.
    crate::ecs::asset_id::ensure_name_resolver();
    macro_rules! bake {
        ($ty:ty, $args_ty:ty) => {{
            let typed = serde_json::from_value::<$args_ty>(args.clone())?;
            Ok(Some(serde_json::to_vec(&<$ty>::bake(typed))?))
        }};
    }
    match ct {
        ComponentType::Camera3D => {
            bake!(crate::assets::Camera3D, crate::assets::Camera3DArgs)
        }
        ComponentType::Room => bake!(crate::assets::Room, crate::assets::RoomArgs),
        ComponentType::File => bake!(crate::assets::File, crate::assets::FileArgs),
        ComponentType::Spawner => bake!(crate::assets::Spawner, crate::assets::SpawnerArgs),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_predicates_follow_origin_and_payload() {
        let reg = |origin, payload| Registration {
            type_name: "T",
            origin,
            payload,
            default_args: None,
        };
        let external = reg(AssetOrigin::External, AssetPayload::Compiled);
        assert!(external.addable());
        assert!(external.serializable());
        assert!(external.runtime_present());
        assert!(external.needs_compilation());

        let runtime = reg(AssetOrigin::RuntimeOnly, AssetPayload::None);
        assert!(!runtime.addable());
        assert!(!runtime.serializable());
        assert!(runtime.runtime_present());
        assert!(!runtime.needs_compilation());

        let build = reg(AssetOrigin::BuildOnly, AssetPayload::None);
        assert!(!build.addable());
        assert!(build.serializable());
        assert!(!build.runtime_present());
        assert!(!build.needs_compilation());
    }

    // The divergent bake produces bytes the type's `from_baked` reconstructs
    // to the same component `from_args` builds at runtime: the baked path and
    // the authored path converge on identical components.
    #[test]
    fn bake_divergent_round_trips_through_from_baked() {
        use crate::ecs::Component;
        use crate::ecs::asset_id;

        let args = serde_json::json!({"size": [16.0, 20.0, 3.5]});
        let bytes = bake_divergent(ComponentType::Room, &args)
            .unwrap()
            .expect("Room bakes divergently");
        let baked = crate::assets::Room::from_baked(&bytes).unwrap();
        // The size shorthand resolved at bake time.
        assert_eq!(baked.half_width, 8.0);
        assert_eq!(baked.half_depth, 10.0);
        assert_eq!(baked.ceiling_height, 3.5);

        let args = serde_json::json!({"position": [1.0, 2.0, 3.0], "yaw": 0.5});
        let bytes = bake_divergent(ComponentType::Camera3D, &args)
            .unwrap()
            .expect("Camera3D bakes divergently");
        let baked = crate::assets::Camera3D::from_baked(&bytes).unwrap();
        assert_eq!(baked.position, [1.0, 2.0, 3.0]);
        // The view matrix composed at bake time.
        let expected = crate::assets::Camera3D::bake(
            serde_json::from_value(serde_json::json!({"position": [1.0, 2.0, 3.0], "yaw": 0.5}))
                .unwrap(),
        );
        assert_eq!(baked.view_matrix, expected.view_matrix);

        let args = serde_json::json!({"path": "tri.obj"});
        let bytes = bake_divergent(ComponentType::File, &args)
            .unwrap()
            .expect("File bakes divergently");
        let baked = crate::assets::File::from_baked(&bytes).unwrap();
        // The kind derived from the extension at bake time.
        assert!(baked.kind.is_some());

        asset_id::reset_interner();
        let args = serde_json::json!({"template": "crate", "interval": -1.0, "lifetime": 2.0});
        let bytes = bake_divergent(ComponentType::Spawner, &args)
            .unwrap()
            .expect("Spawner bakes divergently");
        let baked = crate::assets::Spawner::from_baked(&bytes).unwrap();
        // The interval clamped and the runtime counters zeroed at bake time.
        assert_eq!(baked.interval, 0.0);
        assert_eq!(baked.elapsed, 0.0);
        assert_eq!(baked.count, 0);

        // A pass-through type does not bake divergently.
        assert!(
            bake_divergent(ComponentType::PointLight, &serde_json::json!({}))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn component_types_round_trip_name_and_discriminant() {
        for &ty in ComponentType::all() {
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
        for &ty in ComponentType::all() {
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
        let ty = ComponentType::parse("ProceduralMesh").unwrap();
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
        for &ty in ComponentType::all() {
            let reg = ty.registration();
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
        for &ty in ComponentType::all() {
            let default_args = ty.registration().default_args;
            for &(field, target) in ty.ref_fields() {
                assert!(
                    ComponentType::parse(target).is_some()
                        || crate::resource_type::ResourceAssetType::parse(target).is_some(),
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
