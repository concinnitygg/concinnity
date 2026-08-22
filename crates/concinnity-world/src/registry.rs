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

use crate::result::CnResult;

pub use crate::ecs::{AssetOrigin, AssetPayload};

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

// The bare structural flags: `singleton` (at most one instance belongs to a
// world), `useful_blank` (meaningful when declared with only default args, so
// authoring tools offer a plain add), and `renders` (presence implies the
// world renders). `__meta_useful_blank` / `__meta_renders` are shared with the
// resource-asset registry in `resource_type`.
// The recursive arms are path-qualified so the shared scanners also expand
// from other modules (`resource_type` invokes them by path).
macro_rules! __meta_singleton {
    () => { false };
    (singleton $($r:tt)*) => { true };
    ($t:tt $($r:tt)*) => { crate::registry::__meta_singleton!($($r)*) };
}
macro_rules! __meta_useful_blank {
    () => { false };
    (useful_blank $($r:tt)*) => { true };
    ($t:tt $($r:tt)*) => { crate::registry::__meta_useful_blank!($($r)*) };
}
macro_rules! __meta_renders {
    () => { false };
    (renders $($r:tt)*) => { true };
    ($t:tt $($r:tt)*) => { crate::registry::__meta_renders!($($r)*) };
}
pub(crate) use {__meta_renders, __meta_singleton, __meta_useful_blank};

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
            // serialize the runtime component as postcard. For a pass-through
            // type the args ARE the component; a divergent type (`args:`
            // metadata) routes through its `bake` translation in
            // `bake_divergent`.
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
                            )
                            .map_err(json_args_err)?;
                            Ok(postcard::to_allocvec(&__meta_validate!(typed; $($meta)*))?)
                        }
                    ),+
                }
            }
            // Normalize a JSON args value through the typed args schema: the
            // same deserialize + validate as `reserialize_args`, but back to
            // JSON with defaults filled and references resolved. Authoring
            // tools (`cn add`, the editor form) write this into world.jsonl;
            // the baked postcard bytes cannot round-trip to JSON.
            pub fn normalized_args(
                self,
                args: &serde_json::Value,
            ) -> Result<serde_json::Value, CnResult> {
                crate::ecs::asset_id::ensure_name_resolver();
                match self {
                    $(
                        Self::$variant => {
                            let typed = serde_json::from_value::<__meta_args_ty!($ty; $($meta)*)>(
                                args.clone(),
                            )
                            .map_err(json_args_err)?;
                            serde_json::to_value(&__meta_validate!(typed; $($meta)*))
                                .map_err(json_args_err)
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
            // The structural flags, from the entry's metadata: `singleton`
            // (at most one instance belongs to a world; authoring tools use an
            // edit-or-add flow), `useful_blank` (meaningful when declared with
            // only default args, so authoring tools offer a plain add), and
            // `renders` (presence implies the world renders; drives the
            // GraphicsConfig companion injection at build time).
            pub fn singleton(self) -> bool {
                match self {
                    $( Self::$variant => __meta_singleton!($($meta)*) ),+
                }
            }
            pub fn useful_blank(self) -> bool {
                match self {
                    $( Self::$variant => __meta_useful_blank!($($meta)*) ),+
                }
            }
            pub fn renders(self) -> bool {
                match self {
                    $( Self::$variant => __meta_renders!($($meta)*) ),+
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

// The authored-value trait: the bridge from a typed authoring struct to the
// world line that declares it. Implemented for every declarable asset's args
// schema (the `args:` override where the authored form diverges from the
// component, the component itself where it does not) and for every resource
// asset, so a caller hands the cook a typed value instead of a name/type/args
// triple assembled by hand.
pub trait Authored: serde::Serialize {
    /// The registered asset type, as it appears in a world line's `type`.
    const TYPE: &'static str;
}

// Runtime-only entries are never authored, so they get no impl; every other
// entry (including the `manual` ones, whose hand-written `Component` impl is
// unrelated to authoring) resolves its authored type through `__meta_args_ty`.
macro_rules! __authored_component {
    ( $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? ) => {
        $( __authored_component!(@one $variant $ty { $($meta)* }); )+
    };
    (@one $variant:ident $ty:path { runtime $($rest:tt)* }) => {};
    (@one $variant:ident $ty:path { $($meta:tt)* }) => {
        impl Authored for __meta_args_ty!($ty; $($meta)*) {
            const TYPE: &'static str = stringify!($variant);
        }
    };
}

// Resource assets carry no `args:` divergence: the declared type is the
// authored type.
macro_rules! __authored_resource {
    ( $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? ) => {
        $(
            impl Authored for $ty {
                const TYPE: &'static str = stringify!($variant);
            }
        )+
    };
}

concinnity_core::for_each_component!(__authored_component);
concinnity_core::for_each_resource_asset!(__authored_resource);

/// Serialize one authored asset into the world line that declares it, newline
/// included. The caller never names a JSON type: the line is finished text.
pub fn asset_line<T: Authored>(name: &str, value: &T) -> std::io::Result<String> {
    let bad =
        |e: serde_json::Error| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string());
    let args = serde_json::to_value(value).map_err(bad)?;
    let line = serde_json::json!({ "name": name, "type": T::TYPE, "args": args });
    let mut out = serde_json::to_string(&line).map_err(bad)?;
    out.push('\n');
    Ok(out)
}

#[cfg(test)]
mod authored_tests {
    use super::*;

    // The three shapes the generation has to cover: an args-schema override, a
    // pass-through component, and a resource asset.
    #[test]
    fn authored_types_report_their_registered_name() {
        assert_eq!(<crate::assets::RoomArgs as Authored>::TYPE, "Room");
        assert_eq!(
            <crate::assets::DirectionalLight as Authored>::TYPE,
            "DirectionalLight"
        );
        assert_eq!(<crate::assets::Texture as Authored>::TYPE, "Texture");
        assert_eq!(
            <crate::assets::EnvironmentMap as Authored>::TYPE,
            "EnvironmentMap"
        );
    }

    // Every Authored type names a type the registry can actually parse, so a
    // value handed to the cook always lands on a declarable asset.
    #[test]
    fn the_reported_name_round_trips_through_the_registry() {
        for name in [
            <crate::assets::RoomArgs as Authored>::TYPE,
            <crate::assets::Camera3DArgs as Authored>::TYPE,
            <crate::assets::DirectionalLight as Authored>::TYPE,
        ] {
            assert!(
                ComponentType::parse(name).is_some(),
                "{name} is not a registered component type"
            );
        }
    }
}

// JSON args that fail the typed schema are an authoring error. Core dropped
// its `From<serde_json::Error>` conversion along with runtime JSON parsing,
// so the build side maps the error here.
fn json_args_err(e: serde_json::Error) -> CnResult {
    tracing::error!("JSON args error: {}", e);
    CnResult::InvalidArgument
}

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
            let typed = serde_json::from_value::<$args_ty>(args.clone()).map_err(json_args_err)?;
            Ok(Some(postcard::to_allocvec(&<$ty>::bake(typed))?))
        }};
    }
    match ct {
        ComponentType::Camera3D => {
            bake!(crate::assets::Camera3D, crate::assets::Camera3DArgs)
        }
        ComponentType::Room => bake!(crate::assets::Room, crate::assets::RoomArgs),
        ComponentType::File => bake!(crate::assets::File, crate::assets::FileArgs),
        ComponentType::Spawner => bake!(crate::assets::Spawner, crate::assets::SpawnerArgs),
        ComponentType::Application => {
            bake!(crate::assets::Application, crate::assets::ApplicationArgs)
        }
        _ => Ok(None),
    }
}

// Whether an asset type's presence implies the world renders: the registry's
// `renders` flag, across both the component and resource registries. Matches
// by normalized name (case-insensitive, underscores stripped) so cook's
// companion pass and authoring tools classify the same way.
pub fn type_renders(asset_type: &str) -> bool {
    let norm: String = asset_type.chars().filter(|c| *c != '_').collect();
    let matches = |name: &str| name.eq_ignore_ascii_case(&norm);
    ComponentType::all()
        .iter()
        .any(|t| t.renders() && matches(t.as_str()))
        || crate::resource_type::ResourceAssetType::all()
            .iter()
            .any(|t| t.renders() && matches(t.as_str()))
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
        let back: crate::assets::ProceduralMesh = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.source.as_deref(), Some("a.glb"));
        assert_eq!(
            ty.reserialize_args(&serde_json::json!({ "source": 42 }))
                .unwrap_err(),
            CnResult::InvalidArgument
        );
    }

    #[test]
    fn normalized_args_fills_defaults_and_rejects_bad_types() {
        let ty = ComponentType::parse("ProceduralMesh").unwrap();
        let back = ty
            .normalized_args(&serde_json::json!({ "generator": "box" }))
            .unwrap();
        assert_eq!(back["generator"], "box");
        assert!(back.get("half_width").is_some(), "defaults fill in");
        assert_eq!(
            ty.normalized_args(&serde_json::json!({ "generator": 42 }))
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

    // The structural flags mark the curated sets: the world-config singletons,
    // the render-implying types (which must match the companion pass's
    // GraphicsConfig triggers), and the blank-useful addables. Flag rules: a
    // flagged type must be declarable (singletons and blank-addables are
    // authored), and the two picker sets stay disjoint (a singleton uses the
    // edit-or-add flow, never the plain add).
    #[test]
    fn structural_flags_mark_the_curated_sets() {
        let flagged = |f: fn(ComponentType) -> bool| -> Vec<&'static str> {
            ComponentType::all()
                .iter()
                .copied()
                .filter(|&t| f(t))
                .map(ComponentType::as_str)
                .collect()
        };
        assert_eq!(
            flagged(ComponentType::singleton),
            [
                "Window",
                "GraphicsConfig",
                "PostProcessConfig",
                "StreamingConfig",
                "PhysicsConfig",
                "Application",
                "Variables",
                "LoadingOverlay",
            ]
        );
        assert_eq!(
            flagged(ComponentType::renders),
            [
                "GraphicsConfig",
                "Prop",
                "TextLabel",
                "InstancedProp",
                "VoxelWorld",
                "Sprite",
                "WaterSurface",
                "SdfVolume",
                "LayoutContainer",
                "StatHud",
                "MainMenu",
                "DebugHud",
                "TextInput",
                "LoadingOverlay",
            ]
        );
        for &ty in ComponentType::all() {
            if ty.useful_blank() {
                assert!(
                    ty.addable(),
                    "{} is offered for a plain add but is not External",
                    ty.as_str()
                );
            }
            if ty.singleton() {
                assert!(
                    ty.registration().serializable(),
                    "{} is a singleton but never declarable",
                    ty.as_str()
                );
            }
            assert!(
                !(ty.singleton() && ty.useful_blank()),
                "{} cannot be both a singleton and a plain addable",
                ty.as_str()
            );
        }
    }

    // The two-registry render classifier: exact names, forgiving spellings,
    // resource-registry types, and non-renderers.
    #[test]
    fn type_renders_spans_both_registries() {
        assert!(type_renders("TextLabel"));
        assert!(type_renders("text_label"));
        assert!(type_renders("GraphicsConfig"));
        assert!(type_renders("EnvironmentMap"));
        // A skinned mesh is placed directly and rendered, so its presence
        // renders even without any static Mesh/Prop in the world.
        assert!(type_renders("SkinnedMesh"));
        assert!(type_renders("skinned_mesh"));
        assert!(!type_renders("Window"));
        // A raw Mesh is inert geometry (rendered only through a Prop/Model), so
        // unlike SkinnedMesh it does not by itself render.
        assert!(!type_renders("Mesh"));
        assert!(!type_renders("NotARealType"));
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
