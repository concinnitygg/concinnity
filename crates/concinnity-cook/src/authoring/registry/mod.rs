//! The authoring metadata registry: `RegisteredType`, the enum of every asset
//! type paired with its on-disk discriminant, plus the authoring-only operations
//! over it (name parsing, arg reserialization, enum-field probing, and
//! reference-field listing). Consumed by the build pipeline and the in-engine
//! editor; never by the runtime ECS, which loads components straight from their
//! blob discriminants (`concinnity_core::ecs::ComponentAsset::from_baked`).
//!
//! The vocabulary arrives in three groups. The two the runtime can reach -- the
//! components a world stores and the resources the cook compiles into the blob
//! -- are the single source of truth in `concinnity_core::ecs::registry` (the
//! `for_each_component!` macro). The third, the authoring-only types the cook
//! expands away, is this crate's own: it lives in [`build_only`], and
//! `for_each_authored_type!` composes the two so `RegisteredType` spans the
//! whole declarable vocabulary from one enum.

pub mod build_only;

use crate::result::CnResult;
use concinnity_core::platform::Platform;

pub use build_only::BuildOnlyAsset;
pub use concinnity_core::ecs::{AssetOrigin, AssetPayload};

/// Static authoring metadata for an asset type: how it is declared, whether it
/// compiles a payload, and its default args JSON. Derived from the registry
/// entry's metadata block -- the runtime `Component` trait carries none of it
/// (blob records carry everything a shipped game loads).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Registration {
    /// The asset type's registry name.
    pub type_name: &'static str,
    /// Where the asset comes from and whether it persists.
    pub origin: AssetOrigin,
    /// Whether the asset compiles a binary payload.
    pub payload: AssetPayload,
    /// Default args JSON, for types that declare one.
    pub default_args: Option<serde_json::Value>,
}

impl Registration {
    /// Whether a world may declare this type directly.
    pub fn addable(&self) -> bool {
        self.origin == AssetOrigin::External
    }

    /// Whether the build must compile a payload for this type.
    pub(crate) fn needs_compilation(&self) -> bool {
        self.payload == AssetPayload::Compiled
    }
}

/// What a component name denotes at tick time, for the behavior `scope` and
/// `queries` checks.
///
/// A world may declare far more types than a running world holds in a column:
/// the build expands some away, compiles others into the resource stream, and a
/// load-time pass drains the rest during `World::start`. Only [`Self::Column`]
/// can be matched against entities once the world is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeResolution {
    /// A column that still holds entities at tick time. Carries the type the
    /// name resolves to, which differs from the name itself where a load-time
    /// pass leaves a runtime marker behind (`Prop` resolves to `PropInstance`).
    Column(RegisteredType),
    /// A load-time pass drains this column during `World::start`, leaving
    /// nothing to match.
    Consumed,
    /// The build expands this type into the components it stands for; no
    /// record of it reaches a world.
    Expanded,
    /// Compiled into the blob's resource stream and reached by handle rather
    /// than stored in a column.
    Resource,
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
pub(crate) struct NoArgs {}

// Metadata scanners over a registry entry's `{ ... }` flag tokens. Each walks
// the token stream for its key and falls back to a default when absent; the
// generic `$t:tt` arm skips unrecognized tokens (other flags, `,`, `:`, and
// bracketed lists are each one token tree).

// The authoring origin of a stored entry: `external` / `runtime` (RuntimeOnly is
// also the fallback for entries with no origin flag). The other two groups take
// theirs from the group itself, in `__group_origin`.
macro_rules! __meta_origin {
    () => { AssetOrigin::RuntimeOnly };
    (external $($r:tt)*) => { AssetOrigin::External };
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
// runtime-only entries, or the authoring form of the asset the `args: <Asset>`
// override names, for the types whose authored shape diverges from the runtime
// component.
macro_rules! __meta_args_ty {
    ($default:path;) => { $default };
    ($default:path; runtime $($r:tt)*) => { NoArgs };
    ($default:path; args: $a:ident $($r:tt)*) => { concinnity_core::components::cook::$a };
    ($default:path; $t:tt $($r:tt)*) => { __meta_args_ty!($default; $($r)*) };
}

// The args schema's NAME, for the docs pipeline (which renders the args
// struct's fields, keyed by the struct's own name). A divergent asset's schema
// is declared as `<Asset>Args` and exposed under the asset's name in `cook`, so
// the entry's `args: <Asset>` yields both.
macro_rules! __meta_args_name {
    ($default:ident;) => { stringify!($default) };
    ($default:ident; args: $a:ident $($r:tt)*) => { concat!(stringify!($a), "Args") };
    ($default:ident; $t:tt $($r:tt)*) => { __meta_args_name!($default; $($r)*) };
}

// Apply the entry's bake-time validator (`validate: <fn>`, from
// `crate::authoring::validate`) to a typed value; identity when the entry
// declares none. A `validate_for: <fn>` entry takes the cooked shader platform
// alongside the value.
macro_rules! __meta_validate {
    ($val:expr, $p:expr;) => { $val };
    ($val:expr, $p:expr; validate: $f:ident $($r:tt)*) => {
        crate::authoring::validate::$f($val)
    };
    ($val:expr, $p:expr; validate_for: $f:ident $($r:tt)*) => {
        crate::authoring::validate::$f($val, $p)
    };
    ($val:expr, $p:expr; $t:tt $($r:tt)*) => { __meta_validate!($val, $p; $($r)*) };
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
    ($t:tt $($r:tt)*) => { crate::authoring::registry::__meta_singleton!($($r)*) };
}
macro_rules! __meta_useful_blank {
    () => { false };
    (useful_blank $($r:tt)*) => { true };
    ($t:tt $($r:tt)*) => { crate::authoring::registry::__meta_useful_blank!($($r)*) };
}
macro_rules! __meta_renders {
    () => { false };
    (renders $($r:tt)*) => { true };
    ($t:tt $($r:tt)*) => { crate::authoring::registry::__meta_renders!($($r)*) };
}
macro_rules! __meta_live {
    () => { false };
    (live $($r:tt)*) => { true };
    ($t:tt $($r:tt)*) => { crate::authoring::registry::__meta_live!($($r)*) };
}
pub(crate) use {__meta_live, __meta_renders, __meta_singleton, __meta_useful_blank};

// The `consumed` flag: whether a load-time pass drains this column during
// `World::start`, and the runtime type that survives in its place when one
// does. The `consumed: <Type>` arm must precede the bare one, which would
// otherwise absorb the substitute.
macro_rules! __meta_surviving {
    ($variant:ident;) => { ScopeResolution::Column(RegisteredType::$variant) };
    ($variant:ident; consumed: $surviving:ident $($r:tt)*) => {
        ScopeResolution::Column(RegisteredType::$surviving)
    };
    ($variant:ident; consumed $($r:tt)*) => { ScopeResolution::Consumed };
    ($variant:ident; $t:tt $($r:tt)*) => { crate::authoring::registry::__meta_surviving!($variant; $($r)*) };
}
pub(crate) use __meta_surviving;

// Keyed on group: only a stored entry can name a column, so the other two
// groups answer from their group alone.
macro_rules! __group_surviving {
    (stored; $variant:ident; $($meta:tt)*) => { __meta_surviving!($variant; $($meta)*) };
    (build_only; $variant:ident; $($meta:tt)*) => { ScopeResolution::Expanded };
    (resource; $variant:ident; $($meta:tt)*) => { ScopeResolution::Resource };
}

// The three below are keyed on which group of the registry list an entry came
// from rather than on its flags, because group membership is what decides them.

// The blob tag: the stored group's `ComponentTag` position. A build-only type is
// expanded away before any record is written, and a resource is addressed by a
// handle into the resource stream, so neither carries a component tag.
macro_rules! __group_discriminant {
    (stored; $variant:ident) => {
        Some(crate::ecs::ComponentTag::$variant as u8)
    };
    (build_only; $variant:ident) => {
        None
    };
    (resource; $variant:ident) => {
        None
    };
}

// The authoring origin: read from the entry's flags for a stored type (which is
// `external` or `runtime`), fixed by the group for the other two.
macro_rules! __group_origin {
    (stored; $($meta:tt)*) => { __meta_origin!($($meta)*) };
    (build_only; $($meta:tt)*) => { AssetOrigin::BuildOnly };
    (resource; $($meta:tt)*) => { AssetOrigin::External };
}

// Whether the build compiles something for this type. A resource always does,
// including the `data` ones, whose compiled bytes ride inline in the record
// rather than in a payload section it points at.
macro_rules! __group_payload {
    (stored; $($meta:tt)*) => { __meta_payload!($($meta)*) };
    (build_only; $($meta:tt)*) => { __meta_payload!($($meta)*) };
    (resource; $($meta:tt)*) => { AssetPayload::Compiled };
}

// The dense per-kind handle space a resource is assigned into, from its
// `resource: <ResourceKind>` flag; `None` for anything outside that group.
macro_rules! __meta_resource_kind {
    () => { None };
    (resource: $kind:ident $($r:tt)*) => { Some(crate::ecs::ResourceKind::$kind) };
    ($t:tt $($r:tt)*) => { __meta_resource_kind!($($r)*) };
}

// Whether a resource's compiled bytes ride inline in its record (`data`) rather
// than in a payload section the record points at.
macro_rules! __meta_is_data {
    () => { false };
    (data $($r:tt)*) => { true };
    ($t:tt $($r:tt)*) => { __meta_is_data!($($r)*) };
}

// Hand a callback the whole declarable vocabulary: concinnity-core's `stored`
// and `resource` groups plus the `build_only` group this crate owns, in one
// invocation shaped like a single list.
//
// Core's list is the outer one (it cannot name the group this crate holds), so
// the composition goes through an adapter: core passes its groups to
// `__append_build_only`, which forwards them as the prefix of
// `for_each_build_only_type!`, which appends its own group and calls the real
// callback.
macro_rules! __append_build_only {
    ($cb:ident; $($core_groups:tt)*) => {
        $crate::for_each_build_only_type!($cb, $($core_groups)*);
    };
}

macro_rules! for_each_authored_type {
    ($cb:ident) => {
        concinnity_core::for_each_component!(__append_build_only; $cb;);
    };
}

// Generate `RegisteredType` and its authoring methods from the composed
// vocabulary. Invoked once, below, via `for_each_authored_type!`. All authoring
// metadata (origin, payload, args schema, validators, reference fields) derives
// from each entry's `{ ... }` metadata block; the runtime `Component` trait
// carries none of it.
macro_rules! define_registered_type {
    // Every registered type is here, whichever group it came from: one registry
    // means one `parse`, so a caller asking "what type is this?" cannot miss a
    // category. The groups are merged into one list tagged by group, so every
    // method below stays a single repetition; what group membership decides
    // reaches them through the `__group_*!` helpers.
    (
        stored: { $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? },
        resource: { $( $rvariant:ident => $rty:path { $($rmeta:tt)* } ),+ $(,)? },
        build_only: { $( $bvariant:ident => $bty:path { $($bmeta:tt)* } ),+ $(,)? } $(,)?
    ) => {
        define_registered_type!(@all
            $( $variant => $ty { $($meta)* } [stored] ),+ ,
            $( $bvariant => $bty { $($bmeta)* } [build_only] ),+ ,
            $( $rvariant => $rty { $($rmeta)* } [resource] ),+
        );
    };

    (@all $( $variant:ident => $ty:path { $($meta:tt)* } [$group:ident] ),+ $(,)? ) => {
        // One variant per registered component type, named for that type.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[expect(missing_docs, reason = "one variant per registered component type, named for that type")]
        pub enum RegisteredType {
            $( $variant ),+
        }

        impl RegisteredType {
            /// The type's registry name.
            pub fn as_str(self) -> &'static str {
                match self { $( Self::$variant => stringify!($variant) ),+ }
            }
            /// The on-disk blob tag / in-memory `ComponentId`, derived from the
            /// shared `ComponentTag` enum (list position) so it matches the
            /// runtime loader exactly.
            ///
            /// `None` for a build-only type: the cook expands it into the
            /// components it stands for, so no record of it is ever written and
            /// it has no tag to carry.
            pub fn discriminant(self) -> Option<u8> {
                match self {
                    $( Self::$variant => __group_discriminant!($group; $variant) ),+
                }
            }
            /// The type carrying a blob discriminant, or `None` if unknown. Only
            /// a stored type has one.
            pub(crate) fn from_discriminant(val: u8) -> Option<Self> {
                $(
                    if __group_discriminant!($group; $variant) == Some(val) {
                        return Some(Self::$variant);
                    }
                )+
                None
            }
            /// What this type denotes at tick time: the column a behavior can
            /// scope or query against, or why there is none.
            ///
            /// Resolves through the substitute where a load-time pass leaves a
            /// runtime marker behind, so `Prop` answers
            /// `Column(RegisteredType::PropInstance)`.
            pub fn scope_resolution(self) -> ScopeResolution {
                match self {
                    $( Self::$variant => __group_surviving!($group; $variant; $($meta)*) ),+
                }
            }

            /// A name either matches a known type or it does not; callers that
            /// want a message supply their own via `ok_or`/`ok_or_else`.
            pub fn parse(s: &str) -> Option<Self> {
                $(
                    if s == stringify!($variant) { return Some(Self::$variant); }
                )+
                None
            }
            /// The name of this type's authored args schema struct: the
            /// component itself for pass-through types, the `args:` override
            /// for the divergent ones. The docs pipeline renders that struct's
            /// fields as the asset's parameters.
            pub fn args_struct_name(self) -> &'static str {
                match self {
                    $( Self::$variant => __meta_args_name!($variant; $($meta)*) ),+
                }
            }
            /// This type's static authoring metadata.
            pub fn registration(self) -> Registration {
                match self {
                    $(
                        Self::$variant => Registration {
                            type_name: stringify!($variant),
                            origin: __group_origin!($group; $($meta)*),
                            payload: __group_payload!($group; $($meta)*),
                            default_args: serde_json::to_value(
                                <__meta_args_ty!($ty; $($meta)*) as Default>::default(),
                            )
                            .ok(),
                        }
                    ),+
                }
            }
            /// Bake a JSON args value into the blob record's component bytes:
            /// deserialize through the typed args schema (interning name-string
            /// cross-references), apply the type's bake-time validator, and
            /// serialize the runtime component as postcard. For a pass-through
            /// type the args ARE the component; a divergent type (`args:`
            /// metadata) routes through its `bake` translation in
            /// `bake_divergent`.
            pub fn reserialize_args(
                self,
                args: &serde_json::Value,
                platform: Platform,
            ) -> Result<Vec<u8>, CnResult> {
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
                            Ok(postcard::to_allocvec(
                                &__meta_validate!(typed, platform; $($meta)*),
                            )?)
                        }
                    ),+
                }
            }
            /// Normalize a JSON args value through the typed args schema: the
            /// same deserialize + validate as `reserialize_args`, but back to
            /// JSON with defaults filled and references resolved. Authoring
            /// tools (`cn add`, the editor form) write this into world.jsonl;
            /// the baked postcard bytes cannot round-trip to JSON.
            pub fn normalized_args(
                self,
                args: &serde_json::Value,
                platform: Platform,
            ) -> Result<serde_json::Value, CnResult> {
                crate::ecs::asset_id::ensure_name_resolver();
                match self {
                    $(
                        Self::$variant => {
                            let typed = serde_json::from_value::<__meta_args_ty!($ty; $($meta)*)>(
                                args.clone(),
                            )
                            .map_err(json_args_err)?;
                            serde_json::to_value(&__meta_validate!(typed, platform; $($meta)*))
                                .map_err(json_args_err)
                        }
                    ),+
                }
            }
            /// The allowed values of a string-enum args field (in declaration
            /// order), or `None` if `field` is a free-form string / absent / not a
            /// string-enum. Probes the typed args by deserializing the defaults
            /// with `field` set to a sentinel: a string-enum yields serde's
            /// "unknown variant ..., expected ..." which `parse_expected_variants`
            /// reads; a free string accepts the sentinel and yields `None`. Used by
            /// authoring tools to offer a picker instead of a free text box; it
            /// degrades to `None` (free text) if serde's phrasing ever changes.
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
            /// The dense per-kind handle space this asset is assigned into, or
            /// `None` if it is not a resource asset. Cook assigns the handle;
            /// the runtime addresses the resource by it.
            pub(crate) fn resource_kind(self) -> Option<crate::ecs::ResourceKind> {
                match self {
                    $( Self::$variant => __meta_resource_kind!($($meta)*) ),+
                }
            }
            /// Whether this type is a resource asset: compiled into the blob's
            /// resource stream and reached by a handle, rather than stored in a
            /// component column.
            pub fn is_resource(self) -> bool {
                self.resource_kind().is_some()
            }
            /// Whether this resource's compiled bytes ride inline in its record
            /// rather than in a payload section the record points at. False for
            /// everything that is not a resource asset.
            pub(crate) fn is_data(self) -> bool {
                match self {
                    $( Self::$variant => __meta_is_data!($($meta)*) ),+
                }
            }
            /// The asset-reference fields of this type, as (field, target_type),
            /// from the entry's `refs:` metadata.
            pub fn ref_fields(self) -> &'static [(&'static str, &'static str)] {
                match self {
                    $(
                        Self::$variant => __meta_refs!($($meta)*)
                    ),+
                }
            }
            /// The structural flags, from the entry's metadata: `singleton`
            /// (at most one instance belongs to a world; authoring tools use an
            /// edit-or-add flow), `useful_blank` (meaningful when declared with
            /// only default args, so authoring tools offer a plain add), and
            /// `renders` (presence implies the world renders; drives the
            /// GraphicsConfig companion injection at build time).
            pub fn singleton(self) -> bool {
                match self {
                    $( Self::$variant => __meta_singleton!($($meta)*) ),+
                }
            }
            /// Whether declaring the type with no args still does something useful.
            pub fn useful_blank(self) -> bool {
                match self {
                    $( Self::$variant => __meta_useful_blank!($($meta)*) ),+
                }
            }
            /// Whether declaring the type implies the world renders.
            pub fn renders(self) -> bool {
                match self {
                    $( Self::$variant => __meta_renders!($($meta)*) ),+
                }
            }
            /// Whether the running world re-reads this type's column every
            /// frame. An editing tool holding a live world can overwrite such
            /// a component in place and see the change on the next draw,
            /// instead of reloading the world to apply it.
            ///
            /// Flagging a type asserts two things: its column still holds
            /// entities at tick time and some system reads them afresh, AND no
            /// build-time expansion reads its args -- an in-place write never
            /// runs the expansion, so a type another asset is generated from
            /// would leave that generated asset standing on the old values.
            ///
            /// An expansion is not the only thing that reads args at build
            /// time; the reference graph that decides how payloads pack and the
            /// cross-asset validator do too. A type carrying one of those can
            /// still be flagged, so long as the writer declines the edits that
            /// would move it.
            pub fn live(self) -> bool {
                match self {
                    $( Self::$variant => __meta_live!($($meta)*) ),+
                }
            }
            /// Whether a world may declare this type directly.
            pub fn addable(self) -> bool {
                self.registration().addable()
            }
            /// Every registered component type, in list order.
            pub fn all() -> &'static [RegisteredType] {
                &[ $( Self::$variant ),+ ]
            }
            /// Every type a world may declare, with its registration metadata.
            pub fn addable_types() -> impl Iterator<Item = (RegisteredType, Registration)> {
                Self::all()
                    .iter()
                    .map(|t| (*t, t.registration()))
                    .filter(|(_, reg)| reg.addable())
            }
        }
    };
}

for_each_authored_type!(define_registered_type);

/// The authored-value trait: the bridge from a typed authoring struct to the
/// world line that declares it. Implemented for every declarable asset's args
/// schema (the `args:` override where the authored form diverges from the
/// component, the component itself where it does not) and for every resource
/// asset, so a caller hands the cook a typed value instead of a name/type/args
/// triple assembled by hand.
pub trait Authored: serde::Serialize {
    /// The registered asset type, as it appears in a world line's `type`.
    const TYPE: &'static str;
}

// Runtime-only entries are never authored, so they get no impl; every other
// entry (including the `manual` ones, whose hand-written `Component` impl is
// unrelated to authoring) resolves its authored type through `__meta_args_ty`.
macro_rules! __authored_component {
    (
        stored: { $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? },
        resource: { $( $rvariant:ident => $rty:path { $($rmeta:tt)* } ),+ $(,)? },
        build_only: { $( $bvariant:ident => $bty:path { $($bmeta:tt)* } ),+ $(,)? } $(,)?
    ) => {
        $( __authored_component!(@one $variant $ty { $($meta)* }); )+
        $( __authored_component!(@one $rvariant $rty { $($rmeta)* }); )+
        $( __authored_component!(@one $bvariant $bty { $($bmeta)* }); )+
    };
    (@one $variant:ident $ty:path { runtime $($rest:tt)* }) => {};
    (@one $variant:ident $ty:path { $($meta:tt)* }) => {
        impl Authored for __meta_args_ty!($ty; $($meta)*) {
            const TYPE: &'static str = stringify!($variant);
        }
    };
}

for_each_authored_type!(__authored_component);

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

/// Write a reference into an already-serialized asset line: `field` is set to
/// the asset name `target`, the form the compile resolves to a handle. A
/// typed authored value cannot carry the name itself (a reference field holds
/// the resolved handle), so a builder names it after the fact.
pub fn set_reference(line: &str, field: &str, target: &str) -> std::io::Result<String> {
    let bad = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);
    let mut value: serde_json::Value =
        serde_json::from_str(line).map_err(|e| bad(format!("asset line: {e}")))?;
    value
        .get_mut("args")
        .and_then(|args| args.as_object_mut())
        .ok_or_else(|| bad(format!("asset line has no args to hold '{field}'")))?
        .insert(
            field.to_string(),
            serde_json::Value::String(target.to_string()),
        );
    let mut out = serde_json::to_string(&value).map_err(|e| bad(e.to_string()))?;
    out.push('\n');
    Ok(out)
}

#[cfg(test)]
mod authored_tests {
    use super::*;

    #[test]
    fn set_reference_names_a_field_the_typed_value_cannot_carry() {
        let line = asset_line("hero_shape", &crate::components::CharacterShape::default())
            .expect("serializes");
        let patched = set_reference(&line, "target", "hero").expect("patched");
        assert!(patched.ends_with('\n'));
        let value: serde_json::Value = serde_json::from_str(&patched).expect("parses");
        assert_eq!(value["args"]["target"], "hero");
        assert_eq!(value["name"], "hero_shape");
        assert_eq!(value["type"], "CharacterShape");
        // A line that is not an asset declaration is refused, not mangled.
        let err = set_reference("7", "target", "hero").expect_err("not an asset line");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let err = set_reference(r#"{"name":"x"}"#, "target", "hero").expect_err("no args");
        assert!(err.to_string().contains("no args"), "{err}");
    }

    // The three shapes the generation has to cover: an args-schema override, a
    // pass-through component, and a resource asset.
    #[test]
    fn authored_types_report_their_registered_name() {
        assert_eq!(
            <concinnity_core::components::cook::Room as Authored>::TYPE,
            "Room"
        );
        assert_eq!(
            <crate::components::DirectionalLight as Authored>::TYPE,
            "DirectionalLight"
        );
        assert_eq!(<crate::components::Texture as Authored>::TYPE, "Texture");
        assert_eq!(
            <crate::components::EnvironmentMap as Authored>::TYPE,
            "EnvironmentMap"
        );
    }

    // Every Authored type names a type the registry can actually parse, so a
    // value handed to the cook always lands on a declarable asset.
    #[test]
    fn the_reported_name_round_trips_through_the_registry() {
        for name in [
            <concinnity_core::components::cook::Room as Authored>::TYPE,
            <concinnity_core::components::cook::Camera3D as Authored>::TYPE,
            <crate::components::DirectionalLight as Authored>::TYPE,
        ] {
            assert!(
                RegisteredType::parse(name).is_some(),
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

/// Bake the runtime component for the asset types whose baked form diverges
/// from their authored args (the entries with `args:` metadata): run the type's
/// `bake` translation at build time and serialize the component itself, which
/// the type's `from_baked` deserializes at load. Pass-through types return
/// `None`; cook reserializes their args (which ARE the component).
pub fn bake_divergent(
    ct: RegisteredType,
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
        RegisteredType::Camera3D => {
            bake!(
                crate::components::Camera3D,
                concinnity_core::components::cook::Camera3D
            )
        }
        RegisteredType::Room => bake!(
            crate::components::Room,
            concinnity_core::components::cook::Room
        ),
        RegisteredType::File => bake!(
            crate::components::File,
            concinnity_core::components::cook::File
        ),
        RegisteredType::Spawner => {
            bake!(
                crate::components::Spawner,
                concinnity_core::components::cook::Spawner
            )
        }
        RegisteredType::AppConfig => {
            bake!(
                crate::components::AppConfig,
                concinnity_core::components::cook::AppConfig
            )
        }
        _ => Ok(None),
    }
}

/// Whether an asset type's presence implies the world renders: the registry's
/// `renders` flag, across both the component and resource registries. Matches
/// by normalized name (case-insensitive, underscores stripped) so cook's
/// companion pass and authoring tools classify the same way.
pub fn type_renders(asset_type: &str) -> bool {
    let norm: String = asset_type.chars().filter(|c| *c != '_').collect();
    let matches = |name: &str| name.eq_ignore_ascii_case(&norm);
    RegisteredType::all()
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
        assert!(external.needs_compilation());

        let runtime = reg(AssetOrigin::RuntimeOnly, AssetPayload::None);
        assert!(!runtime.addable());
        assert!(!runtime.needs_compilation());

        let build = reg(AssetOrigin::BuildOnly, AssetPayload::None);
        assert!(!build.addable());
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
        let bytes = bake_divergent(RegisteredType::Room, &args)
            .unwrap()
            .expect("Room bakes divergently");
        let baked = crate::components::Room::from_baked(&bytes).unwrap();
        // The size shorthand resolved at bake time.
        assert_eq!(baked.half_width, 8.0);
        assert_eq!(baked.half_depth, 10.0);
        assert_eq!(baked.ceiling_height, 3.5);

        let args = serde_json::json!({"position": [1.0, 2.0, 3.0], "yaw": 0.5});
        let bytes = bake_divergent(RegisteredType::Camera3D, &args)
            .unwrap()
            .expect("Camera3D bakes divergently");
        let baked = crate::components::Camera3D::from_baked(&bytes).unwrap();
        assert_eq!(baked.position, [1.0, 2.0, 3.0]);
        // The view matrix composed at bake time.
        let expected = crate::components::Camera3D::bake(
            serde_json::from_value(serde_json::json!({"position": [1.0, 2.0, 3.0], "yaw": 0.5}))
                .unwrap(),
        );
        assert_eq!(baked.view_matrix, expected.view_matrix);

        let args = serde_json::json!({"path": "tri.obj"});
        let bytes = bake_divergent(RegisteredType::File, &args)
            .unwrap()
            .expect("File bakes divergently");
        let baked = crate::components::File::from_baked(&bytes).unwrap();
        // The kind derived from the extension at bake time.
        assert!(baked.kind.is_some());

        asset_id::reset_interner();
        let args = serde_json::json!({"template": "crate", "interval": -1.0, "lifetime": 2.0});
        let bytes = bake_divergent(RegisteredType::Spawner, &args)
            .unwrap()
            .expect("Spawner bakes divergently");
        let baked = crate::components::Spawner::from_baked(&bytes).unwrap();
        // The interval clamped and the runtime counters zeroed at bake time.
        assert_eq!(baked.interval, 0.0);
        assert_eq!(baked.elapsed, 0.0);
        assert_eq!(baked.count, 0);

        // A pass-through type does not bake divergently.
        assert!(
            bake_divergent(RegisteredType::PointLight, &serde_json::json!({}))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn component_types_round_trip_name_and_discriminant() {
        for &ty in RegisteredType::all() {
            assert_eq!(RegisteredType::parse(ty.as_str()), Some(ty));
            // Only a stored type carries a tag, so only those round trip
            // through one. That the rest have none is the assertion for them.
            match ty.discriminant() {
                Some(d) => assert_eq!(RegisteredType::from_discriminant(d), Some(ty)),
                None => assert!(
                    ty.registration().origin == AssetOrigin::BuildOnly || ty.is_resource(),
                    "{} has no discriminant but is neither build-only nor a resource",
                    ty.as_str()
                ),
            }
        }
        assert_eq!(RegisteredType::parse("NotARealComponent"), None);
        assert_eq!(RegisteredType::from_discriminant(255), None);
    }

    // Exactly one group has a column. A build-only type is expanded away by the
    // cook and a resource compiles into the resource stream, so neither may be
    // handed a component tag; the facts must not drift apart.
    #[test]
    fn only_stored_types_carry_a_discriminant() {
        for &ty in RegisteredType::all() {
            let stored = !(ty.registration().origin == AssetOrigin::BuildOnly || ty.is_resource());
            assert_eq!(
                ty.discriminant().is_some(),
                stored,
                "{} disagrees about whether it is stored in a column",
                ty.as_str()
            );
        }
    }

    // On-disk discriminants must stay unique and inside the component range; the
    // only iterator over the full list is `RegisteredType::all`, so the invariant
    // is checked here even though the discriminants are a runtime/blob concern.
    #[test]
    fn component_discriminants_are_unique_and_in_range() {
        let mut seen = std::collections::HashSet::new();
        for &ty in RegisteredType::all() {
            let Some(d) = ty.discriminant() else { continue };
            assert!(
                d < 128,
                "{} discriminant {} outside the component range",
                ty.as_str(),
                d
            );
            assert!(seen.insert(d), "duplicate discriminant {d}");
        }
    }

    // The docs pipeline renders an asset's parameters from the fields of the
    // struct this names, looked up by the struct's own name in the extracted
    // schema. A divergent asset's registry entry names the asset (`args: Room`)
    // and its schema is declared as `RoomArgs`, so the two are
    // bridged by that naming convention; a rename on either side that broke it
    // would silently render an empty parameter table.
    #[test]
    fn a_divergent_asset_names_the_schema_struct_the_docs_render() {
        assert_eq!(RegisteredType::Room.args_struct_name(), "RoomArgs");
        assert_eq!(RegisteredType::Camera3D.args_struct_name(), "Camera3DArgs");
        assert_eq!(RegisteredType::File.args_struct_name(), "FileArgs");
        assert_eq!(RegisteredType::Spawner.args_struct_name(), "SpawnerArgs");
        assert_eq!(
            RegisteredType::AppConfig.args_struct_name(),
            "AppConfigArgs"
        );
        // A pass-through asset's schema is the asset itself, whichever group it
        // is in.
        assert_eq!(RegisteredType::PointLight.args_struct_name(), "PointLight");
        assert_eq!(RegisteredType::Prefab.args_struct_name(), "Prefab");
        assert_eq!(RegisteredType::Texture.args_struct_name(), "Texture");
    }

    #[test]
    fn reserialize_args_round_trips_and_rejects_bad_types() {
        let ty = RegisteredType::parse("ProceduralMesh").unwrap();
        let bytes = ty
            .reserialize_args(&serde_json::json!({ "source": "a.glb" }), Platform::Metal)
            .unwrap();
        let back: crate::components::ProceduralMesh = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.source.as_deref(), Some("a.glb"));
        assert_eq!(
            ty.reserialize_args(&serde_json::json!({ "source": 42 }), Platform::Metal)
                .unwrap_err(),
            CnResult::InvalidArgument
        );
    }

    #[test]
    fn normalized_args_fills_defaults_and_rejects_bad_types() {
        let ty = RegisteredType::parse("ProceduralMesh").unwrap();
        let back = ty
            .normalized_args(&serde_json::json!({ "generator": "box" }), Platform::Metal)
            .unwrap();
        assert_eq!(back["generator"], "box");
        assert!(back.get("half_width").is_some(), "defaults fill in");
        assert_eq!(
            ty.normalized_args(&serde_json::json!({ "generator": 42 }), Platform::Metal)
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
        for &ty in RegisteredType::all() {
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
            RegisteredType::Sprite.field_enum_variants("fit"),
            Some(vec!["fit".into(), "cover".into(), "bottom".into()])
        );
        assert_eq!(
            RegisteredType::TextLabel.field_enum_variants("align"),
            Some(vec!["left".into(), "center".into(), "right".into()])
        );
        // A two-variant enum uses serde's "expected `a` or `b`" phrasing.
        assert_eq!(
            RegisteredType::AudioCue.field_enum_variants("kind"),
            Some(vec!["music".into(), "sound".into()])
        );
        // A free-form string field is not an enum.
        assert_eq!(
            RegisteredType::HitRegion.field_enum_variants("action"),
            None
        );
        assert_eq!(RegisteredType::KeyBinding.field_enum_variants("key"), None);
        // An absent field yields None (not a panic).
        assert_eq!(RegisteredType::Sprite.field_enum_variants("nope"), None);
        // A non-string field (probing it errors on type, not "unknown variant").
        assert_eq!(RegisteredType::Sprite.field_enum_variants("x"), None);
    }

    // `ref_fields` reports each type's asset-reference fields and their targets;
    // every referenced target must itself be a real component type.
    #[test]
    fn ref_fields_name_real_target_types() {
        assert_eq!(
            RegisteredType::Decal.ref_fields(),
            &[("texture", "Texture")]
        );
        assert_eq!(
            RegisteredType::AudioEmitter.ref_fields(),
            &[("clip", "AudioClip"), ("prop", "Prop")]
        );
        // A type without references reports none.
        assert!(RegisteredType::PointLight.ref_fields().is_empty());
        // Every declared ref field names an existing arg key and a real target
        // type -- either a component or a resource-only asset (e.g. AudioClip,
        // which has left the component registry).
        for &ty in RegisteredType::all() {
            let default_args = ty.registration().default_args;
            for &(field, target) in ty.ref_fields() {
                assert!(
                    RegisteredType::parse(target).is_some(),
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
        let flagged = |f: fn(RegisteredType) -> bool| -> Vec<&'static str> {
            RegisteredType::all()
                .iter()
                .copied()
                .filter(|&t| f(t))
                .map(RegisteredType::as_str)
                .collect()
        };
        assert_eq!(
            flagged(RegisteredType::singleton),
            [
                "Window",
                "GraphicsConfig",
                "PostProcessConfig",
                "StreamingConfig",
                "PhysicsConfig",
                "AppConfig",
                "Variables",
                "LoadingOverlay",
                "EngineDefaults",
            ]
        );
        assert_eq!(
            flagged(RegisteredType::renders),
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
                "DebugHud",
                "TextInput",
                "LoadingOverlay",
                // The build-only group sorts after the stored one, and the
                // resource group after that.
                "MainMenu",
                "EnvironmentMap",
                "SkinnedMesh",
            ]
        );
        for &ty in RegisteredType::all() {
            if ty.useful_blank() {
                assert!(
                    ty.addable(),
                    "{} is offered for a plain add but is not External",
                    ty.as_str()
                );
            }
            if ty.singleton() {
                assert!(
                    ty.registration().origin != AssetOrigin::RuntimeOnly,
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
            RegisteredType::Transform,
            RegisteredType::MeshRenderer,
            RegisteredType::ModelRenderer,
            RegisteredType::Collider,
            RegisteredType::Interactable,
            RegisteredType::Pickup,
            RegisteredType::Parent,
            RegisteredType::Children,
            RegisteredType::SceneMember,
            RegisteredType::GlobalTransform,
            RegisteredType::RenderHandle,
            RegisteredType::Held,
        ] {
            assert!(
                !ty.registration().addable(),
                "{} must be RuntimeOnly (not declarable)",
                ty.as_str()
            );
        }
    }
}
