//! Serde and default coverage driven by the component registry: every authored
//! type, stored or resource, is checked the same way, so a type cannot join
//! the registry without it.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::ecs::{Component, PayloadLocator};

fn json<T: Serialize>(value: &T) -> serde_json::Value {
    serde_json::to_value(value).expect("the value serializes to JSON")
}

// The least an authored type can be written as: `{}` unless it has a key
// with no default.
fn minimal_json(name: &str) -> &'static str {
    match name {
        "Shader" => r#"{"fragment":""}"#,
        "File" => r#"{"path":""}"#,
        _ => "{}",
    }
}

// An authored type reads its own default back from its minimal JSON, and its
// default survives a trip through JSON unchanged.
fn check_authored<T: Default + Serialize + DeserializeOwned>(
    name: &str,
    failures: &mut Vec<String>,
) {
    let default = json(&T::default());
    let minimal = minimal_json(name);
    match serde_json::from_str::<T>(minimal) {
        Ok(bare) if json(&bare) == default => {}
        Ok(_) => failures.push(format!("{name}: `{minimal}` is not the default")),
        Err(e) => failures.push(format!("{name}: `{minimal}` does not parse: {e}")),
    }
    match serde_json::from_value::<T>(default.clone()) {
        Ok(back) if json(&back) == default => {}
        Ok(_) => failures.push(format!("{name}: the default JSON is not stable")),
        Err(e) => failures.push(format!("{name}: its default JSON does not parse: {e}")),
    }
}

// A stored type's default round-trips through its baked form and the
// `Component` hooks, and a frame carrying one byte too many is refused.
fn check_baked<C: Component + Default + Serialize>(failures: &mut Vec<String>) {
    let name = C::NAME;
    let bytes = postcard::to_allocvec(&C::default()).expect("the default bakes");
    match C::from_baked(&bytes) {
        Ok(mut comp) => comp.inject_locator(PayloadLocator {
            blob_index: 0,
            offset: 0,
            len: 0,
        }),
        Err(e) => failures.push(format!("{name}: its baked default does not load: {e:?}")),
    }
    let mut widened = bytes;
    widened.push(0);
    if C::from_baked(&widened).is_ok() {
        failures.push(format!(
            "{name}: a record with an unread trailing byte loads"
        ));
    }
}

macro_rules! check_registry {
    (
        stored: { $( $variant:ident => $ty:path { $($meta:tt)* } ),+ $(,)? },
        resource: { $( $rvariant:ident => $rty:path { $($rmeta:tt)* } ),+ $(,)? } $(,)?
    ) => {
        // Checks every authored entry, returning how many it reached.
        fn check_every_type(failures: &mut Vec<String>) -> usize {
            let mut checked = 0;
            $( checked += check_registry!(@stored failures, $variant, $ty, $($meta)*); )+
            $(
                check_authored::<$rty>(stringify!($rvariant), failures);
                checked += 1;
            )+
            checked
        }
    };
    (@stored $failures:ident, $variant:ident, $ty:path, gen, external $($flags:tt)*) => {{
        check_registry!(@external $failures, $variant, $ty, [$($flags)*]);
        1
    }};
    (@stored $failures:ident, $variant:ident, $ty:path, $($meta:tt)*) => { 0 };
    // A type naming `args:` is authored as that form and compiled into the
    // stored type, which has no default of its own.
    (@external $failures:ident, $variant:ident, $ty:path, [args: $args:ident $($rest:tt)*]) => {
        check_authored::<crate::components::cook::$args>(stringify!($variant), $failures);
    };
    (@external $failures:ident, $variant:ident, $ty:path, [$head:tt $($rest:tt)*]) => {
        check_registry!(@external $failures, $variant, $ty, [$($rest)*]);
    };
    (@external $failures:ident, $variant:ident, $ty:path, []) => {
        check_authored::<$ty>(stringify!($variant), $failures);
        check_baked::<$ty>($failures);
    };
}

crate::for_each_component!(check_registry);

#[test]
fn every_authored_type_reads_its_default_from_nothing_and_round_trips() {
    let mut failures = Vec::new();
    let checked = check_every_type(&mut failures);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    // Every authored entry: a type that drops out of the walk fails here.
    assert_eq!(checked, 67, "the walk reached {checked} authored types");
}
