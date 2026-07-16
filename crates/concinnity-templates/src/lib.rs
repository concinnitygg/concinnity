// concinnity-templates
//
// Engine-owned content templates, expressed as typed data rather than JSON strings.
// Two namespaces:
//
//   * `asset`  -- reusable element constructors (Sprite, TextLabel, TextInput,
//                 Panel, HitRegion, Screen, and the scene lighting / atmosphere
//                 pieces). Each builder returns an `AssetSpec`: a `{name, type,
//                 args}` world-line entry as data.
//   * `world`  -- named bundles (`cn add --template <name>`) composed from the
//                 asset builders.
//
// A consumer turns a spec into the engine's `serde_json::Value` (or a typed
// component) without parsing any JSON. The crate is `#![no_std]` (using only
// `core` + `alloc`) with zero external dependencies, so it can never drift into
// engine logic and is consumable from std and no_std alike.
//
// To add an asset builder: add a function to the matching `asset` submodule that
// sets only the fields it changes (every other field takes the type's serde
// default). To add a world template: add a module under `world` returning its
// `Vec<AssetSpec>` and list it in `world::TEMPLATES`.

#![no_std]

extern crate alloc;

// The test harness links std; name it so `#[cfg(test)]` modules can use
// `std`-pathed helpers. The library target itself pulls in nothing beyond
// `core` + `alloc`.
#[cfg(test)]
extern crate std;

pub mod asset;
pub mod ui;
pub mod world;

mod spec;

pub use spec::{ArgValue, AssetSpec};
pub use world::{TEMPLATES, WorldTemplate, by_name};
