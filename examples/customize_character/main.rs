//! A customizable humanoid: the body the engine's character tooling is built
//! around, shaped by a `CharacterShape` at runtime.
//!
//! `world.rs` beside this file declares the whole scene as typed asset
//! structs. `body` is a `CharacterModel` over `base_humanoid.glb` (the
//! Blender generator under `private/scripts/blender/base_humanoid.py`), which
//! conforms to the bundled `builtin:humanoid` schema: that is what validates
//! it and synthesizes the regional sliders such as `biceps` or `thigh_girth`
//! from the mesh. `body_shape` drives fifteen of those sliders live under
//! the `idle` clip, including the `face` show/hide slider that reveals the
//! facial relief on the otherwise smooth head. Lower levels of detail are
//! decimated at build time from `lod_levels`; walk back with the free-fly
//! camera to see them switch at 6 m and 12 m.
//!
//! `cargo run --example customize_character --features cook`

use concinnity::{App, cook};

mod world;

/// The body this example shapes, beside this source file. Absolute, so the
/// world resolves it whatever directory the example is run from.
pub(crate) const BODY_GLB: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/customize_character/base_humanoid.glb"
);

fn main() {
    let mut spec = cook::world();
    world::declare(&mut spec);
    let world = spec.compile().expect("the declared world compiles");
    App::from_world(world).run().expect("the app runs");
}

#[cfg(test)]
mod tests {
    use super::*;

    // The world names the body by an absolute path built from the manifest
    // root; a wrong root would send the import somewhere silently empty.
    #[test]
    fn the_body_sits_beside_this_source_file() {
        assert!(
            std::path::Path::new(BODY_GLB).is_file(),
            "{BODY_GLB} is wrong"
        );
    }

    // The whole scene compiles: the body imports, the built-in humanoid schema
    // validates it, the sliders synthesize, and every reference the typed
    // structs cannot carry themselves resolves. An unresolved reference fails
    // validation, so a clean compile is the assertion.
    #[test]
    fn the_declared_world_compiles_with_its_references_resolved() {
        let mut spec = cook::world();
        world::declare(&mut spec);
        spec.compile().expect("the declared world compiles");
    }
}
