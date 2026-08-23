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

use std::io;

use concinnity::{App, cook, paths};

mod world;

concinnity::install_global_allocator!();

/// The body this example shapes, beside this source file. Absolute, so the
/// world resolves it whatever directory the example is run from.
pub(crate) const BODY_GLB: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/customize_character/base_humanoid.glb"
);

fn main() -> io::Result<()> {
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf))
    {
        paths::set_root(dir);
    }

    let mut spec = cook::world();
    world::declare(&mut spec);
    App::from_world(spec.compile()?).run()
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

    // The whole scene compiles, and every reference the typed structs cannot
    // carry themselves resolved: a shape whose target went unresolved would
    // still compile and simply leave the body unshaped.
    #[test]
    fn the_declared_world_compiles_with_its_references_resolved() {
        use concinnity::assets::{Animation, CharacterShape, Prop};

        // Keep the compile's payload cache out of the source tree, the same
        // way `main` does for a run.
        paths::set_root(std::env::temp_dir().join("concinnity-customize-character-test"));

        let mut spec = cook::world();
        world::declare(&mut spec);
        let world = spec.compile().expect("the declared world compiles");

        let shapes: Vec<CharacterShape> = world.query::<CharacterShape>().cloned().collect();
        assert_eq!(shapes.len(), 1);
        assert!(shapes[0].target.is_some(), "body_shape names its target");
        assert_eq!(shapes[0].sliders.len(), 15);

        let clip = world.query::<Animation>().next().expect("the idle clip");
        assert!(clip.target.is_some(), "body_idle names its target");
        assert!(clip.duration > 0.0, "the clip imported from the body");

        let floor = world.query::<Prop>().next().expect("the floor");
        assert!(floor.mesh.is_some() && floor.material.is_some());
    }
}
