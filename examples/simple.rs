//! The smallest runnable world: a window and one text label.

use concinnity::assets::{GraphicsConfig, TextLabel};
use concinnity::{App, World};

fn main() -> std::io::Result<()> {
    let mut world = World::new();
    world.add_component(GraphicsConfig::default());
    world.add_component(TextLabel {
        content: "Hello, world!".to_string(),
        ..Default::default()
    });

    App::from_world(world).run()
}
