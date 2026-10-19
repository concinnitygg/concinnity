use super::{Core, SYSTEMS};
use concinnity_core::ecs::{Phase, SystemEntry};

const ENTRIES: &[SystemEntry] = SYSTEMS.entries;

fn position(name: &str) -> usize {
    ENTRIES
        .iter()
        .position(|e| e.name == name)
        .unwrap_or_else(|| panic!("{name} is a table entry"))
}

// The table is well formed, holds every core entry, and keeps core's order.
#[test]
fn the_table_holds_every_core_system_in_core_order() {
    assert_eq!(SYSTEMS.check_complete(Core::ALL), Ok(()));
}

// The frame is submitted after everything that moves what it draws: every
// simulation entry runs before StreamingSystem, whose `CameraRelativeView` is
// the pose the draw takes, and only the report follows GraphicsSystem.
#[test]
fn the_frame_follows_the_simulation() {
    let streaming = position("StreamingSystem");
    let graphics = position("GraphicsSystem");
    for (i, entry) in ENTRIES.iter().enumerate() {
        if entry.phase <= Phase::Late {
            assert!(i < streaming, "{} runs before streaming", entry.name);
        }
    }
    assert!(streaming < graphics);
    let after: Vec<&str> = ENTRIES[graphics + 1..].iter().map(|e| e.name).collect();
    assert_eq!(after, ["FrameReportSystem"]);
}
