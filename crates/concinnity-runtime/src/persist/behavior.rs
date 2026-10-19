// Persisted behavior state, the one thing this crate lends
// `concinnity_core::behavior::BehaviorSystem`: a `state` file under the host's
// save directory. The state's shape and its keying are the system's (see
// `concinnity_core::behavior::BehaviorState`); what this owns is the file,
// written off the sim thread.

use concinnity_core::behavior::{BehaviorState, BehaviorStore};
use concinnity_host::store::paths::StateTree;
use concinnity_host::store::writer::FileWriter;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(crate) struct FileStore {
    dir: PathBuf,
    writer: FileWriter,
}

impl FileStore {
    // The store in `tree`'s save directory.
    pub(crate) fn new(tree: &StateTree) -> Self {
        Self::at(&tree.saves_dir())
    }

    pub(crate) fn at(dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
            writer: FileWriter::new("cn-behav-save"),
        }
    }
}

impl BehaviorStore for FileStore {
    fn read(&self) -> Option<BehaviorState> {
        self.writer.flush();
        concinnity_host::store::cbor::read(&state_file(&self.dir), "BehaviorSystem: saved state")
    }

    fn write(&self, state: &BehaviorState) {
        self.writer
            .write_cbor(state_file(&self.dir), state, "BehaviorSystem: state");
    }
}

fn state_file(dir: &Path) -> PathBuf {
    dir.join("state")
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::{
        Behavior, BehaviorExpr, BehaviorNode, BehaviorSource, Variables,
    };
    use concinnity_core::ecs::World;

    // The store this host attaches is reached by a `save` node and read back at the
    // next start: the seam between the system and the file is live, not just
    // constructible.
    #[test]
    fn a_saving_world_restores_its_variable_through_the_file_store() {
        use concinnity_core::behavior::{BehaviorStore, BehaviorSystem};
        use concinnity_core::components::{BehaviorLiteral, VariableDecl};
        use concinnity_core::ecs::System;

        let tree = concinnity_testing::TempTree::new();
        let dir = tree.join("state");

        let saver = || Behavior {
            on: BehaviorSource::Start,
            body: vec![
                BehaviorNode::Set {
                    var: "visits".into(),
                    value: BehaviorExpr::Int(1),
                    add: true,
                },
                BehaviorNode::Save,
            ],
            ..Default::default()
        };
        let declared = || Variables {
            vars: vec![VariableDecl {
                name: "visits".into(),
                value: BehaviorLiteral::Int(0),
            }],
        };

        // One run: the `save` node's write reaches the file.
        let run = |expected: i32, what: &str| {
            let mut world = World::new();
            world.add_component(saver());
            world.add_component(declared());
            let mut system = BehaviorSystem::new().with_store(Box::new(FileStore::at(&dir)));
            system.init(&mut world.context());
            system.step(&mut world.context());
            // Dropping the system flushes its pending save.
            drop(system);

            let state = FileStore::at(&dir)
                .read()
                .expect("the save node wrote through the store");
            assert_eq!(
                state.vars.get("visits"),
                Some(&BehaviorLiteral::Int(expected)),
                "{what}"
            );
        };

        run(1, "the tick's value reached the file");
        run(2, "the second run started from what the first stored");
        std::fs::remove_dir_all(&dir).ok();
    }

    // A written state reads back from its file.
    #[test]
    fn state_round_trips_through_the_file() {
        use concinnity_core::behavior::{BehaviorState, BehaviorStore};
        use concinnity_core::components::BehaviorLiteral;

        let tree = concinnity_testing::TempTree::new();
        let dir = tree.join("state");
        let store = FileStore::at(&dir);
        assert!(store.read().is_none(), "nothing written yet");

        let mut state = BehaviorState::default();
        state.vars.insert("score".into(), BehaviorLiteral::Int(12));
        state.fired.push((3, 0xfeed));
        store.write(&state);
        let back = store.read().expect("state readable");
        assert!(state_file(&dir).exists());
        assert_eq!(back.vars.get("score"), Some(&BehaviorLiteral::Int(12)));
        assert_eq!(back.fired, vec![(3, 0xfeed)]);
    }
}
