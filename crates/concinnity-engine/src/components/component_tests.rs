//! Behavior checks for data-only components whose `Component` impls are
//! generated centrally (see `cn_impl_components!`): the clamped accessors of
//! `VoxelWorld`, and the runtime-only state of `TextInput`. One submodule per
//! component.

use crate::components::*;

mod voxel_world {
    use super::*;

    #[test]
    fn degenerate_args_are_floored_and_clamped() {
        let w = VoxelWorld {
            chunk_blocks: [0, 0, 0],
            block_size: -1.0,
            view_radius: 9999,
            load_budget: 0,
            ..VoxelWorld::default()
        };
        assert_eq!(w.chunk_blocks(), [1, 1, 1]);
        assert!(w.block_size() > 0.0);
        assert_eq!(w.view_radius(), 32);
        assert_eq!(w.load_budget(), 1);
    }

    #[test]
    fn a_zero_impostor_radius_disables_the_far_band() {
        let w = VoxelWorld {
            impostor_radius: 0,
            ..VoxelWorld::default()
        };
        // Clamped up to the view radius, which leaves no far band.
        assert_eq!(w.impostor_radius(), w.view_radius());
        assert!(!w.impostors_enabled());
    }

    #[test]
    fn impostor_radius_enables_the_far_band_and_clamps() {
        let w = VoxelWorld {
            view_radius: 5,
            impostor_radius: 16,
            impostor_step: 0,
            ..VoxelWorld::default()
        };
        assert_eq!(w.impostor_radius(), 16);
        assert!(w.impostors_enabled());
        // step floored at 1.
        assert_eq!(w.impostor_step(), 1);

        // An impostor radius below the view radius disables impostors.
        let w2 = VoxelWorld {
            view_radius: 8,
            impostor_radius: 4,
            ..VoxelWorld::default()
        };
        assert_eq!(w2.impostor_radius(), w2.view_radius());
        assert!(!w2.impostors_enabled());
    }
}

mod text_input {
    use super::*;

    #[test]
    fn runtime_state_is_not_serialized() {
        // `focused` / `caret` are runtime-only, so `args` (the
        // public schema) never carries them.
        let t = TextInput {
            focused: true,
            caret: 3,
            ..Default::default()
        };
        let v = serde_json::to_value(&t).unwrap();
        assert!(v.get("focused").is_none());
        assert!(v.get("caret").is_none());
        assert!(v.is_object());
    }
}
