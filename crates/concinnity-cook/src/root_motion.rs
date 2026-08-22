// src/root_motion.rs
//
// Root-motion bake for Animation clips: strip the root joint's travel out of
// the pose tracks into the asset's `root_track`. Build-time only -- the
// runtime samples the finished curve (`gfx::root_motion::RootTrack`), it
// never re-derives it.

use crate::assets::Animation;
use concinnity_core::gfx::root_motion::RootKey;

// Strip the root joint's travel out of `tracks` and bake it into
// `root_track`, per the `root_motion` / `root_motion_y` flags. Runs once
// after any glTF import (a non-empty `root_track` marks the strip as already
// done, so re-running a build pass is a no-op). The root joint is joint 0
// (skeletons are parents-before-children); a clip with no track on it gains
// an empty curve and a warning upstream.
pub(crate) fn bake_root_motion(anim: &mut Animation) {
    if !anim.root_motion || !anim.root_track.is_empty() {
        return;
    }
    let Some(track) = anim.tracks.iter_mut().find(|t| t.joint == 0) else {
        return;
    };
    let strip_y = anim.root_motion_y;
    for key in &mut track.keyframes {
        let t = key.pose.translation;
        anim.root_track.push(RootKey {
            time: key.time,
            translation: [t[0], if strip_y { t[1] } else { 0.0 }, t[2]],
        });
        key.pose.translation = [0.0, if strip_y { 0.0 } else { t[1] }, 0.0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 1s clip whose root (joint 0) walks +2 X while bobbing 1.0 -> 1.2 Y.
    fn walking_clip() -> Animation {
        crate::ecs::asset_id::reset_interner();
        serde_json::from_value(serde_json::json!({
            "target": "hero",
            "duration": 1.0,
            "root_motion": true,
            "tracks": [{"joint": 0, "keyframes": [
                {"time": 0.0, "translation": [0.0, 1.0, 0.0]},
                {"time": 1.0, "translation": [2.0, 1.2, 0.0]}
            ]}]
        }))
        .unwrap()
    }

    #[test]
    fn bake_root_motion_strips_xz_and_keeps_y_in_the_pose() {
        let mut a = walking_clip();
        bake_root_motion(&mut a);
        assert_eq!(a.root_track.len(), 2);
        assert_eq!(a.root_track[1].translation, [2.0, 0.0, 0.0]);
        // The pose keeps the vertical bob but stays anchored horizontally.
        assert_eq!(a.tracks[0].keyframes[1].pose.translation, [0.0, 1.2, 0.0]);
        let clip = a.to_clip();
        assert!(clip.root.is_some());
    }

    #[test]
    fn bake_root_motion_y_moves_vertical_travel_too() {
        let mut a = walking_clip();
        a.root_motion_y = true;
        bake_root_motion(&mut a);
        assert_eq!(a.root_track[1].translation, [2.0, 1.2, 0.0]);
        assert_eq!(a.tracks[0].keyframes[1].pose.translation, [0.0; 3]);
    }

    #[test]
    fn bake_root_motion_respects_flag_and_prior_bake() {
        let mut plain = walking_clip();
        plain.root_motion = false;
        bake_root_motion(&mut plain);
        assert!(plain.root_track.is_empty());
        assert_eq!(
            plain.tracks[0].keyframes[1].pose.translation,
            [2.0, 1.2, 0.0]
        );

        let mut baked = walking_clip();
        bake_root_motion(&mut baked);
        let track = baked.root_track.clone();
        bake_root_motion(&mut baked);
        assert_eq!(
            baked.root_track.len(),
            track.len(),
            "second bake is a no-op"
        );
    }
}
