use super::*;
use crate::gfx::streaming::texture::{self, PayloadSource};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::render::backend::recording::{Call, recording_backend};
use concinnity_core::render::ops::RenderOps;
use concinnity_core::render::queues::RenderQueues;
use concinnity_core::render::render_slots::RenderSlots;
use concinnity_core::render::scene_residency::{CHANNEL_SHADER, SceneResidency};
use concinnity_core::render::stream_pools::{
    DriveView, StreamPool, StreamingPools, StreamingRuntime,
};
use std::sync::Arc;

// A source yielding a fixed 1x1 texture for any id.
struct ConstTexture;
impl PayloadSource for ConstTexture {
    fn fetch(&self, _id: usize) -> Result<TextureImage, String> {
        Ok(TextureImage::rgba8(1, 1, vec![1, 2, 3, 4]))
    }
}

// A two-slot texture pool on a real load thread.
fn texture_runtime() -> StreamingRuntime {
    let centers = vec![vec![[0.0, 0.0, 0.0]], vec![[100.0, 0.0, 0.0]]];
    StreamingRuntime::new(
        StreamingState::new(
            StreamingPools {
                texture: Some(StreamPool::new(centers, 4, 8)),
                ..Default::default()
            },
            2,
        ),
        Box::new(StreamWorkers {
            texture: Some(texture::spawn_worker(Arc::new(ConstTexture)).expect("thread")),
            ..Default::default()
        }),
    )
}

fn queues() -> RenderQueues {
    RenderQueues {
        ops: RenderOps::default(),
        slots: RenderSlots::new(0, true, &[]),
    }
}

fn view(pins: Option<&[AssetId]>) -> DriveView<'_> {
    DriveView {
        cam_pos: [0.0; 3],
        view: concinnity_core::transform::IDENTITY,
        world_hidden: false,
        scene_pins: pins,
        planet: None,
    }
}

// Requests cross to the load thread after the drive and come back through the
// next frames' deliveries, uploading through the recorded ops.
#[test]
fn the_texture_pool_loads_through_its_thread() {
    let (recorded, mut backend) = recording_backend();
    let mut runtime = texture_runtime();
    concinnity_testing::eventually("both textures to load", || {
        let mut q = queues();
        runtime.drive(&mut q, view(None));
        q.ops.replay(&mut backend);
        (runtime.state.streaming_stats().texture == Some((2, 0, 0))).then_some(())
    });
    let s = recorded.lock();
    for slot in [0, 1] {
        assert!(s.saw(&Call::UpdateTextureSlot { slot, w: 1, h: 1 }));
    }
}

// A pinned scene owning a shader bucket keeps loading until the bucket's
// off-thread build comes back and its install is recorded.
#[test]
fn a_pinned_scene_loads_once_its_shader_bucket_is_installed() {
    use crate::gfx::streaming::shader::{DeferredBucket, ShaderPayloadSource};
    use concinnity_core::components::ShaderPrograms;

    let cooked = ShaderPrograms {
        name: "cooked".into(),
        ..Default::default()
    };
    let scene = AssetId(81);
    let mut runtime = StreamingRuntime::new(
        StreamingState::new(
            StreamingPools {
                scene_residency: Some(SceneResidency::new(vec![(
                    scene,
                    vec![(CHANNEL_SHADER, 1)],
                )])),
                ..Default::default()
            },
            2,
        ),
        Box::new(StreamWorkers {
            shaders: Some(ShaderWarmup::new(
                vec![DeferredBucket {
                    bucket: 1,
                    source: ShaderPayloadSource::Bytes(cooked.encode().unwrap()),
                }],
                None,
            )),
            ..Default::default()
        }),
    );
    let (recorded, mut backend) = recording_backend();
    let pins = [scene];
    let mut drive = |runtime: &mut StreamingRuntime| {
        let mut q = queues();
        runtime.drive(&mut q, view(Some(&pins)));
        q.ops.replay(&mut backend);
    };
    drive(&mut runtime);
    let loading = |r: &StreamingRuntime| r.state.scene_residency().unwrap().any_loading();
    assert!(
        loading(&runtime),
        "the dispatch alone does not load the scene"
    );
    concinnity_testing::eventually("the scene to load", || {
        if !loading(&runtime) {
            return Some(());
        }
        drive(&mut runtime);
        None
    });
    assert_eq!(
        recorded.lock().calls,
        [Call::InstallWorldShader {
            bucket: 1,
            name: "cooked".into(),
            prepared: None,
        }]
    );
}
