//! Which streamed content each scene owns, for scene residency: texture
//! members from the blob's baked per-scene groups (a streamed texture's slot is
//! its resource handle), mesh members from each streamed draw's `SceneMember`
//! entity, and shader members from the deferred buckets (a bucket's pipeline
//! is built when its scene pins).

use alloc::vec;
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;
use hashbrown::HashMap;

use crate::components::{RenderHandle, SceneMember};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{BlobSceneGroups, PipelineContext, ResourceKind};
use crate::gfx::render_types::DrawIndex;
use crate::render::streaming::scene_residency::{
    CHANNEL_MESH, CHANNEL_SHADER, CHANNEL_TEXTURE, SceneResidency,
};

/// The streamed content a world holds, by pool.
#[derive(Debug, Clone, Copy)]
pub struct StreamedContent<'a> {
    /// The streamed texture pool's slot count, when textures stream.
    pub texture_slots: Option<usize>,
    /// The draw each streamed mesh id stands for, when meshes stream.
    pub mesh_draws: Option<&'a [DrawIndex]>,
    /// Whether shader pipelines stream.
    pub shaders_stream: bool,
    /// Each deferred shader bucket with the scene that owns it.
    pub deferred_shaders: &'a [(u32, AssetId)],
}

/// The residency over `scenes` that `content` implies, or `None` when nothing
/// streams.
pub fn scene_residency(
    ctx: &PipelineContext<'_>,
    scenes: &[AssetId],
    content: StreamedContent<'_>,
) -> Option<SceneResidency> {
    if content.texture_slots.is_none() && content.mesh_draws.is_none() && !content.shaders_stream {
        return None;
    }
    let scene_idx: HashMap<AssetId, usize> =
        scenes.iter().enumerate().map(|(i, &s)| (s, i)).collect();
    let mut members: Vec<Vec<(u8, u32)>> = vec![Vec::new(); scenes.len()];

    if let Some(slots) = content.texture_slots
        && let Some(groups) = ctx.resource::<BlobSceneGroups>()
    {
        for group in &groups.0 {
            let Some(&idx) = scene_idx.get(&group.scene) else {
                continue;
            };
            for &(kind, handle) in &group.resources {
                if kind == ResourceKind::Texture && (handle as usize) < slots {
                    members[idx].push((CHANNEL_TEXTURE, handle));
                }
            }
        }
    }

    if let Some(mesh_draws) = content.mesh_draws {
        let mut scene_of_draw = HashMap::new();
        for (_, member, handle) in ctx.join2::<SceneMember, RenderHandle>() {
            for &slot in &handle.draws {
                scene_of_draw.insert(slot, member.0);
            }
        }
        for (stream_id, draw_idx) in mesh_draws.iter().enumerate() {
            if let Some(scene) = scene_of_draw.get(draw_idx)
                && let Some(&idx) = scene_idx.get(scene)
            {
                members[idx].push((CHANNEL_MESH, u32::saturating_from(stream_id)));
            }
        }
    }

    for &(bucket, scene) in content.deferred_shaders {
        if let Some(&idx) = scene_idx.get(&scene) {
            members[idx].push((CHANNEL_SHADER, bucket));
        }
    }

    let owned: usize = members.iter().map(Vec::len).sum();
    tracing::info!(
        "scene residency enabled ({} scenes, {} owned members)",
        scenes.len(),
        owned,
    );
    Some(SceneResidency::new(
        scenes.iter().copied().zip(members).collect(),
    ))
}

#[cfg(test)]
mod tests;
