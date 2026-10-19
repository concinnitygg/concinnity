//! The one-of-a-kind media a world's graphics load reads: the environment map
//! and color LUT payloads, whether the environment map draws as the
//! background, and the textures the sprite atlas pool holds.

use alloc::vec::Vec;

use crate::components::{Sprite, Story};
use crate::ecs::{PipelineContext, TextureHandle};
use crate::resource::{ColorLutTable, EnvironmentMapTable, ResourceEntry};

/// The environment map's payload: `Some(None)` with none declared, `None`
/// (failing the load) when it cannot be read.
pub fn environment_map_payload(ctx: &mut PipelineContext<'_>) -> Option<Option<Vec<u8>>> {
    let first = ctx
        .resource::<EnvironmentMapTable>()
        .map(|t| (t.len(), t.0.first().cloned()));
    first_payload(ctx, first, "EnvironmentMap")
}

/// The color LUT's payload, as [`environment_map_payload`].
pub fn color_lut_payload(ctx: &mut PipelineContext<'_>) -> Option<Option<Vec<u8>>> {
    let first = ctx
        .resource::<ColorLutTable>()
        .map(|t| (t.len(), t.0.first().cloned()));
    first_payload(ctx, first, "ColorLut")
}

// The payload of the sole resource of its kind, from `first` (the table's
// length and its first entry): the bytes a world baked for itself, else read
// through its locator. Extra declarations are logged and ignored.
#[expect(
    clippy::option_option,
    reason = "`None` fails the load through `?`; `Some(None)` is a world without one"
)]
fn first_payload(
    ctx: &mut PipelineContext<'_>,
    first: Option<(usize, Option<ResourceEntry>)>,
    what: &str,
) -> Option<Option<Vec<u8>>> {
    let Some((count, Some(entry))) = first else {
        return Some(None);
    };
    if count > 1 {
        tracing::warn!("graphics: {count} {what}s declared; only the first is used");
    }
    if let Some(baked) = entry.baked_bytes() {
        return Some(Some(baked.to_vec()));
    }
    let Some(locator) = entry.payload else {
        return Some(None);
    };
    match ctx.read_payload(&locator) {
        Ok(bytes) => Some(Some(bytes.to_vec())),
        Err(e) => {
            tracing::error!("graphics: failed to read {what} payload: {e}");
            None
        }
    }
}

/// Whether the environment map draws as the background: yes with none
/// declared. `None` (failing the load) when its record cannot be decoded.
pub fn environment_background(ctx: &PipelineContext<'_>) -> Option<bool> {
    match ctx.resource::<EnvironmentMapTable>().map(|t| t.record(0)) {
        None => Some(true),
        Some(Ok(record)) => Some(record.background),
        Some(Err(e)) => {
            tracing::error!(
                "graphics: EnvironmentMap handle 0 runtime record failed to decode: {e}"
            );
            None
        }
    }
}

/// Every distinct texture the sprite atlas pool holds, in handle order: each
/// one a Sprite draws, and each image a Story's stages show. The story swaps
/// its images onto the stage sprites at runtime, so they are resident before
/// any sprite names them.
pub fn sprite_texture_handles(ctx: &PipelineContext<'_>) -> Vec<TextureHandle> {
    let mut ids: Vec<TextureHandle> = ctx.query::<Sprite>().filter_map(|s| s.texture).collect();
    for story in ctx.query::<Story>() {
        let stages = story.nodes.iter().flat_map(|n| {
            n.pages
                .iter()
                .map(|p| &p.stage)
                .chain(core::iter::once(&n.choice_stage))
        });
        for stage in stages {
            let images = [&stage.bg, &stage.left, &stage.center, &stage.right];
            ids.extend(images.into_iter().flatten().map(|image| image.texture));
        }
    }
    ids.sort_unstable_by_key(|id| id.0);
    ids.dedup();
    ids
}

#[cfg(test)]
mod tests;
