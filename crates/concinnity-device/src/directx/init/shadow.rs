//! Shadows: the cascade shadow map array with its depth-only pipeline, and the
//! spot shadow array with its per-slice projections.

use concinnity_core::gfx::render_types::{
    self, LightUniforms, NUM_SHADOW_CASCADES, ShadowUniforms, SpotShadowData,
};
use concinnity_core::render::backend_init::ShadowParams;
use concinnity_core::render::csm;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::lights;
use windows::Win32::Graphics::Direct3D12::*;

use super::heap_layout::{DSV_SHADOW_BASE_SLOT, DSV_SPOT_SHADOW_BASE_SLOT};
use super::pipelines::{create_shadow_pso, create_shadow_root_signature};
use super::{InitGpu, heaps};
use crate::directx::context::{
    DxDescriptors, DxTargets, ShadowState, SpotShadowState, align256, dump_on_err,
};
use crate::directx::draw::upload_static_records;
use crate::directx::slang_builtins::{self, SlangCompile};
use crate::directx::texture::{
    create_buffer, create_fallback_shadow_array, create_shadow_map_array,
};

pub(super) fn build_shadow(
    gpu: &InitGpu<'_>,
    descriptors: &DxDescriptors,
    targets: &DxTargets,
    shadows: &ShadowParams,
    light_uniforms: &LightUniforms,
) -> RenderResult<ShadowState> {
    let hw = gpu.hw;
    let dsv_descriptor_size = heaps::descriptor_size(&hw.device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    // Shadow map array
    // Real path: NUM_SHADOW_CASCADES-slice Texture2DArray with per-slice DSVs.
    // Fallback: 1x1 single-slice R32_FLOAT array with value 0.0 (LESS_EQUAL
    // always passes, so fully lit), declared as Texture2DArray so the shader's
    // binding type stays identical between disabled and enabled cases.
    // CSM is gated on `shadow_map_size` (from GraphicsConfig; 0 disables
    // shadows). The shadow vertex shader is engine-internal
    // (`slang_builtins::SHADOW_VERT`). Mirrors the Metal internal-shadow
    // path.
    let effective_shadow_size = shadows.map_size;
    let (shadow_resource_opt, shadow_dsvs, shadow_srv_gpu) = if effective_shadow_size > 0 {
        let (sm, dsvs) = create_shadow_map_array(
            &hw.device,
            effective_shadow_size,
            NUM_SHADOW_CASCADES as u32,
            heaps::cpu_handle(
                &targets.depth.heap,
                dsv_descriptor_size,
                DSV_SHADOW_BASE_SLOT,
            ),
            dsv_descriptor_size,
            descriptors.slot_cpu(0),
            descriptors.slot_gpu(0),
        )?;
        (Some(sm), dsvs, descriptors.slot_gpu(0))
    } else {
        let fb = create_fallback_shadow_array(
            &hw.alloc,
            descriptors.slot_cpu(0),
            descriptors.slot_gpu(0),
        )?;
        (Some(fb), Vec::new(), descriptors.slot_gpu(0))
    };

    // Only build the shadow PSO when shadows are enabled; the shadow pass
    // keys off `shadow.pso.is_some()`, so passing `None` when
    // `effective_shadow_size == 0` keeps a shadow-disabled world from
    // rendering into nonexistent cascade DSVs.
    let shadow_vs = slang_builtins::SHADOW_VERT.compile(gpu.hot_reload)?;
    let shadow_vs_for_pso = if effective_shadow_size > 0 {
        Some(shadow_vs.as_slice())
    } else {
        None
    };
    let (root_sig, pso) =
        build_shadow_pipeline(&hw.device, hw.info_queue.as_ref(), shadow_vs_for_pso)?;

    Ok(ShadowState {
        resource: shadow_resource_opt,
        dsvs: shadow_dsvs,
        map_size: effective_shadow_size,
        srv_gpu: shadow_srv_gpu,
        // The first directional light's direction, for per-frame CSM updates.
        light_dir: lights::sun_direction(light_uniforms),
        update: shadows.update,
        distance: shadows.distance,
        cascades: shadows.cascades,
        scheduler: Default::default(),
        render_mask: 0,
        uniforms: csm::empty_shadow_uniforms(),
        root_sig,
        pso,
    })
}

fn build_shadow_pipeline(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    shadow_vs: Option<&[u8]>,
) -> Result<(Option<ID3D12RootSignature>, Option<ID3D12PipelineState>), String> {
    if let Some(svs) = shadow_vs {
        let sr = dump_on_err(info_queue, create_shadow_root_signature(device))?;
        let sp = dump_on_err(info_queue, create_shadow_pso(device, &sr, svs))?;
        Ok((Some(sr), Some(sp)))
    } else {
        Ok((None, None))
    }
}

pub(super) fn build_spot_shadow(
    gpu: &InitGpu<'_>,
    descriptors: &DxDescriptors,
    targets: &DxTargets,
    spot_shadows: &[SpotShadowData],
    shadow_map_size: u32,
) -> RenderResult<SpotShadowState> {
    let hw = gpu.hw;
    let spot_shadow_srv_slot = descriptors.layout.spot_shadow_srv_slot;
    let dsv_descriptor_size = heaps::descriptor_size(&hw.device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    // Spot shadow map array: one slice per shadow-casting spot light, at a
    // quarter the cascade resolution (a spot slice covers a single cone, not
    // a view-frustum slab). Local lights are static, so the slice count and
    // every light-space matrix are fixed here; only the depth refreshes.
    // A world with no shadowed spot still binds a 1x1 fallback so the main
    // pass's SRV is never unwritten.
    let spot_shadow_slice_size = render_types::spot_shadow_slice_size(shadow_map_size);
    let (spot_shadow_resource, spot_shadow_dsvs) = if spot_shadows.is_empty() {
        let fb = create_fallback_shadow_array(
            &hw.alloc,
            descriptors.slot_cpu(spot_shadow_srv_slot),
            descriptors.slot_gpu(spot_shadow_srv_slot),
        )?;
        (Some(fb), Vec::new())
    } else {
        let (sm, dsvs) = create_shadow_map_array(
            &hw.device,
            spot_shadow_slice_size,
            spot_shadows.len() as u32,
            heaps::cpu_handle(
                &targets.depth.heap,
                dsv_descriptor_size,
                DSV_SPOT_SHADOW_BASE_SLOT,
            ),
            dsv_descriptor_size,
            descriptors.slot_cpu(spot_shadow_srv_slot),
            descriptors.slot_gpu(spot_shadow_srv_slot),
        )?;
        (Some(sm), dsvs)
    };
    // The per-slice projections, uploaded once. A scene with no shadowed
    // spot gets a one-element identity buffer: the shader never indexes it
    // (every `shadow_index` is -1) but the root SRV must still be valid.
    let spot_shadow_data = if spot_shadows.is_empty() {
        vec![SpotShadowData::ZERO]
    } else {
        spot_shadows.to_vec()
    };
    let spot_shadow_buffer = {
        let size = align256((spot_shadow_data.len() * size_of::<SpotShadowData>()) as u64);
        let buf = create_buffer(
            &hw.alloc,
            size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        upload_static_records(&buf, &spot_shadow_data, "spot-shadow")?;
        buf
    };
    // One `ShadowUniforms` per slice, each carrying that spot's matrix in
    // `light_vps[0]`, so the shared shadow vertex shader renders a spot
    // slice by pushing cascade_idx = 0. Written once: the projections never
    // change, so unlike the cascade UBO this needs no per-frame ring.
    let spot_shadow_ubo_stride = align256(size_of::<ShadowUniforms>() as u64);
    let spot_shadow_ubo = {
        let slots = spot_shadows.len().max(1) as u64;
        let buf = create_buffer(
            &hw.alloc,
            spot_shadow_ubo_stride * slots,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
        // local that receives the mapping.
        unsafe { buf.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| format!("map spot-shadow UBO: {e}"))?;
        for (i, sd) in spot_shadows.iter().enumerate() {
            let mut u = csm::empty_shadow_uniforms();
            u.light_vps[0] = sd.light_vp;
            u.active_cascades = 1;
            // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload,
            // and the source is a separate allocation, so the ranges cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &u as *const ShadowUniforms as *const u8,
                    (ptr as *mut u8).add(i * spot_shadow_ubo_stride as usize),
                    size_of::<ShadowUniforms>(),
                );
            }
        }
        // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping
        // past this call.
        unsafe { buf.Unmap(0, None) };
        buf
    };

    Ok(SpotShadowState {
        resource: spot_shadow_resource,
        dsvs: spot_shadow_dsvs,
        srv_gpu: descriptors.slot_gpu(spot_shadow_srv_slot),
        buffer: spot_shadow_buffer,
        ubo: spot_shadow_ubo,
        ubo_stride: spot_shadow_ubo_stride,
        slice_size: spot_shadow_slice_size,
        scheduler: Default::default(),
        render_mask: 0,
    })
}
