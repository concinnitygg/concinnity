//! Composite: the fullscreen tonemap pipeline that draws the HDR scene onto the
//! swapchain back buffer.

use concinnity_core::render::error::RenderResult;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

use super::InitGpu;
use crate::directx::context::{CompositeState, dump_on_err};
use crate::directx::pipeline::{
    compile_composite_shaders, create_composite_pso, create_composite_root_signature,
};

pub(super) fn build_composite(
    gpu: &InitGpu<'_>,
    swap_format: DXGI_FORMAT,
) -> RenderResult<CompositeState> {
    let hw = gpu.hw;
    let info_queue = hw.info_queue.as_ref();
    let root_sig = dump_on_err(info_queue, create_composite_root_signature(&hw.device))?;
    let (composite_vs, composite_ps) = compile_composite_shaders(gpu.hot_reload)?;
    let pso = dump_on_err(
        info_queue,
        create_composite_pso(
            &hw.device,
            &root_sig,
            &composite_vs,
            &composite_ps,
            swap_format,
        ),
    )?;
    Ok(CompositeState { root_sig, pso })
}
