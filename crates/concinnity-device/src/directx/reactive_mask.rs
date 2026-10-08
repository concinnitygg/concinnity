// The reactive mask on D3D12: the R8 render-resolution target the particle and
// transparent passes write beside the scene and TAA, the upscalers and the
// reactive view read. It rests in PIXEL_SHADER_RESOURCE like every sampled
// color target, and the frame graph moves it while it is in the frame; on a
// frame it is not, its writers bind a null view in its place, so their writes
// go nowhere and it never leaves that state.

use concinnity_core::render::error::RenderResult;
use concinnity_core::render::reactive_mask::ReactiveWrite;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::descriptor_slot::SrvSlot;
use super::pso::{Blend, GraphicsPso};
use super::texture::{create_rt_target, transition_barrier, write_format_rtv, write_format_srv};

pub(in crate::directx) const REACTIVE_MASK_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8_UNORM;

// Declare the mask's target, after the scene's, on a writer PSO: max-blended,
// so the most reactive layer over a pixel wins whatever order the layers draw
// in.
pub(in crate::directx) fn mask_target(pso: GraphicsPso<'_>) -> GraphicsPso<'_> {
    pso.target(REACTIVE_MASK_FORMAT, Blend::Max)
}

// The descriptor slots the mask is viewed through, reserved by the heap layout.
#[derive(Clone, Copy)]
pub(in crate::directx) struct ReactiveMaskSlots {
    pub rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub null_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub srv_gpu: SrvSlot,
}

pub(in crate::directx) struct ReactiveMask {
    pub(in crate::directx) resource: ID3D12Resource,
    slots: ReactiveMaskSlots,
}

impl ReactiveMask {
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        (width, height): (u32, u32),
        slots: ReactiveMaskSlots,
    ) -> RenderResult<Self> {
        let null_desc = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: REACTIVE_MASK_FORMAT,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        // SAFETY: a null-resource view with a fully specified descriptor, written
        // into a slot this context reserved for it in a heap it owns.
        unsafe { device.CreateRenderTargetView(None, Some(&null_desc), slots.null_rtv) };
        let mut mask = Self {
            resource: create_rt_target(device, width, height, REACTIVE_MASK_FORMAT)?,
            slots,
        };
        mask.write_views(device);
        Ok(mask)
    }

    // Recreate the target at a new render resolution; the views keep their
    // slots.
    pub(in crate::directx) fn resize(
        &mut self,
        device: &ID3D12Device,
        (width, height): (u32, u32),
    ) -> RenderResult<()> {
        self.resource = create_rt_target(device, width, height, REACTIVE_MASK_FORMAT)?;
        self.write_views(device);
        Ok(())
    }

    fn write_views(&mut self, device: &ID3D12Device) {
        write_format_rtv(device, &self.resource, self.slots.rtv, REACTIVE_MASK_FORMAT);
        write_format_srv(
            device,
            &self.resource,
            self.slots.srv_cpu,
            REACTIVE_MASK_FORMAT,
        );
    }

    pub(in crate::directx) fn srv_gpu(&self) -> SrvSlot {
        self.slots.srv_gpu
    }

    // The view a writer binds beside the scene: the mask's own while it is in
    // the frame, the null view otherwise.
    pub(in crate::directx) fn rtv(&self, write: ReactiveWrite) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        if write.stores() {
            self.slots.rtv
        } else {
            self.slots.null_rtv
        }
    }

    // Clear the mask for the frame's first writer, which the graph has already
    // put it in RENDER_TARGET for. A no-op unless `write` clears.
    pub(in crate::directx) fn clear(&self, cmd: &ID3D12GraphicsCommandList, write: ReactiveWrite) {
        if write == ReactiveWrite::Clear {
            // SAFETY: the command list is in the recording state and the view
            // names the live mask, which the graph transitioned to RENDER_TARGET.
            unsafe { cmd.ClearRenderTargetView(self.slots.rtv, &[0.0_f32; 4], None) };
        }
    }

    // Clear the mask on a frame the graph does not carry it, for a reader that
    // must have one, and leave it in the compute read: from its resting state
    // to RENDER_TARGET, cleared, then to NON_PIXEL_SHADER_RESOURCE.
    pub(in crate::directx) fn clear_outside_graph(&self, cmd: &ID3D12GraphicsCommandList) {
        let to_rt = transition_barrier(
            &self.resource,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let to_read = transition_barrier(
            &self.resource,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        // SAFETY: the command list is in the recording state, and the mask rests in
        // PIXEL_SHADER_RESOURCE on a frame the graph does not carry it.
        unsafe {
            cmd.ResourceBarrier(&[to_rt]);
            cmd.ClearRenderTargetView(self.slots.rtv, &[0.0_f32; 4], None);
            cmd.ResourceBarrier(&[to_read]);
        }
    }

    // Return a mask `clear_outside_graph` prepared to its resting state.
    pub(in crate::directx) fn rest_after_compute_read(&self, cmd: &ID3D12GraphicsCommandList) {
        let back = transition_barrier(
            &self.resource,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        // SAFETY: the command list is in the recording state, and the mask is in the compute read
        // `clear_outside_graph` left it in.
        unsafe { cmd.ResourceBarrier(&[back]) };
    }
}
