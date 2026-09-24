// The reflection-probe set as DirectX binds it: one cube-map array holding every
// probe's prefiltered radiance, a cube per probe, behind a single SRV slot, and a
// per-frame upload ring of parallax records bound as a root SRV beside the
// count's root CBV. Every pass that reads probes -- the bindless main pass, SSR,
// the RT resolve and the transparent pass -- binds those three.
//
// Between bakes every subresource rests in PIXEL_SHADER_RESOURCE. A bake moves
// only its own cube's subresources to UNORDERED_ACCESS and back, so the frames
// sampling the cubes installed before it are never reading a subresource in a
// write state.

use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::reflection_probe::PrefilterPlan;
use concinnity_core::render::uniforms::{ProbeUniforms, grown_probe_capacity};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::PooledBuffer;
use super::com;
use super::context::DxContext;
use super::error::map_hresult;
use super::probe_prefilter::PROBE_CUBE_FORMAT;

// The state every subresource of the array rests in between bakes.
pub(in crate::directx) const PROBE_CUBES_STATE: D3D12_RESOURCE_STATES =
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

// The subresource index of `mip` of array slice `slice` in a resource with `mips`
// levels, plane 0: D3D12CalcSubresource.
const fn subresource(mip: u32, slice: u32, mips: u32) -> u32 {
    mip + slice * mips
}

// Every subresource of cube `cube`: its six slices at every mip.
fn cube_subresources(cube: u32, mips: u32) -> impl Iterator<Item = u32> {
    (6 * cube..6 * cube + 6)
        .flat_map(move |slice| (0..mips).map(move |mip| subresource(mip, slice, mips)))
}

// The shape and initial state of a committed probe cube resource: `cubes`
// cubes of `mips` levels, UAV + SRV capable.
#[derive(Clone, Copy)]
pub(in crate::directx) struct CubeResource<'a> {
    pub face_size: u32,
    pub mips: u32,
    pub cubes: usize,
    pub state: D3D12_RESOURCE_STATES,
    pub label: &'a str,
}

// A committed cube-map (array) resource. Committed rather than pooled: the
// suballocator refuses GPU-written descs, because a placed resource needs
// re-initializing every time it claims memory and the pool does not do that.
pub(in crate::directx) fn create_cube_resource(
    device: &ID3D12Device,
    shape: CubeResource<'_>,
) -> RenderResult<ID3D12Resource> {
    let CubeResource {
        face_size,
        mips,
        cubes,
        state,
        label,
    } = shape;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: face_size as u64,
        Height: face_size,
        DepthOrArraySize: u16::try_from(6 * cubes)
            .map_err(|_| RenderError::Other(format!("{label}: {cubes} cubes exceed an array")))?,
        MipLevels: mips as u16,
        Format: PROBE_CUBE_FORMAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
    // new COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            state,
            None,
            &mut resource,
        )
    }
    .map_err(|e| map_hresult(e.code(), &format!("create {label}")))?;
    resource.ok_or_else(|| RenderError::Other(format!("create {label} returned None")))
}

// A single-mip TEXTURE2DARRAY UAV over cube `cube`'s six faces of `resource`. A
// cube is a six-slice array, so this is what lets a kernel address (x, y, face)
// directly.
pub(in crate::directx) fn write_cube_mip_uav(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    cube: usize,
    mip: u32,
    uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: PROBE_CUBE_FORMAT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                MipSlice: mip,
                FirstArraySlice: 6 * cube as u32,
                ArraySize: 6,
                PlaneSlice: 0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateUnorderedAccessView(resource, None, Some(&desc), uav_cpu) };
}

// A committed cube-map array, `capacity` cubes of `mips` levels.
pub(in crate::directx) struct ProbeCubeArray {
    resource: ID3D12Resource,
    capacity: usize,
    mips: u32,
}

impl ProbeCubeArray {
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        plan: &PrefilterPlan,
        capacity: usize,
    ) -> RenderResult<ProbeCubeArray> {
        Self::create(device, plan.face_size(), plan.mips(), capacity)
    }

    // A one-texel array holding one cube, bound until a world places a probe. No
    // shader samples it: the probe count is 0.
    pub(in crate::directx) fn stand_in(device: &ID3D12Device) -> RenderResult<ProbeCubeArray> {
        let mut array = Self::create(device, 1, 1, 1)?;
        array.capacity = 0;
        Ok(array)
    }

    fn create(
        device: &ID3D12Device,
        face_size: u32,
        mips: u32,
        cubes: usize,
    ) -> RenderResult<ProbeCubeArray> {
        let resource = create_cube_resource(
            device,
            CubeResource {
                face_size,
                mips,
                cubes,
                state: PROBE_CUBES_STATE,
                label: "probe cube array",
            },
        )?;
        Ok(ProbeCubeArray {
            resource,
            capacity: cubes,
            mips,
        })
    }

    pub(in crate::directx) fn capacity(&self) -> usize {
        self.capacity
    }

    // The TEXTURECUBEARRAY SRV every pass samples the array through.
    pub(in crate::directx) fn write_srv(
        &self,
        device: &ID3D12Device,
        srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: PROBE_CUBE_FORMAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBEARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                TextureCubeArray: D3D12_TEXCUBE_ARRAY_SRV {
                    MostDetailedMip: 0,
                    MipLevels: self.mips,
                    First2DArrayFace: 0,
                    NumCubes: self.capacity.max(1) as u32,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        // SAFETY: the view descriptor and the resource it names are live for the call,
        // and the destination handle addresses a slot this context reserved for the
        // view in a heap it owns.
        unsafe { device.CreateShaderResourceView(&self.resource, Some(&desc), srv_cpu) };
    }

    // A single-mip TEXTURE2DARRAY UAV over cube `cube`'s six faces, the
    // destination of one convolution dispatch.
    pub(in crate::directx) fn write_mip_uav(
        &self,
        device: &ID3D12Device,
        cube: usize,
        mip: u32,
        uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        write_cube_mip_uav(device, &self.resource, cube, mip, uav_cpu);
    }

    // Transitions moving every subresource of cube `cube` from `before` to
    // `after`, leaving the other cubes where they are.
    pub(in crate::directx) fn cube_barriers(
        &self,
        cube: usize,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) -> Vec<D3D12_RESOURCE_BARRIER> {
        cube_subresources(cube as u32, self.mips)
            .map(|sub| D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: com::borrowed(&self.resource),
                        StateBefore: before,
                        StateAfter: after,
                        Subresource: sub,
                    }),
                },
            })
            .collect()
    }

    // Mip levels every cube carries.
    pub(in crate::directx) fn mips(&self) -> u32 {
        self.mips
    }
}

// A persistently mapped upload buffer of parallax records with room for
// `capacity`, bound as a root SRV.
pub(in crate::directx) struct ProbeRecords {
    buffer: PooledBuffer,
    mapped: *mut u8,
    capacity: usize,
}

impl ProbeRecords {
    pub(in crate::directx) fn new(
        ctx_alloc: &super::allocator::DeviceAllocator,
        capacity: usize,
    ) -> RenderResult<ProbeRecords> {
        let capacity = capacity.max(1);
        let buffer = ctx_alloc.alloc_buffer(
            (capacity * size_of::<ProbeUniforms>()) as u64,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is
        // a live local that receives the mapping.
        unsafe { buffer.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| map_hresult(e.code(), "map probe records"))?;
        let records = ProbeRecords {
            buffer,
            mapped: ptr.cast(),
            capacity,
        };
        records.write(&vec![
            <ProbeUniforms as bytemuck::Zeroable>::zeroed();
            capacity
        ]);
        Ok(records)
    }

    fn write(&self, records: &[ProbeUniforms]) {
        let bytes: &[u8] = bytemuck::cast_slice(records);
        assert!(
            records.len() <= self.capacity,
            "probe records past the buffer"
        );
        // SAFETY: the mapping covers `capacity` records of an UPLOAD-heap buffer, the
        // assert above keeps the copy inside it, and the source is a separate
        // allocation, so the ranges cannot overlap.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.mapped, bytes.len()) };
    }

    // The GPU address a root SRV binds.
    pub(in crate::directx) fn gpu_va(&self) -> u64 {
        com::gpu_va(&self.buffer)
    }
}

// Everything the probe set binds: the live array (`None` until a world places a
// probe), the stand-in bound in its place, one records buffer per frame in
// flight, and a one-record stand-in the captures bind.
pub(in crate::directx) struct ProbeSetGpu {
    pub(in crate::directx) cubes: Option<ProbeCubeArray>,
    pub(in crate::directx) stand_in: ProbeCubeArray,
    pub(in crate::directx) records: Vec<ProbeRecords>,
    pub(in crate::directx) stand_in_records: ProbeRecords,
}

impl ProbeSetGpu {
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        alloc: &super::allocator::DeviceAllocator,
        frames: usize,
    ) -> RenderResult<ProbeSetGpu> {
        let capacity = concinnity_core::render::uniforms::DEFAULT_PROBE_RECORD_CAPACITY;
        Ok(ProbeSetGpu {
            cubes: None,
            stand_in: ProbeCubeArray::stand_in(device)?,
            records: (0..frames)
                .map(|_| ProbeRecords::new(alloc, capacity))
                .collect::<RenderResult<_>>()?,
            stand_in_records: ProbeRecords::new(alloc, 1)?,
        })
    }

    // The array the SRV slot names: the live one, or the stand-in.
    pub(in crate::directx) fn bound_cubes(&self) -> &ProbeCubeArray {
        self.cubes.as_ref().unwrap_or(&self.stand_in)
    }
}

impl DxContext {
    // The capacity the cube array must be replaced at to hold `placements`
    // cubes, or `None` while the live array already holds them.
    fn grown_probe_cube_capacity(&self, placements: usize) -> Option<usize> {
        let have = self
            .probe
            .gpu
            .cubes
            .as_ref()
            .map_or(0, ProbeCubeArray::capacity);
        grown_probe_capacity(have, placements, 1)
    }

    // Make room for `placements` cubes, replacing the array when the list
    // outgrows it and rewriting its SRV slot. Replacing a live array idles the
    // GPU first, so no submitted list still reads the old array or the slot.
    pub(in crate::directx) fn reserve_probe_cubes(
        &mut self,
        plan: &PrefilterPlan,
        placements: usize,
    ) -> RenderResult<()> {
        let Some(capacity) = self.grown_probe_cube_capacity(placements) else {
            return Ok(());
        };
        if self.probe.gpu.cubes.is_some() {
            self.wait_idle();
        }
        self.probe.gpu.cubes = Some(ProbeCubeArray::new(&self.hw.device, plan, capacity)?);
        self.write_probe_cubes_srv();
        tracing::debug!("reflection probes: cube array grown to {capacity}");
        Ok(())
    }

    // Point the probe SRV slot at whichever array is bound.
    pub(in crate::directx) fn write_probe_cubes_srv(&self) {
        let slot = self
            .descriptors
            .slot_cpu(self.descriptors.layout.probe_cubes_srv_slot);
        self.probe
            .gpu
            .bound_cubes()
            .write_srv(&self.hw.device, slot);
    }

    // Write this frame's probe count and records. The frame's records buffer grows
    // when the installed count outgrows it, to the array's capacity so it grows
    // once per placement list; its fence has retired, and the old buffer's free is
    // deferred behind the frames in flight anyway.
    pub(in crate::directx) fn upload_probe_set(&mut self, frame_idx: usize) -> RenderResult<()> {
        let set = self.probe.book.header();
        // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant
        // buffer that init sized for this payload, and the source is a separate live
        // value, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytemuck::bytes_of(&set).as_ptr(),
                self.uniforms.probe_set_cbv_ptrs[frame_idx],
                size_of_val(&set),
            );
        }
        let floor = self.probe.gpu.bound_cubes().capacity();
        let have = self.probe.gpu.records[frame_idx].capacity;
        if let Some(capacity) = grown_probe_capacity(have, self.probe.book.count(), floor) {
            self.probe.gpu.records[frame_idx] = ProbeRecords::new(&self.hw.alloc, capacity)?;
        }
        self.probe.gpu.records[frame_idx].write(self.probe.book.records());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // D3D12CalcSubresource for plane 0: the mips of slice 0, then slice 1's.
    #[test]
    fn subresources_count_mips_within_each_slice() {
        assert_eq!(subresource(0, 0, 10), 0);
        assert_eq!(subresource(3, 0, 10), 3);
        assert_eq!(subresource(0, 1, 10), 10);
        assert_eq!(subresource(9, 5, 10), 59);
    }

    // A cube's barriers cover exactly its six slices at every mip, so a bake
    // moves nothing a frame is sampling from another cube.
    #[test]
    fn a_cube_covers_its_six_slices_at_every_mip_and_nothing_else() {
        let mips = 4;
        let second: Vec<u32> = cube_subresources(1, mips).collect();
        assert_eq!(second.len(), 6 * mips as usize);
        assert_eq!(second.first(), Some(&subresource(0, 6, mips)));
        assert_eq!(second.last(), Some(&subresource(mips - 1, 11, mips)));
        let first: Vec<u32> = cube_subresources(0, mips).collect();
        assert!(first.iter().all(|s| !second.contains(s)));
        assert!(first.iter().chain(&second).all(|&s| s < 12 * mips));
    }
}
