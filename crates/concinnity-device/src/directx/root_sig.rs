//! Root-signature builder. A signature is described as data -- root constants,
//! root descriptors, descriptor tables over [`Range`]s, and [`StaticSampler`]s,
//! in parameter order -- and turned into the D3D12 desc only inside
//! [`RootSig::build`], so every range pointer the desc carries is taken after
//! the last range was added.

use bytemuck::NoUninit;
use concinnity_core::render::error::{RenderError, RenderResult};
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::depth::shadow_sample_compare;
use crate::directx::error::map_hresult;
use crate::directx::root_constants::root_dwords;

// The shader stages a root parameter or static sampler is visible to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::directx) enum Visibility {
    All,
    Vertex,
    Pixel,
}

impl Visibility {
    fn raw(self) -> D3D12_SHADER_VISIBILITY {
        match self {
            Visibility::All => D3D12_SHADER_VISIBILITY_ALL,
            Visibility::Vertex => D3D12_SHADER_VISIBILITY_VERTEX,
            Visibility::Pixel => D3D12_SHADER_VISIBILITY_PIXEL,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::directx) enum RangeKind {
    Srv,
    Uav,
    Sampler,
}

impl RangeKind {
    fn raw(self) -> D3D12_DESCRIPTOR_RANGE_TYPE {
        match self {
            RangeKind::Srv => D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            RangeKind::Uav => D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            RangeKind::Sampler => D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        }
    }
}

// One descriptor range of a table: `count` registers of `kind` from `base` in
// `space`, appended after the previous range unless placed at the table start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::directx) struct Range {
    pub kind: RangeKind,
    pub base: u32,
    pub count: u32,
    pub space: u32,
    pub at_table_start: bool,
}

impl Range {
    // The descriptor count of an unbounded range.
    pub(in crate::directx) const UNBOUNDED: u32 = u32::MAX;

    pub(in crate::directx) const fn new(kind: RangeKind, base: u32, count: u32) -> Self {
        Self {
            kind,
            base,
            count,
            space: 0,
            at_table_start: false,
        }
    }

    pub(in crate::directx) const fn srv(base: u32, count: u32) -> Self {
        Self::new(RangeKind::Srv, base, count)
    }

    pub(in crate::directx) const fn uav(base: u32, count: u32) -> Self {
        Self::new(RangeKind::Uav, base, count)
    }

    pub(in crate::directx) const fn sampler(base: u32, count: u32) -> Self {
        Self::new(RangeKind::Sampler, base, count)
    }

    // An unbounded SRV array at `t0` of `space`, starting the table.
    pub(in crate::directx) const fn bindless_srv(space: u32) -> Self {
        Self {
            space,
            at_table_start: true,
            ..Self::srv(0, Self::UNBOUNDED)
        }
    }

    fn raw(self) -> D3D12_DESCRIPTOR_RANGE {
        D3D12_DESCRIPTOR_RANGE {
            RangeType: self.kind.raw(),
            NumDescriptors: self.count,
            BaseShaderRegister: self.base,
            RegisterSpace: self.space,
            OffsetInDescriptorsFromTableStart: if self.at_table_start {
                0
            } else {
                D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
            },
        }
    }
}

// The fixed sampler states the engine bakes into its root signatures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::directx) enum SamplerState {
    // Trilinear, clamp to edge.
    LinearClamp,
    // Trilinear, repeat.
    LinearWrap,
    // As `LinearClamp`, with a transparent black border.
    LinearClampTransparentBorder,
    // Bilinear shadow-map depth compare, clamp to edge.
    ShadowCompare,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::directx) struct StaticSampler {
    pub state: SamplerState,
    pub register: u32,
    pub visibility: Visibility,
}

impl StaticSampler {
    fn raw(self) -> D3D12_STATIC_SAMPLER_DESC {
        let (filter, address, compare, border) = match self.state {
            SamplerState::LinearClamp => (
                D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                D3D12_COMPARISON_FUNC_ALWAYS,
                D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
            ),
            SamplerState::LinearWrap => (
                D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                D3D12_COMPARISON_FUNC_ALWAYS,
                D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
            ),
            SamplerState::LinearClampTransparentBorder => (
                D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                D3D12_COMPARISON_FUNC_ALWAYS,
                D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
            ),
            SamplerState::ShadowCompare => (
                D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                shadow_sample_compare(),
                D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
            ),
        };
        D3D12_STATIC_SAMPLER_DESC {
            Filter: filter,
            AddressU: address,
            AddressV: address,
            AddressW: address,
            ComparisonFunc: compare,
            BorderColor: border,
            MinLOD: 0.0,
            MaxLOD: f32::MAX,
            ShaderRegister: self.register,
            RegisterSpace: 0,
            ShaderVisibility: self.visibility.raw(),
            ..Default::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RootDescriptor {
    Cbv,
    Srv,
    Uav,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Param {
    Constants {
        register: u32,
        dwords: u32,
        visibility: Visibility,
    },
    Descriptor {
        kind: RootDescriptor,
        register: u32,
        visibility: Visibility,
    },
    // `len` ranges of the builder's range list from `first`.
    Table {
        first: usize,
        len: usize,
        visibility: Visibility,
    },
}

#[derive(Default)]
pub(in crate::directx) struct RootSig {
    params: Vec<Param>,
    ranges: Vec<D3D12_DESCRIPTOR_RANGE>,
    samplers: Vec<StaticSampler>,
    input_layout: bool,
}

impl RootSig {
    pub(in crate::directx) fn new() -> Self {
        Self::default()
    }

    // The number of root parameters so far, which is the index the next one
    // lands at.
    pub(in crate::directx) fn len(&self) -> u32 {
        self.params.len() as u32
    }

    // `dwords` 32-bit root constants at `b{register}`.
    pub(in crate::directx) fn constant_dwords(
        mut self,
        register: u32,
        dwords: u32,
        visibility: Visibility,
    ) -> Self {
        self.params.push(Param::Constants {
            register,
            dwords,
            visibility,
        });
        self
    }

    // A root-constant block sized to `T` at `b{register}`.
    pub(in crate::directx) fn constants<T: NoUninit>(
        self,
        register: u32,
        visibility: Visibility,
    ) -> Self {
        self.constant_dwords(register, root_dwords::<T>(), visibility)
    }

    fn descriptor(mut self, kind: RootDescriptor, register: u32, visibility: Visibility) -> Self {
        self.params.push(Param::Descriptor {
            kind,
            register,
            visibility,
        });
        self
    }

    // A root CBV at `b{register}`.
    pub(in crate::directx) fn cbv(self, register: u32, visibility: Visibility) -> Self {
        self.descriptor(RootDescriptor::Cbv, register, visibility)
    }

    // A root SRV at `t{register}`.
    pub(in crate::directx) fn srv(self, register: u32, visibility: Visibility) -> Self {
        self.descriptor(RootDescriptor::Srv, register, visibility)
    }

    // A root UAV at `u{register}`.
    pub(in crate::directx) fn uav(self, register: u32, visibility: Visibility) -> Self {
        self.descriptor(RootDescriptor::Uav, register, visibility)
    }

    // A descriptor table over `ranges`, in order.
    pub(in crate::directx) fn table(mut self, ranges: &[Range], visibility: Visibility) -> Self {
        self.params.push(Param::Table {
            first: self.ranges.len(),
            len: ranges.len(),
            visibility,
        });
        self.ranges.extend(ranges.iter().map(|r| r.raw()));
        self
    }

    // A table of `count` SRVs from `t{base}`.
    pub(in crate::directx) fn srv_table(
        self,
        base: u32,
        count: u32,
        visibility: Visibility,
    ) -> Self {
        self.table(&[Range::srv(base, count)], visibility)
    }

    // A table of `count` UAVs from `u{base}`.
    pub(in crate::directx) fn uav_table(
        self,
        base: u32,
        count: u32,
        visibility: Visibility,
    ) -> Self {
        self.table(&[Range::uav(base, count)], visibility)
    }

    // A table of `count` samplers from `s{base}`.
    pub(in crate::directx) fn sampler_table(
        self,
        base: u32,
        count: u32,
        visibility: Visibility,
    ) -> Self {
        self.table(&[Range::sampler(base, count)], visibility)
    }

    pub(in crate::directx) fn static_sampler(
        mut self,
        state: SamplerState,
        register: u32,
        visibility: Visibility,
    ) -> Self {
        self.samplers.push(StaticSampler {
            state,
            register,
            visibility,
        });
        self
    }

    // Let the input assembler feed the vertex stage from an input layout.
    pub(in crate::directx) fn input_layout(mut self) -> Self {
        self.input_layout = true;
        self
    }

    fn flags(&self) -> D3D12_ROOT_SIGNATURE_FLAGS {
        if self.input_layout {
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
        } else {
            D3D12_ROOT_SIGNATURE_FLAG_NONE
        }
    }

    // The D3D12 root parameters, each table pointing into `self`'s range list.
    // Borrowing `self` keeps the ranges in place while the parameters live.
    fn raw_params(&self) -> Vec<D3D12_ROOT_PARAMETER> {
        self.params
            .iter()
            .map(|&param| match param {
                Param::Constants {
                    register,
                    dwords,
                    visibility,
                } => D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: register,
                            RegisterSpace: 0,
                            Num32BitValues: dwords,
                        },
                    },
                    ShaderVisibility: visibility.raw(),
                },
                Param::Descriptor {
                    kind,
                    register,
                    visibility,
                } => D3D12_ROOT_PARAMETER {
                    ParameterType: match kind {
                        RootDescriptor::Cbv => D3D12_ROOT_PARAMETER_TYPE_CBV,
                        RootDescriptor::Srv => D3D12_ROOT_PARAMETER_TYPE_SRV,
                        RootDescriptor::Uav => D3D12_ROOT_PARAMETER_TYPE_UAV,
                    },
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR {
                            ShaderRegister: register,
                            RegisterSpace: 0,
                        },
                    },
                    ShaderVisibility: visibility.raw(),
                },
                Param::Table {
                    first,
                    len,
                    visibility,
                } => D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                            NumDescriptorRanges: len as u32,
                            pDescriptorRanges: self.ranges[first..first + len].as_ptr(),
                        },
                    },
                    ShaderVisibility: visibility.raw(),
                },
            })
            .collect()
    }

    fn raw_samplers(&self) -> Vec<D3D12_STATIC_SAMPLER_DESC> {
        self.samplers.iter().map(|s| s.raw()).collect()
    }

    // Serialize and create the signature; `label` names it in a failure.
    pub(in crate::directx) fn build(
        &self,
        device: &ID3D12Device,
        label: &str,
    ) -> RenderResult<ID3D12RootSignature> {
        let params = self.raw_params();
        let samplers = self.raw_samplers();
        let desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: samplers.len() as u32,
            pStaticSamplers: samplers.as_ptr(),
            Flags: self.flags(),
        };
        serialize_desc_and_create(device, &desc, label)
    }
}

fn serialize_desc_and_create(
    device: &ID3D12Device,
    desc: &D3D12_ROOT_SIGNATURE_DESC,
    label: &str,
) -> RenderResult<ID3D12RootSignature> {
    let mut blob: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
    let mut error: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe {
        windows::Win32::Graphics::Direct3D12::D3D12SerializeRootSignature(
            desc,
            windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1,
            &mut blob,
            Some(&mut error),
        )
    }
    .map_err(|e| {
        let msg = error
            .as_ref()
            .map(|b| {
                // SAFETY: a property query on a live `ID3DBlob`; it only reads.
                let p = unsafe { b.GetBufferPointer() } as *const u8;
                // SAFETY: a property query on a live `ID3DBlob`; it only reads.
                let n = unsafe { b.GetBufferSize() };
                // SAFETY: `ID3DBlob` owns a non-null buffer of `GetBufferSize()` bytes that stays
                // live while `b` is held, and the text is copied out before the blob is released.
                String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(p, n) }).into_owned()
            })
            .unwrap_or_default();
        map_hresult(e.code(), &format!("serialize {label}: {msg}"))
    })?;

    let b = blob.ok_or_else(|| RenderError::Other(format!("{label}: no blob after serialize")))?;
    // SAFETY: a property query on a live `ID3DBlob`; it only reads.
    let ptr = unsafe { b.GetBufferPointer() };
    // SAFETY: a property query on a live `ID3DBlob`; it only reads.
    let len = unsafe { b.GetBufferSize() };
    // SAFETY: `ID3DBlob` owns a non-null buffer of `GetBufferSize()` bytes that stays live while
    // `b` is held, and `b` outlives the `CreateRootSignature` call that reads the slice.
    let sig_bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };

    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe { device.CreateRootSignature(0, sig_bytes) }
        .map_err(|e| map_hresult(e.code(), &format!("create {label}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(bytemuck::NoUninit, Clone, Copy)]
    #[repr(C)]
    struct ThreeDwords {
        a: u32,
        b: f32,
        c: u32,
    }

    fn register_of(param: &D3D12_ROOT_PARAMETER) -> u32 {
        match param.ParameterType {
            // SAFETY: a 32-bit-constants parameter carries `Constants`.
            D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS => unsafe {
                param.Anonymous.Constants.ShaderRegister
            },
            // SAFETY: a root descriptor parameter carries `Descriptor`.
            _ => unsafe { param.Anonymous.Descriptor.ShaderRegister },
        }
    }

    #[test]
    fn parameters_keep_their_order_kind_register_and_visibility() {
        let sig = RootSig::new()
            .constants::<ThreeDwords>(0, Visibility::All)
            .cbv(1, Visibility::Pixel)
            .srv(4, Visibility::Vertex)
            .uav(2, Visibility::All)
            .srv_table(3, 2, Visibility::Pixel);
        let params = sig.raw_params();
        let kinds: Vec<_> = params.iter().map(|p| p.ParameterType).collect();
        assert_eq!(
            kinds,
            [
                D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                D3D12_ROOT_PARAMETER_TYPE_CBV,
                D3D12_ROOT_PARAMETER_TYPE_SRV,
                D3D12_ROOT_PARAMETER_TYPE_UAV,
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            ]
        );
        let visibility: Vec<_> = params.iter().map(|p| p.ShaderVisibility).collect();
        assert_eq!(
            visibility,
            [
                D3D12_SHADER_VISIBILITY_ALL,
                D3D12_SHADER_VISIBILITY_PIXEL,
                D3D12_SHADER_VISIBILITY_VERTEX,
                D3D12_SHADER_VISIBILITY_ALL,
                D3D12_SHADER_VISIBILITY_PIXEL,
            ]
        );
        let registers: Vec<_> = params[..4].iter().map(register_of).collect();
        assert_eq!(registers, [0, 1, 4, 2]);
        // SAFETY: parameter 0 is the 32-bit-constants block.
        let constants = unsafe { params[0].Anonymous.Constants };
        assert_eq!(constants.Num32BitValues, 3);
        assert_eq!(sig.len(), 5);
    }

    #[test]
    fn tables_point_at_their_own_ranges_in_order() {
        let sig = RootSig::new()
            .table(&[Range::srv(0, 1), Range::srv(5, 2)], Visibility::Pixel)
            .table(&[Range::bindless_srv(1)], Visibility::Pixel)
            .sampler_table(1, 2, Visibility::Pixel)
            .uav_table(0, 1, Visibility::All);
        let params = sig.raw_params();
        // SAFETY: every parameter here is a descriptor table.
        let tables: Vec<_> = params
            .iter()
            .map(|p| unsafe { p.Anonymous.DescriptorTable })
            .collect();
        let expected_first = [0usize, 2, 3, 4];
        let expected_len = [2u32, 1, 1, 1];
        for (i, table) in tables.iter().enumerate() {
            assert_eq!(table.NumDescriptorRanges, expected_len[i], "table {i}");
            assert_eq!(
                table.pDescriptorRanges, &sig.ranges[expected_first[i]] as *const _,
                "table {i}"
            );
        }
        let r = &sig.ranges;
        assert_eq!(
            (r[1].RangeType, r[1].BaseShaderRegister, r[1].NumDescriptors),
            (D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 5, 2)
        );
        assert_eq!(
            r[1].OffsetInDescriptorsFromTableStart,
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
        );
        assert_eq!(r[2].RegisterSpace, 1);
        assert_eq!(r[2].NumDescriptors, u32::MAX);
        assert_eq!(r[2].OffsetInDescriptorsFromTableStart, 0);
        assert_eq!(r[3].RangeType, D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER);
        assert_eq!(r[4].RangeType, D3D12_DESCRIPTOR_RANGE_TYPE_UAV);
        assert_eq!(r.iter().filter(|r| r.RegisterSpace == 0).count(), 4);
    }

    #[test]
    fn static_samplers_carry_their_state() {
        let sig = RootSig::new()
            .static_sampler(SamplerState::LinearClamp, 0, Visibility::Pixel)
            .static_sampler(SamplerState::LinearWrap, 1, Visibility::Pixel)
            .static_sampler(SamplerState::ShadowCompare, 2, Visibility::All)
            .static_sampler(
                SamplerState::LinearClampTransparentBorder,
                3,
                Visibility::Pixel,
            );
        let s = sig.raw_samplers();
        assert_eq!(s[0].Filter, D3D12_FILTER_MIN_MAG_MIP_LINEAR);
        assert_eq!(s[0].AddressU, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
        assert_eq!(s[0].BorderColor, D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK);
        assert_eq!(s[0].MaxLOD, f32::MAX);
        assert_eq!(s[1].AddressW, D3D12_TEXTURE_ADDRESS_MODE_WRAP);
        assert_eq!(s[1].ShaderRegister, 1);
        assert_eq!(s[2].ComparisonFunc, D3D12_COMPARISON_FUNC_GREATER_EQUAL);
        assert_eq!(
            s[2].Filter,
            D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT
        );
        assert_eq!(s[2].ShaderVisibility, D3D12_SHADER_VISIBILITY_ALL);
        assert_eq!(
            s[3].BorderColor,
            D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK
        );
        assert_eq!(s[3].AddressV, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
        assert!(s.iter().all(|s| s.RegisterSpace == 0));
    }

    #[test]
    fn the_input_layout_flag_is_opt_in() {
        assert_eq!(RootSig::new().flags(), D3D12_ROOT_SIGNATURE_FLAG_NONE);
        assert_eq!(
            RootSig::new().input_layout().flags(),
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
        );
    }
}
