//! Pipeline-state builders. [`GraphicsPso`] describes a graphics pipeline as
//! data -- stages, color targets with their blend, depth, raster and sample
//! count -- over the fixed state every engine pass shares (triangle lists, all
//! samples enabled, no stencil), and creates it through the pipeline library.
//! [`compute_pso`] is the compute counterpart.

use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::device::PostBlend;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::com;
use crate::directx::error::map_pso_hresult;

// How a color target combines the fragment with what it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::directx) enum Blend {
    // Overwrite.
    Opaque,
    // `dst + src`.
    Additive,
    // `src + (1 - src.a) * dst`.
    PremultipliedOver,
    // `src.a * src + (1 - src.a) * dst`, alpha included.
    AlphaOver,
}

impl From<PostBlend> for Blend {
    fn from(blend: PostBlend) -> Self {
        match blend {
            PostBlend::Replace => Blend::Opaque,
            PostBlend::Additive => Blend::Additive,
            PostBlend::PremultipliedOver => Blend::PremultipliedOver,
        }
    }
}

impl Blend {
    fn raw(self) -> D3D12_RENDER_TARGET_BLEND_DESC {
        let mask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        let (src, dst) = match self {
            Blend::Opaque => {
                return D3D12_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: false.into(),
                    RenderTargetWriteMask: mask,
                    ..Default::default()
                };
            }
            Blend::Additive => (D3D12_BLEND_ONE, D3D12_BLEND_ONE),
            Blend::PremultipliedOver => (D3D12_BLEND_ONE, D3D12_BLEND_INV_SRC_ALPHA),
            Blend::AlphaOver => (D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA),
        };
        D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            SrcBlend: src,
            DestBlend: dst,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: src,
            DestBlendAlpha: dst,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            RenderTargetWriteMask: mask,
            ..Default::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::directx) enum CompareOp {
    Less,
    LessEqual,
    Always,
}

impl CompareOp {
    fn raw(self) -> D3D12_COMPARISON_FUNC {
        match self {
            CompareOp::Less => D3D12_COMPARISON_FUNC_LESS,
            CompareOp::LessEqual => D3D12_COMPARISON_FUNC_LESS_EQUAL,
            CompareOp::Always => D3D12_COMPARISON_FUNC_ALWAYS,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::directx) enum Depth {
    Off,
    Test { compare: CompareOp, write: bool },
}

impl Depth {
    // The opaque-geometry test: nearer fragments pass and write.
    pub(in crate::directx) const LESS_WRITE: Depth = Depth::Test {
        compare: CompareOp::Less,
        write: true,
    };

    fn raw(self) -> D3D12_DEPTH_STENCIL_DESC {
        let (enable, compare, write) = match self {
            Depth::Off => (false, CompareOp::Always, false),
            Depth::Test { compare, write } => (true, compare, write),
        };
        D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: enable.into(),
            DepthWriteMask: if write {
                D3D12_DEPTH_WRITE_MASK_ALL
            } else {
                D3D12_DEPTH_WRITE_MASK_ZERO
            },
            DepthFunc: compare.raw(),
            StencilEnable: false.into(),
            ..Default::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::directx) enum Cull {
    None,
    Front,
}

impl Cull {
    fn raw(self) -> D3D12_CULL_MODE {
        match self {
            Cull::None => D3D12_CULL_MODE_NONE,
            Cull::Front => D3D12_CULL_MODE_FRONT,
        }
    }
}

// Slope-scaled depth bias, in the rasterizer's own units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(in crate::directx) struct DepthBias {
    pub constant: i32,
    pub clamp: f32,
    pub slope: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::directx) struct Raster {
    pub cull: Cull,
    pub front_ccw: bool,
    pub depth_clip: bool,
    pub wireframe: bool,
    pub multisample: bool,
    pub bias: DepthBias,
}

impl Default for Raster {
    fn default() -> Self {
        Self {
            cull: Cull::None,
            front_ccw: true,
            depth_clip: true,
            wireframe: false,
            multisample: false,
            bias: DepthBias::default(),
        }
    }
}

impl Raster {
    fn raw(self) -> D3D12_RASTERIZER_DESC {
        D3D12_RASTERIZER_DESC {
            FillMode: if self.wireframe {
                D3D12_FILL_MODE_WIREFRAME
            } else {
                D3D12_FILL_MODE_SOLID
            },
            CullMode: self.cull.raw(),
            FrontCounterClockwise: self.front_ccw.into(),
            DepthBias: self.bias.constant,
            DepthBiasClamp: self.bias.clamp,
            SlopeScaledDepthBias: self.bias.slope,
            DepthClipEnable: self.depth_clip.into(),
            MultisampleEnable: self.multisample.into(),
            ..Default::default()
        }
    }
}

// Most color targets one pipeline writes.
const MAX_TARGETS: usize = 8;

// The RTV formats and blend state of `targets`, in order. Blending is
// independent only when the targets disagree.
fn color_targets(
    targets: &[(DXGI_FORMAT, Blend)],
) -> ([DXGI_FORMAT; MAX_TARGETS], D3D12_BLEND_DESC) {
    let mut formats = [DXGI_FORMAT_UNKNOWN; MAX_TARGETS];
    let mut blends = [D3D12_RENDER_TARGET_BLEND_DESC::default(); MAX_TARGETS];
    for (i, &(format, blend)) in targets.iter().take(MAX_TARGETS).enumerate() {
        formats[i] = format;
        blends[i] = blend.raw();
    }
    let independent = targets.windows(2).any(|w| w[0].1 != w[1].1);
    let desc = D3D12_BLEND_DESC {
        IndependentBlendEnable: independent.into(),
        RenderTarget: blends,
        ..Default::default()
    };
    (formats, desc)
}

pub(in crate::directx) struct GraphicsPso<'a> {
    root_sig: &'a ID3D12RootSignature,
    vs: &'a [u8],
    // Empty for a depth-only pipeline.
    ps: &'a [u8],
    input_layout: &'a [D3D12_INPUT_ELEMENT_DESC],
    targets: Vec<(DXGI_FORMAT, Blend)>,
    depth_format: DXGI_FORMAT,
    depth: Depth,
    samples: u32,
    raster: Raster,
}

impl<'a> GraphicsPso<'a> {
    // No targets, no depth, one sample, the default raster state.
    pub(in crate::directx) fn new(
        root_sig: &'a ID3D12RootSignature,
        vs: &'a [u8],
        ps: &'a [u8],
    ) -> Self {
        Self {
            root_sig,
            vs,
            ps,
            input_layout: &[],
            targets: Vec::new(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            depth: Depth::Off,
            samples: 1,
            raster: Raster::default(),
        }
    }

    // A vertex-buffer-less fullscreen triangle into one single-sample target.
    pub(in crate::directx) fn fullscreen(
        root_sig: &'a ID3D12RootSignature,
        vs: &'a [u8],
        ps: &'a [u8],
        format: DXGI_FORMAT,
        blend: Blend,
    ) -> Self {
        Self::new(root_sig, vs, ps).target(format, blend)
    }

    pub(in crate::directx) fn input_layout(
        mut self,
        layout: &'a [D3D12_INPUT_ELEMENT_DESC],
    ) -> Self {
        self.input_layout = layout;
        self
    }

    // Append a color target.
    pub(in crate::directx) fn target(mut self, format: DXGI_FORMAT, blend: Blend) -> Self {
        self.targets.push((format, blend));
        self
    }

    pub(in crate::directx) fn depth(mut self, format: DXGI_FORMAT, depth: Depth) -> Self {
        self.depth_format = format;
        self.depth = depth;
        self
    }

    pub(in crate::directx) fn samples(mut self, count: u32) -> Self {
        self.samples = count;
        self
    }

    pub(in crate::directx) fn raster(mut self, raster: Raster) -> Self {
        self.raster = raster;
        self
    }

    // The D3D12 desc. Its pointers borrow `self`'s root signature, bytecode and
    // input layout, so it must not outlive them.
    fn raw(&self) -> D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        let bytecode = |code: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: if code.is_empty() {
                std::ptr::null()
            } else {
                code.as_ptr().cast()
            },
            BytecodeLength: code.len(),
        };
        let (formats, blend) = color_targets(&self.targets);
        D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: com::borrowed(self.root_sig),
            VS: bytecode(self.vs),
            PS: bytecode(self.ps),
            BlendState: blend,
            SampleMask: u32::MAX,
            RasterizerState: self.raster.raw(),
            DepthStencilState: self.depth.raw(),
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: if self.input_layout.is_empty() {
                    std::ptr::null()
                } else {
                    self.input_layout.as_ptr()
                },
                NumElements: self.input_layout.len() as u32,
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: self.targets.len().min(MAX_TARGETS) as u32,
            RTVFormats: formats,
            DSVFormat: self.depth_format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: self.samples,
                Quality: 0,
            },
            ..Default::default()
        }
    }

    // Create the pipeline through the pipeline library; `label` names it in a
    // failure.
    pub(in crate::directx) fn build(
        &self,
        device: &ID3D12Device,
        label: &str,
    ) -> RenderResult<ID3D12PipelineState> {
        let desc = self.raw();
        // SAFETY: `desc` borrows the root signature, bytecode and input layout `self` holds, all of
        // which outlive this synchronous call.
        unsafe { crate::directx::pso_library::create_graphics(device, &desc) }
            .map_err(|e| map_pso_hresult(e.code(), &format!("create {label} PSO")))
    }
}

// A compute pipeline running `cs` under `root_sig`; `label` names it in a
// failure.
pub(in crate::directx) fn compute_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    cs: &[u8],
    label: &str,
) -> RenderResult<ID3D12PipelineState> {
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: com::borrowed(root_sig),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: cs.as_ptr().cast(),
            BytecodeLength: cs.len(),
        },
        ..Default::default()
    };
    // SAFETY: `desc` borrows the root signature and kernel bytecode, both of which outlive this
    // synchronous call.
    unsafe { crate::directx::pso_library::create_compute(device, &desc) }
        .map_err(|e| map_pso_hresult(e.code(), &format!("create {label} PSO")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blends_set_their_factors_and_opaque_disables_blending() {
        let opaque = Blend::Opaque.raw();
        assert!(!opaque.BlendEnable.as_bool());
        assert_eq!(
            opaque.SrcBlend,
            D3D12_RENDER_TARGET_BLEND_DESC::default().SrcBlend
        );
        let over = Blend::AlphaOver.raw();
        assert!(over.BlendEnable.as_bool());
        assert_eq!(
            (
                over.SrcBlend,
                over.DestBlend,
                over.SrcBlendAlpha,
                over.DestBlendAlpha
            ),
            (
                D3D12_BLEND_SRC_ALPHA,
                D3D12_BLEND_INV_SRC_ALPHA,
                D3D12_BLEND_SRC_ALPHA,
                D3D12_BLEND_INV_SRC_ALPHA
            )
        );
        let pre = Blend::from(PostBlend::PremultipliedOver).raw();
        assert_eq!(
            (pre.SrcBlend, pre.DestBlend),
            (D3D12_BLEND_ONE, D3D12_BLEND_INV_SRC_ALPHA)
        );
        let add = Blend::from(PostBlend::Additive).raw();
        assert_eq!(
            (add.SrcBlend, add.DestBlend),
            (D3D12_BLEND_ONE, D3D12_BLEND_ONE)
        );
        for b in [
            Blend::Opaque,
            Blend::Additive,
            Blend::PremultipliedOver,
            Blend::AlphaOver,
        ] {
            assert_eq!(
                b.raw().RenderTargetWriteMask,
                D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8
            );
        }
    }

    #[test]
    fn color_targets_fill_in_order_and_blend_independently_only_when_they_differ() {
        let (formats, blend) = color_targets(&[
            (DXGI_FORMAT_R16G16B16A16_FLOAT, Blend::Opaque),
            (DXGI_FORMAT_R8_UNORM, Blend::Opaque),
        ]);
        assert_eq!(
            &formats[..3],
            &[
                DXGI_FORMAT_R16G16B16A16_FLOAT,
                DXGI_FORMAT_R8_UNORM,
                DXGI_FORMAT_UNKNOWN
            ]
        );
        assert!(!blend.IndependentBlendEnable.as_bool());
        assert_eq!(
            blend.RenderTarget[1].RenderTargetWriteMask,
            D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8
        );
        assert_eq!(blend.RenderTarget[2].RenderTargetWriteMask, 0);
        let (_, mixed) = color_targets(&[
            (DXGI_FORMAT_R16G16B16A16_FLOAT, Blend::Opaque),
            (DXGI_FORMAT_R16G16B16A16_FLOAT, Blend::Additive),
        ]);
        assert!(mixed.IndependentBlendEnable.as_bool());
        assert!(mixed.RenderTarget[1].BlendEnable.as_bool());
    }

    #[test]
    fn depth_modes_map_to_enable_write_and_compare() {
        let off = Depth::Off.raw();
        assert!(!off.DepthEnable.as_bool());
        assert_eq!(off.DepthWriteMask, D3D12_DEPTH_WRITE_MASK_ZERO);
        let less = Depth::LESS_WRITE.raw();
        assert!(less.DepthEnable.as_bool());
        assert_eq!(less.DepthWriteMask, D3D12_DEPTH_WRITE_MASK_ALL);
        assert_eq!(less.DepthFunc, D3D12_COMPARISON_FUNC_LESS);
        let read = Depth::Test {
            compare: CompareOp::LessEqual,
            write: false,
        }
        .raw();
        assert_eq!(read.DepthWriteMask, D3D12_DEPTH_WRITE_MASK_ZERO);
        assert_eq!(read.DepthFunc, D3D12_COMPARISON_FUNC_LESS_EQUAL);
        assert!(!read.StencilEnable.as_bool());
    }

    #[test]
    fn the_default_raster_is_solid_double_sided_ccw_and_clipped() {
        let r = Raster::default().raw();
        assert_eq!(r.FillMode, D3D12_FILL_MODE_SOLID);
        assert_eq!(r.CullMode, D3D12_CULL_MODE_NONE);
        assert!(r.FrontCounterClockwise.as_bool());
        assert!(r.DepthClipEnable.as_bool());
        assert!(!r.MultisampleEnable.as_bool());
        assert_eq!(r.DepthBias, 0);
        let biased = Raster {
            bias: DepthBias {
                constant: 3,
                clamp: 0.5,
                slope: 2.0,
            },
            wireframe: true,
            cull: Cull::Front,
            ..Raster::default()
        }
        .raw();
        assert_eq!(biased.FillMode, D3D12_FILL_MODE_WIREFRAME);
        assert_eq!(biased.CullMode, D3D12_CULL_MODE_FRONT);
        assert_eq!(
            (
                biased.DepthBias,
                biased.DepthBiasClamp,
                biased.SlopeScaledDepthBias
            ),
            (3, 0.5, 2.0)
        );
    }
}
