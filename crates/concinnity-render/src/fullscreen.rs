//! Backend-agnostic fullscreen-pass encoder seam, the first pilot of a hardware
//! abstraction layer over the three render backends. The bloom
//! prefilter -> downsample -> upsample chain is structurally identical on every
//! backend, so its orchestration lives here once and each backend implements
//! `BloomEncoder` to bind + draw one sub-pass in its own command stream.
//!
//! Two associated types absorb the only real divergence, so the trait names no
//! backend types: `Rec` hides the per-backend command recorder, and `Args`
//! carries the per-invocation binding context (DirectX passes the scene-colour
//! SRV its prefilter samples; Vulkan threads the frame-in-flight index that
//! selects its per-frame framebuffers + descriptor sets). Everything else each
//! impl reads from `&self`, consistent with the read-only parallel-encode
//! contract.
//!
//! Implemented by DirectX + Vulkan. Metal keeps its hand-rolled `encode_bloom`,
//! already factored through its own `fullscreen_pass`, so this seam is unused
//! (dead code) on a Metal build.

use crate::render_types::TextDrawCall;

/// Convert a `TextDrawCall.clip_rect` (a rectangle `[x, y, w, h]` in overlay
/// units, already mapped through the overlay transform by
/// `gfx::text::band_to_window`) into an integer scissor rect `(x, y, w, h)` in
/// attachment pixels, clamped to the attachment's bounds. Returns `None` when the
/// clamped rectangle is empty (a row scrolled fully out of its band), so the
/// caller skips the draw entirely.
///
/// `ui` is the overlay's logical size (see `RenderBackend::logical_size`) and
/// `attach` the pixel size of the target the text pass writes. The two are equal
/// wherever a window's logical units are pixels (Windows, unscaled X11), leaving
/// a pure clamp; on a hi-DPI surface (macOS retina, scaled Wayland) the
/// attachment is larger by the backing scale and the rect scales up with it. A
/// zero logical dimension (minimised / mid-resize) falls back to a 1.0 scale
/// rather than dividing by zero.
pub fn clip_rect_to_scissor(
    clip: [f32; 4],
    ui: (f32, f32),
    attach: (u32, u32),
) -> Option<(i32, i32, u32, u32)> {
    let aw = attach.0 as f32;
    let ah = attach.1 as f32;
    let sx = if ui.0 > 0.0 { aw / ui.0 } else { 1.0 };
    let sy = if ui.1 > 0.0 { ah / ui.1 } else { 1.0 };
    let x0 = (clip[0] * sx).floor().clamp(0.0, aw);
    let y0 = (clip[1] * sy).floor().clamp(0.0, ah);
    let x1 = ((clip[0] + clip[2]) * sx).ceil().clamp(0.0, aw);
    let y1 = ((clip[1] + clip[3]) * sy).ceil().clamp(0.0, ah);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some((x0 as i32, y0 as i32, (x1 - x0) as u32, (y1 - y0) as u32))
}

/// Round `offset` up to the next multiple of `align` (a power of two).
pub fn align_up(offset: u64, align: u64) -> u64 {
    (offset + align - 1) & !(align - 1)
}

/// Total bytes a frame's text geometry occupies in a backend's per-frame upload
/// buffer, once each label's vertex and index blocks start on an `align`-byte
/// boundary. Every sub-allocation aligns its start up and a prior aligned start
/// plus an aligned size stays aligned, so this sum is an exact upper bound on the
/// buffer cursor after all of a frame's blocks are appended: a slot reserved to
/// it can never overflow mid-frame.
///
/// `align` is per backend: the alignment its buffer bindings require of a
/// sub-range's offset.
pub fn text_upload_bytes(text_calls: &[TextDrawCall], align: u64) -> u64 {
    text_calls
        .iter()
        .map(|c| {
            let v = std::mem::size_of_val(c.vertices.as_slice()) as u64;
            let i = std::mem::size_of_val(c.indices.as_slice()) as u64;
            align_up(v, align) + align_up(i, align)
        })
        .sum()
}

/// Per-backend hooks the shared bloom driver encodes through.
pub trait BloomEncoder {
    /// Per-backend command recorder (DX `ID3D12GraphicsCommandList`, VK `vk::CommandBuffer`).
    type Rec;
    /// Per-invocation binding context (DX scene-colour SRV handle, VK frame index).
    type Args;

    /// Number of bloom mips; zero means bloom is off and the driver no-ops.
    fn bloom_mip_count(&self) -> usize;
    /// One-time per-encode preamble (DX root signature / heap / IA state; VK no-op).
    fn begin_bloom(&self, rec: &Self::Rec, args: &Self::Args);
    /// Prefilter: scene colour -> mip 0 (soft-knee threshold + Karis average).
    fn bloom_prefilter(&self, rec: &Self::Rec, args: &Self::Args);
    /// Downsample: mip `dst - 1` -> mip `dst`.
    fn bloom_downsample(&self, rec: &Self::Rec, args: &Self::Args, dst: usize);
    /// Upsample: mip `dst + 1` -> mip `dst`, additively blended.
    fn bloom_upsample(&self, rec: &Self::Rec, args: &Self::Args, dst: usize);
}

/// The bloom chain orchestration, previously hand-duplicated in each backend's
/// `encode_bloom`. On return, mip 0 holds the accumulated glow the composite pass
/// samples.
pub fn encode_bloom_chain<E: BloomEncoder>(enc: &E, rec: &E::Rec, args: E::Args) {
    let n = enc.bloom_mip_count();
    if n == 0 {
        return;
    }
    enc.begin_bloom(rec, &args);
    // Prefilter: scene -> mip 0.
    enc.bloom_prefilter(rec, &args);
    // Downsample chain: mip i-1 -> mip i.
    for dst in 1..n {
        enc.bloom_downsample(rec, &args, dst);
    }
    // Upsample chain: mip i+1 -> mip i, walking back down to mip 0.
    for dst in (0..n - 1).rev() {
        enc.bloom_upsample(rec, &args, dst);
    }
}

/// The composite pass: tonemap (+ optional LUT grade) the post-stack scene onto
/// the swapchain image, then layer the text overlay on top in the same pass. Its
/// begin -> composite-draw -> text-loop -> end shape is identical on every
/// backend; the swapchain target lifecycle, the descriptor binding, and the
/// text-geometry uploads stay backend-specific behind the trait. `Args`
/// carries the per-frame binding context each backend needs (DX: the swapchain
/// back-buffer + its RTV, the scene SRV, the window size, the frame slot; VK: the
/// acquired image index + the frame slot).
///
/// Every backend uploads a frame's text geometry into one persistent buffer per
/// frame-in-flight slot, reserved up front with [`text_upload_bytes`] and
/// appended to per call, and binds sub-ranges of it: no GPU buffer is created
/// per label per frame anywhere. DX and VK append inside `text_draw`; Metal
/// (which drives its own composite loop rather than this trait) writes the whole
/// frame's geometry into its slot before the render graph runs.
pub trait CompositeEncoder {
    /// Per-backend command recorder (DX `ID3D12GraphicsCommandList`, VK `vk::CommandBuffer`).
    type Rec;
    /// Per-invocation binding context (see the trait doc).
    type Args;

    /// Begin the pass: target the swapchain image (DX transitions it to
    /// RENDER_TARGET + binds the RTV; VK begins the composite render pass) and set
    /// the full-window viewport / scissor.
    fn begin_composite(&self, rec: &Self::Rec, args: &Self::Args);
    /// The fullscreen tonemap draw: bind the composite pipeline + its inputs
    /// (scene, bloom, LUT) + push constants, draw the fullscreen triangle.
    fn composite_draw(&self, rec: &Self::Rec, args: &Self::Args);
    /// Bind the text pipeline + any one-time text state. Returns false when text
    /// is inert (no pipeline or no atlases), so the driver skips the call loop.
    fn begin_text(&self, rec: &Self::Rec, args: &Self::Args) -> bool;
    /// Encode one text draw call: append its vertex/index geometry to this frame
    /// slot's persistent upload buffer, bind the atlas plus the two sub-ranges,
    /// and draw.
    fn text_draw(
        &self,
        rec: &Self::Rec,
        args: &Self::Args,
        call: &TextDrawCall,
    ) -> Result<(), String>;
    /// End the pass: DX transitions the back-buffer back to PRESENT; VK ends the
    /// render pass.
    fn end_composite(&self, rec: &Self::Rec, args: &Self::Args);
}

/// The composite + text orchestration, previously hand-duplicated in each
/// backend's `encode_composite_and_text`. An error mid-text propagates without
/// closing the pass, matching the prior DX/VK behaviour (the frame fails either
/// way: the target is just left mis-stated). This is unused on Metal, where a
/// render encoder must be `endEncoding`-ed before the command buffer commits:
/// skipping `end_composite` on a text error would crash at commit, so Metal's
/// `encode_composite_and_text` ends the encoder on any `?` with a `ScopedEncoder`
/// RAII guard instead.
pub fn encode_composite_chain<E: CompositeEncoder>(
    enc: &E,
    rec: &E::Rec,
    args: &E::Args,
    text_calls: &[TextDrawCall],
) -> Result<(), String> {
    enc.begin_composite(rec, args);
    enc.composite_draw(rec, args);
    if !text_calls.is_empty() && enc.begin_text(rec, args) {
        for call in text_calls {
            enc.text_draw(rec, args, call)?;
        }
    }
    enc.end_composite(rec, args);
    Ok(())
}

/// A single-draw fullscreen post pass (SSR resolve, TAA resolve, ...): target a
/// render target, bind a pipeline + inputs, draw one fullscreen triangle, restore.
/// Unlike the bloom + composite chains (whose drivers hold a mip / text loop), a
/// fullscreen pass has no loop, so the driver is a fixed begin -> draw -> end. The
/// value is the shared per-backend lifecycle factored behind begin/end (DX: the
/// PSR<->RENDER_TARGET barrier bracket + render-target bind; VK: the render-pass
/// bracket), reused across every such pass instead of re-pasted per pass.
///
/// The inert-pass guard lives at each backend's call site: it resolves the pass's
/// resources (returning early if a required one is absent) BEFORE constructing the
/// encoder, so the driver always runs all three steps over a fully-resolved pass
/// and can never leave a render pass / barrier half-open. There is no `Args`: each
/// backend's encoder is a small struct holding the already-resolved references +
/// per-call scalars, so the trait names no backend types (like `BloomEncoder`).
///
/// Implemented by DirectX + Vulkan. Metal keeps its own `fullscreen_pass` helper,
/// which already factors this begin/draw/end skeleton, so this seam is unused
/// (dead code) on a Metal build.
pub trait FullscreenPass {
    /// Per-backend command recorder (DX `ID3D12GraphicsCommandList`, VK `vk::CommandBuffer`).
    type Rec;

    /// Begin: bind the target render target + set the full-resolution viewport /
    /// scissor. DX transitions the target PIXEL_SHADER_RESOURCE -> RENDER_TARGET,
    /// binds its RTV + the SRV heap; VK begins the pass's render pass.
    fn begin(&self, rec: &Self::Rec);
    /// Bind the pipeline + inputs + per-frame params and draw the fullscreen
    /// triangle (3 vertices; the vertex shader builds the triangle from the id).
    fn draw(&self, rec: &Self::Rec);
    /// End: DX transitions the target back to PIXEL_SHADER_RESOURCE; VK ends the
    /// render pass.
    fn end(&self, rec: &Self::Rec);
}

/// The fullscreen-pass orchestration. Trivial by design (a single draw), but kept
/// as a driver so every fullscreen post pass shares one begin -> draw -> end
/// contract across backends, matching `encode_bloom_chain` / `encode_composite_chain`.
pub fn encode_fullscreen<E: FullscreenPass>(enc: &E, rec: &E::Rec) {
    enc.begin(rec);
    enc.draw(rec);
    enc.end(rec);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_types::TextDrawCall;
    use std::cell::RefCell;

    #[test]
    fn clip_inside_attachment_passes_through() {
        // Logical units are attachment pixels (Windows, unscaled X11): 1:1.
        assert_eq!(
            clip_rect_to_scissor([100.0, 50.0, 300.0, 200.0], (1280.0, 720.0), (1280, 720)),
            Some((100, 50, 300, 200))
        );
    }

    #[test]
    fn clip_scales_from_logical_units_to_a_hi_dpi_attachment() {
        // A 2x backing scale (macOS retina, scaled Wayland): the band covers the
        // same fraction of an attachment twice the logical size.
        assert_eq!(
            clip_rect_to_scissor([100.0, 50.0, 300.0, 200.0], (1024.0, 768.0), (2048, 1536)),
            Some((200, 100, 600, 400))
        );
        // A non-integer scale still lands on whole pixels, rounded outward so a
        // band never crops the glyphs it should show.
        assert_eq!(
            clip_rect_to_scissor([10.0, 10.0, 100.0, 100.0], (1000.0, 1000.0), (1500, 1500)),
            Some((15, 15, 150, 150))
        );
    }

    #[test]
    fn clip_is_clamped_to_attachment_bounds() {
        // A band hanging off the right / bottom edge is clamped to the target.
        assert_eq!(
            clip_rect_to_scissor([1200.0, 700.0, 400.0, 400.0], (1280.0, 720.0), (1280, 720)),
            Some((1200, 700, 80, 20))
        );
        // A negative origin is clamped to zero, shrinking the width/height.
        assert_eq!(
            clip_rect_to_scissor([-40.0, -10.0, 100.0, 100.0], (1280.0, 720.0), (1280, 720)),
            Some((0, 0, 60, 90))
        );
        // The clamp is against the attachment, after scaling.
        assert_eq!(
            clip_rect_to_scissor([600.0, 350.0, 200.0, 200.0], (640.0, 360.0), (1280, 720)),
            Some((1200, 700, 80, 20))
        );
    }

    #[test]
    fn fully_offscreen_clip_is_skipped() {
        // A band entirely past the attachment yields no scissor (skip the draw).
        assert_eq!(
            clip_rect_to_scissor([2000.0, 50.0, 100.0, 100.0], (1280.0, 720.0), (1280, 720)),
            None
        );
        // A zero-area band is also skipped.
        assert_eq!(
            clip_rect_to_scissor([10.0, 10.0, 0.0, 50.0], (1280.0, 720.0), (1280, 720)),
            None
        );
    }

    #[test]
    fn a_zero_logical_size_falls_back_to_an_unscaled_clip() {
        // Minimised / mid-resize: no divide by zero, and the rect is still
        // clamped into the attachment.
        assert_eq!(
            clip_rect_to_scissor([10.0, 20.0, 100.0, 100.0], (0.0, 0.0), (1280, 720)),
            Some((10, 20, 100, 100))
        );
    }

    // A text-only draw call for the composite driver: the drivers never inspect
    // its contents, so the geometry is empty.
    fn text_call() -> TextDrawCall {
        TextDrawCall {
            vertices: Vec::new(),
            indices: Vec::new(),
            atlas_slot: 0,
            clip_rect: None,
            layer: 0,
        }
    }

    // A call carrying `glyphs` quads: 4 vertices + 6 indices each, the shape
    // `gfx::text::build_text_calls` emits.
    fn glyph_call(glyphs: usize) -> TextDrawCall {
        TextDrawCall {
            vertices: vec![
                crate::render_types::TextVertex {
                    pos: [0.0; 2],
                    uv: [0.0; 2],
                    color: [0.0; 3],
                    mode: 0.0,
                };
                glyphs * 4
            ],
            indices: vec![0u16; glyphs * 6],
            atlas_slot: 0,
            clip_rect: None,
            layer: 0,
        }
    }

    #[test]
    fn align_up_rounds_to_multiple() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(257, 256), 512);
    }

    #[test]
    fn text_upload_bytes_is_zero_without_calls() {
        assert_eq!(text_upload_bytes(&[], 256), 0);
        // An empty call still contributes nothing: both blocks are zero bytes.
        assert_eq!(text_upload_bytes(&[text_call()], 256), 0);
    }

    #[test]
    fn text_upload_bytes_aligns_each_block() {
        // One glyph: 4 * 32 B of vertices (already a multiple of 16) and 12 B of
        // indices (rounded up).
        assert_eq!(text_upload_bytes(&[glyph_call(1)], 16), 128 + 16);
        assert_eq!(text_upload_bytes(&[glyph_call(1)], 256), 256 + 256);
    }

    // The reserved size must be an upper bound on the cursor after a run of
    // appends (an aligned start plus an aligned size stays aligned), so a slot
    // reserved to it can never overflow mid-frame.
    #[test]
    fn text_upload_bytes_bounds_a_simulated_cursor() {
        let calls = [glyph_call(3), glyph_call(1), glyph_call(17), glyph_call(0)];
        for align in [16u64, 256] {
            let total = text_upload_bytes(&calls, align);
            let mut cursor = 0u64;
            for c in &calls {
                for block in [
                    std::mem::size_of_val(c.vertices.as_slice()) as u64,
                    std::mem::size_of_val(c.indices.as_slice()) as u64,
                ] {
                    cursor = align_up(cursor, align) + block;
                    assert!(cursor <= total, "cursor {cursor} exceeded reserved {total}");
                }
            }
        }
    }

    // A mock bloom encoder recording each sub-pass in call order. The trait's
    // associated types name no backend types, so both are `()`.
    struct MockBloom {
        mips: usize,
        log: RefCell<Vec<String>>,
    }

    impl BloomEncoder for MockBloom {
        type Rec = ();
        type Args = ();

        fn bloom_mip_count(&self) -> usize {
            self.mips
        }
        fn begin_bloom(&self, _rec: &(), _args: &()) {
            self.log.borrow_mut().push("begin".into());
        }
        fn bloom_prefilter(&self, _rec: &(), _args: &()) {
            self.log.borrow_mut().push("prefilter".into());
        }
        fn bloom_downsample(&self, _rec: &(), _args: &(), dst: usize) {
            self.log.borrow_mut().push(format!("down{dst}"));
        }
        fn bloom_upsample(&self, _rec: &(), _args: &(), dst: usize) {
            self.log.borrow_mut().push(format!("up{dst}"));
        }
    }

    #[test]
    fn bloom_chain_encodes_prefilter_downsample_upsample_in_order() {
        // 3 mips: prefilter, then the downsample chain 1..3, then the upsample
        // chain walking back down (1, 0).
        let enc = MockBloom {
            mips: 3,
            log: RefCell::new(Vec::new()),
        };
        encode_bloom_chain(&enc, &(), ());
        assert_eq!(
            *enc.log.borrow(),
            ["begin", "prefilter", "down1", "down2", "up1", "up0"]
        );
    }

    #[test]
    fn bloom_chain_with_zero_mips_is_a_noop() {
        // Bloom off: the driver returns before touching the encoder at all.
        let enc = MockBloom {
            mips: 0,
            log: RefCell::new(Vec::new()),
        };
        encode_bloom_chain(&enc, &(), ());
        assert!(enc.log.borrow().is_empty());
    }

    // A mock composite encoder. `text_ready` is the `begin_text` return; when
    // `fail_at` matches a text-draw index that draw returns an error.
    struct MockComposite {
        text_ready: bool,
        fail_at: Option<usize>,
        log: RefCell<Vec<String>>,
        text_seen: RefCell<usize>,
    }

    impl MockComposite {
        fn new(text_ready: bool, fail_at: Option<usize>) -> Self {
            Self {
                text_ready,
                fail_at,
                log: RefCell::new(Vec::new()),
                text_seen: RefCell::new(0),
            }
        }
    }

    impl CompositeEncoder for MockComposite {
        type Rec = ();
        type Args = ();

        fn begin_composite(&self, _rec: &(), _args: &()) {
            self.log.borrow_mut().push("begin".into());
        }
        fn composite_draw(&self, _rec: &(), _args: &()) {
            self.log.borrow_mut().push("draw".into());
        }
        fn begin_text(&self, _rec: &(), _args: &()) -> bool {
            self.log.borrow_mut().push("begin_text".into());
            self.text_ready
        }
        fn text_draw(&self, _rec: &(), _args: &(), _call: &TextDrawCall) -> Result<(), String> {
            let mut n = self.text_seen.borrow_mut();
            self.log.borrow_mut().push(format!("text{}", *n));
            let fail = self.fail_at == Some(*n);
            *n += 1;
            if fail {
                return Err("text upload failed".into());
            }
            Ok(())
        }
        fn end_composite(&self, _rec: &(), _args: &()) {
            self.log.borrow_mut().push("end".into());
        }
    }

    #[test]
    fn composite_chain_orders_passes_then_text_then_end() {
        let enc = MockComposite::new(true, None);
        let calls = [text_call(), text_call()];
        let r = encode_composite_chain(&enc, &(), &(), &calls);
        assert!(r.is_ok());
        assert_eq!(
            *enc.log.borrow(),
            ["begin", "draw", "begin_text", "text0", "text1", "end"]
        );
    }

    #[test]
    fn composite_chain_propagates_text_error_without_ending() {
        // The first text draw fails: the error propagates and, matching the
        // prior DX/VK behaviour, the pass is left open (no `end_composite`) and
        // the remaining text calls are skipped.
        let enc = MockComposite::new(true, Some(0));
        let calls = [text_call(), text_call()];
        let r = encode_composite_chain(&enc, &(), &(), &calls);
        assert_eq!(r, Err("text upload failed".into()));
        let log = enc.log.borrow();
        assert_eq!(*log, ["begin", "draw", "begin_text", "text0"]);
        assert!(!log.contains(&"end".to_string()), "pass must stay open");
    }

    #[test]
    fn composite_chain_with_no_text_skips_the_text_loop() {
        // Empty text: `begin_text` is never called, but the pass still ends.
        let enc = MockComposite::new(true, None);
        let r = encode_composite_chain(&enc, &(), &(), &[]);
        assert!(r.is_ok());
        assert_eq!(*enc.log.borrow(), ["begin", "draw", "end"]);
    }

    #[test]
    fn composite_chain_skips_draws_when_text_is_inert() {
        // `begin_text` returns false (no pipeline / atlases): no per-call draws,
        // but the pass still ends cleanly.
        let enc = MockComposite::new(false, None);
        let calls = [text_call()];
        let r = encode_composite_chain(&enc, &(), &(), &calls);
        assert!(r.is_ok());
        assert_eq!(*enc.log.borrow(), ["begin", "draw", "begin_text", "end"]);
    }

    // A mock single-draw fullscreen pass recording its lifecycle.
    struct MockFullscreen {
        log: RefCell<Vec<String>>,
    }

    impl FullscreenPass for MockFullscreen {
        type Rec = ();

        fn begin(&self, _rec: &()) {
            self.log.borrow_mut().push("begin".into());
        }
        fn draw(&self, _rec: &()) {
            self.log.borrow_mut().push("draw".into());
        }
        fn end(&self, _rec: &()) {
            self.log.borrow_mut().push("end".into());
        }
    }

    #[test]
    fn fullscreen_encodes_begin_draw_end() {
        let enc = MockFullscreen {
            log: RefCell::new(Vec::new()),
        };
        encode_fullscreen(&enc, &());
        assert_eq!(*enc.log.borrow(), ["begin", "draw", "end"]);
    }
}
