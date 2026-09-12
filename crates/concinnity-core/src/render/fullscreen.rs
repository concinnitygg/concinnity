//! What order the two multi-draw post passes encode their sub-passes in, shared
//! by every backend.
//!
//! The bloom prefilter -> downsample -> upsample chain and the composite ->
//! text overlay pass are structurally identical on every backend, so each one's
//! loop lives here once and a backend implements the hooks that bind + draw one
//! sub-pass in its own command stream. All three backends implement both.
//!
//! This shares orchestration only: every resource, layout and bind stays per
//! backend, so a pass written against it is still written three times. The
//! deeper seam beside it, `render::post`, shares the whole pass instead
//! (pipeline, target, and one fullscreen draw over slot-indexed binds), so a
//! pass written against that one is written once. A new post pass belongs
//! there.
//!
//! What is left here is the two passes that are not a single draw. Converging
//! them is open work, and neither is close: the composite writes the swapchain
//! image and draws per-label geometry under its own scissor, and bloom's chain
//! is a run of mips the render graph owns, none of which that seam models
//! today.
//!
//! Two associated types absorb the only real divergence, so the traits name no
//! backend types: `Rec` hides the per-backend command recorder, and `Args`
//! carries the per-invocation binding context (DirectX passes the scene-color
//! SRV its prefilter samples; Vulkan threads the frame-in-flight index that
//! selects its per-frame framebuffers + descriptor sets; Metal, which binds per
//! sub-pass encoder, carries what it needs on the encoder value itself and
//! passes `()`). Everything else each impl reads from `&self`, consistent with
//! the read-only parallel-encode contract.

use crate::gfx::render_types::TextDrawCall;
use crate::math::{ceil, floor};
use alloc::string::String;

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
/// zero logical dimension (minimized / mid-resize) falls back to a 1.0 scale
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
    let x0 = floor(clip[0] * sx).clamp(0.0, aw);
    let y0 = floor(clip[1] * sy).clamp(0.0, ah);
    let x1 = ceil((clip[0] + clip[2]) * sx).clamp(0.0, aw);
    let y1 = ceil((clip[1] + clip[3]) * sy).clamp(0.0, ah);
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
            let v = core::mem::size_of_val(c.vertices.as_slice()) as u64;
            let i = core::mem::size_of_val(c.indices.as_slice()) as u64;
            align_up(v, align) + align_up(i, align)
        })
        .sum()
}

/// Per-backend hooks the shared bloom driver encodes through.
pub trait BloomEncoder {
    /// Per-backend command recorder (DX `ID3D12GraphicsCommandList`, VK
    /// `vk::CommandBuffer`, Metal the command buffer each sub-pass opens its own
    /// render encoder on).
    type Rec: ?Sized;
    /// Per-invocation binding context (DX scene-color SRV handle, VK frame index).
    type Args;

    /// Number of bloom mips; zero means bloom is off and the driver no-ops.
    fn bloom_mip_count(&self) -> usize;
    /// One-time per-encode preamble, run once before the sub-passes and on the
    /// same recorder: state every sub-pass shares belongs here, not in the
    /// per-mip hooks (DX root signature / heap / IA state and the post-process
    /// root constants; VK the post-process push constants). Metal binds per
    /// sub-pass encoder, so it has nothing to do here.
    fn begin_bloom(&self, rec: &Self::Rec, args: &Self::Args) -> Result<(), String>;
    /// Prefilter: scene color -> mip 0 (soft-knee threshold + Karis average).
    fn bloom_prefilter(&self, rec: &Self::Rec, args: &Self::Args) -> Result<(), String>;
    /// Downsample: mip `dst - 1` -> mip `dst`.
    fn bloom_downsample(
        &self,
        rec: &Self::Rec,
        args: &Self::Args,
        dst: usize,
    ) -> Result<(), String>;
    /// Upsample: mip `dst + 1` -> mip `dst`, additively blended.
    fn bloom_upsample(&self, rec: &Self::Rec, args: &Self::Args, dst: usize) -> Result<(), String>;
}

/// The bloom chain orchestration, previously hand-duplicated in each backend's
/// `encode_bloom`. On return, mip 0 holds the accumulated glow the composite pass
/// samples.
///
/// A sub-pass reports failure where opening its own encoder can fail (Metal),
/// which abandons the chain: the mips below the failure hold no glow, and the
/// caller fails the frame rather than compositing a half-built chain.
pub fn encode_bloom_chain<E: BloomEncoder>(
    enc: &E,
    rec: &E::Rec,
    args: E::Args,
) -> Result<(), String> {
    let n = enc.bloom_mip_count();
    if n == 0 {
        return Ok(());
    }
    enc.begin_bloom(rec, &args)?;
    // Prefilter: scene -> mip 0.
    enc.bloom_prefilter(rec, &args)?;
    // Downsample chain: mip i-1 -> mip i.
    for dst in 1..n {
        enc.bloom_downsample(rec, &args, dst)?;
    }
    // Upsample chain: mip i+1 -> mip i, walking back down to mip 0.
    for dst in (0..n - 1).rev() {
        enc.bloom_upsample(rec, &args, dst)?;
    }
    Ok(())
}

/// The text-overlay state one draw call would set that the previous call in the
/// same pass already left bound. A heads-up display is overwhelmingly many labels
/// sharing one atlas and one full-window scissor, so tracking the last value bound
/// turns a per-label bind into a per-change bind.
///
/// The scissor is the canonical `(x, y, w, h)` in attachment pixels that
/// [`clip_rect_to_scissor`] returns and every backend's own rect type converts
/// from, so one cache serves all three backends.
///
/// A cache starts empty rather than seeded with whatever the pass began with, so
/// the first call of each kind always binds and the cache never has to assume what
/// the surrounding pass left in place.
///
/// Both queries record as they answer, so a caller must ask only where it goes on
/// to bind: asking and then skipping the bind desynchronizes the cache from the
/// recorder.
#[derive(Default)]
pub struct TextBindCache {
    atlas: Option<usize>,
    scissor: Option<(i32, i32, u32, u32)>,
}

impl TextBindCache {
    /// An empty cache: the first query of each kind reports a change.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the atlas at `idx` still needs binding, recording it as bound.
    pub fn atlas_changed(&mut self, idx: usize) -> bool {
        self.atlas.replace(idx) != Some(idx)
    }

    /// Whether `rect` still needs setting as the scissor, recording it as set.
    pub fn scissor_changed(&mut self, rect: (i32, i32, u32, u32)) -> bool {
        self.scissor.replace(rect) != Some(rect)
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
    /// Per-backend command recorder (DX `ID3D12GraphicsCommandList`, VK
    /// `vk::CommandBuffer`, Metal the render encoder the caller opened for the
    /// whole chain).
    type Rec: ?Sized;
    /// Per-invocation binding context (see the trait doc).
    type Args;

    /// Begin the pass: target the swapchain image (DX transitions it to
    /// RENDER_TARGET + binds the RTV; VK begins the composite render pass) and set
    /// the full-window viewport / scissor. Nothing to do on a backend whose
    /// recorder is itself the open pass, which is how Metal reaches the chain.
    fn begin_composite(&self, rec: &Self::Rec, args: &Self::Args);
    /// The fullscreen tonemap draw: bind the composite pipeline + its inputs
    /// (scene, bloom, LUT) + push constants, draw the fullscreen triangle.
    fn composite_draw(&self, rec: &Self::Rec, args: &Self::Args);
    /// Bind the text pipeline + any one-time text state. Returns false when text
    /// is inert (no pipeline or no atlases), so the driver skips the call loop.
    fn begin_text(&self, rec: &Self::Rec, args: &Self::Args) -> bool;
    /// Encode one text draw call: append its vertex/index geometry to this frame
    /// slot's persistent upload buffer, bind the atlas plus the two sub-ranges,
    /// and draw. `cache` carries what the previous call in this pass left bound;
    /// consult it for the atlas and the scissor so a run of labels sharing either
    /// binds it once (see [`TextBindCache`]).
    ///
    /// `idx` is the call's position in the frame's list, for a backend whose
    /// geometry was uploaded before the pass and is addressed by that position
    /// (Metal). A backend that appends here ignores it. Every call is offered in
    /// order, including one a backend then skips, so the position always matches.
    fn text_draw(
        &self,
        rec: &Self::Rec,
        args: &Self::Args,
        idx: usize,
        call: &TextDrawCall,
        cache: &mut TextBindCache,
    ) -> Result<(), String>;
    /// End the pass: DX transitions the back-buffer back to PRESENT; VK ends the
    /// render pass. Runs however the chain leaves, a failed text draw included,
    /// so no backend is left with a pass or a resource state half-open. Nothing
    /// to do on a backend whose recorder ends its own pass when it drops, which
    /// is how Metal reaches the chain.
    fn end_composite(&self, rec: &Self::Rec, args: &Self::Args);
}

/// The composite + text orchestration, previously hand-duplicated in each
/// backend's `encode_composite_and_text`.
///
/// A failed text draw ends the pass before it propagates: the frame fails either
/// way, but a backend is never left holding an open pass or a mis-stated target
/// (Metal rejects a command buffer committed with an encoder still open, which
/// is what kept it off this driver while the error path skipped the end).
pub fn encode_composite_chain<E: CompositeEncoder>(
    enc: &E,
    rec: &E::Rec,
    args: &E::Args,
    text_calls: &[TextDrawCall],
) -> Result<(), String> {
    enc.begin_composite(rec, args);
    enc.composite_draw(rec, args);
    let mut result = Ok(());
    if !text_calls.is_empty() && enc.begin_text(rec, args) {
        // One cache per pass: the text calls are encoded back to back into the
        // same recorder, so what one call binds is still bound for the next.
        let mut cache = TextBindCache::new();
        for (idx, call) in text_calls.iter().enumerate() {
            result = enc.text_draw(rec, args, idx, call, &mut cache);
            if result.is_err() {
                break;
            }
        }
    }
    enc.end_composite(rec, args);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::TextDrawCall;
    use core::cell::RefCell;

    use alloc::format;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;
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
        // Minimized / mid-resize: no divide by zero, and the rect is still
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
                crate::gfx::render_types::TextVertex {
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
                    core::mem::size_of_val(c.vertices.as_slice()) as u64,
                    core::mem::size_of_val(c.indices.as_slice()) as u64,
                ] {
                    cursor = align_up(cursor, align) + block;
                    assert!(cursor <= total, "cursor {cursor} exceeded reserved {total}");
                }
            }
        }
    }

    // A mock bloom encoder recording each sub-pass in call order. The trait's
    // associated types name no backend types, so both are `()`. `fail_at` is the
    // log entry whose sub-pass reports failure, for the abandon-the-chain test.
    struct MockBloom {
        mips: usize,
        log: RefCell<Vec<String>>,
        fail_at: Option<&'static str>,
    }

    impl MockBloom {
        fn new(mips: usize) -> Self {
            Self {
                mips,
                log: RefCell::new(Vec::new()),
                fail_at: None,
            }
        }

        // Record one sub-pass, reporting failure where the test asked for it.
        fn step(&self, entry: String) -> Result<(), String> {
            let failed = self.fail_at == Some(entry.as_str());
            self.log.borrow_mut().push(entry);
            if failed {
                return Err("sub-pass failed".to_string());
            }
            Ok(())
        }
    }

    impl BloomEncoder for MockBloom {
        type Rec = ();
        type Args = ();

        fn bloom_mip_count(&self) -> usize {
            self.mips
        }
        fn begin_bloom(&self, _rec: &(), _args: &()) -> Result<(), String> {
            self.step("begin".to_string())
        }
        fn bloom_prefilter(&self, _rec: &(), _args: &()) -> Result<(), String> {
            self.step("prefilter".to_string())
        }
        fn bloom_downsample(&self, _rec: &(), _args: &(), dst: usize) -> Result<(), String> {
            self.step(format!("down{dst}"))
        }
        fn bloom_upsample(&self, _rec: &(), _args: &(), dst: usize) -> Result<(), String> {
            self.step(format!("up{dst}"))
        }
    }

    #[test]
    fn bloom_chain_encodes_prefilter_downsample_upsample_in_order() {
        // 3 mips: prefilter, then the downsample chain 1..3, then the upsample
        // chain walking back down (1, 0).
        let enc = MockBloom::new(3);
        assert!(encode_bloom_chain(&enc, &(), ()).is_ok());
        assert_eq!(
            *enc.log.borrow(),
            ["begin", "prefilter", "down1", "down2", "up1", "up0"]
        );
    }

    #[test]
    fn bloom_chain_begins_once_whatever_the_mip_count() {
        // Backends push the shared post-process constants in `begin_bloom` and
        // rely on them surviving every sub-pass, so the preamble must run
        // exactly once per chain, ahead of the first draw.
        for mips in 1..8 {
            let enc = MockBloom::new(mips);
            assert!(encode_bloom_chain(&enc, &(), ()).is_ok());
            let log = enc.log.borrow();
            assert_eq!(log.iter().filter(|e| *e == "begin").count(), 1);
            assert_eq!(log[0], "begin");
        }
    }

    #[test]
    fn bloom_chain_with_zero_mips_is_a_noop() {
        // Bloom off: the driver returns before touching the encoder at all.
        let enc = MockBloom::new(0);
        assert!(encode_bloom_chain(&enc, &(), ()).is_ok());
        assert!(enc.log.borrow().is_empty());
    }

    #[test]
    fn a_failed_sub_pass_abandons_the_rest_of_the_bloom_chain() {
        // A backend that opens an encoder per sub-pass can fail mid-chain. The
        // mips past the failure are left unwritten rather than half-built, and
        // the caller sees the error.
        let mut enc = MockBloom::new(3);
        enc.fail_at = Some("down1");
        assert!(encode_bloom_chain(&enc, &(), ()).is_err());
        assert_eq!(*enc.log.borrow(), ["begin", "prefilter", "down1"]);
    }

    // A mock composite encoder. `text_ready` is the `begin_text` return; when
    // `fail_at` matches a text-draw index that draw returns an error. `binds`
    // records what the cache answered per call, so a test can see which calls
    // would have rebound the atlas.
    struct MockComposite {
        text_ready: bool,
        fail_at: Option<usize>,
        log: RefCell<Vec<String>>,
        text_seen: RefCell<usize>,
        binds: RefCell<Vec<bool>>,
    }

    impl MockComposite {
        fn new(text_ready: bool, fail_at: Option<usize>) -> Self {
            Self {
                text_ready,
                fail_at,
                log: RefCell::new(Vec::new()),
                text_seen: RefCell::new(0),
                binds: RefCell::new(Vec::new()),
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
        fn text_draw(
            &self,
            _rec: &(),
            _args: &(),
            idx: usize,
            call: &TextDrawCall,
            cache: &mut TextBindCache,
        ) -> Result<(), String> {
            let mut n = self.text_seen.borrow_mut();
            // The driver's index and the encoder's own call count must agree, so
            // a backend addressing pre-uploaded geometry by position can trust it.
            assert_eq!(idx, *n);
            self.binds
                .borrow_mut()
                .push(cache.atlas_changed(call.atlas_slot));
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
    fn composite_chain_ends_the_pass_before_propagating_a_text_error() {
        // The first text draw fails: the remaining calls are skipped, the pass
        // still ends (Metal rejects a command buffer committed with an encoder
        // open), and the error reaches the caller.
        let enc = MockComposite::new(true, Some(0));
        let calls = [text_call(), text_call()];
        let r = encode_composite_chain(&enc, &(), &(), &calls);
        assert_eq!(r, Err("text upload failed".into()));
        assert_eq!(
            *enc.log.borrow(),
            ["begin", "draw", "begin_text", "text0", "end"]
        );
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

    #[test]
    fn an_empty_cache_reports_the_first_bind_of_each_kind() {
        let mut cache = TextBindCache::new();
        assert!(cache.atlas_changed(0));
        assert!(cache.scissor_changed((0, 0, 1280, 720)));
    }

    #[test]
    fn a_repeated_value_is_not_rebound() {
        // The heads-up-display case: every label on one atlas, none clipped, so
        // only the first call of the run binds either.
        let mut cache = TextBindCache::new();
        let full = (0, 0, 1280, 720);
        assert!(cache.atlas_changed(2));
        assert!(cache.scissor_changed(full));
        for _ in 0..100 {
            assert!(!cache.atlas_changed(2));
            assert!(!cache.scissor_changed(full));
        }
    }

    #[test]
    fn a_changed_value_rebinds_and_then_settles() {
        // A clipped call in the middle of a run sets its own band and the next
        // unclipped call restores the full-window rect; a third unclipped call
        // then rides the restored one.
        let mut cache = TextBindCache::new();
        let full = (0, 0, 1280, 720);
        let band = (10, 20, 300, 100);
        assert!(cache.scissor_changed(full));
        assert!(cache.scissor_changed(band));
        assert!(cache.scissor_changed(full));
        assert!(!cache.scissor_changed(full));
        // The two kinds are tracked independently.
        assert!(cache.atlas_changed(0));
        assert!(!cache.atlas_changed(0));
        assert!(cache.atlas_changed(1));
        assert!(cache.atlas_changed(0));
    }

    #[test]
    fn a_cache_distinguishes_rects_that_differ_in_one_field() {
        let mut cache = TextBindCache::new();
        assert!(cache.scissor_changed((0, 0, 100, 100)));
        assert!(cache.scissor_changed((1, 0, 100, 100)));
        assert!(cache.scissor_changed((1, 2, 100, 100)));
        assert!(cache.scissor_changed((1, 2, 101, 100)));
        assert!(cache.scissor_changed((1, 2, 101, 99)));
        assert!(!cache.scissor_changed((1, 2, 101, 99)));
    }

    #[test]
    fn one_cache_spans_the_whole_text_loop() {
        // The driver must hand every call in a pass the same cache, or nothing
        // is ever deduplicated: three calls on one atlas bind it once.
        let enc = MockComposite::new(true, None);
        let calls = [text_call(), text_call(), text_call()];
        let r = encode_composite_chain(&enc, &(), &(), &calls);
        assert!(r.is_ok());
        assert_eq!(*enc.binds.borrow(), [true, false, false]);
    }

    #[test]
    fn each_pass_starts_from_an_empty_cache() {
        // A cache must not outlive its pass: the next frame records into a fresh
        // recorder that has none of the previous frame's state bound.
        for _ in 0..2 {
            let enc = MockComposite::new(true, None);
            let calls = [text_call(), text_call()];
            assert!(encode_composite_chain(&enc, &(), &(), &calls).is_ok());
            assert_eq!(*enc.binds.borrow(), [true, false]);
        }
    }
}
