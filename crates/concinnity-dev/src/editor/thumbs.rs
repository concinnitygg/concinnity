// src/editor/thumbs.rs
//
// The editor's view of the baked thumbnail set (entries of the build cache
// segment, `cache/1`, written by the cook's disk-build tail): reads the set's
// name -> key map, seeks to the PNG each key addresses, assigns every decoded
// image a reserved `TextureHandle`, and hands them to HUD injection as the
// `OverlayImages` resource so the Content panel's cell sprites can sample
// them. Loads are cached and re-checked by the set's own revision, not by the
// segment file: the payload cache shares that file, so a build would bump it
// whether or not a thumbnail moved. The set actually resident in the atlas is
// the one captured at the last injection (the pool is built once per world
// rebuild), so panels bind through `injected()`, never the raw disk state.
//
// Best effort throughout: a build replaces the segment by rename while this
// may be reading it, so a key that resolves to nothing (or to bytes of the
// file that replaced it) costs that asset its preview and leaves the panel to
// its typed icon.

use crate::ecs::{OverlayImage, OverlayImages, TextureHandle};
use concinnity_cook::cache::thumbnails::Thumbnails;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

// Reserved handle space for overlay images: far above any compiled world's
// dense texture handles.
const HANDLE_BASE: u32 = 0x4000_0000;

// Upper bound on loaded thumbnails (128x128 RGBA is 64 KB each).
const MAX_THUMBS: usize = 256;

// One loaded thumbnail: its atlas handle and pixel dimensions (the grid needs
// the aspect to letterbox non-square images).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Thumb {
    pub handle: TextureHandle,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Default)]
pub(crate) struct ThumbSet {
    by_name: HashMap<String, Thumb>,
    images: Vec<OverlayImage>,
}

impl ThumbSet {
    pub(crate) fn get(&self, name: &str) -> Option<Thumb> {
        self.by_name.get(name).copied()
    }

    pub(crate) fn overlay_images(&self) -> OverlayImages {
        OverlayImages(self.images.clone())
    }
}

struct Cache {
    revision: Option<u64>,
    current: Arc<ThumbSet>,
    injected: Arc<ThumbSet>,
}

fn cache() -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(Cache {
            revision: None,
            current: Arc::new(ThumbSet::default()),
            injected: Arc::new(ThumbSet::default()),
        })
    })
}

// The freshest loaded set, reloading when the segment's thumbnail set moved.
// A deleted `cache/` opens to nothing and empties the set, which is what
// returns the panel to typed icons without a restart.
fn current() -> Arc<ThumbSet> {
    let thumbs = Thumbnails::open();
    let revision = thumbs.as_ref().map(Thumbnails::revision);
    let mut cache = cache().lock().unwrap();
    if cache.revision != revision {
        cache.current = Arc::new(match &thumbs {
            Some(thumbs) => load(thumbs.names(), &|key| thumbs.png(key)),
            None => ThumbSet::default(),
        });
        cache.revision = revision;
    }
    cache.current.clone()
}

// Capture the freshest set for a world (re)build and return its images for
// the `OverlayImages` resource. Called by HUD injection, which runs right
// before the graphics init that builds the atlas pool.
pub(crate) fn capture_for_injection() -> OverlayImages {
    let set = current();
    let images = set.overlay_images();
    tracing::info!(
        "thumbnails: {} baked image(s) available to the Content panel",
        images.0.len()
    );
    cache().lock().unwrap().injected = set;
    images
}

// The set resident in the live atlas: what the panels bind cell sprites to.
pub(crate) fn injected() -> Arc<ThumbSet> {
    cache().lock().unwrap().injected.clone()
}

// Decode the set `names` addresses, in bake order, up to the atlas budget.
// `png` reads one entry, so an asset whose entry is gone is skipped rather
// than failing the load.
fn load(names: &[(String, String)], png: &dyn Fn(&str) -> Option<Vec<u8>>) -> ThumbSet {
    let mut set = ThumbSet::default();
    for (name, key) in names {
        if set.images.len() >= MAX_THUMBS {
            tracing::info!(
                "thumbnails: cap of {MAX_THUMBS} reached; remaining assets show typed icons"
            );
            break;
        }
        let Some((w, h, rgba)) = png(key).and_then(|bytes| decode_png(&bytes)) else {
            continue;
        };
        let handle = TextureHandle(HANDLE_BASE + set.images.len() as u32);
        set.by_name.insert(
            name.clone(),
            Thumb {
                handle,
                width: w,
                height: h,
            },
        );
        set.images.push(OverlayImage {
            handle,
            width: w,
            height: h,
            rgba,
        });
    }
    set
}

fn decode_png(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
        return None;
    }
    buf.truncate(info.buffer_size());
    Some((info.width, info.height, buf))
}

// Fit an image of `w` x `h` into `rect` preserving aspect, centered.
pub(crate) fn fit_rect(rect: [f32; 4], w: u32, h: u32) -> [f32; 4] {
    if w == 0 || h == 0 {
        return rect;
    }
    let scale = (rect[2] / w as f32).min(rect[3] / h as f32);
    let (fw, fh) = (w as f32 * scale, h as f32 * scale);
    [
        rect[0] + (rect[2] - fw) * 0.5,
        rect[1] + (rect[3] - fh) * 0.5,
        fw,
        fh,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_rect_letterboxes_and_centers() {
        let cell = [10.0, 10.0, 80.0, 80.0];
        assert_eq!(fit_rect(cell, 128, 128), cell);
        let wide = fit_rect(cell, 128, 64);
        assert_eq!(wide, [10.0, 30.0, 80.0, 40.0]);
        let tall = fit_rect(cell, 64, 128);
        assert_eq!(tall, [30.0, 10.0, 40.0, 80.0]);
        assert_eq!(fit_rect(cell, 0, 5), cell, "degenerate dims fill the cell");
    }

    fn png_of(w: u32, h: u32) -> Vec<u8> {
        let mut out = Vec::new();
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()
            .unwrap()
            .write_image_data(&vec![128u8; (w * h * 4) as usize])
            .unwrap();
        out
    }

    fn names(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(n, k)| ((*n).to_string(), (*k).to_string()))
            .collect()
    }

    #[test]
    fn load_reads_the_name_map_and_assigns_reserved_handles() {
        let entries: HashMap<&str, Vec<u8>> = [("aaa", png_of(8, 4)), ("bbb", png_of(4, 4))].into();
        let set = load(
            &names(&[("tex_one", "aaa"), ("tex_two", "bbb"), ("missing", "nope")]),
            &|key| entries.get(key).cloned(),
        );
        assert_eq!(set.images.len(), 2, "the missing entry is skipped");
        let one = set.get("tex_one").unwrap();
        assert_eq!((one.width, one.height), (8, 4));
        assert!(one.handle.0 >= HANDLE_BASE);
        let two = set.get("tex_two").unwrap();
        assert_ne!(one.handle, two.handle);
        assert!(set.get("missing").is_none());
        let images = set.overlay_images();
        assert_eq!(images.0.len(), 2);
        assert_eq!(images.0[0].rgba.len(), 8 * 4 * 4);
    }

    // Two assets may share one entry, and each still gets a cell sprite.
    #[test]
    fn two_names_may_share_one_image() {
        let entries: HashMap<&str, Vec<u8>> = [("aaa", png_of(4, 4))].into();
        let set = load(&names(&[("one", "aaa"), ("two", "aaa")]), &|key| {
            entries.get(key).cloned()
        });
        assert_eq!(set.images.len(), 2);
        assert_ne!(
            set.get("one").unwrap().handle,
            set.get("two").unwrap().handle
        );
    }

    // A build replaces the segment by rename while this may be reading it, so
    // a key that resolves to bytes of another file must cost a preview and
    // nothing more.
    #[test]
    fn unreadable_entries_fall_back_to_typed_icons() {
        let set = load(&names(&[("one", "aaa"), ("two", "bbb")]), &|key| {
            (key == "aaa").then(|| b"not a png at all".to_vec())
        });
        assert!(set.images.is_empty());
        assert!(set.get("one").is_none());
    }

    // The atlas budget is a cap on what loads, not on what the set names.
    #[test]
    fn the_cap_bounds_what_loads() {
        let png = png_of(4, 4);
        let pairs: Vec<(String, String)> = (0..MAX_THUMBS + 8)
            .map(|i| (format!("asset{i}"), format!("key{i}")))
            .collect();
        let set = load(&pairs, &|_| Some(png.clone()));
        assert_eq!(set.images.len(), MAX_THUMBS);
        assert!(set.get(&format!("asset{}", MAX_THUMBS)).is_none());
    }

    #[test]
    fn a_missing_set_loads_empty() {
        let set = load(&[], &|_| None);
        assert!(set.images.is_empty());
    }
}
