// src/editor/thumbs.rs
//
// The editor's view of the baked thumbnail store (the state root's `thumbnails/`,
// written by the cook's disk-build tail): loads `index.json` plus its PNGs,
// assigns each a reserved `TextureHandle`, and hands the decoded images to
// HUD injection as the `OverlayImages` resource so the Content panel's cell
// sprites can sample them. Loads are cached and re-checked by the index
// file's modification stamp; the set actually resident in the atlas is the
// one captured at the last injection (the pool is built once per world
// rebuild), so panels bind through `injected()`, never the raw disk state.

use crate::ecs::{OverlayImage, OverlayImages, TextureHandle};
use std::collections::HashMap;
use std::path::Path;
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
    stamp: Option<std::time::SystemTime>,
    current: Arc<ThumbSet>,
    injected: Arc<ThumbSet>,
}

fn cache() -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(Cache {
            stamp: None,
            current: Arc::new(ThumbSet::default()),
            injected: Arc::new(ThumbSet::default()),
        })
    })
}

// The freshest loaded set, reloading when the index file changed on disk.
fn current() -> Arc<ThumbSet> {
    let Some(dir) = concinnity_store::paths::thumbnails_dir() else {
        return Arc::new(ThumbSet::default());
    };
    let stamp = std::fs::metadata(dir.join("index.json"))
        .and_then(|m| m.modified())
        .ok();
    let mut cache = cache().lock().unwrap();
    if cache.stamp != stamp {
        cache.current = Arc::new(load(&dir));
        cache.stamp = stamp;
    }
    cache.current.clone()
}

// Capture the freshest set for a world (re)build and return its images for
// the `OverlayImages` resource. Called by HUD injection, which runs right
// before the graphics init that builds the atlas pool.
pub(crate) fn capture_for_injection() -> OverlayImages {
    let set = current();
    let images = set.overlay_images();
    cache().lock().unwrap().injected = set;
    images
}

// The set resident in the live atlas: what the panels bind cell sprites to.
pub(crate) fn injected() -> Arc<ThumbSet> {
    cache().lock().unwrap().injected.clone()
}

fn load(dir: &Path) -> ThumbSet {
    let mut set = ThumbSet::default();
    let Ok(text) = std::fs::read_to_string(dir.join("index.json")) else {
        return set;
    };
    let Ok(serde_json::Value::Object(map)) = serde_json::from_str(&text) else {
        return set;
    };
    for (name, key) in map {
        if set.images.len() >= MAX_THUMBS {
            tracing::info!(
                "thumbnails: cap of {MAX_THUMBS} reached; remaining assets show typed icons"
            );
            break;
        }
        let Some(key) = key.as_str() else { continue };
        let Some((w, h, rgba)) = read_png(&dir.join(format!("{key}.png"))) else {
            continue;
        };
        let handle = TextureHandle(HANDLE_BASE + set.images.len() as u32);
        set.by_name.insert(
            name,
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

fn read_png(path: &Path) -> Option<(u32, u32, Vec<u8>)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
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

    #[test]
    fn load_reads_index_and_assigns_reserved_handles() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, w: u32, h: u32| {
            let file = std::fs::File::create(dir.path().join(format!("{name}.png"))).unwrap();
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header()
                .unwrap()
                .write_image_data(&vec![128u8; (w * h * 4) as usize])
                .unwrap();
        };
        write("aaa", 8, 4);
        write("bbb", 4, 4);
        std::fs::write(
            dir.path().join("index.json"),
            r#"{ "tex_one": "aaa", "tex_two": "bbb", "missing": "nope" }"#,
        )
        .unwrap();
        let set = load(dir.path());
        assert_eq!(set.images.len(), 2, "the missing png is skipped");
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

    #[test]
    fn a_missing_store_loads_empty() {
        let dir = tempfile::tempdir().unwrap();
        let set = load(&dir.path().join("nothing"));
        assert!(set.images.is_empty());
    }
}
