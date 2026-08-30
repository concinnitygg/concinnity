//! glTF container loading shared by every glTF consumer in the cook: parses a
//! `.glb` or text `.gltf` file and eagerly resolves every buffer -- the GLB
//! binary chunk, external `.bin` files, and base64 data URIs -- so downstream
//! readers never care which container the data arrived in. External URIs
//! resolve relative to the source file's own directory, per the glTF spec.
//!
//! Percent-encoding in URIs is decoded; scheme URIs (`http://...`), absolute
//! paths, and `..` traversal are rejected so a cook never reads outside the
//! source's directory tree.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

/// A parsed glTF document with every buffer resolved to bytes. Buffers are
/// `Arc`-backed so a clone shares them; the editor hot-reload caches clone one
/// doc across many mesh / texture entries.
#[derive(Clone, Debug)]
pub struct GltfDoc {
    /// The parsed glTF document.
    pub doc: gltf::Gltf,
    // Indexed by glTF buffer index. `None` when the buffer could not be
    // resolved (a GLB whose BIN chunk is absent); accessor reads then fail
    // with their own contextual errors.
    buffers: Vec<Option<Arc<[u8]>>>,
    // Directory external URIs resolve against; `None` for in-memory parses.
    base_dir: Option<PathBuf>,
}

impl GltfDoc {
    // Parse a `.glb` or `.gltf` at `path`, which is read as given. A caller
    // holding an authored `source` string resolves it against the build's
    // asset search root first (see `crate::import::glb::resolve_source`).
    pub(crate) fn parse_file(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("failed to read '{}': {}", path, e))?;
        let base_dir = Path::new(path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(PathBuf::from);
        Self::from_slice(&bytes, base_dir, path)
    }

    /// Parse from bytes already in memory. `base_dir` anchors external URIs;
    /// without one, only the GLB binary chunk and data URIs resolve.
    pub fn from_slice(
        bytes: &[u8],
        base_dir: Option<PathBuf>,
        origin: &str,
    ) -> Result<Self, String> {
        let mut doc = gltf::Gltf::from_slice(bytes)
            .map_err(|e| format!("'{}': not a valid glTF/GLB file: {}", origin, e))?;
        let blob: Option<Arc<[u8]>> = doc.blob.take().map(Arc::from);

        let mut buffers: Vec<Option<Arc<[u8]>>> = Vec::new();
        for buffer in doc.document.buffers() {
            let resolved = match buffer.source() {
                gltf::buffer::Source::Bin => blob.clone(),
                gltf::buffer::Source::Uri(uri) => {
                    let bytes = resolve_uri_bytes(uri, base_dir.as_deref())
                        .map_err(|e| format!("'{}': buffer {}: {}", origin, buffer.index(), e))?;
                    Some(Arc::from(bytes))
                }
            };
            buffers.push(resolved);
        }

        Ok(Self {
            doc,
            buffers,
            base_dir,
        })
    }

    // Bytes for one glTF buffer, or `None` when it did not resolve. Shaped for
    // the `gltf` crate's reader closures: `primitive.reader(|b| doc.buffer_bytes(b))`.
    pub(crate) fn buffer_bytes(&self, buffer: gltf::Buffer<'_>) -> Option<&[u8]> {
        self.buffers.get(buffer.index()).and_then(|b| b.as_deref())
    }

    // Read an external (non-buffer) URI, e.g. a glTF image file. Data URIs
    // decode inline; file URIs resolve against the source's directory.
    pub(crate) fn external_bytes(&self, uri: &str) -> Result<Vec<u8>, String> {
        resolve_uri_bytes(uri, self.base_dir.as_deref())
    }

    // Encoded bytes of one glTF image plus its MIME type, whether the image
    // lives in a `bufferView`, in a file beside the source, or in a data URI.
    // The bytes are still PNG / JPEG here; picking a decoder is the caller's
    // job. Errors name the image index but not the container, so callers add
    // their own source prefix.
    pub(crate) fn image_bytes(
        &self,
        image_index: u32,
    ) -> Result<(Vec<u8>, Option<String>), String> {
        let image = self
            .doc
            .document
            .images()
            .nth(image_index as usize)
            .ok_or_else(|| format!("image_index {} is out of range", image_index))?;

        match image.source() {
            gltf::image::Source::View { view, mime_type } => {
                let backing = self.buffer_bytes(view.buffer()).ok_or_else(|| {
                    format!(
                        "image {} references buffer data the container does not carry",
                        image_index
                    )
                })?;
                let start = view.offset();
                let end = start + view.length();
                if end > backing.len() {
                    return Err(format!(
                        "image {} bufferView [{}, {}) exceeds buffer size {}",
                        image_index,
                        start,
                        end,
                        backing.len()
                    ));
                }
                Ok((backing[start..end].to_vec(), Some(mime_type.to_string())))
            }
            gltf::image::Source::Uri { uri, mime_type } => {
                let bytes = self
                    .external_bytes(uri)
                    .map_err(|e| format!("image {}: {}", image_index, e))?;
                let mime = mime_type.map(str::to_string).or_else(|| mime_from_uri(uri));
                Ok((bytes, mime))
            }
        }
    }
}

// Infer an image MIME type from a URI's extension (or a data URI's header)
// when the glTF omits `mimeType`, which is optional for URI-sourced images.
fn mime_from_uri(uri: &str) -> Option<String> {
    if let Some(rest) = uri.strip_prefix("data:") {
        return rest.split(&[';', ','][..]).next().map(str::to_string);
    }
    let lower = uri.to_lowercase();
    if lower.ends_with(".png") {
        Some("image/png".to_string())
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        Some("image/jpeg".to_string())
    } else {
        None
    }
}

// External files a `.gltf` reads beyond itself: URI-referenced buffers and
// images, resolved against the source's directory. Used to fold sibling-file
// contents into cache keys, so the result is best-effort: a source that is
// missing or fails to parse contributes nothing (the compile will surface the
// real error). Memoized by `FileStamp` like `cache::file_content_hash`, since
// cache-key computation calls this once per asset sharing the file.
pub(crate) fn referenced_files(source: &str, assets_dir: Option<&Path>) -> Vec<String> {
    use crate::file_stamp::FileStamp;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    type Memo = Mutex<HashMap<String, (FileStamp, Vec<String>)>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();

    let path = crate::import::glb::resolve_source(source, assets_dir);
    let Some(stamp) = FileStamp::read(&path) else {
        return Vec::new();
    };
    // Decided before the scan, so a write racing it lands on a later mtime and
    // misses this entry rather than matching it.
    let memoizable = stamp.settled();
    let memo = MEMO.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some((s, files)) = memo
        .lock()
        .expect("glTF source memo lock is not poisoned")
        .get(&path)
        && *s == stamp
    {
        return files.clone();
    }

    let files = scan_referenced_files(&path);
    if memoizable {
        memo.lock()
            .expect("glTF source memo lock is not poisoned")
            .insert(path, (stamp, files.clone()));
    }
    files
}

fn scan_referenced_files(path: &str) -> Vec<String> {
    let Ok(bytes) = std::fs::read(path) else {
        return Vec::new();
    };
    let Ok(doc) = gltf::Gltf::from_slice(&bytes) else {
        return Vec::new();
    };
    let Some(base) = Path::new(path)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    else {
        return Vec::new();
    };

    let buffer_uris = doc.document.buffers().filter_map(|b| match b.source() {
        gltf::buffer::Source::Uri(uri) => Some(uri.to_string()),
        gltf::buffer::Source::Bin => None,
    });
    let image_uris = doc.document.images().filter_map(|i| match i.source() {
        gltf::image::Source::Uri { uri, .. } => Some(uri.to_string()),
        gltf::image::Source::View { .. } => None,
    });

    let mut out = Vec::new();
    for uri in buffer_uris.chain(image_uris) {
        if uri.starts_with("data:") {
            continue;
        }
        if let Ok(rel) = safe_relative_path(&uri) {
            out.push(base.join(rel).to_string_lossy().into_owned());
        }
    }
    out.sort();
    out.dedup();
    out
}

// Resolve one URI to bytes: base64 data URIs decode inline, anything else is
// a percent-encoded path relative to `base_dir`.
fn resolve_uri_bytes(uri: &str, base_dir: Option<&Path>) -> Result<Vec<u8>, String> {
    if let Some(rest) = uri.strip_prefix("data:") {
        let Some((header, payload)) = rest.split_once(',') else {
            return Err("malformed data URI (no comma separator)".to_string());
        };
        if !header.ends_with(";base64") {
            return Err(format!(
                "data URI encoding '{}' is unsupported; only base64 is handled",
                header
            ));
        }
        return decode_base64(payload).map_err(|e| format!("invalid base64 data URI: {}", e));
    }

    let rel = safe_relative_path(uri)?;
    let base = base_dir.ok_or_else(|| {
        format!(
            "external URI '{}' cannot resolve without a source directory",
            uri
        )
    })?;
    let full = base.join(rel);
    std::fs::read(&full)
        .map_err(|e| format!("failed to read external file '{}': {}", full.display(), e))
}

// Percent-decode a URI and validate it stays inside the source's directory:
// scheme URIs, absolute paths, and `..` components are rejected.
fn safe_relative_path(uri: &str) -> Result<PathBuf, String> {
    if uri.contains("://") {
        return Err(format!(
            "URI '{}' uses a URL scheme; only files relative to the source are supported",
            uri
        ));
    }
    let decoded = percent_decode(uri);
    let path = PathBuf::from(&decoded);
    if path.is_absolute() {
        return Err(format!(
            "URI '{}' is an absolute path; only relative paths are supported",
            uri
        ));
    }
    for component in path.components() {
        match component {
            Component::ParentDir => {
                return Err(format!(
                    "URI '{}' escapes the source directory ('..' is not allowed)",
                    uri
                ));
            }
            Component::Prefix(_) | Component::RootDir => {
                return Err(format!(
                    "URI '{}' is an absolute path; only relative paths are supported",
                    uri
                ));
            }
            Component::CurDir | Component::Normal(_) => {}
        }
    }
    Ok(path)
}

// Decode `%XX` escapes; malformed escapes pass through literally so a path
// containing a stray '%' still resolves as written.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && let (Some(hi), Some(lo)) = (
                bytes.get(i + 1).and_then(|b| (*b as char).to_digit(16)),
                bytes.get(i + 2).and_then(|b| (*b as char).to_digit(16)),
            )
        {
            out.push((hi * 16 + lo) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

// Decode standard-alphabet base64 (RFC 4648, `+` / `/`, optional `=` padding).
// Hand-rolled like the DDS / TGA / BCn decoders: the algorithm is small enough
// that a dependency is not worth its supply-chain surface.
fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    fn value(c: u8) -> Result<u32, String> {
        match c {
            b'A'..=b'Z' => Ok((c - b'A') as u32),
            b'a'..=b'z' => Ok((c - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((c - b'0' + 52) as u32),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("invalid base64 character '{}'", c as char)),
        }
    }

    let bytes: Vec<u8> = input
        .bytes()
        .filter(|b| !matches!(b, b'\r' | b'\n'))
        .collect();
    let trimmed = match bytes.iter().position(|&b| b == b'=') {
        Some(pos) => {
            if bytes[pos..].iter().any(|&b| b != b'=') || bytes.len() - pos > 2 {
                return Err("misplaced base64 padding".to_string());
            }
            &bytes[..pos]
        }
        None => &bytes[..],
    };
    if trimmed.len() % 4 == 1 {
        return Err("truncated base64 input".to_string());
    }

    let mut out = Vec::with_capacity(trimmed.len() / 4 * 3 + 2);
    for chunk in trimmed.chunks(4) {
        let mut acc: u32 = 0;
        for &c in chunk {
            acc = (acc << 6) | value(c)?;
        }
        match chunk.len() {
            4 => out.extend_from_slice(&[(acc >> 16) as u8, (acc >> 8) as u8, acc as u8]),
            3 => {
                acc <<= 6;
                out.extend_from_slice(&[(acc >> 16) as u8, (acc >> 8) as u8]);
            }
            2 => {
                acc <<= 12;
                out.push((acc >> 16) as u8);
            }
            _ => unreachable!("chunk length 1 rejected above"),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // base64

    #[test]
    fn base64_round_trips_all_padding_lengths() {
        // "" / "f" / "fo" / "foo" / "foob" cover every padding case.
        let cases: [(&str, &[u8]); 5] = [
            ("", b""),
            ("Zg==", b"f"),
            ("Zm8=", b"fo"),
            ("Zm9v", b"foo"),
            ("Zm9vYg==", b"foob"),
        ];
        for (encoded, expected) in cases {
            assert_eq!(
                decode_base64(encoded).expect(encoded),
                expected,
                "decoding '{encoded}'"
            );
        }
    }

    #[test]
    fn base64_accepts_unpadded_input() {
        assert_eq!(decode_base64("Zm9vYg").unwrap(), b"foob");
        assert_eq!(decode_base64("Zm8").unwrap(), b"fo");
    }

    #[test]
    fn base64_decodes_binary_values() {
        // 0xFB 0xFF exercises the '+' and '/' alphabet tail.
        assert_eq!(decode_base64("+/8=").unwrap(), vec![0xfb, 0xff]);
    }

    #[test]
    fn base64_rejects_invalid_characters_and_padding() {
        assert!(decode_base64("Zm9v!").is_err());
        assert!(decode_base64("Zg=X").is_err());
        assert!(decode_base64("Z").is_err());
        assert!(decode_base64("Zg===").is_err());
    }

    #[test]
    fn base64_skips_line_breaks() {
        assert_eq!(decode_base64("Zm9v\r\nYg==").unwrap(), b"foob");
    }

    #[test]
    fn base64_names_the_offending_character() {
        // A full 4-character group so the length check passes and the
        // per-character alphabet lookup is what rejects the input.
        let err = decode_base64("Zm9!").unwrap_err();
        assert_eq!(err, "invalid base64 character '!'");
        assert_eq!(
            decode_base64("Zm 9").unwrap_err(),
            "invalid base64 character ' '"
        );
    }

    // image bytes

    #[test]
    fn mime_from_uri_infers_a_type_from_the_extension_or_data_header() {
        assert_eq!(mime_from_uri("albedo.PNG").as_deref(), Some("image/png"));
        assert_eq!(mime_from_uri("albedo.jpg").as_deref(), Some("image/jpeg"));
        assert_eq!(mime_from_uri("albedo.jpeg").as_deref(), Some("image/jpeg"));
        assert_eq!(
            mime_from_uri("data:image/png;base64,iVBOR").as_deref(),
            Some("image/png")
        );
        assert_eq!(mime_from_uri("albedo.webp"), None);
    }

    #[test]
    fn image_bytes_reads_a_data_uri_image_and_infers_its_type() {
        let mut json = crate::import::glb::test_fixtures::static_triangle_json();
        json["images"] = serde_json::json!([{"uri": "data:image/png;base64,Zm9vYg=="}]);
        let glb = crate::import::glb::test_fixtures::make_glb(
            &json,
            Some(&crate::import::glb::test_fixtures::static_triangle_bin()),
        );
        let doc = GltfDoc::from_slice(&glb, None, "t.glb").expect("parse");
        let (bytes, mime) = doc.image_bytes(0).expect("image bytes");
        assert_eq!(bytes, b"foob");
        assert_eq!(mime.as_deref(), Some("image/png"));
    }

    #[test]
    fn image_bytes_reports_an_index_past_the_end() {
        let doc = GltfDoc::from_slice(
            &crate::import::glb::test_fixtures::static_triangle_glb(),
            None,
            "t.glb",
        )
        .expect("parse");
        let err = doc.image_bytes(0).unwrap_err();
        assert_eq!(err, "image_index 0 is out of range");
    }

    // percent decoding

    #[test]
    fn percent_decode_handles_escapes_and_passthrough() {
        assert_eq!(percent_decode("my%20file.bin"), "my file.bin");
        assert_eq!(percent_decode("plain.bin"), "plain.bin");
        assert_eq!(percent_decode("odd%2"), "odd%2");
        assert_eq!(percent_decode("100%"), "100%");
    }

    // safe_relative_path

    #[test]
    fn safe_relative_path_accepts_plain_relative_paths() {
        assert_eq!(safe_relative_path("a.bin").unwrap(), PathBuf::from("a.bin"));
        assert_eq!(
            safe_relative_path("tex/albedo.png").unwrap(),
            PathBuf::from("tex/albedo.png")
        );
    }

    #[test]
    fn safe_relative_path_rejects_escapes() {
        assert!(safe_relative_path("../a.bin").is_err());
        assert!(safe_relative_path("x/../../a.bin").is_err());
        assert!(safe_relative_path("/abs/a.bin").is_err());
        assert!(safe_relative_path("http://host/a.bin").is_err());
    }

    // resolve_uri_bytes

    #[test]
    fn resolve_uri_decodes_a_base64_data_uri() {
        let out = resolve_uri_bytes("data:application/octet-stream;base64,Zm9vYg==", None).unwrap();
        assert_eq!(out, b"foob");
    }

    #[test]
    fn resolve_uri_rejects_non_base64_data_uris() {
        let err = resolve_uri_bytes("data:text/plain,hello", None).unwrap_err();
        assert!(err.contains("only base64"), "got: {err}");
    }

    #[test]
    fn resolve_uri_rejects_a_data_uri_without_a_comma() {
        let err = resolve_uri_bytes("data:application/octet-stream;base64", None).unwrap_err();
        assert_eq!(err, "malformed data URI (no comma separator)");
    }

    #[test]
    fn resolve_uri_reports_undecodable_base64() {
        let err = resolve_uri_bytes("data:application/octet-stream;base64,Zm9!", None).unwrap_err();
        assert!(err.starts_with("invalid base64 data URI"), "got: {err}");
    }

    #[test]
    fn resolve_uri_reads_a_sibling_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("geo.bin"), b"payload").unwrap();
        let out = resolve_uri_bytes("geo.bin", Some(dir.path())).unwrap();
        assert_eq!(out, b"payload");
    }

    #[test]
    fn resolve_uri_reports_a_missing_sibling_file() {
        let dir = tempfile::tempdir().unwrap();
        let err = resolve_uri_bytes("missing.bin", Some(dir.path())).unwrap_err();
        assert!(err.contains("failed to read external file"), "got: {err}");
    }

    #[test]
    fn resolve_uri_rejects_a_path_escaping_the_source_directory() {
        let dir = tempfile::tempdir().unwrap();
        let err = resolve_uri_bytes("../secrets.bin", Some(dir.path())).unwrap_err();
        assert!(err.contains("escapes the source directory"), "got: {err}");
    }

    #[test]
    fn resolve_uri_without_a_base_dir_rejects_file_uris() {
        let err = resolve_uri_bytes("geo.bin", None).unwrap_err();
        assert!(err.contains("without a source directory"), "got: {err}");
    }

    // GltfDoc

    // A text .gltf JSON referencing `geo.bin` (one triangle, indexed) written
    // beside it, mirroring crate::import::glb::test_fixtures::static_triangle_*.
    fn triangle_gltf_json(buffer_uri: &str) -> serde_json::Value {
        let mut json = crate::import::glb::test_fixtures::static_triangle_json();
        json["buffers"][0]["uri"] = buffer_uri.into();
        json
    }

    #[test]
    fn parse_file_resolves_an_external_buffer() {
        let dir = tempfile::tempdir().unwrap();
        let bin = crate::import::glb::test_fixtures::static_triangle_bin();
        std::fs::write(dir.path().join("geo.bin"), &bin).unwrap();
        let gltf_path = dir.path().join("tri.gltf");
        std::fs::write(
            &gltf_path,
            serde_json::to_vec(&triangle_gltf_json("geo.bin")).unwrap(),
        )
        .unwrap();

        let doc = GltfDoc::parse_file(gltf_path.to_str().unwrap()).expect("parse");
        let buffer = doc.doc.document.buffers().next().unwrap();
        assert_eq!(doc.buffer_bytes(buffer), Some(bin.as_slice()));
    }

    #[test]
    fn parse_file_resolves_a_data_uri_buffer() {
        let dir = tempfile::tempdir().unwrap();
        let bin = crate::import::glb::test_fixtures::static_triangle_bin();
        let mut encoded = String::new();
        for chunk in bin.chunks(3) {
            encoded.push_str(&encode_base64_chunk(chunk));
        }
        let uri = format!("data:application/octet-stream;base64,{encoded}");
        let gltf_path = dir.path().join("tri.gltf");
        std::fs::write(
            &gltf_path,
            serde_json::to_vec(&triangle_gltf_json(&uri)).unwrap(),
        )
        .unwrap();

        let doc = GltfDoc::parse_file(gltf_path.to_str().unwrap()).expect("parse");
        let buffer = doc.doc.document.buffers().next().unwrap();
        assert_eq!(doc.buffer_bytes(buffer), Some(bin.as_slice()));
    }

    #[test]
    fn parse_file_reports_a_missing_external_buffer() {
        let dir = tempfile::tempdir().unwrap();
        let gltf_path = dir.path().join("tri.gltf");
        std::fs::write(
            &gltf_path,
            serde_json::to_vec(&triangle_gltf_json("missing.bin")).unwrap(),
        )
        .unwrap();
        let err = GltfDoc::parse_file(gltf_path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("missing.bin"), "got: {err}");
        assert!(err.contains("buffer 0"), "got: {err}");
    }

    #[test]
    fn from_slice_still_reads_a_glb_binary_chunk() {
        let glb = crate::import::glb::test_fixtures::static_triangle_glb();
        let doc = GltfDoc::from_slice(&glb, None, "t.glb").expect("parse");
        let buffer = doc.doc.document.buffers().next().unwrap();
        assert!(doc.buffer_bytes(buffer).is_some());
    }

    #[test]
    fn from_slice_leaves_a_missing_glb_chunk_unresolved() {
        let glb = crate::import::glb::test_fixtures::make_glb(
            &crate::import::glb::test_fixtures::static_triangle_json(),
            None,
        );
        let doc = GltfDoc::from_slice(&glb, None, "t.glb").expect("parse");
        let buffer = doc.doc.document.buffers().next().unwrap();
        assert!(doc.buffer_bytes(buffer).is_none());
    }

    // referenced_files

    #[test]
    fn referenced_files_lists_buffer_and_image_uris() {
        let dir = tempfile::tempdir().unwrap();
        let mut json = triangle_gltf_json("geo.bin");
        json["images"] = serde_json::json!([
            {"uri": "tex%20map.png"},
            {"uri": "data:image/png;base64,AAAA"},
            {"uri": "../outside.png"},
            {"bufferView": 1, "mimeType": "image/png"}
        ]);
        let gltf_path = dir.path().join("tri.gltf");
        std::fs::write(&gltf_path, serde_json::to_vec(&json).unwrap()).unwrap();

        let files = referenced_files(gltf_path.to_str().unwrap(), None);
        assert_eq!(
            files.len(),
            2,
            "data URIs, bufferView images, and paths outside the source \
             directory must not be listed: {files:?}"
        );
        assert!(files.iter().any(|f| f.ends_with("geo.bin")), "{files:?}");
        assert!(
            files.iter().any(|f| f.ends_with("tex map.png")),
            "{files:?}"
        );
    }

    #[test]
    fn referenced_files_is_empty_for_glb_and_missing_sources() {
        let dir = tempfile::tempdir().unwrap();
        let glb_path = dir.path().join("t.glb");
        std::fs::write(
            &glb_path,
            crate::import::glb::test_fixtures::static_triangle_glb(),
        )
        .unwrap();
        assert!(referenced_files(glb_path.to_str().unwrap(), None).is_empty());
        assert!(referenced_files("/no/such/file.gltf", None).is_empty());
    }

    #[test]
    fn referenced_files_tracks_source_edits() {
        let dir = tempfile::tempdir().unwrap();
        let gltf_path = dir.path().join("tri.gltf");
        std::fs::write(
            &gltf_path,
            serde_json::to_vec(&triangle_gltf_json("a.bin")).unwrap(),
        )
        .unwrap();
        let first = referenced_files(gltf_path.to_str().unwrap(), None);
        assert!(first[0].ends_with("a.bin"));

        std::fs::write(
            &gltf_path,
            serde_json::to_vec(&triangle_gltf_json("b.bin")).unwrap(),
        )
        .unwrap();
        let second = referenced_files(gltf_path.to_str().unwrap(), None);
        assert!(
            second[0].ends_with("b.bin"),
            "memo must not serve stale: {second:?}"
        );
    }

    #[test]
    fn scan_referenced_files_tolerates_unreadable_and_unparseable_sources() {
        let dir = tempfile::tempdir().unwrap();
        assert!(scan_referenced_files("/no/such/file.gltf").is_empty());

        let junk = dir.path().join("junk.gltf");
        std::fs::write(&junk, b"{ not glTF at all").unwrap();
        assert!(scan_referenced_files(junk.to_str().unwrap()).is_empty());
    }

    #[test]
    fn encode_base64_chunk_matches_the_decoder_for_every_chunk_length() {
        for payload in [&b"f"[..], b"fo", b"foo"] {
            let encoded = encode_base64_chunk(payload);
            assert_eq!(encoded.len(), 4, "always a padded quad: {encoded}");
            assert_eq!(decode_base64(&encoded).unwrap(), payload, "{encoded}");
        }
    }

    // Minimal encoder for test fixtures only.
    fn encode_base64_chunk(chunk: &[u8]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut acc: u32 = 0;
        for (i, &b) in chunk.iter().enumerate() {
            acc |= (b as u32) << (16 - 8 * i);
        }
        let mut out = String::new();
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[((acc >> (18 - 6 * i)) & 0x3f) as usize] as char);
            } else {
                out.push('=');
            }
        }
        out
    }
}
