//! The engine shader artifacts an exported app ships: every built-in program
//! compiled ahead of time for the player's graphics APIs, in one read-only file
//! beside its data. A built-in program's artifact comes from here when it was
//! compiled from the exact source the renderer assembles, so a player needs no
//! shader compiler; otherwise it comes from the runtime cache or a compile, as
//! in a development run, which ships none.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

use concinnity_core::platform::Platform;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::shader::source::source_digest;
use concinnity_shader::HlslTarget;

use super::stamp::SHIPPED_SHADER_STAMP;

// Leads the file, with the format's version and the stamp of the build that
// compiled it after it.
const MAGIC: &[u8; 8] = b"cnshadr\0";
const VERSION: u32 = 2;

/// One shipped artifact: the digest of the source it was compiled from, and
/// its bytes.
pub(crate) type Shipped = (u64, &'static [u8]);

// The shipped artifacts, keyed by platform and artifact name.
type Artifacts = BTreeMap<(String, String), (u64, Vec<u8>)>;

static SHIPPED: OnceLock<Artifacts> = OnceLock::new();

/// Load the engine shader artifacts an export shipped at `path`, for the life
/// of the process. Call it before the first backend starts; a missing,
/// unreadable or mismatched file leaves every program to the runtime cache and
/// a compile.
pub fn use_shipped_shaders(path: &Path) {
    let artifacts = match std::fs::read(path) {
        Ok(bytes) => match decode(&bytes, SHIPPED_SHADER_STAMP) {
            Ok(artifacts) => artifacts,
            Err(e) => {
                tracing::warn!("shipped shaders at {} are unusable: {e}", path.display());
                return;
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => {
            tracing::warn!("shipped shaders at {} unreadable: {e}", path.display());
            return;
        }
    };
    let count = artifacts.len();
    if SHIPPED.set(artifacts).is_ok() {
        tracing::info!("shipped shaders: {count} artifacts from {}", path.display());
    }
}

// The shipped artifact `name` for `platform`, if the export shipped one.
fn lookup(platform: Platform, name: &str) -> Option<Shipped> {
    SHIPPED
        .get()?
        .get(&(platform.key().to_string(), name.to_string()))
        .map(|(digest, bytes)| (*digest, bytes.as_slice()))
}

/// Compile every engine program for `platforms` into the file an export ships,
/// returning its bytes and how many artifacts it holds.
///
/// # Errors
///
/// When this build has no backend for one of `platforms`, or a program fails
/// to compile.
pub fn precompile_shipped_shaders(platforms: &[Platform]) -> Result<(Vec<u8>, usize), String> {
    use rayon::prelude::*;
    let mut set = ShippedSet::default();
    for &platform in platforms {
        let failed = |e: RenderError| format!("{}: {e}", platform.key());
        let plan = plan(platform).map_err(failed)?;
        let compiled: Vec<RenderResult<Vec<u8>>> = plan
            .par_iter()
            .map(|planned| planned.compile().map_err(|e| e.context(&planned.name)))
            .collect();
        for (planned, bytes) in plan.into_iter().zip(compiled) {
            let digest = source_digest(&planned.source);
            set.insert(platform, planned.name, digest, bytes.map_err(failed)?);
        }
    }
    Ok((set.encode(), set.len()))
}

// Every artifact `platform`'s backend fetches by name, as a player assembles it.
fn plan(platform: Platform) -> RenderResult<Vec<Planned>> {
    #[cfg(backend_metal)]
    if platform == Platform::Metal {
        return Ok(crate::metal::builtin_shaders::shipped_plan());
    }
    #[cfg(backend_dx)]
    if platform == Platform::DirectX {
        return Ok(crate::directx::builtin_shaders::shipped_plan());
    }
    #[cfg(backend_vk)]
    if platform == Platform::Vulkan {
        return Ok(crate::vulkan::builtin_shaders::shipped_plan());
    }
    Err(RenderError::Other(format!(
        "this build has no backend for {}",
        platform.key()
    )))
}

/// One built-in artifact as a player fetches it: the name it is looked up by,
/// the exact source a shipped copy must have been compiled from, and how that
/// source compiles. The runtime fetch and an export's precompile both build it
/// through the same constructor, so the two cannot key different text.
pub(crate) struct Planned {
    pub(crate) name: String,
    pub(crate) source: String,
    pub(crate) compile: Compile,
}

/// How a planned artifact's source compiles.
#[derive(Clone, Copy)]
pub(crate) enum Compile {
    /// One HLSL entry through dxc to `target`.
    Hlsl {
        file: &'static str,
        entry: &'static str,
        target: HlslTarget,
    },
    /// MSL text through the Metal toolchain to a metallib.
    #[cfg(backend_metal)]
    Msl,
}

impl Planned {
    /// The copy of this artifact an export shipped for `platform`, if any.
    pub(crate) fn shipped(&self, platform: Platform) -> Option<Shipped> {
        lookup(platform, &self.name)
    }

    /// Compile the source, bypassing every cache.
    pub(crate) fn compile(&self) -> RenderResult<Vec<u8>> {
        match self.compile {
            Compile::Hlsl {
                file,
                entry,
                target,
            } => super::compile::compile(file, entry, &self.source, target),
            #[cfg(backend_metal)]
            Compile::Msl => crate::metal::msl_cache::compile_to_metallib(&self.source, &self.name),
        }
    }
}

/// Artifacts being collected for a file to ship.
#[derive(Default)]
pub(crate) struct ShippedSet(Artifacts);

impl ShippedSet {
    /// Add `name`'s artifact for `platform`, compiled from source of `digest`.
    pub(crate) fn insert(&mut self, platform: Platform, name: String, digest: u64, bytes: Vec<u8>) {
        self.0
            .insert((platform.key().to_string(), name), (digest, bytes));
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// The file holding every artifact collected.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&SHIPPED_SHADER_STAMP.to_le_bytes());
        put_len(&mut out, self.0.len());
        for ((platform, name), (digest, bytes)) in &self.0 {
            put_bytes(&mut out, platform.as_bytes());
            put_bytes(&mut out, name.as_bytes());
            out.extend_from_slice(&digest.to_le_bytes());
            put_bytes(&mut out, bytes);
        }
        out
    }
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_len(out, bytes.len());
    out.extend_from_slice(bytes);
}

// The artifacts in a shipped file, which must carry `stamp`.
fn decode(bytes: &[u8], stamp: u64) -> Result<Artifacts, String> {
    let mut reader = Reader(bytes);
    if reader.take(MAGIC.len())? != MAGIC {
        return Err("not a shipped shader file".to_string());
    }
    let version = u32::from_le_bytes(reader.array()?);
    if version != VERSION {
        return Err(format!("format version {version}, not {VERSION}"));
    }
    let shipped = u64::from_le_bytes(reader.array()?);
    if shipped != stamp {
        return Err(format!(
            "compiled by a build stamped {shipped:016x}, not this player's {stamp:016x}; \
             export it again with the `concinnity` built beside this player"
        ));
    }
    let count = reader.len()?;
    let mut artifacts = Artifacts::new();
    for _ in 0..count {
        let platform = reader.text()?;
        let name = reader.text()?;
        let digest = u64::from_le_bytes(reader.array()?);
        let len = reader.len()?;
        let bytes = reader.take(len)?.to_vec();
        artifacts.insert((platform, name), (digest, bytes));
    }
    if !reader.0.is_empty() {
        return Err("trailing bytes".to_string());
    }
    Ok(artifacts)
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.0.len() {
            return Err("truncated".to_string());
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn len(&mut self) -> Result<usize, String> {
        usize::try_from(u64::from_le_bytes(self.array()?)).map_err(|e| e.to_string())
    }

    fn text(&mut self) -> Result<String, String> {
        let len = self.len()?;
        String::from_utf8(self.take(len)?.to_vec()).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shipped_file_reads_back_what_was_written() {
        let mut set = ShippedSet::default();
        set.insert(Platform::DirectX, "a.hlsl".into(), 7, vec![1, 2, 3]);
        set.insert(Platform::Vulkan, "a.hlsl".into(), 9, vec![4]);
        let decoded = decode(&set.encode(), SHIPPED_SHADER_STAMP).unwrap();
        assert_eq!(decoded, set.0);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn a_damaged_file_is_refused() {
        let mut set = ShippedSet::default();
        set.insert(Platform::Metal, "b".into(), 1, vec![0; 16]);
        let bytes = set.encode();
        assert!(decode(&bytes[..bytes.len() - 1], SHIPPED_SHADER_STAMP).is_err());
        assert!(decode(b"something else entirely", SHIPPED_SHADER_STAMP).is_err());
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(decode(&longer, SHIPPED_SHADER_STAMP).is_err());
        let mut other_version = bytes.clone();
        other_version[MAGIC.len()] = 1;
        assert!(decode(&other_version, SHIPPED_SHADER_STAMP).is_err());
        assert!(decode(&bytes, SHIPPED_SHADER_STAMP ^ 1).is_err());
    }

    #[test]
    fn nothing_is_shipped_until_a_file_is_loaded() {
        assert!(lookup(Platform::Metal, "never-shipped").is_none());
    }
}
