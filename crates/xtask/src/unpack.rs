//! Unpacking a downloaded archive into its `vendor/` directory.

use std::fs;
use std::io::{self, Cursor};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;

use crate::tools::{remove_dir_quietly, staging_dir};
use crate::{Context, Result, fail};

/// Extract `archive` into `target`, replacing it only once it holds `payload`.
///
/// A release unpacks its payload either at the archive root or under one
/// directory named for the release; whichever holds it becomes `target`.
pub(crate) fn unpack(
    archive: &[u8],
    url: &str,
    payload: &str,
    target: &Path,
    keep: Option<&[&str]>,
) -> Result {
    let staging = staging_dir(target);
    remove_dir_quietly(&staging);
    fs::create_dir_all(&staging).context(staging.display())?;
    let placed = extract(archive, url, &staging, keep).and_then(|()| {
        let Some(root) = payload_root(&staging, payload) else {
            return fail(format!("{url}: contains no {payload}"));
        };
        remove_dir_quietly(target);
        fs::rename(&root, target).context(target.display())
    });
    remove_dir_quietly(&staging);
    placed
}

fn payload_root(staging: &Path, payload: &str) -> Option<PathBuf> {
    let subdirs = fs::read_dir(staging)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir());
    std::iter::once(staging.to_path_buf())
        .chain(subdirs)
        .find(|root| root.join(payload).is_file())
}

// By content, not by name: a NuGet package is a zip called `.nupkg`, and a
// release asset is free to be named anything at all.
fn extract(archive: &[u8], url: &str, into: &Path, keep: Option<&[&str]>) -> Result {
    if archive.starts_with(b"PK") {
        extract_zip(archive, url, into, keep)
    } else if archive.starts_with(&[0x1f, 0x8b]) {
        extract_tar_gz(archive, url, into, keep)
    } else {
        fail(format!("{url}: not a zip or a gzip archive"))
    }
}

fn extract_zip(archive: &[u8], url: &str, into: &Path, keep: Option<&[&str]>) -> Result {
    let mut zip = zip::ZipArchive::new(Cursor::new(archive)).context(url)?;
    for index in 0..zip.len() {
        let mut member = zip.by_index(index).context(url)?;
        let name = member.name().context(url)?.into_owned();
        if member.is_dir() || !wanted(keep, &name) {
            continue;
        }
        let Some(relative) = member.enclosed_name() else {
            return fail(format!("{url}: {name} escapes the archive"));
        };
        let path = into.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).context(parent.display())?;
        }
        let mut file = fs::File::create(&path).context(path.display())?;
        io::copy(&mut member, &mut file).context(path.display())?;
        mark_executable(&path)?;
    }
    Ok(())
}

// Zip carries no executable bit, so a file named like a program gets one.
#[cfg(unix)]
fn mark_executable(path: &Path) -> Result {
    use std::os::unix::fs::PermissionsExt;
    let program = path
        .extension()
        .is_none_or(|extension| extension.eq_ignore_ascii_case("exe"));
    if !program {
        return Ok(());
    }
    let mut permissions = fs::metadata(path).context(path.display())?.permissions();
    permissions.set_mode(permissions.mode() | 0o111);
    fs::set_permissions(path, permissions).context(path.display())
}

#[cfg(not(unix))]
#[expect(clippy::unnecessary_wraps, reason = "matches the unix signature")]
fn mark_executable(_path: &Path) -> Result {
    Ok(())
}

fn extract_tar_gz(archive: &[u8], url: &str, into: &Path, keep: Option<&[&str]>) -> Result {
    let mut tar = tar::Archive::new(GzDecoder::new(archive));
    for entry in tar.entries().context(url)? {
        let mut entry = entry.context(url)?;
        let name = entry.path().context(url)?.to_string_lossy().into_owned();
        if wanted(keep, &name) {
            // `unpack_in` refuses a member that would land outside `into`.
            entry.unpack_in(into).context(url)?;
        }
    }
    Ok(())
}

/// Whether to unpack the member `name`: every member, or those `keep` names.
///
/// A pattern is matched against the member's path both as it stands and below
/// its first directory, since a release unpacks either at the archive root or
/// under one directory named for the release.
pub(crate) fn wanted(keep: Option<&[&str]>, name: &str) -> bool {
    let Some(patterns) = keep else {
        return true;
    };
    let normalized = name.replace('\\', "/");
    let full = normalized.trim_matches('/');
    let below_first = full.split_once('/').map_or("", |(_, rest)| rest);
    patterns
        .iter()
        .any(|pattern| glob(pattern, full) || glob(pattern, below_first))
}

/// Shell-style matching: `*` matches any run of characters, slashes included,
/// `?` any one, and `[...]` one of a set (`!` negates, `a-z` is a range).
fn glob(pattern: &str, name: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let name: Vec<char> = name.chars().collect();
    glob_chars(&pattern, &name)
}

fn glob_chars(pattern: &[char], name: &[char]) -> bool {
    match pattern.split_first() {
        None => name.is_empty(),
        Some(('*', rest)) => (0..=name.len()).any(|skip| glob_chars(rest, &name[skip..])),
        Some(('?', rest)) => !name.is_empty() && glob_chars(rest, &name[1..]),
        Some(('[', rest)) => match class(rest) {
            Some((set, after)) => name
                .split_first()
                .is_some_and(|(c, tail)| set.matches(*c) && glob_chars(after, tail)),
            // An unclosed bracket is a literal one.
            None => name.first() == Some(&'[') && glob_chars(rest, &name[1..]),
        },
        Some((c, rest)) => name.first() == Some(c) && glob_chars(rest, &name[1..]),
    }
}

struct Class<'a> {
    negated: bool,
    items: &'a [char],
}

impl Class<'_> {
    fn matches(&self, c: char) -> bool {
        let mut found = false;
        let mut i = 0;
        while i < self.items.len() {
            match (self.items[i], self.items.get(i + 1), self.items.get(i + 2)) {
                (low, Some('-'), Some(&high)) => {
                    found |= (low..=high).contains(&c);
                    i += 3;
                }
                (single, ..) => {
                    found |= single == c;
                    i += 1;
                }
            }
        }
        found != self.negated
    }
}

// The set after an opening `[`, and the pattern after its closing `]`. A `]`
// right after the opening (or its `!`) belongs to the set.
fn class(rest: &[char]) -> Option<(Class<'_>, &[char])> {
    let negated = rest.first() == Some(&'!');
    let body = if negated { &rest[1..] } else { rest };
    let close = body.iter().skip(1).position(|&c| c == ']')? + 1;
    Some((
        Class {
            negated,
            items: &body[..close],
        },
        &body[close + 1..],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const LINUX_DXC: &[&str] = &[
        "bin/dxc",
        "lib/libdxcompiler.so",
        "lib/libdxil.so",
        "LICEN[CS]E*",
    ];
    const WINDOWS_DXC: &[&str] = &[
        "bin/x64/dxc.exe",
        "bin/x64/dxcompiler.dll",
        "bin/x64/dxil.dll",
        "LICEN[CS]E*",
    ];

    #[test]
    fn the_linux_dxc_archive_is_cut_to_the_compiler() {
        let members = [
            ("linux_dxc_2026_07_29.x86_x64/bin/dxc", true),
            ("linux_dxc_2026_07_29.x86_x64/lib/libdxcompiler.so", true),
            ("linux_dxc_2026_07_29.x86_x64/LICENCE-MIT.txt", true),
            ("linux_dxc_2026_07_29.x86_x64/LICENSE-LLVM.txt", true),
            ("linux_dxc_2026_07_29.x86_x64/lib/libLLVMSupport.a", false),
            ("linux_dxc_2026_07_29.x86_x64/bin/opt", false),
            ("bin/dxc", true),
        ];
        for (name, want) in members {
            assert_eq!(wanted(Some(LINUX_DXC), name), want, "{name}");
        }
    }

    #[test]
    fn the_windows_dxc_archive_is_cut_to_the_compiler() {
        let members = [
            ("bin\\x64\\dxc.exe", true),
            ("bin\\x64\\dxil.dll", true),
            ("bin\\arm64\\dxc.exe", false),
            ("inc\\hlsl\\LICENCE.txt", false),
            ("LICENSE-MS.txt", true),
        ];
        for (name, want) in members {
            assert_eq!(wanted(Some(WINDOWS_DXC), name), want, "{name}");
        }
    }

    #[test]
    fn naming_no_members_unpacks_all_of_them() {
        assert!(wanted(None, "anything/at/all"));
    }

    #[test]
    fn globs_follow_shell_rules() {
        assert!(glob("a*c", "abbc"));
        assert!(glob("a*", "a/b/c"));
        assert!(glob("a?c", "abc"));
        assert!(!glob("a?c", "ac"));
        assert!(glob("[a-c]x", "bx"));
        assert!(!glob("[!a-c]x", "bx"));
        assert!(glob("[]]", "]"));
        assert!(glob("a[b", "a[b"));
        assert!(!glob("abc", "abcd"));
    }

    fn zip_of(members: &[(&str, &[u8])]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, bytes) in members {
            writer
                .start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(bytes).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    fn tar_gz_of(members: &[(&str, &[u8])]) -> Vec<u8> {
        let gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        let mut builder = tar::Builder::new(gz);
        for (name, bytes) in members {
            let mut header = tar::Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o755);
            builder.append_data(&mut header, name, *bytes).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap()
    }

    #[test]
    fn a_release_directory_inside_the_archive_becomes_the_target() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("vendor/sdk-1.0-windows-x86_64");
        let archive = zip_of(&[("SDK-v1.0/bin/sdk.dll", b"dll"), ("SDK-v1.0/README", b"")]);
        unpack(&archive, "url", "bin/sdk.dll", &target, None).unwrap();
        assert_eq!(fs::read(target.join("bin/sdk.dll")).unwrap(), b"dll");
        assert!(!staging_dir(&target).exists());
    }

    #[test]
    fn a_tar_gz_unpacks_only_what_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("dxc-1-linux-x86_64");
        let archive = tar_gz_of(&[
            ("linux_dxc/bin/dxc", b"exe"),
            ("linux_dxc/bin/opt", b"exe"),
            ("linux_dxc/LICENSE-MIT.txt", b"mit"),
        ]);
        unpack(&archive, "url", "bin/dxc", &target, Some(LINUX_DXC)).unwrap();
        assert!(target.join("bin/dxc").is_file());
        assert!(target.join("LICENSE-MIT.txt").is_file());
        assert!(!target.join("bin/opt").exists());
    }

    #[cfg(unix)]
    #[test]
    fn zipped_programs_come_out_executable() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("tool");
        unpack(
            &zip_of(&[("bin/tool", b"")]),
            "url",
            "bin/tool",
            &target,
            None,
        )
        .unwrap();
        let mode = fs::metadata(target.join("bin/tool"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o111, 0o111);
    }

    #[test]
    fn an_archive_without_the_payload_leaves_the_target_alone() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("sdk");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("kept"), b"").unwrap();
        let archive = zip_of(&[("other.dll", b"")]);
        let Err(failure) = unpack(&archive, "url", "bin/sdk.dll", &target, None) else {
            panic!("an archive without its payload unpacked");
        };
        assert!(failure.0.contains("contains no bin/sdk.dll"));
        assert!(target.join("kept").is_file());
        assert!(!staging_dir(&target).exists());
    }

    #[test]
    fn something_neither_zip_nor_gzip_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let result = unpack(b"<html>", "url", "x", &dir.path().join("t"), None);
        assert!(result.is_err());
    }
}
