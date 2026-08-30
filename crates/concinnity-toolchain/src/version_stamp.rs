//! The version stamp a build bakes into what it produces: the commit the
//! source was built from, and the day.
//!
//! A checkout answers both from git. A source tree that is not a checkout --
//! a crates.io install, an unpacked archive -- has no commit to name, so only
//! the build date is stamped, and the renderer says so.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// The env vars the stamp is read back through. Both are always emitted;
/// `COMMIT` is empty when the source tree is not a checkout.
const COMMIT: &str = "CONCINNITY_COMMIT";
const DATE: &str = "CONCINNITY_STAMP_DATE";

/// Emit the calling package's version stamp as build-script env vars, plus the
/// rerun directives that restamp it: the git refs when there are any, and the
/// package sources otherwise, so a rebuilt tree never reports a stale date.
pub fn emit_version_stamp() {
    let dir =
        std::env::var("CARGO_MANIFEST_DIR").map_or_else(|_| PathBuf::from("."), PathBuf::from);
    let stamp = git_stamp(&dir);

    for path in watched_paths(&dir, stamp.is_some()) {
        println!("cargo::rerun-if-changed={}", path.display());
    }

    let (commit, date) = stamp.unwrap_or_else(|| (String::new(), today()));
    println!("cargo::rustc-env={COMMIT}={commit}");
    println!("cargo::rustc-env={DATE}={date}");
}

// The commit `dir` sits on and the day it was authored, or `None` when git is
// absent, the tree is not a checkout, or the repository has no commits yet.
fn git_stamp(dir: &Path) -> Option<(String, String)> {
    let out = Command::new("git")
        .args(["-C"])
        .arg(dir)
        .args(["log", "-1", "--abbrev=9", "--date=short", "--format=%h %cd"])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| parse_stamp(&String::from_utf8_lossy(&out.stdout)))?
}

// "<short hash> <YYYY-MM-DD>" as git prints it. Anything else is discarded
// rather than stamped: a malformed stamp reads as a real one downstream.
fn parse_stamp(output: &str) -> Option<(String, String)> {
    let (commit, date) = output.trim().split_once(' ')?;
    let hex = !commit.is_empty() && commit.chars().all(|c| c.is_ascii_hexdigit());
    (hex && is_iso_date(date)).then(|| (commit.to_string(), date.to_string()))
}

fn is_iso_date(date: &str) -> bool {
    let parts: Vec<&str> = date.split('-').collect();
    parts.len() == 3
        && parts
            .iter()
            .zip([4, 2, 2])
            .all(|(part, width)| part.len() == width && part.chars().all(|c| c.is_ascii_digit()))
}

// What a change to must restamp the build. Under git that is the refs, so a
// commit or a checkout moves the hash; without it the package's own sources,
// which is the closest thing to "this build is new".
fn watched_paths(dir: &Path, versioned: bool) -> Vec<PathBuf> {
    let candidates = if versioned {
        let git = git_dir(dir);
        vec![git.join("HEAD"), git.join("refs"), git.join("packed-refs")]
    } else {
        vec![dir.join("src")]
    };
    // Cargo treats a directive naming a missing path as always-changed, which
    // would re-run the script on every build.
    candidates
        .into_iter()
        .filter(|path| path.exists())
        .collect()
}

fn git_dir(dir: &Path) -> PathBuf {
    let out = Command::new("git")
        .args(["-C"])
        .arg(dir)
        .args(["rev-parse", "--absolute-git-dir"])
        .output()
        .ok()
        .filter(|out| out.status.success());
    out.map_or_else(
        || dir.join(".git"),
        |out| PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()),
    )
}

// Today, UTC. A clock behind the epoch is not a date worth reporting, so it
// stamps the epoch itself rather than failing the build.
fn today() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    iso_date(secs / 86_400)
}

// Days since 1970-01-01 as YYYY-MM-DD, by Howard Hinnant's civil_from_days.
fn iso_date(days: u64) -> String {
    let z = days as i64 + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_git_line_splits_into_a_commit_and_a_date() {
        assert_eq!(
            parse_stamp("c980f4866 2026-06-30\n"),
            Some(("c980f4866".to_string(), "2026-06-30".to_string()))
        );
    }

    #[test]
    fn a_malformed_git_line_is_not_a_stamp() {
        for line in [
            "",
            "c980f4866",
            "zzzzzzzzz 2026-06-30",
            "c980f4866 30-06-2026",
            "c980f4866 2026-6-30",
            "fatal: not a git repository",
        ] {
            assert_eq!(parse_stamp(line), None, "{line:?} should not stamp");
        }
    }

    #[test]
    fn days_since_the_epoch_render_as_civil_dates() {
        for (days, date) in [
            (0, "1970-01-01"),
            (58, "1970-02-28"),
            (365, "1971-01-01"),
            (10_957, "2000-01-01"),
            (19_782, "2024-02-29"),
            (20_696, "2026-08-31"),
        ] {
            assert_eq!(iso_date(days), date, "{days} days after the epoch");
        }
    }

    #[test]
    fn today_is_an_iso_date() {
        let today = today();
        assert!(is_iso_date(&today), "{today} is not YYYY-MM-DD");
    }

    // The whole git path end to end: a real repository, its own commit read
    // back, and the ref files that restamp it. Skipped where git is absent,
    // which is the same fallback the build script takes.
    #[test]
    fn a_checkout_stamps_its_own_commit() {
        let dir = tempfile::tempdir().expect("temp dir");
        let git = |args: &[&str]| {
            Command::new("git")
                .arg("-C")
                .arg(dir.path())
                .args(args)
                .output()
        };
        let Ok(init) = git(&["init", "--quiet"]) else {
            eprintln!("[skip] a_checkout_stamps_its_own_commit: no git on PATH");
            return;
        };
        assert!(init.status.success(), "git init failed");
        assert!(
            git_stamp(dir.path()).is_none(),
            "an empty repo has no commit"
        );

        std::fs::write(dir.path().join("a.txt"), "stamp").expect("write");
        assert!(git(&["add", "."]).expect("git add").status.success());
        let commit = git(&[
            "-c",
            "user.email=ci@concinnity.gg",
            "-c",
            "user.name=ci",
            "commit",
            "--quiet",
            "-m",
            "stamp",
        ])
        .expect("git commit");
        assert!(commit.status.success(), "git commit failed");

        let (hash, date) = git_stamp(dir.path()).expect("a committed tree stamps");
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()), "{hash}");
        assert!(is_iso_date(&date), "{date}");

        let watched = watched_paths(dir.path(), true);
        assert!(
            watched.iter().any(|path| path.ends_with("HEAD")),
            "a checkout should restamp when HEAD moves: {watched:?}"
        );
    }

    // Without the filter Cargo would see an always-changed input and re-run
    // the stamp on every build.
    #[test]
    fn only_existing_paths_are_watched() {
        let dir = tempfile::tempdir().expect("temp dir");
        assert!(watched_paths(dir.path(), false).is_empty());
        std::fs::create_dir(dir.path().join("src")).expect("src");
        assert_eq!(
            watched_paths(dir.path(), false),
            vec![dir.path().join("src")]
        );
    }
}
