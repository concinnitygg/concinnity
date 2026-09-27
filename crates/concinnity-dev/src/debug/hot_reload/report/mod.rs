//! What became of a reload, shared by every kind of source a save recompiles:
//! a world `Shader` and an `SdfVolume`'s field. Each report names its subject
//! by kind and asset name, is logged, toasted in an editor session, and
//! published to the board an editor's panels read.

mod board;
mod failure;
mod toast;

use concinnity_cook::compile::program::Diagnostic;
use std::time::Duration;

pub(crate) use board::{ReloadReports, ReportBoard};
pub(crate) use failure::ReloadFailure;

// The kind of asset a reload rebuilt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum SubjectKind {
    Shader,
    SdfVolume,
}

impl SubjectKind {
    // The asset type, as a message names one of it.
    pub(crate) fn label(self) -> &'static str {
        match self {
            SubjectKind::Shader => "Shader",
            SubjectKind::SdfVolume => "SdfVolume",
        }
    }

    // The asset type, as a message names several.
    fn plural(self) -> &'static str {
        match self {
            SubjectKind::Shader => "Shaders",
            SubjectKind::SdfVolume => "SdfVolumes",
        }
    }

    // What a swap rebuilt.
    fn rebuilt(self) -> &'static str {
        match self {
            SubjectKind::Shader => "pipeline",
            SubjectKind::SdfVolume => "pipelines",
        }
    }

    // Why nothing was swapped when nothing was resident, and when the edit
    // lands instead.
    fn not_resident(self) -> &'static str {
        match self {
            SubjectKind::Shader => "its scene is not loaded, so the edit installs when it loads",
            SubjectKind::SdfVolume => {
                "it has no live pipeline, so the edit applies when the world is next built"
            }
        }
    }

    // The same, short enough for a toast.
    fn applies_when(self) -> &'static str {
        match self {
            SubjectKind::Shader => "applies when its scene loads",
            SubjectKind::SdfVolume => "applies when the world is next built",
        }
    }
}

// The asset a reload rebuilt: its kind and its name. Two kinds may share a
// name, so a name alone does not identify a subject.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ReloadSubject {
    pub kind: SubjectKind,
    pub name: String,
}

impl ReloadSubject {
    pub(crate) fn shader(name: &str) -> Self {
        Self {
            kind: SubjectKind::Shader,
            name: name.to_string(),
        }
    }

    pub(crate) fn sdf_volume(name: &str) -> Self {
        Self {
            kind: SubjectKind::SdfVolume,
            name: name.to_string(),
        }
    }
}

// What became of one subject's reload. Warnings and errors name the subject's
// own files by the resolved path the recompile read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReloadOutcome {
    // The live pipelines were rebuilt from the edit, taking `frame_time` on
    // the frame thread.
    Swapped {
        frame_time: Duration,
        warnings: Vec<Diagnostic>,
    },
    // Nothing the edit feeds is resident; it applies when the subject is next
    // built (a Shader's scene loading, a volume's world building).
    AppliesOnLoad {
        warnings: Vec<Diagnostic>,
    },
    // The live pipelines keep their previous source.
    Failed(ReloadFailure),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReloadReport {
    pub subject: ReloadSubject,
    pub outcome: ReloadOutcome,
}

// Log each report and, in an editor session, toast them, one toast for the
// subjects of a kind that came to the same end.
pub(crate) fn report(reports: &[ReloadReport], notify: Option<&crate::editor::notify::Notifier>) {
    for report in reports {
        log(report);
    }
    let Some(n) = notify else {
        return;
    };
    for toast in toast::toasts(reports) {
        if toast.error {
            n.error_with(&toast.text, crate::editor::notify::Action::OpenConsole);
        } else {
            n.success(&toast.text);
        }
    }
}

fn log(ReloadReport { subject, outcome }: &ReloadReport) {
    let (kind, name) = (subject.kind, &subject.name);
    let what = kind.label();
    match outcome {
        ReloadOutcome::Swapped {
            frame_time,
            warnings,
        } => tracing::info!(
            "{what} hot-reload: '{name}' recompiled{}, {} swapped ({:.1} ms on the frame thread)",
            warned(warnings),
            kind.rebuilt(),
            frame_time.as_secs_f64() * 1000.0
        ),
        ReloadOutcome::AppliesOnLoad { warnings } => tracing::info!(
            "{what} hot-reload: '{name}' recompiled{}; {}",
            warned(warnings),
            kind.not_resident()
        ),
        ReloadOutcome::Failed(e) => tracing::error!(
            "{what} hot-reload: '{name}' failed: {e} (live {} kept {} previous source)",
            kind.rebuilt(),
            if kind == SubjectKind::Shader {
                "its"
            } else {
                "their"
            }
        ),
    }
}

// " with N warnings", or nothing. Each warning is logged where it was compiled.
fn warned(warnings: &[Diagnostic]) -> String {
    match warnings.len() {
        0 => String::new(),
        1 => " with 1 warning".to_string(),
        n => format!(" with {n} warnings"),
    }
}
