use crate::build_only::story::model::{Directive, Gate, ScriptLine, Stage, VarOp};
use crate::build_only::story::script::parse_script_line;

// Media and script state in document order: the music and stage current from
// the most recent directives, plus the one-shot sounds, variable ops, and
// gates waiting for the next page or choice menu. A directive with nothing
// after it to attach to is dead and rejected at end of parse.
#[derive(Default)]
pub(super) struct PendingMedia {
    music: Option<String>,
    stage: Stage,
    sounds: Vec<String>,
    ops: Vec<VarOp>,
    gates: Vec<Gate>,
    pub(super) unconsumed_directive: Option<usize>,
}

// What a page or choice menu picks up when it shows.
pub(super) struct Attachment {
    pub(super) music: Option<String>,
    pub(super) stage: Stage,
    pub(super) sounds: Vec<String>,
    pub(super) ops: Vec<VarOp>,
    pub(super) gates: Vec<Gate>,
}

impl PendingMedia {
    // Music and stage persist to later pages; the one-shots are consumed.
    pub(super) fn take_pending(&mut self) -> Attachment {
        self.unconsumed_directive = None;
        Attachment {
            music: self.music.clone(),
            stage: self.stage.clone(),
            sounds: std::mem::take(&mut self.sounds),
            ops: std::mem::take(&mut self.ops),
            gates: std::mem::take(&mut self.gates),
        }
    }

    pub(super) fn apply_directive(&mut self, directive: Directive) {
        match directive {
            Directive::Music(path) => self.music = Some(path),
            Directive::Sound(path) => self.sounds.push(path),
            // A backdrop change is a scene change: the portraits leave with
            // the old scene.
            Directive::Bg(path) => {
                self.stage = Stage {
                    bg: Some(path),
                    ..Stage::default()
                };
            }
            Directive::Left(path) => self.stage.left = Some(path),
            Directive::Center(path) => self.stage.center = Some(path),
            Directive::Right(path) => self.stage.right = Some(path),
        }
    }

    // Queue a closed ```story fence's ops and gates; `line` is the fence's
    // source line, for errors and the unconsumed check.
    pub(super) fn queue_script(&mut self, text: &str, line: usize) -> Result<(), String> {
        for raw in text.lines() {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                continue;
            }
            match parse_script_line(trimmed) {
                Ok(ScriptLine::Op(op)) => self.ops.push(op),
                Ok(ScriptLine::Gate(gate)) => self.gates.push(gate),
                Err(e) => return Err(format!("line {}: {}", line, e)),
            }
        }
        self.unconsumed_directive = Some(line);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loaded() -> PendingMedia {
        let mut pending = PendingMedia::default();
        pending.apply_directive(Directive::Music("theme.ogg".to_string()));
        pending.apply_directive(Directive::Bg("inn.png".to_string()));
        pending.apply_directive(Directive::Left("a.png".to_string()));
        pending.apply_directive(Directive::Sound("door.wav".to_string()));
        pending
            .queue_script("set met\n\nif met -> #wood\n", 4)
            .unwrap();
        pending
    }

    #[test]
    fn take_pending_drains_one_shots_and_keeps_music_and_stage() {
        let mut pending = loaded();
        assert_eq!(pending.unconsumed_directive, Some(4));

        let first = pending.take_pending();
        assert_eq!(first.music.as_deref(), Some("theme.ogg"));
        assert_eq!(first.stage.bg.as_deref(), Some("inn.png"));
        assert_eq!(first.stage.left.as_deref(), Some("a.png"));
        assert_eq!(first.sounds, ["door.wav"]);
        assert_eq!(first.ops[0].name, "met");
        assert_eq!(first.gates[0].target, "wood");
        assert_eq!(pending.unconsumed_directive, None);

        let second = pending.take_pending();
        assert_eq!(second.music.as_deref(), Some("theme.ogg"));
        assert_eq!(second.stage.left.as_deref(), Some("a.png"));
        assert!(second.sounds.is_empty() && second.ops.is_empty() && second.gates.is_empty());
    }

    #[test]
    fn a_backdrop_directive_clears_the_portraits() {
        let mut pending = loaded();
        pending.apply_directive(Directive::Bg("shore.png".to_string()));
        let attachment = pending.take_pending();
        assert_eq!(attachment.stage.bg.as_deref(), Some("shore.png"));
        assert_eq!(attachment.stage.left, None);
    }

    #[test]
    fn a_bad_script_line_reports_the_fence_line() {
        let mut pending = PendingMedia::default();
        let err = pending.queue_script("set BAD", 9).unwrap_err();
        assert!(err.starts_with("line 9: 'BAD'"), "{err}");
    }
}
