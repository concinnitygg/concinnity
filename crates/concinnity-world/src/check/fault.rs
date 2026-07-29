// src/check/fault.rs
//
// A validation failure and where in an asset's args it was found. A checker
// walks authored JSON by descending into it, so the walk already knows where it
// is: each frame attaches the one hop it descended through as the error unwinds,
// and the location assembles itself outermost-first without any frame having to
// know the whole path. An editor can then point at the value at fault instead of
// leaving the author to find it from the message alone.

/// One hop into an asset's authored args: an object key or an array index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// An object key.
    Field(String),
    /// An array index.
    Index(usize),
}

/// A failed check: what is wrong, and where in the args it was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fault {
    /// The checker's own message, which is what a build reports verbatim.
    pub message: String,
    /// Hops from the args root to the value at fault, outermost first. Empty
    /// when nothing narrower than the whole asset is to blame.
    pub at: Vec<Step>,
}

impl Fault {
    pub(crate) fn new(message: impl Into<String>) -> Fault {
        Fault {
            message: message.into(),
            at: Vec::new(),
        }
    }

    // Record that this fault was found inside `step`. Called as the error
    // unwinds, so the outermost hop is attached last and lands first.
    pub(crate) fn within(mut self, step: Step) -> Fault {
        self.at.insert(0, step);
        self
    }

    // Say what the message is about, keeping the location: the caller's label
    // then the message, which is how a nested complaint reads as one sentence
    // ("`let` is missing", "Behavior 'chase': unknown node `teleport`").
    pub(crate) fn about(mut self, what: &str) -> Fault {
        self.message = format!("{what} {}", self.message);
        self
    }
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

pub(crate) fn field(name: &str) -> Step {
    Step::Field(name.to_string())
}

// Attach one hop to whatever a nested check reported. Implemented on `Result` so
// a descent reads as the call plus where it went: `check(..).at_field("cond")?`.
pub(crate) trait Locate<T> {
    fn at_field(self, name: &str) -> Result<T, Fault>;
    fn at_index(self, index: usize) -> Result<T, Fault>;
}

impl<T> Locate<T> for Result<T, Fault> {
    fn at_field(self, name: &str) -> Result<T, Fault> {
        self.map_err(|f| f.within(field(name)))
    }

    fn at_index(self, index: usize) -> Result<T, Fault> {
        self.map_err(|f| f.within(Step::Index(index)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The walk attaches hops as the error unwinds, so a fault raised deep in a
    // body comes out addressing that spot from the root.
    #[test]
    fn hops_attached_while_unwinding_read_outermost_first() {
        let deep: Result<(), Fault> = Err(Fault::new("is missing"));
        let out = deep
            .at_field("cond")
            .at_field("if")
            .at_index(1)
            .at_field("do")
            .unwrap_err();
        assert_eq!(
            out.at,
            vec![field("do"), Step::Index(1), field("if"), field("cond")],
        );
    }

    #[test]
    fn a_fault_with_nothing_to_blame_carries_no_location() {
        let f = Fault::new("duplicate name");
        assert!(f.at.is_empty());
        assert_eq!(f.to_string(), "duplicate name");
    }

    // Labels compose into one sentence, and they never disturb the location:
    // the build reads the message and the editor reads the place.
    #[test]
    fn labels_compose_and_leave_the_location_alone() {
        let f = Fault::new("is missing")
            .within(field("value"))
            .about("`let`")
            .about("Behavior 'chase':");
        assert_eq!(f.message, "Behavior 'chase': `let` is missing");
        assert_eq!(f.at, vec![field("value")]);
    }

    #[test]
    fn success_passes_through_untouched() {
        let ok: Result<u8, Fault> = Ok(7);
        assert_eq!(ok.at_field("cond").at_index(0).unwrap(), 7);
    }
}
