//! Rows a type adds to the add / edit form beyond its schema fields: a
//! heading or note, a checkbox, or a choice cycled on click. The form lists
//! them after the fields and reports a press on one by the id its type gave
//! it. What a row means is the type's business (`hook/form_extras.rs`).

// What a row carries on its right.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExtraControl {
    // Nothing: the caption alone, a heading or a note.
    Label,
    Check {
        on: bool,
        enabled: bool,
    },
    // Advances to the next option on click.
    Choice {
        options: Vec<String>,
        selected: usize,
    },
}

// One row. `detail` is dim text in the control column, after a checkbox's box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExtraRow {
    pub(crate) id: usize,
    pub(crate) caption: String,
    pub(crate) indent: bool,
    pub(crate) control: ExtraControl,
    pub(crate) detail: Option<String>,
}

impl ExtraRow {
    pub(crate) fn label(caption: impl Into<String>, detail: Option<String>) -> Self {
        Self {
            id: 0,
            caption: caption.into(),
            indent: false,
            control: ExtraControl::Label,
            detail,
        }
    }

    pub(crate) fn indented(mut self) -> Self {
        self.indent = true;
        self
    }
}
