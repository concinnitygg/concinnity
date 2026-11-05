//! The shader panels: the Shaders list (`list`, `list_panel`), the Shader
//! source panel it spawns (`source`, `source_panel`, with `diagnostics` for what
//! a compile marks), the shader reference (`reference`, `reference_panel`), and
//! the new-shader form (`form`, `kind`, `templates`) with the edits it commits
//! (`edit`).

pub(crate) mod diagnostics;
pub(crate) mod edit;
pub(crate) mod form;
pub(crate) mod kind;
pub(crate) mod list;
pub(crate) mod list_panel;
pub(crate) mod reference;
pub(crate) mod reference_panel;
pub(crate) mod source;
pub(crate) mod source_panel;
pub(crate) mod templates;
