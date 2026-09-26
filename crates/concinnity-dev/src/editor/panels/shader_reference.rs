//! The data half of the reference column beside a Shader file: every name the
//! engine provides a Shader, from the engine's vocabulary table, grouped by
//! kind under headings that fold. Clicking a name inserts it at the caret, a
//! helper as a call naming its parameters. Its layout is
//! `shader_reference_panel.rs`.

use concinnity_core::render::shader_programs::vocabulary::{Entry, Kind};

// The kinds the column groups names under, in the order it lists them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Group {
    Helpers,
    Blocks,
    Record,
    Varyings,
}

impl Group {
    pub(crate) const ALL: [Group; 4] = [
        Group::Helpers,
        Group::Blocks,
        Group::Record,
        Group::Varyings,
    ];

    fn of(kind: Kind) -> Group {
        match kind {
            Kind::Helper => Group::Helpers,
            Kind::BlockField(_) => Group::Blocks,
            Kind::RecordField => Group::Record,
            Kind::Varying => Group::Varyings,
        }
    }

    pub(crate) fn title(self) -> &'static str {
        match self {
            Group::Helpers => "Helpers",
            Group::Blocks => "VIEW and LIGHTS",
            Group::Record => "Material record (od)",
            Group::Varyings => "Varyings (v)",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

// One row of the column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum RefRow<'a> {
    Heading {
        group: Group,
        count: usize,
        folded: bool,
    },
    Name(&'a Entry),
}

impl RefRow<'_> {
    // What the row reads.
    pub(crate) fn text(&self) -> String {
        match self {
            RefRow::Heading {
                group,
                count,
                folded,
            } => {
                let mark = if *folded { '+' } else { '-' };
                format!("{mark} {} ({count})", group.title())
            }
            RefRow::Name(e) => match e.kind.owner() {
                Some(owner) => format!("{owner}.{}", e.name),
                None => format!("{}()", e.name),
            },
        }
    }

    // What the status line reads while the row is hovered.
    pub(crate) fn describe(&self) -> Option<String> {
        match self {
            RefRow::Heading { .. } => None,
            RefRow::Name(e) => Some(format!("{}: {}", e.signature, e.summary)),
        }
    }
}

// What a click on a row does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RefClick {
    Fold(Group),
    Insert(String),
}

pub(crate) fn click(row: &RefRow) -> RefClick {
    match row {
        RefRow::Heading { group, .. } => RefClick::Fold(*group),
        RefRow::Name(e) => RefClick::Insert(e.usage()),
    }
}

// Whether the column is shown, which groups are folded, and its first shown
// row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Reference {
    pub(crate) open: bool,
    folded: [bool; Group::ALL.len()],
    pub(crate) scroll: usize,
}

impl Default for Reference {
    fn default() -> Self {
        Self {
            open: true,
            folded: [false; Group::ALL.len()],
            scroll: 0,
        }
    }
}

impl Reference {
    // The rows for `entries`: each group's heading, then its names unless it
    // is folded. A group with no names is left out.
    pub(crate) fn rows<'a>(&self, entries: &'a [Entry]) -> Vec<RefRow<'a>> {
        let mut rows = Vec::new();
        for group in Group::ALL {
            let names: Vec<&Entry> = entries
                .iter()
                .filter(|e| Group::of(e.kind) == group)
                .collect();
            if names.is_empty() {
                continue;
            }
            let folded = self.folded[group.index()];
            rows.push(RefRow::Heading {
                group,
                count: names.len(),
                folded,
            });
            if !folded {
                rows.extend(names.into_iter().map(RefRow::Name));
            }
        }
        rows
    }

    pub(crate) fn toggle(&mut self, group: Group) {
        self.folded[group.index()] ^= true;
    }

    // Move the window by `delta` rows over `total`, `shown` at a time.
    pub(crate) fn scroll_by(&mut self, delta: isize, total: usize, shown: usize) {
        self.scroll = self.scroll.saturating_add_signed(delta);
        self.clamp(total, shown);
    }

    pub(crate) fn clamp(&mut self, total: usize, shown: usize) {
        self.scroll = self.scroll.min(total.saturating_sub(shown));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::render::shader_programs::vocabulary::ENTRIES;

    fn names<'a>(rows: &[RefRow<'a>]) -> Vec<String> {
        rows.iter().map(RefRow::text).collect()
    }

    #[test]
    fn rows_group_every_entry_under_its_kind() {
        let rows = Reference::default().rows(ENTRIES);
        let headings: Vec<Group> = rows
            .iter()
            .filter_map(|r| match r {
                RefRow::Heading { group, .. } => Some(*group),
                RefRow::Name(_) => None,
            })
            .collect();
        assert_eq!(headings, Group::ALL);
        assert_eq!(rows.len(), ENTRIES.len() + Group::ALL.len());
        let text = names(&rows);
        assert!(text[0].starts_with("- Helpers ("));
        assert!(text.contains(&"shade_surface()".to_string()));
        assert!(text.contains(&"VIEW.elapsed".to_string()));
        assert!(text.contains(&"od.tint_roughness".to_string()));
        assert!(text.contains(&"v.world_pos".to_string()));
    }

    #[test]
    fn a_folded_group_keeps_its_heading_and_hides_its_names() {
        let mut r = Reference::default();
        r.toggle(Group::Helpers);
        let rows = r.rows(ENTRIES);
        assert!(matches!(
            rows[0],
            RefRow::Heading {
                group: Group::Helpers,
                folded: true,
                ..
            }
        ));
        assert!(matches!(
            rows[1],
            RefRow::Heading {
                group: Group::Blocks,
                ..
            }
        ));
        assert!(names(&rows)[0].starts_with("+ Helpers"));
        r.toggle(Group::Helpers);
        assert_eq!(r.rows(ENTRIES).len(), ENTRIES.len() + Group::ALL.len());
    }

    #[test]
    fn a_group_with_no_names_is_left_out() {
        let only_helpers: Vec<Entry> = ENTRIES
            .iter()
            .filter(|e| e.kind == Kind::Helper)
            .copied()
            .collect();
        let rows = Reference::default().rows(&only_helpers);
        assert_eq!(rows.len(), only_helpers.len() + 1);
    }

    #[test]
    fn clicking_a_name_inserts_its_usage_and_a_heading_folds() {
        let rows = Reference::default().rows(ENTRIES);
        let find = |t: &str| rows.iter().find(|r| r.text() == t).unwrap();
        assert_eq!(
            click(find("shade_surface()")),
            RefClick::Insert("shade_surface(v, od)".to_string())
        );
        assert_eq!(
            click(find("LIGHTS.num_dir")),
            RefClick::Insert("LIGHTS.num_dir".to_string())
        );
        assert_eq!(click(&rows[0]), RefClick::Fold(Group::Helpers));
        let described = find("pool_sample()").describe().unwrap();
        assert!(described.starts_with("float4 pool_sample(uint index, float2 uv): "));
        assert_eq!(rows[0].describe(), None);
    }

    #[test]
    fn scrolling_stays_within_the_rows() {
        let mut r = Reference::default();
        r.scroll_by(5, 40, 10);
        assert_eq!(r.scroll, 5);
        r.scroll_by(100, 40, 10);
        assert_eq!(r.scroll, 30);
        r.scroll_by(-100, 40, 10);
        assert_eq!(r.scroll, 0);
        r.scroll = 30;
        r.clamp(12, 10);
        assert_eq!(r.scroll, 2, "folding pulls the window back over the rows");
    }
}
