//! The data half of the reference column beside a Shader file or an SDF field:
//! every name the engine provides the file, from the engine's vocabulary table
//! for its kind, grouped by kind under headings that fold. Clicking a name
//! inserts it at the caret, a helper as a call naming its parameters and a
//! function to define as its prototype. Its layout is
//! `reference_panel.rs`.

use concinnity_engine::render::shader::programs::vocabulary::{Entry, Kind};

// The kinds the column groups names under, in the order it lists them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Group {
    Hooks,
    Helpers,
    Types,
    Returns,
    Blocks,
    Record,
    Varyings,
}

impl Group {
    pub(crate) const ALL: [Self; 7] = [
        Self::Hooks,
        Self::Helpers,
        Self::Types,
        Self::Returns,
        Self::Blocks,
        Self::Record,
        Self::Varyings,
    ];

    fn of(kind: Kind) -> Self {
        match kind {
            Kind::Hook => Self::Hooks,
            Kind::Helper => Self::Helpers,
            Kind::Type => Self::Types,
            Kind::ReturnField(_) => Self::Returns,
            Kind::BlockField(_) => Self::Blocks,
            Kind::RecordField => Self::Record,
            Kind::Varying => Self::Varyings,
        }
    }

    pub(crate) fn title(self) -> &'static str {
        match self {
            Self::Hooks => "Functions to define",
            Self::Helpers => "Helpers",
            Self::Types => "Types",
            Self::Returns => "Returned fields",
            Self::Blocks => "VIEW and LIGHTS",
            Self::Record => "Material record (od)",
            Self::Varyings => "Varyings (v)",
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
            RefRow::Name(e) => match (e.kind, e.kind.owner()) {
                (_, Some(owner)) => format!("{owner}.{}", e.name),
                (Kind::ReturnField(s), None) => format!("{s}.{}", e.name),
                (Kind::Type, None) => e.name.to_string(),
                (_, None) => format!("{}()", e.name),
            },
        }
    }
}

// What a click on a row does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RefClick {
    Fold(Group),
    Insert(String),
}

pub(crate) fn click(row: &RefRow<'_>) -> RefClick {
    match row {
        RefRow::Heading { group, .. } => RefClick::Fold(*group),
        RefRow::Name(e) => RefClick::Insert(e.usage()),
    }
}

// Whether the column is shown, and which groups are folded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Reference {
    pub(crate) open: bool,
    folded: [bool; Group::ALL.len()],
}

impl Default for Reference {
    fn default() -> Self {
        Self {
            open: true,
            folded: [false; Group::ALL.len()],
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

    // Row `i` of what `rows` would list for `entries`, found without listing
    // them.
    pub(crate) fn row_at<'a>(&self, entries: &'a [Entry], mut i: usize) -> Option<RefRow<'a>> {
        for group in Group::ALL {
            let names = || entries.iter().filter(move |e| Group::of(e.kind) == group);
            let count = names().count();
            if count == 0 {
                continue;
            }
            let folded = self.folded[group.index()];
            if i == 0 {
                return Some(RefRow::Heading {
                    group,
                    count,
                    folded,
                });
            }
            i -= 1;
            if !folded {
                if let Some(entry) = names().nth(i) {
                    return Some(RefRow::Name(entry));
                }
                i -= count;
            }
        }
        None
    }

    // Which groups are folded, one bit each, so a reader knows when the rows
    // change.
    pub(crate) fn folds(&self) -> u8 {
        const _: () = assert!(Group::ALL.len() <= 8, "one bit per group");
        self.folded
            .iter()
            .enumerate()
            .fold(0, |bits, (i, &f)| bits | (u8::from(f) << i))
    }

    pub(crate) fn toggle(&mut self, group: Group) {
        self.folded[group.index()] ^= true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::render::shader::programs::vocabulary::{ENTRIES, sdf};

    fn names(rows: &[RefRow<'_>]) -> Vec<String> {
        rows.iter().map(RefRow::text).collect()
    }

    #[test]
    fn a_row_is_found_without_listing_them() {
        let mut r = Reference::default();
        r.toggle(Group::Helpers);
        let rows = r.rows(ENTRIES);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(r.row_at(ENTRIES, i).as_ref(), Some(row), "row {i}");
        }
        assert_eq!(r.row_at(ENTRIES, rows.len()), None);
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
        let surface = [
            Group::Helpers,
            Group::Blocks,
            Group::Record,
            Group::Varyings,
        ];
        assert_eq!(headings, surface);
        assert_eq!(rows.len(), ENTRIES.len() + surface.len());
        let text = names(&rows);
        assert!(text[0].starts_with("- Helpers ("));
        assert!(text.contains(&"shade_surface()".to_string()));
        assert!(text.contains(&"VIEW.elapsed".to_string()));
        assert!(text.contains(&"od.tint_roughness".to_string()));
        assert!(text.contains(&"v.world_pos".to_string()));
    }

    // A distance field lists what it defines first, then what it calls, the
    // types it names and the fields it returns; a click on a function to define
    // writes its prototype.
    #[test]
    fn a_field_lists_its_own_groups() {
        let rows = Reference::default().rows(sdf::ENTRIES);
        let headings: Vec<Group> = rows
            .iter()
            .filter_map(|r| match r {
                RefRow::Heading { group, .. } => Some(*group),
                RefRow::Name(_) => None,
            })
            .collect();
        assert_eq!(
            headings,
            [Group::Hooks, Group::Helpers, Group::Types, Group::Returns]
        );
        let text = names(&rows);
        assert!(text.contains(&"map()".to_string()));
        assert!(text.contains(&"SdfParams".to_string()));
        assert!(text.contains(&"SdfSurface.albedo".to_string()));
        let map = rows.iter().find(|r| r.text() == "map()").unwrap();
        assert_eq!(
            click(map),
            RefClick::Insert("float map(float3 p, SdfParams params, float time)".to_string())
        );
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
        assert_eq!(r.rows(ENTRIES).len(), ENTRIES.len() + 4);
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
    }
}
