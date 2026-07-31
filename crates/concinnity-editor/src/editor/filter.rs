// src/editor/filter.rs
//
// The ranked substring filter shared by the editor's pick lists (the Behavior
// palette, the Content browser). The point of typing is that the first answer
// is the one wanted, so matches are ranked rather than merely collected: a
// caption the query starts is offered before one that merely contains it, and
// before an entry whose hint mentions it at all.
//
// Underscores are dropped from both sides of a caption comparison, so
// "foreach" reaches `for_each` without the author having to remember where the
// engine puts its separators. Hints are sentences, so those match as written.

// Ranks, best first. Kept as an enum-free scale because the only thing that
// reads them is the sort.
const STARTS: u8 = 0;
const CONTAINS: u8 = 1;
const IN_HINT: u8 = 2;

// The candidates `query` keeps, best first, as indices into `candidates`.
// Each candidate is `(caption, hint)`: the caption is what is drawn (and what
// starts/contains rank against); the hint is descriptive text a last-resort
// match may live in. An empty query keeps everything in declaration order.
pub(crate) fn ranked<'a>(
    candidates: impl IntoIterator<Item = (&'a str, &'a str)>,
    query: &str,
) -> Vec<usize> {
    let typed = query.trim();
    let caption_query = squashed(typed);
    let hint_query = typed.to_lowercase();
    let mut hits: Vec<(u8, usize)> = candidates
        .into_iter()
        .enumerate()
        .filter_map(|(i, (caption, hint))| {
            rank(caption, hint, &caption_query, &hint_query).map(|r| (r, i))
        })
        .collect();
    // The index is the tiebreak, so candidates of equal rank keep declaration
    // order.
    hits.sort_by_key(|&(rank, i)| (rank, i));
    hits.into_iter().map(|(_, i)| i).collect()
}

// How well a candidate answers the query, lower being better, `None` for no
// match.
fn rank(caption: &str, hint: &str, caption_query: &str, hint_query: &str) -> Option<u8> {
    let caption = squashed(caption);
    if caption.starts_with(caption_query) {
        return Some(STARTS);
    }
    if caption.contains(caption_query) {
        return Some(CONTAINS);
    }
    let hint = hint.to_lowercase();
    (!hint_query.is_empty() && hint.contains(hint_query)).then_some(IN_HINT)
}

fn squashed(text: &str) -> String {
    text.chars()
        .filter(|c| *c != '_')
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ITEMS: [(&str, &str); 4] = [
        ("stone_wall", "a mesh"),
        ("wall_lamp", "a light fixture"),
        ("floor", "walls and floors ship together"),
        ("ceiling", "nothing relevant"),
    ];

    fn names(query: &str) -> Vec<&'static str> {
        ranked(ITEMS, query)
            .into_iter()
            .map(|i| ITEMS[i].0)
            .collect()
    }

    #[test]
    fn empty_query_keeps_declaration_order() {
        assert_eq!(names(""), ["stone_wall", "wall_lamp", "floor", "ceiling"]);
        assert_eq!(names("   "), names(""));
    }

    #[test]
    fn starts_beats_contains_beats_hint() {
        assert_eq!(names("wall"), ["wall_lamp", "stone_wall", "floor"]);
    }

    #[test]
    fn underscores_and_case_are_ignored_in_captions() {
        assert_eq!(names("STONEWALL"), ["stone_wall"]);
        assert_eq!(names("stone_wall"), ["stone_wall"]);
    }

    #[test]
    fn no_match_keeps_nothing() {
        assert!(ranked(ITEMS, "zzz").is_empty());
        assert!(ranked([], "wall").is_empty());
    }
}
