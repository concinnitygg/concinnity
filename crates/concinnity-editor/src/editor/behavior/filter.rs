// src/editor/behavior/filter.rs
//
// Which of a row's options a typed query keeps, and in what order: the shared
// ranked filter (`editor::filter`) applied to the palette's picks.

use super::edit::{self, Pick};

// The options `query` keeps, best first, as indices into `picks`. An empty query
// keeps every option in the vocabulary's own order. The ranking itself is the
// shared `editor::filter`; this wrapper matches the CAPTION, not the raw verb:
// the clear-to-unset entry draws as a word it does not carry, and matching what
// is on screen is what the author expects.
pub(crate) fn matching(picks: &[Pick], query: &str) -> Vec<usize> {
    crate::editor::filter::ranked(
        picks
            .iter()
            .map(|p| (edit::pick_caption(p), p.hint.as_str())),
        query,
    )
}

#[cfg(test)]
mod tests {
    use super::super::outline::{Kind, List};
    use super::*;

    fn nodes() -> Vec<Pick> {
        edit::picks(&Kind::List(List::Nodes), &[])
    }

    fn verbs(picks: &[Pick], query: &str) -> Vec<&'static str> {
        matching(picks, query)
            .into_iter()
            .map(|i| picks[i].verb)
            .collect()
    }

    #[test]
    fn an_empty_query_keeps_the_whole_vocabulary_in_order() {
        let picks = nodes();
        let kept = matching(&picks, "");
        assert_eq!(kept, (0..picks.len()).collect::<Vec<_>>());
        // Whitespace alone is not a query either.
        assert_eq!(matching(&picks, "   "), kept);
    }

    // The rank is what makes typing then committing reliable: the option the
    // query starts comes before one that merely contains it.
    #[test]
    fn a_verb_the_query_starts_is_offered_first() {
        let picks = nodes();
        let kept = verbs(&picks, "set");
        assert_eq!(kept[0], "set", "an exact start wins");
        assert!(kept.contains(&"set_local"));
        assert!(kept.contains(&"set_transform"));
        // `set` appears inside no other node verb, so nothing ranks below.
        assert!(kept.iter().all(|v| v.starts_with("set")), "{kept:?}");
    }

    #[test]
    fn a_query_matching_inside_a_verb_still_reaches_it() {
        let picks = nodes();
        let kept = verbs(&picks, "parent");
        assert_eq!(kept.first(), Some(&"reparent"));
    }

    // Separators are the engine's business, not the author's.
    #[test]
    fn underscores_do_not_have_to_be_typed() {
        let picks = nodes();
        assert_eq!(verbs(&picks, "foreach").first(), Some(&"for_each"));
        assert_eq!(verbs(&picks, "setlocal").first(), Some(&"set_local"));
        assert_eq!(verbs(&picks, "for_each").first(), Some(&"for_each"));
    }

    #[test]
    fn case_does_not_matter() {
        let picks = nodes();
        assert_eq!(verbs(&picks, "SPAWN").first(), Some(&"spawn"));
        assert_eq!(verbs(&picks, "Hide").first(), Some(&"hide"));
    }

    // A hint match is a last resort, so it never displaces a verb match.
    #[test]
    fn a_hint_match_ranks_below_every_verb_match() {
        let picks = nodes();
        let kept = verbs(&picks, "variable");
        assert!(
            kept.contains(&"set"),
            "`set` writes a world variable: {kept:?}"
        );
        // Nothing has "variable" in its verb, so the hint matches are all there
        // is -- and a query matching both would put the verb first.
        let mixed = verbs(&picks, "spawn");
        assert_eq!(mixed[0], "spawn", "the verb outranks any hint: {mixed:?}");
    }

    #[test]
    fn a_query_nothing_answers_keeps_nothing() {
        let picks = nodes();
        assert!(matching(&picks, "zzzz").is_empty());
        assert!(matching(&[], "set").is_empty());
    }

    // The clear-to-unset entry draws as a word its verb does not carry, so that
    // word is what has to match.
    #[test]
    fn the_unset_entry_matches_the_word_it_draws_as() {
        let picks = edit::picks(&Kind::Expr { optional: true }, &[]);
        assert_eq!(
            matching(&picks, "unset")
                .first()
                .map(|&i| edit::pick_caption(&picks[i])),
            Some("unset"),
        );
    }
}
