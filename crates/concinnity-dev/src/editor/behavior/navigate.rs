// src/editor/behavior/navigate.rs
//
// Where a step of the selection lands. A list is stepped one place at a time; a
// chart is not a list, so a step there is the nearest card in the direction
// pressed -- measured along that axis first and across it only to break a tie,
// which is what makes a sideways step follow the chain and a vertical one cross
// between the branches stacked under a branching node. Both rules read
// positions alone, so neither needs a panel to be exercised.

use super::graph::Card;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dir {
    Up,
    Down,
    Left,
    Right,
}

impl Dir {
    pub(crate) fn horizontal(self) -> bool {
        matches!(self, Self::Left | Self::Right)
    }

    // The list step this direction makes, if it makes one: a list runs down the
    // panel, so only Up and Down move through it.
    pub(crate) fn list_step(self) -> Option<isize> {
        match self {
            Self::Up => Some(-1),
            Self::Down => Some(1),
            _ => None,
        }
    }
}

// One step through `total` items, stopping at either end rather than wrapping.
// With nothing selected the step starts from the end it is coming from.
pub(crate) fn step(current: Option<usize>, delta: isize, total: usize) -> Option<usize> {
    let last = total.checked_sub(1)?;
    Some(match current {
        Some(i) => (i as isize + delta).clamp(0, last as isize) as usize,
        None if delta < 0 => last,
        None => 0,
    })
}

// The card `dir` reaches from `from`, or `None` when nothing lies that way.
// With nothing selected the first card is where a chart starts.
pub(crate) fn nearest(cards: &[Card], from: Option<usize>, dir: Dir) -> Option<usize> {
    let Some(from) = from else {
        return (!cards.is_empty()).then_some(0);
    };
    let at = cards.get(from)?;
    cards
        .iter()
        .enumerate()
        .filter_map(|(i, c)| reach(at, c, dir).map(|d| (d, i)))
        .min()
        .map(|(_, i)| i)
}

// How far `to` lies from `at`: along the axis pressed, then across it. `None`
// when `to` is not that way at all, which is what keeps a step from doubling
// back on itself.
fn reach(at: &Card, to: &Card, dir: Dir) -> Option<(usize, usize)> {
    Some(match dir {
        Dir::Right => (
            to.column.checked_sub(at.column + 1)?,
            at.row.abs_diff(to.row),
        ),
        Dir::Left => (
            at.column.checked_sub(to.column + 1)?,
            at.row.abs_diff(to.row),
        ),
        Dir::Down => (
            to.row.checked_sub(at.row + 1)?,
            at.column.abs_diff(to.column),
        ),
        Dir::Up => (
            at.row.checked_sub(to.row + 1)?,
            at.column.abs_diff(to.column),
        ),
    })
}

// The scroll that brings `index` inside a window of `pool` rows, moving no
// further than it has to.
pub(crate) fn scroll_to(index: usize, scroll: usize, pool: usize) -> usize {
    let pool = pool.max(1);
    if index < scroll {
        index
    } else if index >= scroll + pool {
        index + 1 - pool
    } else {
        scroll
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::graph;
    use super::*;

    // A branch and the node that resumes past it: enough shape for a step to
    // have somewhere wrong to go in every direction.
    fn branching() -> Vec<Card> {
        graph::chart(&json!({
            "on": "tick",
            "do": [
                {"if": {
                    "cond": {"bool": true},
                    "then": [{"show": {"target": "self"}}],
                    "else": [{"hide": {"target": "self"}}],
                }},
                {"save": {}},
            ],
        }))
        .cards
    }

    fn card(cards: &[Card], title: &str) -> usize {
        cards
            .iter()
            .position(|c| c.title == title)
            .unwrap_or_else(|| panic!("no card titled {title}"))
    }

    #[test]
    fn a_list_step_stops_at_either_end() {
        assert_eq!(step(Some(1), 1, 4), Some(2));
        assert_eq!(step(Some(3), 1, 4), Some(3));
        assert_eq!(step(Some(0), -1, 4), Some(0));
        assert_eq!(step(Some(0), 1, 1), Some(0));
        assert_eq!(step(None, 1, 0), None);
        assert_eq!(step(Some(2), 1, 0), None);
    }

    #[test]
    fn a_list_step_from_nothing_starts_at_the_end_it_comes_from() {
        assert_eq!(step(None, 1, 4), Some(0));
        assert_eq!(step(None, -1, 4), Some(3));
    }

    // A sideways step is along the chain, so it reaches the first node of a
    // branch rather than the node that resumes past the whole branch.
    #[test]
    fn a_sideways_step_follows_the_chain() {
        let cards = branching();
        let at = |i: usize| cards[i].title.clone();
        assert_eq!(
            nearest(&cards, Some(0), Dir::Right).map(at),
            Some("if".into())
        );
        let branch = card(&cards, "if");
        assert_eq!(
            nearest(&cards, Some(branch), Dir::Right).map(at),
            Some("show".into()),
        );
        assert_eq!(
            nearest(&cards, Some(card(&cards, "show")), Dir::Left).map(at),
            Some("if".into()),
        );
    }

    // A vertical step crosses between the branches a node stacks, which is the
    // only way the chart puts two cards in the same column.
    #[test]
    fn a_vertical_step_crosses_between_stacked_branches() {
        let cards = branching();
        let at = |i: usize| cards[i].title.clone();
        assert_eq!(
            nearest(&cards, Some(card(&cards, "show")), Dir::Down).map(at),
            Some("hide".into()),
        );
        assert_eq!(
            nearest(&cards, Some(card(&cards, "hide")), Dir::Up).map(at),
            Some("show".into()),
        );
    }

    #[test]
    fn a_step_with_nothing_that_way_stays_put() {
        let cards = branching();
        assert_eq!(nearest(&cards, Some(0), Dir::Left), None);
        assert_eq!(nearest(&cards, Some(0), Dir::Up), None);
        assert_eq!(nearest(&cards, Some(card(&cards, "show")), Dir::Up), None);
        assert_eq!(nearest(&cards, Some(99), Dir::Right), None);
    }

    #[test]
    fn a_chart_step_from_nothing_starts_at_the_first_card() {
        let cards = branching();
        assert_eq!(nearest(&cards, None, Dir::Right), Some(0));
        assert_eq!(nearest(&cards, None, Dir::Up), Some(0));
        assert_eq!(nearest(&[], None, Dir::Right), None);
    }

    #[test]
    fn a_scroll_moves_no_further_than_the_window_needs() {
        assert_eq!(scroll_to(4, 2, 10), 2);
        assert_eq!(scroll_to(1, 2, 10), 1);
        assert_eq!(scroll_to(12, 2, 10), 3);
        assert_eq!(scroll_to(0, 0, 0), 0);
    }
}
