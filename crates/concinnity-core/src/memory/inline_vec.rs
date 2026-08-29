// A sequence that keeps its first element inline.
//
// Some populations hold exactly one item almost everywhere: an entity's draw
// slots (one per mesh, and most props are one mesh), a parent's children. A
// `Vec` charges each of those a heap block -- twenty thousand entities, twenty
// thousand blocks of four bytes -- and a pointer chase to reach it on every
// frame that walks them. Holding the first element inline removes both for the
// common case while keeping the heap for the rest, so a caller does not have to
// know which case it is in.
//
// Capacity one rather than a general N: one is expressible as an enum, which is
// why this file contains no `unsafe`. A general inline capacity needs
// `MaybeUninit` and a hand-written drop, and nothing has asked for it.
//
// Equality and ordering read the contents, never the representation: a
// one-element inline value equals a one-element spilled one.

use alloc::vec::Vec;
use core::fmt;
use core::ops::{Deref, DerefMut};

/// A sequence holding its first element inline and spilling to the heap beyond
/// that.
///
/// Derefs to `[T]`, so slice methods (`iter`, `len`, `contains`, indexing) are
/// available directly.
///
/// Spilled storage is kept once acquired: removing elements does not move the
/// remainder back inline, so a value that has grown never silently reallocates
/// when it shrinks.
pub struct InlineVec<T>(Repr<T>);

#[derive(Clone)]
enum Repr<T> {
    Empty,
    One(T),
    Spilled(Vec<T>),
}

impl<T> InlineVec<T> {
    /// An empty sequence, holding no allocation.
    pub const fn new() -> Self {
        Self(Repr::Empty)
    }

    /// A sequence holding exactly `value`, inline.
    pub const fn one(value: T) -> Self {
        Self(Repr::One(value))
    }

    // Whether the contents live on the heap rather than inline.
    #[cfg(test)]
    pub(crate) fn spilled(&self) -> bool {
        matches!(self.0, Repr::Spilled(_))
    }

    /// The contents as a slice.
    pub fn as_slice(&self) -> &[T] {
        match &self.0 {
            Repr::Empty => &[],
            Repr::One(value) => core::slice::from_ref(value),
            Repr::Spilled(values) => values,
        }
    }

    /// The contents as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match &mut self.0 {
            Repr::Empty => &mut [],
            Repr::One(value) => core::slice::from_mut(value),
            Repr::Spilled(values) => values,
        }
    }

    /// Append `value`, spilling to the heap if this is the second element.
    pub fn push(&mut self, value: T) {
        match &mut self.0 {
            Repr::Empty => self.0 = Repr::One(value),
            Repr::Spilled(values) => values.push(value),
            Repr::One(_) => {
                let Repr::One(first) = core::mem::replace(&mut self.0, Repr::Empty) else {
                    unreachable!("the arm above matched One")
                };
                self.0 = Repr::Spilled(alloc::vec![first, value]);
            }
        }
    }

    /// Remove and return the last element, or `None` when empty.
    pub fn pop(&mut self) -> Option<T> {
        match &mut self.0 {
            Repr::Empty => None,
            Repr::Spilled(values) => values.pop(),
            Repr::One(_) => match core::mem::replace(&mut self.0, Repr::Empty) {
                Repr::One(value) => Some(value),
                _ => unreachable!("the arm above matched One"),
            },
        }
    }

    /// Drop every element for which `keep` returns false.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        match &mut self.0 {
            Repr::Empty => {}
            Repr::Spilled(values) => values.retain(|value| keep(value)),
            Repr::One(value) => {
                if !keep(value) {
                    self.0 = Repr::Empty;
                }
            }
        }
    }

    /// Drop every element and release any heap storage.
    pub fn clear(&mut self) {
        self.0 = Repr::Empty;
    }

    // The contents as an owned `Vec`, allocating when they were held inline.
    pub(crate) fn into_vec(self) -> Vec<T> {
        match self.0 {
            Repr::Empty => Vec::new(),
            Repr::One(value) => alloc::vec![value],
            Repr::Spilled(values) => values,
        }
    }
}

impl<T> Default for InlineVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Clone for InlineVec<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Deref for InlineVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for InlineVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

// Contents, not representation: `One(x)` and `Spilled(vec![x])` hold the same
// sequence and must not be distinguishable.
impl<T: PartialEq> PartialEq for InlineVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for InlineVec<T> {}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<[U; N]> for InlineVec<T> {
    fn eq(&self, other: &[U; N]) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<U>, U> PartialEq<[U]> for InlineVec<T> {
    fn eq(&self, other: &[U]) -> bool {
        self.as_slice() == other
    }
}

impl<T: PartialEq<U>, U> PartialEq<Vec<U>> for InlineVec<T> {
    fn eq(&self, other: &Vec<U>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: fmt::Debug> fmt::Debug for InlineVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl<T> From<Vec<T>> for InlineVec<T> {
    // Normalizes: a `Vec` of at most one element gives up its allocation.
    fn from(values: Vec<T>) -> Self {
        let mut values = values;
        match values.len() {
            0 => Self::new(),
            1 => Self(Repr::One(values.pop().expect("length is one"))),
            _ => Self(Repr::Spilled(values)),
        }
    }
}

impl<T, const N: usize> From<[T; N]> for InlineVec<T> {
    fn from(values: [T; N]) -> Self {
        values.into_iter().collect()
    }
}

impl<T> From<InlineVec<T>> for Vec<T> {
    fn from(inline: InlineVec<T>) -> Self {
        inline.into_vec()
    }
}

impl<T> FromIterator<T> for InlineVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        let Some(first) = iter.next() else {
            return Self::new();
        };
        let Some(second) = iter.next() else {
            return Self(Repr::One(first));
        };
        let mut values = Vec::with_capacity(iter.size_hint().0 + 2);
        values.push(first);
        values.push(second);
        values.extend(iter);
        Self(Repr::Spilled(values))
    }
}

impl<T> Extend<T> for InlineVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.push(value);
        }
    }
}

/// By-value iterator over an [`InlineVec`], produced by `into_iter`.
pub enum IntoIter<T> {
    #[doc(hidden)]
    Inline(core::option::IntoIter<T>),
    #[doc(hidden)]
    Spilled(alloc::vec::IntoIter<T>),
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match self {
            Self::Inline(iter) => iter.next(),
            Self::Spilled(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Inline(iter) => iter.size_hint(),
            Self::Spilled(iter) => iter.size_hint(),
        }
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}

impl<T> IntoIterator for InlineVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        match self.0 {
            Repr::Empty => IntoIter::Inline(None.into_iter()),
            Repr::One(value) => IntoIter::Inline(Some(value).into_iter()),
            Repr::Spilled(values) => IntoIter::Spilled(values.into_iter()),
        }
    }
}

impl<'a, T> IntoIterator for &'a InlineVec<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> core::slice::Iter<'a, T> {
        self.as_slice().iter()
    }
}

impl<'a, T> IntoIterator for &'a mut InlineVec<T> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;

    fn into_iter(self) -> core::slice::IterMut<'a, T> {
        self.as_mut_slice().iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn an_empty_value_holds_nothing_inline() {
        let empty = InlineVec::<u32>::new();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert!(!empty.spilled());
        assert_eq!(empty.as_slice(), &[] as &[u32]);
    }

    // The whole point: the common case must not reach the heap.
    #[test]
    fn a_single_element_stays_inline() {
        let mut one = InlineVec::new();
        one.push(7u32);
        assert!(!one.spilled(), "one element must not allocate");
        assert_eq!(one.as_slice(), [7]);
        assert_eq!(InlineVec::one(7u32).as_slice(), [7]);
    }

    #[test]
    fn a_second_element_spills_to_the_heap_in_order() {
        let mut many = InlineVec::new();
        many.push(1u32);
        many.push(2);
        many.push(3);
        assert!(many.spilled());
        assert_eq!(many.as_slice(), [1, 2, 3]);
    }

    #[test]
    fn collecting_picks_the_representation_from_the_length() {
        let none: InlineVec<u32> = core::iter::empty().collect();
        assert!(none.is_empty() && !none.spilled());

        let one: InlineVec<u32> = core::iter::once(5).collect();
        assert!(!one.spilled());
        assert_eq!(one.as_slice(), [5]);

        let many: InlineVec<u32> = (0..4).collect();
        assert!(many.spilled());
        assert_eq!(many.as_slice(), [0, 1, 2, 3]);
    }

    // A `Vec` handed in gives up an allocation it no longer needs, so a caller
    // building through `Vec` is not stuck with the block forever.
    #[test]
    fn a_short_vec_gives_up_its_allocation() {
        assert!(!InlineVec::from(vec![9u32]).spilled());
        assert!(InlineVec::from(vec![9u32, 10]).spilled());
        assert!(!InlineVec::<u32>::from(vec![]).spilled());
        assert_eq!(InlineVec::from(vec![9u32, 10]).as_slice(), [9, 10]);
    }

    // Equality is over contents. Both representations can hold one element, and
    // a caller must never be able to tell which it got.
    #[test]
    fn the_two_representations_compare_equal_on_equal_contents() {
        let inline = InlineVec::one(4u32);
        let mut spilled = InlineVec::new();
        spilled.push(4u32);
        spilled.push(5);
        spilled.pop();

        assert!(spilled.spilled() && !inline.spilled());
        assert_eq!(inline, spilled);
        assert_eq!(
            alloc::format!("{inline:?}"),
            alloc::format!("{spilled:?}"),
            "the representation must not show in Debug"
        );
    }

    #[test]
    fn retain_drops_the_inline_element_and_filters_the_spilled_one() {
        let mut one = InlineVec::one(1u32);
        one.retain(|&v| v != 1);
        assert!(one.is_empty());

        let mut many: InlineVec<u32> = (0..6).collect();
        many.retain(|v| v % 2 == 0);
        assert_eq!(many.as_slice(), [0, 2, 4]);
    }

    // Documented behaviour: shrinking keeps the block rather than reallocating.
    #[test]
    fn a_shrunk_value_keeps_its_heap_storage_until_cleared() {
        let mut many: InlineVec<u32> = (0..3).collect();
        many.retain(|&v| v == 0);
        assert!(many.spilled());
        assert_eq!(many.as_slice(), [0]);

        many.clear();
        assert!(!many.spilled());
        assert!(many.is_empty());
    }

    #[test]
    fn pop_returns_elements_from_the_end_of_both_representations() {
        let mut one = InlineVec::one(1u32);
        assert_eq!(one.pop(), Some(1));
        assert_eq!(one.pop(), None);

        let mut many: InlineVec<u32> = (0..3).collect();
        assert_eq!(many.pop(), Some(2));
        assert_eq!(many.pop(), Some(1));
        assert_eq!(many.pop(), Some(0));
        assert_eq!(many.pop(), None);
    }

    #[test]
    fn iteration_covers_every_representation() {
        let cases = [
            InlineVec::<u32>::new(),
            InlineVec::one(1),
            (1..4).collect::<InlineVec<u32>>(),
        ];
        let expected: [&[u32]; 3] = [&[], &[1], &[1, 2, 3]];

        for (case, want) in cases.into_iter().zip(expected) {
            assert_eq!(case.iter().copied().collect::<Vec<_>>(), want);
            assert_eq!((&case).into_iter().copied().collect::<Vec<_>>(), want);
            assert_eq!(case.clone().into_vec(), want);
            assert_eq!(case.into_iter().collect::<Vec<_>>(), want);
        }
    }

    #[test]
    fn mutable_access_reaches_both_representations() {
        let mut one = InlineVec::one(1u32);
        for value in &mut one {
            *value += 1;
        }
        assert_eq!(one.as_slice(), [2]);

        let mut many: InlineVec<u32> = (0..3).collect();
        many.as_mut_slice()[1] = 9;
        assert_eq!(many.as_slice(), [0, 9, 2]);
    }

    #[test]
    fn extend_grows_through_the_same_spill_rule() {
        let mut values = InlineVec::new();
        values.extend([1u32]);
        assert!(!values.spilled());
        values.extend([2u32, 3]);
        assert_eq!(values.as_slice(), [1, 2, 3]);
    }

    // The inline element rides in the `Vec` pointer's niche, so a column of
    // these is no wider than a column of `Vec`s and the saved heap block is not
    // paid for in inline bytes. A representation change that broke the niche
    // would grow every component column that holds one, silently.
    #[test]
    fn the_inline_element_costs_no_more_than_the_vec_it_replaces() {
        assert_eq!(
            size_of::<InlineVec<u32>>(),
            size_of::<Vec<u32>>(),
            "the inline case must ride in the Vec's niche"
        );
        assert_eq!(size_of::<InlineVec<u64>>(), size_of::<Vec<u64>>());
        assert_eq!(
            size_of::<InlineVec<(u32, u32)>>(),
            size_of::<Vec<(u32, u32)>>()
        );
    }

    // Comparing against a literal is what call sites and their tests do, so it
    // reads the same whether the value spilled or not.
    #[test]
    fn arrays_and_vecs_convert_and_compare_across_both_representations() {
        let one: InlineVec<u32> = [1].into();
        let many: InlineVec<u32> = [1, 2].into();
        assert!(!one.spilled() && many.spilled());

        assert_eq!(one, [1]);
        assert_eq!(many, [1, 2]);
        assert_eq!(many, vec![1u32, 2]);
        assert_eq!(many, *[1u32, 2].as_slice());
        assert_ne!(one, [2]);
        assert_ne!(one, [1, 2]);
    }

    // Slice methods arrive through `Deref`, so call sites that held a `Vec`
    // keep reading the same way.
    #[test]
    fn slice_methods_are_available_through_deref() {
        let values: InlineVec<u32> = (10..13).collect();
        assert!(values.contains(&11));
        assert_eq!(values[2], 12);
        assert_eq!(values.first(), Some(&10));
        assert_eq!(values.to_vec(), vec![10, 11, 12]);
    }

    // Ownership is real on both paths: dropping must run element destructors
    // exactly once whether or not the value spilled.
    #[test]
    fn dropping_runs_element_destructors_once() {
        use alloc::rc::Rc;

        let witness = Rc::new(());
        let mut values = InlineVec::new();
        values.push(Rc::clone(&witness));
        assert_eq!(Rc::strong_count(&witness), 2);

        values.push(Rc::clone(&witness));
        assert_eq!(Rc::strong_count(&witness), 3, "the spill must not leak");

        drop(values);
        assert_eq!(Rc::strong_count(&witness), 1);
    }
}
