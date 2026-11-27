// The window a backend context holds, which a live world reload lends to the
// context it builds. A successor that fails to build hands the window back as
// it drops, so the outgoing context tears down with its window present.

use std::cell::Cell;
use std::rc::{Rc, Weak};

pub(crate) struct HeldWindow<W> {
    window: Option<W>,
    // Where the window returns if this holder drops while its loan is open.
    lender: Weak<Cell<Option<W>>>,
}

// The outgoing context's claim on a lent window. Dropping it, once the
// successor is built, leaves the window with the successor for good.
pub(crate) struct WindowLoan<W>(Rc<Cell<Option<W>>>);

impl<W> HeldWindow<W> {
    pub(crate) fn new(window: W) -> Self {
        Self {
            window: Some(window),
            lender: Weak::new(),
        }
    }

    // The window, or `None` while it is lent to a successor.
    pub(crate) fn get(&self) -> Option<&W> {
        self.window.as_ref()
    }

    pub(crate) fn get_mut(&mut self) -> Option<&mut W> {
        self.window.as_mut()
    }

    // Whether this holder has the window with no loan open on it: a fresh
    // holder, a successor whose loan closed, or a lender that reclaimed it.
    #[cfg(any(backend_vk, test))]
    pub(crate) fn is_owned(&self) -> bool {
        self.window.is_some() && self.lender.strong_count() == 0
    }

    // Lend the window to a successor's holder, or `None` when it is already
    // lent.
    pub(crate) fn lend(&mut self) -> Option<(Self, WindowLoan<W>)> {
        let window = self.window.take()?;
        let slot = Rc::new(Cell::new(None));
        let held = Self {
            window: Some(window),
            lender: Rc::downgrade(&slot),
        };
        Some((held, WindowLoan(slot)))
    }

    // Take back the window a failed successor returned as it dropped.
    pub(crate) fn reclaim(&mut self, loan: WindowLoan<W>) {
        self.window = Rc::into_inner(loan.0).and_then(Cell::into_inner);
    }
}

impl<W> Drop for HeldWindow<W> {
    fn drop(&mut self) {
        if let Some(lender) = self.lender.upgrade() {
            lender.set(self.window.take());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_holder_owns_its_window() {
        assert!(HeldWindow::new(7).is_owned());
    }

    #[test]
    fn a_lent_window_leaves_its_lender_empty() {
        let mut held = HeldWindow::new(7);
        let (lent, _loan) = held.lend().expect("a held window lends");
        assert_eq!(held.get(), None);
        assert_eq!(lent.get(), Some(&7));
        assert!(held.lend().is_none());
    }

    #[test]
    fn neither_side_owns_the_window_while_the_loan_is_open() {
        let mut held = HeldWindow::new(7);
        let (lent, _loan) = held.lend().expect("a held window lends");
        assert!(!held.is_owned());
        assert!(!lent.is_owned());
    }

    #[test]
    fn a_failed_successor_returns_the_window() {
        let mut held = HeldWindow::new(7);
        let (lent, loan) = held.lend().expect("a held window lends");
        drop(lent);
        held.reclaim(loan);
        assert_eq!(held.get(), Some(&7));
        assert!(held.is_owned());
    }

    #[test]
    fn a_built_successor_keeps_the_window_once_the_loan_closes() {
        let mut held = HeldWindow::new(7);
        let (mut lent, loan) = held.lend().expect("a held window lends");
        drop(loan);
        assert!(lent.is_owned());
        assert!(!held.is_owned());
        assert_eq!(lent.get_mut(), Some(&mut 7));
        drop(lent);
        assert_eq!(held.get(), None);
    }
}
