// concinnity-eas/src/column.rs
//
// A typed component column: the per-type storage primitive the closed-world
// ComponentStorage is built from. It bundles the component data with a
// row-aligned Entity id per row, a row-aligned change tick, and column-level
// tick stamps. All structural edits go through helpers that keep the data, id,
// and tick vectors the same length (checked with a debug assertion).
//
// Column derefs to its data slice, so read paths (iteration, indexing) behave
// like a plain Vec. Whole-column mutable access goes through `values_mut`,
// which stamps the bulk tick because any element may be written; a write aimed
// at one row goes through `value_mut`, which stamps only that row, so a
// consumer can recover exactly which entities were touched.

use alloc::vec::Vec;

use core::ops::Deref;

use crate::entity::Entity;
use crate::tick::Tick;

// How a component type is stored. Table is the default dense column; SparseSet
// is opt-in for high-churn types (see SparseColumn). The engine selects the
// kind per component type.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum StorageKind {
    #[default]
    Table,
    SparseSet,
}

// The tick stamps a column keeps. `changed` is the maximum over every kind of
// write and drives whole-column change detection. `added` marks the last
// appended row. `bulk` marks the last whole-column mutable access, after which
// every row must be assumed written. `structural` marks the last row add or
// removal, after which row positions and membership have moved. A consumer
// that tracks rows individually reads `bulk` and `structural` to decide whether
// the per-row stamps alone still describe what changed.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct ColumnTicks {
    pub changed: Tick,
    pub added: Tick,
    pub bulk: Tick,
    pub structural: Tick,
}

#[derive(Debug)]
pub struct Column<T> {
    data: Vec<T>,
    entities: Vec<Entity>,
    row_changed: Vec<Tick>,
    changed: Tick,
    added: Tick,
    bulk: Tick,
    structural: Tick,
}

impl<T> Default for Column<T> {
    fn default() -> Column<T> {
        Column {
            data: Vec::new(),
            entities: Vec::new(),
            row_changed: Vec::new(),
            changed: Tick::ZERO,
            added: Tick::ZERO,
            bulk: Tick::ZERO,
            structural: Tick::ZERO,
        }
    }
}

impl<T> Column<T> {
    pub fn new() -> Column<T> {
        Column::default()
    }

    // The Entity owning each row, aligned with the data slice.
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    pub fn changed_tick(&self) -> Tick {
        self.changed
    }

    pub fn added_tick(&self) -> Tick {
        self.added
    }

    // Every tick stamp at once, for a consumer that needs more than the coarse
    // change tick to decide how much of the column to re-examine.
    pub fn ticks(&self) -> ColumnTicks {
        ColumnTicks {
            changed: self.changed,
            added: self.added,
            bulk: self.bulk,
            structural: self.structural,
        }
    }

    // The change tick of each row, aligned with the data and entity slices.
    pub fn row_ticks(&self) -> &[Tick] {
        &self.row_changed
    }

    // Pre-allocate capacity for `additional` more rows (data + entity ids),
    // ahead of a bulk load.
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
        self.entities.reserve(additional);
        self.row_changed.reserve(additional);
    }

    // Rows the column can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    // Append a row. Stamps every tick: the row is newly added, the column grew,
    // and the new row is (trivially) changed this tick.
    pub fn push(&mut self, entity: Entity, value: T, tick: Tick) {
        self.data.push(value);
        self.entities.push(entity);
        self.row_changed.push(tick);
        self.added = tick;
        self.changed = tick;
        self.structural = tick;
        debug_assert_eq!(self.data.len(), self.entities.len());
        debug_assert_eq!(self.data.len(), self.row_changed.len());
    }

    // Remove row `index`, moving the last row into its place. Returns the
    // removed value. O(1), but reorders the column: a caller that keys on a row
    // position must treat that position as invalidated. The moved row keeps its
    // own change tick, which travels with it.
    pub fn swap_remove(&mut self, index: usize, tick: Tick) -> T {
        let value = self.data.swap_remove(index);
        self.entities.swap_remove(index);
        self.row_changed.swap_remove(index);
        self.changed = tick;
        self.structural = tick;
        debug_assert_eq!(self.data.len(), self.entities.len());
        debug_assert_eq!(self.data.len(), self.row_changed.len());
        value
    }

    // Take all values, leaving the column empty. Stamps the change tick.
    pub fn drain(&mut self, tick: Tick) -> Vec<T> {
        self.entities.clear();
        self.row_changed.clear();
        self.changed = tick;
        self.structural = tick;
        core::mem::take(&mut self.data)
    }

    // Empty the column without returning the values.
    pub fn clear(&mut self, tick: Tick) {
        self.data.clear();
        self.entities.clear();
        self.row_changed.clear();
        self.changed = tick;
        self.structural = tick;
    }

    // Mutable access to the values. Stamps the bulk tick because the caller may
    // write any element, which leaves the per-row stamps unable to describe the
    // change on their own.
    pub fn values_mut(&mut self, tick: Tick) -> &mut [T] {
        self.changed = tick;
        self.bulk = tick;
        &mut self.data
    }

    // Mutable access to one row, stamping only that row. The targeted
    // counterpart of `values_mut`: a consumer comparing row ticks against its
    // last run recovers exactly which entities were written.
    pub fn value_mut(&mut self, row: usize, tick: Tick) -> Option<&mut T> {
        let value = self.data.get_mut(row)?;
        self.row_changed[row] = tick;
        self.changed = tick;
        Some(value)
    }

    // Iterate rows paired with their owning entity.
    pub fn iter_with_entities(&self) -> impl Iterator<Item = (Entity, &T)> {
        self.entities.iter().copied().zip(self.data.iter())
    }

    // Iterate rows mutably, paired with their owning entity. Stamps the bulk
    // tick because any element may be written.
    pub fn iter_mut_with_entities(&mut self, tick: Tick) -> impl Iterator<Item = (Entity, &mut T)> {
        self.changed = tick;
        self.bulk = tick;
        self.entities.iter().copied().zip(self.data.iter_mut())
    }

    // Rows whose own change tick is newer than `last_run`, paired with their
    // owning entity. Only meaningful when neither `bulk` nor `structural` moved
    // since `last_run`; past either of those the per-row stamps no longer
    // describe the whole change.
    pub fn changed_rows(&self, last_run: Tick) -> impl Iterator<Item = (Entity, &T)> {
        self.row_changed
            .iter()
            .zip(self.entities.iter().copied().zip(self.data.iter()))
            .filter_map(move |(row, pair)| row.is_newer_than(last_run).then_some(pair))
    }

    // Whether the column changed since a system's last run, wrap-safe.
    pub fn changed_since(&self, last_run: Tick) -> bool {
        self.changed.is_newer_than(last_run)
    }

    // Whether a row was added since a system's last run, wrap-safe.
    pub fn added_since(&self, last_run: Tick) -> bool {
        self.added.is_newer_than(last_run)
    }
}

impl<T> Deref for Column<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Entities;

    fn three() -> (Entities, [Entity; 3]) {
        let mut entities = Entities::new();
        let ids = [entities.alloc(), entities.alloc(), entities.alloc()];
        (entities, ids)
    }

    #[test]
    fn push_keeps_rows_aligned_and_stamps_ticks() {
        let (_e, ids) = three();
        let mut col: Column<u32> = Column::new();
        col.push(ids[0], 10, Tick(1));
        col.push(ids[1], 20, Tick(2));
        assert_eq!(col.len(), 2);
        assert_eq!(&col[..], &[10, 20]);
        assert_eq!(col.entities(), &[ids[0], ids[1]]);
        assert_eq!(col.added_tick(), Tick(2));
        assert_eq!(col.changed_tick(), Tick(2));
    }

    #[test]
    fn swap_remove_reorders_and_returns_value() {
        let (_e, ids) = three();
        let mut col: Column<u32> = Column::new();
        col.push(ids[0], 10, Tick(1));
        col.push(ids[1], 20, Tick(1));
        col.push(ids[2], 30, Tick(1));
        let removed = col.swap_remove(0, Tick(5));
        assert_eq!(removed, 10);
        // Last row moved into slot 0; data and entity stay aligned.
        assert_eq!(&col[..], &[30, 20]);
        assert_eq!(col.entities(), &[ids[2], ids[1]]);
        assert_eq!(col.changed_tick(), Tick(5));
    }

    #[test]
    fn drain_empties_and_returns_data() {
        let (_e, ids) = three();
        let mut col: Column<u32> = Column::new();
        col.push(ids[0], 10, Tick(1));
        col.push(ids[1], 20, Tick(1));
        let drained = col.drain(Tick(9));
        assert_eq!(drained, vec![10, 20]);
        assert!(col.is_empty());
        assert!(col.entities().is_empty());
        assert_eq!(col.changed_tick(), Tick(9));
    }

    #[test]
    fn values_mut_stamps_change() {
        let (_e, ids) = three();
        let mut col: Column<u32> = Column::new();
        col.push(ids[0], 10, Tick(1));
        for v in col.values_mut(Tick(7)) {
            *v += 1;
        }
        assert_eq!(&col[..], &[11]);
        assert!(col.changed_since(Tick(6)));
        assert!(!col.changed_since(Tick(7)));
    }

    #[test]
    fn iter_with_entities_pairs_rows() {
        let (_e, ids) = three();
        let mut col: Column<&str> = Column::new();
        col.push(ids[0], "a", Tick(1));
        col.push(ids[1], "b", Tick(1));
        let pairs: Vec<(Entity, &str)> = col.iter_with_entities().map(|(e, v)| (e, *v)).collect();
        assert_eq!(pairs, vec![(ids[0], "a"), (ids[1], "b")]);
    }

    #[test]
    fn value_mut_stamps_only_its_own_row() {
        let (_e, ids) = three();
        let mut col: Column<u32> = Column::new();
        col.push(ids[0], 10, Tick(1));
        col.push(ids[1], 20, Tick(1));
        col.push(ids[2], 30, Tick(1));

        *col.value_mut(1, Tick(7)).unwrap() = 99;
        assert_eq!(&col[..], &[10, 99, 30]);
        assert_eq!(col.row_ticks(), &[Tick(1), Tick(7), Tick(1)]);
        // The column tick still moves, so coarse consumers are unaffected.
        assert_eq!(col.changed_tick(), Tick(7));
        // No whole-column write happened, so the bulk stamp stays put.
        assert_eq!(col.ticks().bulk, Tick::ZERO);

        let changed: Vec<(Entity, u32)> = col.changed_rows(Tick(1)).map(|(e, v)| (e, *v)).collect();
        assert_eq!(changed, vec![(ids[1], 99)]);
    }

    #[test]
    fn value_mut_returns_none_past_the_end() {
        let (_e, ids) = three();
        let mut col: Column<u32> = Column::new();
        col.push(ids[0], 10, Tick(1));
        assert!(col.value_mut(1, Tick(5)).is_none());
        // A miss stamps nothing.
        assert_eq!(col.changed_tick(), Tick(1));
    }

    #[test]
    fn values_mut_stamps_the_bulk_tick_and_leaves_rows_alone() {
        let (_e, ids) = three();
        let mut col: Column<u32> = Column::new();
        col.push(ids[0], 10, Tick(1));
        col.push(ids[1], 20, Tick(1));
        for v in col.values_mut(Tick(6)) {
            *v += 1;
        }
        // Rows are not individually stamped; `bulk` is what says they all moved.
        assert_eq!(col.row_ticks(), &[Tick(1), Tick(1)]);
        assert_eq!(col.ticks().bulk, Tick(6));
        assert_eq!(col.ticks().changed, Tick(6));
    }

    #[test]
    fn push_and_remove_stamp_the_structural_tick() {
        let (_e, ids) = three();
        let mut col: Column<u32> = Column::new();
        col.push(ids[0], 10, Tick(1));
        assert_eq!(col.ticks().structural, Tick(1));
        // A targeted write is not structural.
        col.value_mut(0, Tick(2));
        assert_eq!(col.ticks().structural, Tick(1));
        col.push(ids[1], 20, Tick(3));
        col.swap_remove(0, Tick(4));
        assert_eq!(col.ticks().structural, Tick(4));
        // The surviving row kept the tick it was pushed with.
        assert_eq!(col.row_ticks(), &[Tick(3)]);
        col.clear(Tick(5));
        assert_eq!(col.ticks().structural, Tick(5));
        assert!(col.row_ticks().is_empty());
    }

    #[test]
    fn changed_rows_survives_tick_wraparound() {
        let (_e, ids) = three();
        let mut col: Column<u32> = Column::new();
        col.push(ids[0], 10, Tick(u32::MAX - 1));
        col.push(ids[1], 20, Tick(u32::MAX - 1));
        // A write just past the wrap is still newer than the pre-wrap stamp.
        *col.value_mut(0, Tick(2)).unwrap() = 11;
        let changed: Vec<Entity> = col
            .changed_rows(Tick(u32::MAX - 1))
            .map(|(e, _)| e)
            .collect();
        assert_eq!(changed, vec![ids[0]]);
    }

    #[test]
    fn iter_mut_with_entities_pairs_rows_and_stamps_change() {
        let (_e, ids) = three();
        let mut col: Column<u32> = Column::new();
        col.push(ids[0], 10, Tick(1));
        col.push(ids[1], 20, Tick(1));
        let seen: Vec<Entity> = col
            .iter_mut_with_entities(Tick(4))
            .map(|(e, v)| {
                *v += 1;
                e
            })
            .collect();
        assert_eq!(seen, vec![ids[0], ids[1]]);
        assert_eq!(&col[..], &[11, 21]);
        assert!(col.changed_since(Tick(3)));
        assert!(!col.changed_since(Tick(4)));
    }
}
