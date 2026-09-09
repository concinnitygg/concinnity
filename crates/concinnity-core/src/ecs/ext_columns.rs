// src/ecs/ext_columns.rs
//
// Columns for component types the engine's registry does not list.
//
// The registered set is closed on purpose: one `Column<T>` field per type, so a
// query resolves to a column at compile time and the whole storage is one
// struct. That property is worth keeping, and it is why an unregistered type
// does not get a field: it gets an entry here, keyed by the same `ComponentId`
// the join index uses, holding the same `Column<T>` behind an erased pointer.
//
// Everything above this module is unchanged by that. `ComponentSlot` resolves a
// type to its column either way, so `query`, `join2`, `get_mut`, `despawn` and
// the rest are the same generic code over both halves. The one difference the
// rest of the storage has to carry is that a column here may not exist yet: a
// type nothing has pushed has no entry, which reads as an empty column.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use core::any::Any;
use core::fmt::Debug;

use crate::ecs::{Column, ComponentId, Entity, JoinIndex, Tick};

// What the storage needs of a column whose type it cannot name: enough to sweep
// an entity's row out of it, and the downcast back to the typed column.
trait ErasedColumn: Debug + Send + Sync + 'static {
    fn len(&self) -> usize;
    fn entity_at(&self, row: usize) -> Option<Entity>;
    fn remove_row(&mut self, row: usize, tick: Tick);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Debug + Send + Sync + 'static> ErasedColumn for Column<T> {
    fn len(&self) -> usize {
        self.entities().len()
    }

    fn entity_at(&self, row: usize) -> Option<Entity> {
        self.entities().get(row).copied()
    }

    fn remove_row(&mut self, row: usize, tick: Tick) {
        let _ = self.swap_remove(row, tick);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// The columns of component types the engine's registry does not list, keyed by
/// the same [`ComponentId`] the join index records them under.
///
/// A component type declared outside the engine
/// (`concinnity::components!`) stores its rows here. A type nothing has
/// pushed yet has no column, which every read treats as an empty one, so
/// nothing has to be registered before it is used.
#[derive(Default, Debug)]
pub struct ExtColumns {
    columns: BTreeMap<u8, Box<dyn ErasedColumn>>,
}

impl ExtColumns {
    /// Borrow `T`'s column, or `None` when nothing has pushed a `T` yet.
    ///
    /// Panics when a different type already claims `id`, which means two
    /// component types were declared with the same discriminant.
    pub fn column<T: Debug + Send + Sync + 'static>(&self, id: u8) -> Option<&Column<T>> {
        let column = self.columns.get(&id)?;
        Some(
            column
                .as_any()
                .downcast_ref::<Column<T>>()
                .unwrap_or_else(|| mismatch(id)),
        )
    }

    /// Mutably borrow `T`'s column, creating it if this is the first `T`.
    ///
    /// Panics on the same discriminant collision [`column`](ExtColumns::column)
    /// does.
    pub fn column_mut<T: Debug + Send + Sync + 'static>(&mut self, id: u8) -> &mut Column<T> {
        self.columns
            .entry(id)
            .or_insert_with(|| Box::new(Column::<T>::new()))
            .as_any_mut()
            .downcast_mut::<Column<T>>()
            .unwrap_or_else(|| mismatch(id))
    }

    /// Sweep `entity` out of every column here, patching each moved tail row in
    /// `join` the way the registered columns do.
    pub fn despawn_entity(&mut self, join: &mut JoinIndex, entity: Entity, tick: Tick) {
        for (&id, column) in &mut self.columns {
            let component = ComponentId::new(id);
            let Some(row) = join.row(entity, component) else {
                continue;
            };
            let row = row as usize;
            let last = column.len() - 1;
            let moved = if row != last {
                column.entity_at(last)
            } else {
                None
            };
            column.remove_row(row, tick);
            if let Some(moved) = moved {
                join.set(moved, component, row as u32);
            }
        }
    }

    /// Rows across every column here.
    pub fn len(&self) -> usize {
        self.columns.values().map(|c| c.len()).sum()
    }

    /// Whether every column here is empty.
    pub fn is_empty(&self) -> bool {
        self.columns.values().all(|c| c.len() == 0)
    }

    /// `(discriminant, count)` for each populated column, in discriminant
    /// order, to append to the registered types' census.
    pub fn census(&self) -> impl Iterator<Item = (u8, u32)> + '_ {
        self.columns
            .iter()
            .filter(|(_, c)| c.len() > 0)
            .map(|(&id, c)| (id, c.len() as u32))
    }
}

// A discriminant claimed by two different component types. Nothing can pick a
// winner: both would read each other's rows, so the storage refuses rather than
// handing one of them the wrong column.
fn mismatch(id: u8) -> ! {
    panic!(
        "two component types are declared with discriminant {id}: a component's \
         discriminant has to be its own, since the join index and the access \
         masks address it by that number",
    )
}

#[cfg(test)]
mod tests {
    use super::ExtColumns;
    use crate::ecs::{ComponentId, Entities, JoinIndex, Tick};

    #[derive(Debug, PartialEq)]
    struct Health(u32);

    #[derive(Debug, PartialEq)]
    struct Mana(u32);

    // Nothing pushed means no column, which every read is expected to treat as
    // an empty one. That is what lets a component type be used without being
    // registered first.
    #[test]
    fn an_untouched_type_has_no_column() {
        let ext = ExtColumns::default();
        assert!(ext.column::<Health>(90).is_none());
        assert!(ext.is_empty());
        assert_eq!(ext.len(), 0);
    }

    // The first mutable borrow creates the column; later borrows find it.
    #[test]
    fn the_first_mutable_borrow_creates_the_column() {
        let mut ext = ExtColumns::default();
        let mut entities = Entities::default();
        let entity = entities.alloc();

        ext.column_mut::<Health>(90)
            .push(entity, Health(7), Tick::ZERO);
        assert_eq!(ext.len(), 1);
        assert!(!ext.is_empty());
        assert_eq!(
            ext.column::<Health>(90).map(|c| c.len()),
            Some(1),
            "the column is found on a second borrow"
        );
    }

    // Two types are two columns; the census reports each populated one.
    #[test]
    fn columns_are_independent_and_censused() {
        let mut ext = ExtColumns::default();
        let mut entities = Entities::default();
        let a = entities.alloc();
        let b = entities.alloc();

        ext.column_mut::<Health>(90).push(a, Health(1), Tick::ZERO);
        ext.column_mut::<Mana>(91).push(b, Mana(2), Tick::ZERO);
        ext.column_mut::<Mana>(91).push(a, Mana(3), Tick::ZERO);

        let census: alloc::vec::Vec<(u8, u32)> = ext.census().collect();
        assert_eq!(census, [(90, 1), (91, 2)]);
        assert_eq!(ext.len(), 3);
    }

    // A despawn removes the entity's row and patches the row the swap moved,
    // so a later join probe reads the moved entity's own data.
    #[test]
    fn despawn_patches_the_moved_tail_row() {
        let mut ext = ExtColumns::default();
        let mut join = JoinIndex::new();
        let mut entities = Entities::default();
        let id = ComponentId::new(90);

        let first = entities.alloc();
        let second = entities.alloc();
        ext.column_mut::<Health>(90)
            .push(first, Health(1), Tick::ZERO);
        join.set(first, id, 0);
        ext.column_mut::<Health>(90)
            .push(second, Health(2), Tick::ZERO);
        join.set(second, id, 1);

        ext.despawn_entity(&mut join, first, Tick::ZERO);

        assert_eq!(ext.len(), 1);
        assert_eq!(join.row(second, id), Some(0), "the tail row moved down");
        let column = ext.column::<Health>(90).expect("the column survives");
        assert_eq!(column.first(), Some(&Health(2)));
    }

    // An entity with no row in a column is skipped rather than removing
    // someone else's row.
    #[test]
    fn despawn_skips_a_column_the_entity_is_not_in() {
        let mut ext = ExtColumns::default();
        let mut join = JoinIndex::new();
        let mut entities = Entities::default();

        let holder = entities.alloc();
        let other = entities.alloc();
        ext.column_mut::<Health>(90)
            .push(holder, Health(1), Tick::ZERO);
        join.set(holder, ComponentId::new(90), 0);

        ext.despawn_entity(&mut join, other, Tick::ZERO);
        assert_eq!(ext.len(), 1);
    }

    // Two types claiming one discriminant would read each other's rows, so the
    // store refuses instead of handing one of them the wrong column.
    #[test]
    #[should_panic(expected = "declared with discriminant 90")]
    fn a_shared_discriminant_is_refused() {
        let mut ext = ExtColumns::default();
        let mut entities = Entities::default();
        let entity = entities.alloc();
        ext.column_mut::<Health>(90)
            .push(entity, Health(1), Tick::ZERO);
        let _ = ext.column_mut::<Mana>(90);
    }
}
