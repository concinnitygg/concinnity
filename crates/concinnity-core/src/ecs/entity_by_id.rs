//! Asset identity on entities: the [`EntityById`] index, and the one path that
//! gives an entity its [`Identity`], which writes both so they never disagree.
//!
//! The loader identifies every entity it mints from a blob def; code that
//! creates an asset at runtime identifies it the same way, under a declared id
//! or one minted from the world's [`MintedIds`]. A forward lookup (asset id to
//! component) goes through the index; a reverse read (component to asset id)
//! joins the component's column with [`Identity`].

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::components::Identity;
use crate::ecs::asset_id::{AssetId, AssetIdsExhausted, MintedIds};
use crate::ecs::{
    ComponentId, ComponentMask, ComponentSlot, ComponentStorage, Entity, PipelineContext,
    Resources, World,
};

/// Maps every identified entity's asset id to the entity. Written only by
/// [`PipelineContext::identify`] / [`World::identify`], which insert the
/// entity's [`Identity`] alongside; a despawn through either drops the entry.
#[derive(Debug, Default)]
pub struct EntityById(BTreeMap<AssetId, Entity>);

impl EntityById {
    /// The entity the asset `id` was loaded or created into, if any.
    pub fn get(&self, id: AssetId) -> Option<Entity> {
        self.0.get(&id).copied()
    }

    /// Every indexed asset id with its entity, in id order.
    pub fn iter(&self) -> impl Iterator<Item = (AssetId, Entity)> + '_ {
        self.0.iter().map(|(&id, &entity)| (id, entity))
    }

    /// How many entities the index holds.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the index holds no entity.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// Give `entity` the identity `id`. Refused when the entity is dead, already
// carries an identity, or `id` already names another live entity, so the first
// entity to claim an id keeps it.
fn identify(
    components: &mut ComponentStorage,
    resources: &mut Resources,
    entity: Entity,
    id: AssetId,
) -> bool {
    if !components.is_alive(entity) || components.get::<Identity>(entity).is_some() {
        return false;
    }
    if !resources.contains::<EntityById>() {
        resources.insert(EntityById::default());
    }
    let index = resources
        .get_mut::<EntityById>()
        .expect("the index was just ensured");
    if index
        .get(id)
        .is_some_and(|other| components.is_alive(other))
    {
        return false;
    }
    index.0.insert(id, entity);
    components.insert_typed(entity, Identity(id))
}

// Despawn `entity`, dropping its index entry first so the index never holds an
// id whose entity is gone.
fn despawn(components: &mut ComponentStorage, resources: &mut Resources, entity: Entity) {
    if let Some(&Identity(id)) = components.get::<Identity>(entity)
        && let Some(index) = resources.get_mut::<EntityById>()
        && index.get(id) == Some(entity)
    {
        index.0.remove(&id);
    }
    components.despawn(entity);
}

// Drain every `C`, then retire each owner the drain left holding only its
// Identity: its index entry goes and it despawns, as an entity with no
// components left always has. An owner keeping other components (a Prop's
// entity after decomposition) keeps its identity.
pub(crate) fn drain<C: ComponentSlot>(
    components: &mut ComponentStorage,
    resources: &mut Resources,
) -> Vec<C> {
    let owners: Vec<Entity> = C::column(components)
        .map(|column| column.entities().to_vec())
        .unwrap_or_default();
    let drained = components.drain::<C>();
    let identity_only =
        ComponentMask::with(ComponentId::new(<Identity as ComponentSlot>::DISCRIMINANT));
    for entity in owners {
        if components.mask(entity) == identity_only {
            despawn(components, resources, entity);
        }
    }
    drained
}

// The next id from the world's shared minted counter.
pub(crate) fn mint_id(resources: &mut Resources) -> Result<AssetId, AssetIdsExhausted> {
    if !resources.contains::<MintedIds>() {
        resources.insert(MintedIds::default());
    }
    resources
        .get_mut::<MintedIds>()
        .expect("the counter was just ensured")
        .next_id()
}

impl PipelineContext<'_> {
    /// Give `entity` the asset identity `id`: its [`Identity`] component and
    /// its [`EntityById`] entry. `false`, changing nothing, when the entity is
    /// dead, already identified, or `id` already names another live entity.
    pub fn identify(&mut self, entity: Entity, id: AssetId) -> bool {
        #[cfg(debug_assertions)]
        crate::ecs::access_check::touch(crate::ecs::access_check::Touch::Structural {
            op: "identify",
        });
        identify(self.components, self.resources, entity, id)
    }

    /// Push `c` onto a new entity identified as `id` (see
    /// [`identify`](Self::identify)).
    pub fn push_identified<C: ComponentSlot>(&mut self, id: AssetId, c: C) -> Entity {
        let entity = self.push(c);
        self.identify(entity, id);
        entity
    }

    /// Push `c` onto a new entity identified by a freshly minted id.
    pub fn push_minted<C: ComponentSlot>(
        &mut self,
        c: C,
    ) -> Result<(AssetId, Entity), AssetIdsExhausted> {
        let id = self.mint_id()?;
        Ok((id, self.push_identified(id, c)))
    }

    /// The next id from the world's shared minted counter.
    pub fn mint_id(&mut self) -> Result<AssetId, AssetIdsExhausted> {
        #[cfg(debug_assertions)]
        crate::ecs::access_check::touch(crate::ecs::access_check::Touch::Structural {
            op: "mint_id",
        });
        mint_id(self.resources)
    }

    /// The entity the asset `id` was loaded or created into.
    pub fn entity_of(&self, id: AssetId) -> Option<Entity> {
        self.resource::<EntityById>()?.get(id)
    }

    /// The component `C` of the entity the asset `id` was loaded into.
    pub fn get_by_id<C: ComponentSlot>(&self, id: AssetId) -> Option<&C> {
        let entity = self.entity_of(id)?;
        self.get::<C>(entity)
    }

    /// Mutably borrow the component `C` of the entity the asset `id` was
    /// loaded into.
    pub fn get_mut_by_id<C: ComponentSlot>(&mut self, id: AssetId) -> Option<&mut C> {
        let entity = self.entity_of(id)?;
        self.get_mut::<C>(entity)
    }

    /// Remove and return every `C`, each paired with its entity's asset id, in
    /// column order (see [`drain`](Self::drain)).
    pub fn drain_with_ids<C: ComponentSlot>(&mut self) -> Vec<(Option<AssetId>, C)> {
        let ids: Vec<Option<AssetId>> = self
            .query_with_entity::<C>()
            .map(|(entity, _)| self.get::<Identity>(entity).map(|i| i.id()))
            .collect();
        ids.into_iter().zip(self.drain::<C>()).collect()
    }

    /// Remove an entity entirely: swap-remove its row from every component
    /// column, drop its [`EntityById`] entry, and recycle its id (a stale
    /// handle to it then reads as dead). A no-op on an already-dead or unknown
    /// entity.
    pub fn despawn(&mut self, entity: Entity) {
        #[cfg(debug_assertions)]
        crate::ecs::access_check::touch(crate::ecs::access_check::Touch::Structural {
            op: "despawn",
        });
        despawn(self.components, self.resources, entity);
    }
}

impl World {
    /// Give `entity` the asset identity `id`. Mirror of
    /// [`PipelineContext::identify`].
    pub fn identify(&mut self, entity: Entity, id: AssetId) -> bool {
        let (components, resources) = self.storage_and_resources();
        identify(components, resources, entity, id)
    }

    /// Push `c` onto a new entity identified as `id`. Mirror of
    /// [`PipelineContext::push_identified`].
    pub fn push_identified<C: ComponentSlot>(&mut self, id: AssetId, c: C) -> Entity {
        let entity = self.push(c);
        self.identify(entity, id);
        entity
    }

    /// The entity the asset `id` was loaded or created into. Mirror of
    /// [`PipelineContext::entity_of`].
    pub fn entity_of(&self, id: AssetId) -> Option<Entity> {
        self.resource::<EntityById>()?.get(id)
    }

    /// The component `C` of the entity the asset `id` was loaded into. Mirror
    /// of [`PipelineContext::get_by_id`].
    pub fn get_by_id<C: ComponentSlot>(&self, id: AssetId) -> Option<&C> {
        self.get::<C>(self.entity_of(id)?)
    }

    /// Mutably borrow the component `C` of the entity the asset `id` was loaded
    /// into. Mirror of [`PipelineContext::get_mut_by_id`].
    pub fn get_mut_by_id<C: ComponentSlot>(&mut self, id: AssetId) -> Option<&mut C> {
        let entity = self.entity_of(id)?;
        self.get_mut::<C>(entity)
    }

    /// Despawn an entity (all its components, recycling its id), dropping its
    /// [`EntityById`] entry. Mirror of [`PipelineContext::despawn`].
    pub fn despawn(&mut self, entity: Entity) {
        let (components, resources) = self.storage_and_resources();
        despawn(components, resources, entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Sprite, TextLabel, Transform};

    fn world_with_label(id: u32) -> (World, Entity) {
        let mut world = World::new();
        let entity = world.push_identified(AssetId(id), TextLabel::default());
        (world, entity)
    }

    #[test]
    fn identifying_writes_the_component_and_the_index_together() {
        let (world, entity) = world_with_label(4);
        assert_eq!(
            world.get::<Identity>(entity).map(|i| i.id()),
            Some(AssetId(4))
        );
        assert_eq!(world.entity_of(AssetId(4)), Some(entity));
    }

    #[test]
    fn a_lookup_by_id_reaches_the_entitys_component() {
        let (mut world, _) = world_with_label(4);
        world
            .get_mut_by_id::<TextLabel>(AssetId(4))
            .unwrap()
            .content = "hi".into();
        assert_eq!(
            world.get_by_id::<TextLabel>(AssetId(4)).unwrap().content,
            "hi"
        );
        assert!(world.get_by_id::<Sprite>(AssetId(4)).is_none());
        assert!(world.get_by_id::<TextLabel>(AssetId(5)).is_none());
    }

    #[test]
    fn an_identity_is_set_once() {
        let (mut world, entity) = world_with_label(4);
        assert!(!world.identify(entity, AssetId(9)));
        assert_eq!(
            world.get::<Identity>(entity).map(|i| i.id()),
            Some(AssetId(4))
        );
        assert_eq!(world.entity_of(AssetId(9)), None);
    }

    #[test]
    fn the_first_live_entity_keeps_a_claimed_id() {
        let (mut world, first) = world_with_label(4);
        let second = world.push_identified(AssetId(4), TextLabel::default());
        assert_eq!(world.entity_of(AssetId(4)), Some(first));
        assert!(world.get::<Identity>(second).is_none());
    }

    #[test]
    fn despawning_frees_the_id_for_a_new_entity() {
        let (mut world, first) = world_with_label(4);
        world.despawn(first);
        assert_eq!(world.entity_of(AssetId(4)), None);
        let second = world.push_identified(AssetId(4), TextLabel::default());
        assert_eq!(world.entity_of(AssetId(4)), Some(second));
    }

    #[test]
    fn identity_survives_a_drain_that_leaves_other_components() {
        let mut world = World::new();
        let entity = world.push_identified(AssetId(2), TextLabel::default());
        world.insert(entity, Transform::default());
        world.remove_all::<TextLabel>();
        assert!(world.is_alive(entity));
        assert_eq!(world.entity_of(AssetId(2)), Some(entity));
    }

    // A consumed asset's entity recycles once its last real component drains,
    // and its id stops resolving.
    #[test]
    fn a_drain_retires_an_entity_left_with_only_its_identity() {
        let mut world = World::new();
        let entity = world.push_identified(AssetId(2), Transform::default());
        world.remove_all::<Transform>();
        assert!(!world.is_alive(entity));
        assert_eq!(world.entity_of(AssetId(2)), None);
        assert_eq!(world.query::<Identity>().count(), 0);
    }

    #[test]
    fn a_dead_entity_is_not_identified() {
        let mut world = World::new();
        let entity = world.push(TextLabel::default());
        world.despawn(entity);
        assert!(!world.identify(entity, AssetId(1)));
        assert_eq!(world.entity_of(AssetId(1)), None);
    }

    #[test]
    fn minted_ids_count_up_from_the_minted_base() {
        let mut world = World::new();
        let mut ctx = world.context();
        let (a, ea) = ctx.push_minted(TextLabel::default()).unwrap();
        let (b, _) = ctx.push_minted(TextLabel::default()).unwrap();
        assert!(a.is_minted());
        assert_eq!(b.0, a.0 + 1);
        assert_eq!(ctx.entity_of(a), Some(ea));
    }

    #[test]
    fn a_drain_pairs_each_row_with_its_id_in_column_order() {
        let mut world = World::new();
        world.push_identified(AssetId(7), TextLabel::default());
        world.push(TextLabel::default());
        world.push_identified(AssetId(3), TextLabel::default());
        let ids: Vec<Option<AssetId>> = world
            .context()
            .drain_with_ids::<TextLabel>()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(ids, [Some(AssetId(7)), None, Some(AssetId(3))]);
        assert_eq!(world.query::<TextLabel>().count(), 0);
    }

    #[test]
    fn a_reverse_read_joins_a_column_with_identity_in_column_order() {
        let mut world = World::new();
        for id in [7, 3, 5] {
            world.push_identified(AssetId(id), TextLabel::default());
        }
        let ids: Vec<u32> = world
            .join2::<TextLabel, Identity>()
            .map(|(_, _, identity)| identity.id().0)
            .collect();
        assert_eq!(ids, [7, 3, 5]);
    }
}
