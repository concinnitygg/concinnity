// A world's sculpted ground read from its compiled pages when it loads.

use super::*;
use crate::ecs::{PayloadLocator, PayloadStore, World};
use crate::error::{PayloadError, WorldError};
use alloc::boxed::Box;

// A store handing back one fixed payload for every locator.
struct OnePayload(Vec<u8>);

impl PayloadStore for OnePayload {
    fn read(&mut self, _locator: &PayloadLocator) -> Result<&[u8], PayloadError> {
        Ok(&self.0)
    }

    fn release(&mut self, _blob_index: u32) {}

    fn disk_backed(&self) -> bool {
        false
    }
}

const TOP: TileId = TileId {
    face: CubeFace(2),
    level: 3,
    x: 4,
    y: 3,
};

fn world_with(payload: Vec<u8>) -> World {
    let mut world = World::from_payloads(Box::new(OnePayload(payload)));
    world.push(Planet::default());
    world.push(PlanetGround {
        tile: [2, 3, 4, 3],
        locator: Some(PayloadLocator {
            blob_index: 0,
            offset: 0,
            len: 1,
        }),
    });
    world
}

#[test]
fn the_planet_reads_its_ground_once_and_keeps_it() {
    let page = EditPage::from_samples(&[4.0; PAGE_SAMPLES], &[1.0; PAGE_SAMPLES]).unwrap();
    let mut world = world_with(encode_page(&page));
    load_edits(&mut world.context()).unwrap();
    assert_eq!(
        world.query::<PlanetGround>().count(),
        0,
        "drained into the planet"
    );
    let planet = world.query::<Planet>().next().unwrap();
    let pages = planet.edit_pages.clone().unwrap();
    assert_eq!(pages.get(TOP).map(|p| &**p), Some(&page));
    let shape = planet.shape();
    assert!(shape.edits.as_ref().is_some_and(|e| Arc::ptr_eq(e, &pages)));
    let dir = TOP.direction(0.3, 0.6);
    assert!((shape.height(dir) - shape.with_edits(None).height(dir) - 4.0).abs() < 1e-9);
}

#[test]
fn a_world_without_ground_reads_none() {
    let mut world = World::new();
    world.push(Planet::default());
    load_edits(&mut world.context()).unwrap();
    assert!(world.query::<Planet>().next().unwrap().edit_pages.is_none());
}

#[test]
fn a_payload_that_holds_no_page_refuses_the_world() {
    let mut world = world_with(b"not a page".to_vec());
    let err = load_edits(&mut world.context()).unwrap_err();
    assert!(
        matches!(err, WorldError::PlanetGround(PagePayloadError::Compressed)),
        "{err:?}"
    );
}
