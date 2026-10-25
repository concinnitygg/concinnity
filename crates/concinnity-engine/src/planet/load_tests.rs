// A planet's edits read from its compiled payload when the world loads.

use super::*;
use crate::ecs::{PayloadStore, World};
use crate::error::{PayloadError, WorldError};
use alloc::boxed::Box;

// A store handing back one fixed payload.
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

fn planet_with(payload: Vec<u8>) -> World {
    let len = payload.len() as u64;
    let mut world = World::from_payloads(Box::new(OnePayload(payload)));
    world.push(Planet {
        locator: Some(PayloadLocator {
            blob_index: 0,
            offset: 0,
            len,
        }),
        ..Planet::default()
    });
    world
}

#[test]
fn a_planet_reads_its_edits_once_at_load() {
    let page = EditPage::from_samples(&[4.0; PAGE_SAMPLES], &[1.0; PAGE_SAMPLES]).unwrap();
    let top = TileId::root(CubeFace(2));
    let mut world = planet_with(encode_edits(&EditPages::new([(top, page)])));
    load_edits(&mut world.context()).unwrap();
    let planet = world.query::<Planet>().next().unwrap();
    let pages = planet.edit_pages.clone().unwrap();
    assert!(pages.get(top).is_some());
    let shape = planet.shape();
    assert!(shape.edits.as_ref().is_some_and(|e| Arc::ptr_eq(e, &pages)));
    let dir = top.direction(0.3, 0.6);
    assert!((shape.height(dir) - shape.with_edits(None).height(dir) - 4.0).abs() < 1e-9);
}

#[test]
fn a_planet_without_edits_reads_none() {
    let mut world = planet_with(Vec::new());
    load_edits(&mut world.context()).unwrap();
    assert!(world.query::<Planet>().next().unwrap().edit_pages.is_none());
    let mut bare = World::new();
    bare.push(Planet::default());
    load_edits(&mut bare.context()).unwrap();
}

#[test]
fn a_payload_that_is_not_edits_refuses_the_world() {
    let mut world = planet_with(b"not edits at all".to_vec());
    let err = load_edits(&mut world.context()).unwrap_err();
    assert!(
        matches!(err, WorldError::PlanetEdits(EditFileError::Magic)),
        "{err:?}"
    );
}
