//! A planet's stored edits: pages of height changes and detail weights on
//! quadtree tiles, laid over the generated ground.
//!
//! A point reads only the deepest page over it, bilinear between that page's
//! samples; pages at different levels never add together. A finer page is
//! made from the coarser one over it ([`EditPages::refined`]), so painting
//! detail into a small area starts from what the area already reads.

mod area;
mod page;
mod payload;

use alloc::sync::Arc;
use hashbrown::{HashMap, HashSet};

use concinnity_core::math::SaturatingFrom;

use super::shape::CubeFace;
use super::tile::{MAX_TILE_LEVEL, TileId};

pub use area::EditArea;
pub use page::{EditPage, PAGE_CELLS, PAGE_SAMPLES, PAGE_SIDE, PageFigures, PageSample};
pub use payload::{PagePayloadError, decode_page, encode_page};

// One quadtree level for every level a tile can sit at.
const LEVELS: usize = MAX_TILE_LEVEL as usize + 1;

/// Every edit page of one planet, by the tile each lies on.
///
/// The figures it keeps (the lowest and highest height change and the
/// steepest steps per level) are exact for a set built whole, and only ever
/// widen as pages are inserted, so they bound the edits either way.
#[derive(Clone, PartialEq)]
pub struct EditPages {
    pages: HashMap<TileId, PageEntry>,
    // Every tile some page lies strictly under.
    above: HashSet<TileId>,
    lowest: f64,
    highest: f64,
    // Per level, the steepest step of `e` and of `w` between samples.
    steps: [[f32; 2]; LEVELS],
}

#[derive(Clone, PartialEq)]
struct PageEntry {
    page: Arc<EditPage>,
    figures: PageFigures,
}

/// A page read at one point: the tile it lies on, and its values with their
/// slopes per unit of face coordinate along `s` and `t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EditHit {
    /// The tile of the page read.
    pub tile: TileId,
    /// The values, slopes per unit of face coordinate.
    pub sample: PageSample,
}

impl core::fmt::Debug for EditPages {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EditPages")
            .field("pages", &self.pages.len())
            .field("lowest", &self.lowest)
            .field("highest", &self.highest)
            .finish_non_exhaustive()
    }
}

impl Default for EditPages {
    fn default() -> Self {
        Self {
            pages: HashMap::new(),
            above: HashSet::new(),
            lowest: 0.0,
            highest: 0.0,
            steps: [[0.0; 2]; LEVELS],
        }
    }
}

impl EditPages {
    /// The set of `pages`, each on its tile; a later page on a tile replaces
    /// an earlier one, and a page on a tile no quadtree holds is dropped.
    pub fn new(pages: impl IntoIterator<Item = (TileId, EditPage)>) -> Self {
        let mut set = Self::default();
        for (tile, page) in pages {
            set.insert(tile, Arc::new(page));
        }
        set
    }

    /// Put `page` on `tile`, replacing any page there. Returns whether the
    /// tile is one a quadtree holds; a page on any other is dropped.
    pub fn insert(&mut self, tile: TileId, page: Arc<EditPage>) -> bool {
        if !tile.is_valid() {
            return false;
        }
        let figures = page.figures();
        self.lowest = self.lowest.min(f64::from(figures.e_min));
        self.highest = self.highest.max(f64::from(figures.e_max));
        let steps = &mut self.steps[usize::from(tile.level)];
        steps[0] = steps[0].max(figures.e_step);
        steps[1] = steps[1].max(figures.w_step);
        let mut up = tile.parent();
        while let Some(t) = up {
            if !self.above.insert(t) {
                break;
            }
            up = t.parent();
        }
        self.pages.insert(tile, PageEntry { page, figures });
        true
    }

    /// The page on `tile`.
    pub fn get(&self, tile: TileId) -> Option<&Arc<EditPage>> {
        self.pages.get(&tile).map(|e| &e.page)
    }

    /// The figures of the page on `tile`.
    pub fn figures(&self, tile: TileId) -> Option<PageFigures> {
        self.pages.get(&tile).map(|e| e.figures)
    }

    /// Every page with the tile it lies on, in no order.
    pub fn iter(&self) -> impl Iterator<Item = (TileId, &Arc<EditPage>)> {
        self.pages.iter().map(|(t, e)| (*t, &e.page))
    }

    /// How many pages there are.
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Whether there are no pages.
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// The most any page lowers the ground, in meters: 0 or less.
    pub fn lowest_change(&self) -> f64 {
        self.lowest
    }

    /// The most any page raises the ground, in meters: 0 or more.
    pub fn highest_change(&self) -> f64 {
        self.highest
    }

    /// The steepest step between neighboring samples of any page at `level`:
    /// of the height change, in meters, and of the detail weight.
    pub fn level_steps(&self, level: u8) -> [f64; 2] {
        self.steps
            .get(usize::from(level))
            .map_or([0.0; 2], |s| s.map(f64::from))
    }

    /// The deepest page on `tile` or a tile above it, and the tile it lies on.
    pub fn covering(&self, tile: TileId) -> Option<(TileId, &Arc<EditPage>)> {
        let mut at = Some(tile);
        while let Some(t) = at {
            if let Some(page) = self.get(t) {
                return Some((t, page));
            }
            at = t.parent();
        }
        None
    }

    /// The page for `fine` made from the page over it now: the coarse page's
    /// values sampled at the fine page's samples, or a page that changes
    /// nothing where no page lies over it.
    pub fn refined(&self, fine: TileId) -> EditPage {
        self.covering(fine)
            .and_then(|(coarse, page)| {
                if coarse == fine {
                    Some(EditPage::clone(page))
                } else {
                    page.refine(coarse, fine)
                }
            })
            .unwrap_or_else(EditPage::neutral)
    }

    /// The edits over `tile`, resolved once for every point read on it.
    pub fn over(&self, tile: TileId) -> TileEdits<'_> {
        TileEdits {
            pages: self,
            tile,
            base: self.covering(tile),
            deeper: self.above.contains(&tile),
        }
    }

    /// The deepest page at face coordinates `(s, t)` of `face`, read there.
    pub fn at(&self, face: CubeFace, s: f64, t: f64) -> Option<EditHit> {
        self.over(TileId::root(face)).at(s, t)
    }
}

/// The edits over one tile: the deepest page above or on it, and whether any
/// page lies deeper within it, so a point on a tile with no finer pages reads
/// its page with no lookup.
#[derive(Debug, Clone, Copy)]
pub struct TileEdits<'a> {
    pages: &'a EditPages,
    tile: TileId,
    base: Option<(TileId, &'a Arc<EditPage>)>,
    deeper: bool,
}

impl TileEdits<'_> {
    /// Whether no page lies over any of the tile.
    pub fn is_empty(&self) -> bool {
        self.base.is_none() && !self.deeper
    }

    /// The deepest page at face coordinates `(s, t)`, a point of the tile
    /// (a point past its edges reads as the nearest point on them), read
    /// there.
    pub fn at(&self, s: f64, t: f64) -> Option<EditHit> {
        let mut best = self.base;
        if self.deeper {
            let mut at = self.tile;
            loop {
                let child = child_at(at, s, t);
                if let Some(page) = self.pages.get(child) {
                    best = Some((child, page));
                }
                if !self.pages.above.contains(&child) {
                    break;
                }
                at = child;
            }
        }
        let (tile, page) = best?;
        let (s0, t0, span) = tile.span();
        let mut sample = page.sample((s - s0) / span, (t - t0) / span);
        sample.e_slope = sample.e_slope.map(|d| d / span);
        sample.w_slope = sample.w_slope.map(|d| d / span);
        Some(EditHit { tile, sample })
    }
}

// The child of `tile` under face coordinates `(s, t)`, the nearest one for a
// point outside the tile.
fn child_at(tile: TileId, s: f64, t: f64) -> TileId {
    let level = tile.level + 1;
    let n = f64::from(1u32 << level);
    let index = |c: f64, first: u32| {
        u32::saturating_from(libm::floor((c + 1.0) * 0.5 * n).max(0.0)).clamp(first, first + 1)
    };
    TileId {
        face: tile.face,
        level,
        x: index(s, tile.x * 2),
        y: index(t, tile.y * 2),
    }
}

#[cfg(test)]
mod tests;
