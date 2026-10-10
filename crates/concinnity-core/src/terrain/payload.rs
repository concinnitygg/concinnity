//! The cooked `Terrain` payload: the height grid, then one optional density
//! mask per layer, in the layers' order.
//!
//! Little-endian: `"TRRN"`, `u32` resolution, `(resolution + 1)^2` `f32`
//! heights, `u32` mask count, then per mask a `u32` width and height (both 0
//! for a layer with no mask) followed by its texels. The extent is the
//! component's own, so it is not repeated here.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use super::{DensityMask, TerrainGrid};

const MAGIC: &[u8; 4] = b"TRRN";

/// What a terrain cooks into.
#[derive(Debug, Clone, PartialEq)]
pub struct TerrainPayload {
    /// The height grid.
    pub grid: TerrainGrid,
    /// Each layer's density mask, `None` where a layer grows everywhere.
    pub masks: Vec<Option<DensityMask>>,
}

impl TerrainPayload {
    /// The payload's bytes.
    pub fn encode(&self) -> Vec<u8> {
        let heights = self.grid.heights();
        let mut out = Vec::with_capacity(12 + heights.len() * 4);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.grid.resolution().to_le_bytes());
        for h in heights {
            out.extend_from_slice(&h.to_le_bytes());
        }
        out.extend_from_slice(&(self.masks.len() as u32).to_le_bytes());
        for mask in &self.masks {
            let (w, h) = mask.as_ref().map_or((0, 0), |m| (m.width(), m.height()));
            out.extend_from_slice(&w.to_le_bytes());
            out.extend_from_slice(&h.to_le_bytes());
            if let Some(m) = mask {
                out.extend_from_slice(m.texels());
            }
        }
        out
    }

    /// Read a payload cooked for a terrain spanning `extent`.
    pub fn decode(bytes: &[u8], extent: [f32; 2]) -> Result<Self, String> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(4)? != MAGIC {
            return Err("not a terrain payload".into());
        }
        let resolution = r.u32()?;
        if resolution == 0 || resolution > super::MAX_TERRAIN_RESOLUTION {
            return Err(format!(
                "terrain payload resolution {resolution} is out of range"
            ));
        }
        let side = resolution as usize + 1;
        let heights = r
            .take(side * side * 4)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let grid = TerrainGrid::new(resolution, extent, heights)?;
        let count = r.u32()? as usize;
        let mut masks = Vec::with_capacity(count.min(64));
        for _ in 0..count {
            let (w, h) = (r.u32()?, r.u32()?);
            masks.push(if w == 0 && h == 0 {
                None
            } else {
                if w.max(h) > super::MAX_MASK_SIZE {
                    return Err(format!("terrain payload mask {w}x{h} is too large"));
                }
                let texels = r.take(w as usize * h as usize)?.to_vec();
                Some(DensityMask::new(w, h, texels)?)
            });
        }
        if r.at != bytes.len() {
            return Err("terrain payload has trailing bytes".into());
        }
        Ok(Self { grid, masks })
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len())
            .ok_or("terrain payload is truncated")?;
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn payload() -> TerrainPayload {
        TerrainPayload {
            grid: TerrainGrid::new(2, [3.0, 4.0], (0..9).map(|i| i as f32 * 0.5).collect())
                .unwrap(),
            masks: vec![
                None,
                Some(DensityMask::new(3, 2, vec![0, 1, 2, 3, 4, 5]).unwrap()),
            ],
        }
    }

    #[test]
    fn a_payload_round_trips() {
        let p = payload();
        assert_eq!(TerrainPayload::decode(&p.encode(), [3.0, 4.0]).unwrap(), p);
    }

    #[test]
    fn a_damaged_payload_is_refused() {
        let bytes = payload().encode();
        assert!(TerrainPayload::decode(&bytes[..bytes.len() - 1], [3.0, 4.0]).is_err());
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(TerrainPayload::decode(&extra, [3.0, 4.0]).is_err());
        let mut wrong = bytes.clone();
        wrong[0] = b'X';
        assert!(TerrainPayload::decode(&wrong, [3.0, 4.0]).is_err());
        assert!(TerrainPayload::decode(&bytes, [0.0, 4.0]).is_err());
    }

    #[test]
    fn a_payload_whose_counts_overflow_is_refused() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(TerrainPayload::decode(&bytes, [1.0, 1.0]).is_err());
    }
}
