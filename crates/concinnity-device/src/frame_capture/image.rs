// The shape of a read-back frame and turning its bytes into a PNG.

use concinnity_core::gfx::image_decode::{self, PixelLayout};
use concinnity_core::render::backend::error::{RenderError, RenderResult};

use crate::png_encode::encode_png_rows;

/// How a frame's pixels lie in read-back memory: `height` rows of `width`
/// texels, each row starting `row_pitch` bytes after the last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CaptureImage {
    pub width: u32,
    pub height: u32,
    pub row_pitch: usize,
    pub layout: PixelLayout,
}

impl CaptureImage {
    /// An image whose rows follow each other with no padding.
    pub(crate) fn tight(width: u32, height: u32, layout: PixelLayout) -> Self {
        Self {
            width,
            height,
            row_pitch: width as usize * layout.bytes_per_pixel(),
            layout,
        }
    }

    /// Bytes one row's texels occupy, padding excluded.
    pub(crate) fn row_bytes(&self) -> usize {
        self.width as usize * self.layout.bytes_per_pixel()
    }

    /// Bytes the read-back must hold: every row's pitch but the last's, which
    /// needs only its texels.
    pub(crate) fn byte_len(&self) -> usize {
        match self.height {
            0 => 0,
            h => self.row_pitch * (h as usize - 1) + self.row_bytes(),
        }
    }

    /// The texels of each row of `bytes`, top to bottom, padding left out.
    /// Fails when `bytes` cannot hold the image at its pitch.
    pub(crate) fn rows<'a>(&self, bytes: &'a [u8]) -> RenderResult<impl Iterator<Item = &'a [u8]>> {
        if bytes.len() < self.byte_len() || self.row_pitch < self.row_bytes() {
            return Err(RenderError::Other(format!(
                "capture: {} bytes cannot hold a {}x{} image at a {}-byte pitch",
                bytes.len(),
                self.width,
                self.height,
                self.row_pitch
            )));
        }
        let (pitch, row) = (self.row_pitch, self.row_bytes());
        Ok((0..self.height as usize).map(move |y| &bytes[y * pitch..y * pitch + row]))
    }
}

/// Decode `bytes` laid out as `image` and write them to a PNG at `path`, a row
/// at a time, so no full-size copy of the frame is made.
pub(crate) fn save_png(path: &str, image: &CaptureImage, bytes: &[u8]) -> RenderResult<()> {
    let mut rows = image.rows(bytes)?;
    encode_png_rows(path, image.width, image.height, |_, out| {
        if let Some(row) = rows.next() {
            image_decode::decode_into_rgba8(row, image.layout, out);
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tight_image_has_no_row_padding() {
        let image = CaptureImage::tight(3, 2, PixelLayout::Rgba16F { scrgb: true });
        assert_eq!(image.row_pitch, 24);
        assert_eq!(image.row_bytes(), 24);
        assert_eq!(image.byte_len(), 48);
    }

    #[test]
    fn the_last_padded_row_needs_only_its_texels() {
        let image = CaptureImage {
            row_pitch: 256,
            ..CaptureImage::tight(2, 3, PixelLayout::Bgra8)
        };
        assert_eq!(image.byte_len(), 256 * 2 + 8);
    }

    #[test]
    fn padded_bgra_rows_decode_to_tight_opaque_rgba() {
        let image = CaptureImage {
            row_pitch: 12,
            ..CaptureImage::tight(2, 2, PixelLayout::Bgra8)
        };
        #[rustfmt::skip]
        let bytes = [
            1, 2, 3, 0, 4, 5, 6, 0, 99, 99, 99, 99,
            7, 8, 9, 0, 10, 11, 12, 0,
        ];
        let mut rgba = Vec::new();
        for row in image.rows(&bytes).unwrap() {
            image_decode::decode_into_rgba8(row, image.layout, &mut rgba);
        }
        assert_eq!(
            rgba,
            [3, 2, 1, 255, 6, 5, 4, 255, 9, 8, 7, 255, 12, 11, 10, 255]
        );
    }

    #[test]
    fn a_short_read_back_is_an_error_not_a_panic() {
        let image = CaptureImage::tight(4, 4, PixelLayout::Rgba8);
        assert!(image.rows(&[0; 63]).is_err());
    }

    #[test]
    fn a_pitch_narrower_than_a_row_is_an_error() {
        let image = CaptureImage {
            row_pitch: 4,
            ..CaptureImage::tight(2, 1, PixelLayout::Rgba8)
        };
        assert!(image.rows(&[0; 64]).is_err());
    }
}
