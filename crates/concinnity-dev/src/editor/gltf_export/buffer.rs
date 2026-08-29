// src/editor/gltf_export/buffer.rs
//
// Binary-buffer packing for glTF export: each accessor's data is appended to
// one BIN buffer in its own 4-byte-aligned bufferView, and the JSON half
// (bufferViews / accessors) is described alongside so `json.rs` can emit it
// without re-deriving offsets.

// glTF componentType codes.
pub(crate) const FLOAT: u32 = 5126;
pub(crate) const UNSIGNED_SHORT: u32 = 5123;

// One bufferView into the BIN buffer.
pub(crate) struct View {
    pub offset: usize,
    pub len: usize,
}

// One accessor over a whole view. `min` / `max` are per-component bounds,
// present only where the spec requires them (POSITION data).
pub(crate) struct Accessor {
    pub view: usize,
    pub component_type: u32,
    pub count: usize,
    pub element_type: &'static str,
    pub min: Option<Vec<f32>>,
    pub max: Option<Vec<f32>>,
}

// The BIN buffer under construction plus its view / accessor tables. Every
// `push_*` returns the new accessor's index.
#[derive(Default)]
pub(crate) struct BinBuffer {
    pub bytes: Vec<u8>,
    pub views: Vec<View>,
    pub accessors: Vec<Accessor>,
}

impl BinBuffer {
    // Start a 4-byte-aligned view; returns its index. `finish_view` seals it.
    fn begin_view(&mut self) -> usize {
        while !self.bytes.len().is_multiple_of(4) {
            self.bytes.push(0);
        }
        self.views.push(View {
            offset: self.bytes.len(),
            len: 0,
        });
        self.views.len() - 1
    }

    fn finish_view(&mut self, view: usize) {
        self.views[view].len = self.bytes.len() - self.views[view].offset;
    }

    // Append float elements of `components` each; `with_min_max` adds the
    // per-component bounds POSITION accessors require.
    fn push_f32s(
        &mut self,
        data: &[f32],
        components: usize,
        element_type: &'static str,
        with_min_max: bool,
    ) -> usize {
        let view = self.begin_view();
        for v in data {
            self.bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.finish_view(view);
        let count = data.len() / components;
        let (min, max) = if with_min_max && count > 0 {
            let mut min = vec![f32::INFINITY; components];
            let mut max = vec![f32::NEG_INFINITY; components];
            for element in data.chunks_exact(components) {
                for (c, v) in element.iter().enumerate() {
                    min[c] = min[c].min(*v);
                    max[c] = max[c].max(*v);
                }
            }
            (Some(min), Some(max))
        } else {
            (None, None)
        };
        self.accessors.push(Accessor {
            view,
            component_type: FLOAT,
            count,
            element_type,
            min,
            max,
        });
        self.accessors.len() - 1
    }

    pub(crate) fn push_vec3(&mut self, data: &[[f32; 3]], with_min_max: bool) -> usize {
        let flat: Vec<f32> = data.iter().flatten().copied().collect();
        self.push_f32s(&flat, 3, "VEC3", with_min_max)
    }

    pub(crate) fn push_vec2(&mut self, data: &[[f32; 2]]) -> usize {
        let flat: Vec<f32> = data.iter().flatten().copied().collect();
        self.push_f32s(&flat, 2, "VEC2", false)
    }

    pub(crate) fn push_vec4(&mut self, data: &[[f32; 4]]) -> usize {
        let flat: Vec<f32> = data.iter().flatten().copied().collect();
        self.push_f32s(&flat, 4, "VEC4", false)
    }

    // Column-major 4x4 matrices, flattened column-first as glTF stores them.
    pub(crate) fn push_mat4(&mut self, data: &[[[f32; 4]; 4]]) -> usize {
        let flat: Vec<f32> = data.iter().flatten().flatten().copied().collect();
        self.push_f32s(&flat, 16, "MAT4", false)
    }

    // JOINTS_0 data: VEC4 of unsigned shorts.
    pub(crate) fn push_u16_vec4(&mut self, data: &[[u16; 4]]) -> usize {
        let view = self.begin_view();
        for element in data {
            for v in element {
                self.bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        self.finish_view(view);
        self.accessors.push(Accessor {
            view,
            component_type: UNSIGNED_SHORT,
            count: data.len(),
            element_type: "VEC4",
            min: None,
            max: None,
        });
        self.accessors.len() - 1
    }

    // Triangle indices: SCALAR unsigned shorts.
    pub(crate) fn push_indices(&mut self, data: &[u16]) -> usize {
        let view = self.begin_view();
        for v in data {
            self.bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.finish_view(view);
        self.accessors.push(Accessor {
            view,
            component_type: UNSIGNED_SHORT,
            count: data.len(),
            element_type: "SCALAR",
            min: None,
            max: None,
        });
        self.accessors.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_view_starts_on_a_four_byte_boundary() {
        let mut buf = BinBuffer::default();
        // 3 u16 indices leave the buffer at 6 bytes; the next view must pad.
        buf.push_indices(&[0, 1, 2]);
        let pos = buf.push_vec3(&[[1.0, 2.0, 3.0]], false);
        assert_eq!(buf.views[0].offset, 0);
        assert_eq!(buf.views[0].len, 6);
        assert_eq!(buf.views[1].offset, 8);
        assert_eq!(buf.views[1].len, 12);
        assert_eq!(buf.accessors[pos].count, 1);
        assert_eq!(buf.bytes.len(), 20);
    }

    #[test]
    fn position_bounds_cover_each_component_independently() {
        let mut buf = BinBuffer::default();
        let a = buf.push_vec3(&[[1.0, -2.0, 0.5], [-1.0, 4.0, 0.5]], true);
        let acc = &buf.accessors[a];
        assert_eq!(acc.min.as_deref(), Some(&[-1.0, -2.0, 0.5][..]));
        assert_eq!(acc.max.as_deref(), Some(&[1.0, 4.0, 0.5][..]));
        assert_eq!(acc.element_type, "VEC3");
        assert_eq!(acc.component_type, FLOAT);
    }

    #[test]
    fn non_position_accessors_omit_bounds() {
        let mut buf = BinBuffer::default();
        let uv = buf.push_vec2(&[[0.0, 1.0]]);
        let w = buf.push_vec4(&[[1.0, 0.0, 0.0, 0.0]]);
        let j = buf.push_u16_vec4(&[[3, 0, 0, 0]]);
        for a in [uv, w, j] {
            assert!(buf.accessors[a].min.is_none() && buf.accessors[a].max.is_none());
        }
        assert_eq!(buf.accessors[j].component_type, UNSIGNED_SHORT);
    }

    #[test]
    fn matrices_flatten_column_major() {
        let mut buf = BinBuffer::default();
        let mut m = [[0.0f32; 4]; 4];
        m[3][1] = 7.0; // translation.y in the engine's column-major layout
        let a = buf.push_mat4(&[m]);
        assert_eq!(buf.accessors[a].element_type, "MAT4");
        assert_eq!(buf.accessors[a].count, 1);
        // Column 3, row 1 lands at float index 13.
        let at = 13 * 4;
        let v = f32::from_le_bytes([
            buf.bytes[at],
            buf.bytes[at + 1],
            buf.bytes[at + 2],
            buf.bytes[at + 3],
        ]);
        assert_eq!(v, 7.0);
    }

    #[test]
    fn an_empty_slice_yields_a_zero_count_accessor_without_bounds() {
        let mut buf = BinBuffer::default();
        let a = buf.push_vec3(&[], true);
        assert_eq!(buf.accessors[a].count, 0);
        assert!(buf.accessors[a].min.is_none());
    }
}
