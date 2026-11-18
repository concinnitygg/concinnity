//! Writing a canvas's shapes frame by frame into last frame's list in place,
//! so a frame that only moves them allocates nothing.

use concinnity_engine::components::{TextAlign, UiShape};

// One shape to write: its rectangle, look, text, and tag.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Shape<'a> {
    pub rect: [f32; 4],
    pub fill: [f32; 4],
    pub border: [f32; 4],
    pub border_width: f32,
    pub radius: f32,
    pub text: &'a str,
    pub text_color: [f32; 3],
    pub text_scale: f32,
    pub align: TextAlign,
    pub tag: Option<u32>,
}

impl Default for Shape<'_> {
    fn default() -> Self {
        Self {
            rect: [0.0; 4],
            fill: [0.0; 4],
            border: [0.0; 4],
            border_width: 0.0,
            radius: 0.0,
            text: "",
            text_color: [1.0; 3],
            text_scale: 0.0,
            align: TextAlign::Left,
            tag: None,
        }
    }
}

// Writes shapes over a list from the start, reusing what each slot held.
pub(crate) struct Shapes<'a> {
    list: &'a mut Vec<UiShape>,
    len: usize,
}

impl<'a> Shapes<'a> {
    pub(crate) fn new(list: &'a mut Vec<UiShape>) -> Self {
        Self { list, len: 0 }
    }

    pub(crate) fn push(&mut self, shape: Shape<'_>) {
        if self.len == self.list.len() {
            self.list.push(UiShape::default());
        }
        let slot = &mut self.list[self.len];
        slot.rect = shape.rect;
        slot.fill = shape.fill;
        slot.border = shape.border;
        slot.border_width = shape.border_width;
        slot.radius = shape.radius;
        if slot.text != shape.text {
            slot.text.clear();
            slot.text.push_str(shape.text);
        }
        slot.text_color = shape.text_color;
        slot.text_scale = shape.text_scale;
        slot.align = shape.align;
        slot.tag = shape.tag;
        self.len += 1;
    }

    // Drop whatever last frame wrote past this frame's shapes.
    pub(crate) fn finish(self) {
        self.list.truncate(self.len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_rewrites_the_list_in_place() {
        let mut list = Vec::new();
        let mut w = Shapes::new(&mut list);
        for text in ["a", "b", "c"] {
            w.push(Shape {
                text,
                ..Shape::default()
            });
        }
        w.finish();
        let first = list[0].text.as_ptr();
        let mut w = Shapes::new(&mut list);
        w.push(Shape {
            text: "a",
            tag: Some(1),
            ..Shape::default()
        });
        w.finish();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].tag, Some(1));
        assert_eq!(
            list[0].text.as_ptr(),
            first,
            "the same text kept its storage"
        );
    }
}
