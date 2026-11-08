//! Metal argument-table slots, typed by the kind of resource they hold: a
//! slot's type gives its table row its kind, so a typed constant cannot be
//! listed as another kind than it was declared.

use core::marker::PhantomData;

/// A kind of Metal argument-table slot.
pub trait SlotKind {
    /// The attribute MSL spells the kind with.
    const ATTRIBUTE: &'static str;
}

/// Buffers and inline bytes: `[[buffer(n)]]`.
pub struct Buffer;
/// Textures: `[[texture(n)]]`.
pub struct Texture;
/// Samplers: `[[sampler(n)]]`.
pub struct Sampler;

impl SlotKind for Buffer {
    const ATTRIBUTE: &'static str = "buffer";
}
impl SlotKind for Texture {
    const ATTRIBUTE: &'static str = "texture";
}
impl SlotKind for Sampler {
    const ATTRIBUTE: &'static str = "sampler";
}

/// A named entry parameter at one slot of kind `K`.
pub struct Slot<K> {
    /// The parameter's name in the emitted MSL.
    pub name: &'static str,
    /// The slot index the host encoder binds at.
    pub index: usize,
    kind: PhantomData<K>,
}

impl<K> Clone for Slot<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for Slot<K> {}

impl<K: SlotKind> Slot<K> {
    /// `name` at slot `index`.
    pub const fn new(name: &'static str, index: usize) -> Self {
        Self {
            name,
            index,
            kind: PhantomData,
        }
    }

    /// The slot as a table row.
    pub const fn binding(self) -> MetalBinding {
        MetalBinding {
            name: self.name,
            kind: K::ATTRIBUTE,
            index: self.index,
        }
    }
}

/// One entry parameter a Metal encoder binds, of any kind.
#[derive(Clone, Copy, Debug)]
pub struct MetalBinding {
    /// The parameter's name in the emitted MSL.
    pub name: &'static str,
    /// The attribute: `buffer`, `texture` or `sampler`.
    pub kind: &'static str,
    /// The slot index.
    pub index: usize,
}

/// A buffer row.
pub const fn buffer(name: &'static str, index: usize) -> MetalBinding {
    Slot::<Buffer>::new(name, index).binding()
}

/// A texture row.
pub const fn texture(name: &'static str, index: usize) -> MetalBinding {
    Slot::<Texture>::new(name, index).binding()
}

/// A sampler row.
pub const fn sampler(name: &'static str, index: usize) -> MetalBinding {
    Slot::<Sampler>::new(name, index).binding()
}

impl core::fmt::Display for MetalBinding {
    /// The slot as the MSL attribute spells it: `buffer(0)`.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}({})", self.kind, self.index)
    }
}
