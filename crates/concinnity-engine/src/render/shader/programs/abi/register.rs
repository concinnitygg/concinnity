//! DirectX registers, typed by the class of resource they hold: a register's
//! type gives its table row its class, so a typed constant cannot be listed
//! as another class than it was declared.

use core::marker::PhantomData;

/// A class of HLSL register.
pub trait RegisterClass {
    /// The register letter HLSL spells the class with.
    const LETTER: &'static str;
}

/// Constant buffers and root constants: `b` registers.
pub struct Cbv;
/// Shader resource views: `t` registers.
pub struct Srv;
/// Unordered access views: `u` registers.
pub struct Uav;
/// Samplers: `s` registers.
pub struct Sampler;

impl RegisterClass for Cbv {
    const LETTER: &'static str = "b";
}
impl RegisterClass for Srv {
    const LETTER: &'static str = "t";
}
impl RegisterClass for Uav {
    const LETTER: &'static str = "u";
}
impl RegisterClass for Sampler {
    const LETTER: &'static str = "s";
}

/// A named shader parameter at one register of class `C`.
pub struct Register<C> {
    /// The parameter's name in the HLSL source.
    pub name: &'static str,
    /// The register index.
    pub index: u32,
    /// The register space.
    pub space: u32,
    class: PhantomData<C>,
}

impl<C> Clone for Register<C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C> Copy for Register<C> {}

impl<C: RegisterClass> Register<C> {
    /// `name` at register `index` in space 0.
    pub const fn new(name: &'static str, index: u32) -> Self {
        Self {
            name,
            index,
            space: 0,
            class: PhantomData,
        }
    }

    /// The same register in `space`.
    #[must_use]
    pub const fn in_space(self, space: u32) -> Self {
        Self { space, ..self }
    }

    /// The register as a table row.
    pub const fn binding(self) -> DxBinding {
        DxBinding {
            name: self.name,
            class: C::LETTER,
            index: self.index,
            space: self.space,
        }
    }
}

/// One parameter of a DirectX root signature, of any class.
#[derive(Clone, Copy, Debug)]
pub struct DxBinding {
    /// The parameter's name in the HLSL source.
    pub name: &'static str,
    /// The register letter: `b`, `t`, `u` or `s`.
    pub class: &'static str,
    /// The register index.
    pub index: u32,
    /// The register space.
    pub space: u32,
}

/// A constant buffer row.
pub const fn cbv(name: &'static str, index: u32) -> DxBinding {
    Register::<Cbv>::new(name, index).binding()
}

/// A shader resource view row.
pub const fn srv(name: &'static str, index: u32) -> DxBinding {
    Register::<Srv>::new(name, index).binding()
}

/// An unordered access view row.
pub const fn uav(name: &'static str, index: u32) -> DxBinding {
    Register::<Uav>::new(name, index).binding()
}

/// A sampler row.
pub const fn sampler(name: &'static str, index: u32) -> DxBinding {
    Register::<Sampler>::new(name, index).binding()
}

impl DxBinding {
    /// The same row in `space`.
    #[must_use]
    pub const fn in_space(self, space: u32) -> Self {
        Self { space, ..self }
    }
}

impl core::fmt::Display for DxBinding {
    /// The register as HLSL's `register()` spells it: `t0`, or `t0, space1`.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.space {
            0 => write!(f, "{}{}", self.class, self.index),
            space => write!(f, "{}{}, space{space}", self.class, self.index),
        }
    }
}
