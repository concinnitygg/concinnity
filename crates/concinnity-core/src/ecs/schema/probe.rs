//! The classifiers the derives expand to. Not an API: the generated code calls
//! methods on `&&Probe::<T>::NEW` or `&Probe::<T>::NEW`, and method resolution
//! picks the most specific trait `T` satisfies.
//!
//! A field type is a reference first, then a described type, and otherwise an
//! object with no described shape. A default value is boxed for serializing
//! when its type can be serialized, and dropped otherwise.

use alloc::boxed::Box;

use crate::ecs::asset_fields::ReferenceField;
pub use crate::ecs::asset_fields::probe::Probe;

use super::{DefaultValue, Described, FieldType};

/// A field type naming other assets.
pub trait TypeByReference {
    /// The field's JSON-shaped type.
    fn field_type(&self) -> FieldType;
}

impl<T: ReferenceField> TypeByReference for &&Probe<T> {
    fn field_type(&self) -> FieldType {
        FieldType::Reference(T::TARGETS)
    }
}

/// A field type with its own description.
pub trait TypeByDescription {
    /// The field's JSON-shaped type.
    fn field_type(&self) -> FieldType;
}

impl<T: Described> TypeByDescription for &Probe<T> {
    fn field_type(&self) -> FieldType {
        T::TYPE
    }
}

/// Any other field type: an object with no described shape.
pub trait TypeByNothing {
    /// The field's JSON-shaped type.
    fn field_type(&self) -> FieldType;
}

impl<T: ?Sized> TypeByNothing for Probe<T> {
    fn field_type(&self) -> FieldType {
        FieldType::Object
    }
}

/// A default whose type serializes.
pub trait DefaultBySerialize {
    /// The default's type.
    type Value;
    /// The default, boxed for serializing.
    fn boxed(&self, value: Self::Value) -> Option<DefaultValue>;
}

impl<T: serde::Serialize + 'static> DefaultBySerialize for &Probe<T> {
    type Value = T;
    fn boxed(&self, value: T) -> Option<DefaultValue> {
        Some(Box::new(value))
    }
}

/// A default whose type does not serialize: nothing to state.
pub trait DefaultByNothing {
    /// The default's type.
    type Value;
    /// Nothing.
    fn boxed(&self, value: Self::Value) -> Option<DefaultValue>;
}

impl<T> DefaultByNothing for Probe<T> {
    type Value = T;
    fn boxed(&self, _: T) -> Option<DefaultValue> {
        None
    }
}
