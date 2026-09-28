//! A typed reference kept as the target's authored `$id`.
//!
//! [`Ref<T>`](super::Ref) resolves its name to a dense id as it deserializes,
//! which is what a runtime component wants. A schema the build reads by name
//! (one it expands into other assets, which it hands the name on to) holds a
//! [`NameRef<T>`] instead: the same reference target, so the authoring
//! registry sees the field as a reference to `T`, with the name kept as text.
//! An empty name references nothing.

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

use alloc::string::String;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::ecs::asset_fields::ReferenceField;
use crate::ecs::reference::RefTarget;

/// A reference to a `T` asset by its authored `$id`.
pub struct NameRef<T: RefTarget> {
    name: String,
    _target: PhantomData<fn() -> T>,
}

impl<T: RefTarget> NameRef<T> {
    /// A reference to the asset named `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            _target: PhantomData,
        }
    }

    /// The referenced asset's `$id`; empty when the field names nothing.
    pub fn as_str(&self) -> &str {
        &self.name
    }

    /// Whether the field names nothing.
    pub fn is_empty(&self) -> bool {
        self.name.is_empty()
    }
}

impl<T: RefTarget> core::ops::Deref for NameRef<T> {
    type Target = str;

    fn deref(&self) -> &str {
        &self.name
    }
}

impl<T: RefTarget> ReferenceField for NameRef<T> {
    const TARGETS: &'static [&'static str] = T::TYPES;
}

impl<T: RefTarget> From<&str> for NameRef<T> {
    fn from(name: &str) -> Self {
        Self::new(name)
    }
}

impl<T: RefTarget> From<&String> for NameRef<T> {
    fn from(name: &String) -> Self {
        Self::new(name.clone())
    }
}

impl<T: RefTarget> From<String> for NameRef<T> {
    fn from(name: String) -> Self {
        Self::new(name)
    }
}

// Hand-written so no impl asks anything of `T` beyond being a target.
impl<T: RefTarget> Default for NameRef<T> {
    fn default() -> Self {
        Self::new(String::new())
    }
}

impl<T: RefTarget> Clone for NameRef<T> {
    fn clone(&self) -> Self {
        Self::new(self.name.clone())
    }
}

impl<T: RefTarget> PartialEq for NameRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl<T: RefTarget> Eq for NameRef<T> {}

impl<T: RefTarget> PartialEq<str> for NameRef<T> {
    fn eq(&self, other: &str) -> bool {
        self.name == other
    }
}

impl<T: RefTarget> PartialEq<&str> for NameRef<T> {
    fn eq(&self, other: &&str) -> bool {
        self.name == *other
    }
}

impl<T: RefTarget> Hash for NameRef<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl<T: RefTarget> fmt::Debug for NameRef<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.name, f)
    }
}

impl<T: RefTarget> fmt::Display for NameRef<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

impl<T: RefTarget> Serialize for NameRef<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.name.serialize(s)
    }
}

// Authored input may leave a reference null, which names nothing like an
// empty string; the non-self-describing blob form always carries the string.
impl<'de, T: RefTarget> Deserialize<'de> for NameRef<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if !d.is_human_readable() {
            return String::deserialize(d).map(Self::new);
        }
        Option::<String>::deserialize(d).map(|name| Self::new(name.unwrap_or_default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::AssetFields;
    use crate::ecs::reference::AnyAsset;

    struct Screen;

    impl RefTarget for Screen {
        const TYPES: &'static [&'static str] = &["Screen"];
    }

    #[derive(Debug, Default, serde::Serialize, serde::Deserialize, AssetFields)]
    #[serde(default)]
    struct Holder {
        screen: NameRef<Screen>,
        any: alloc::vec::Vec<NameRef<AnyAsset>>,
    }

    #[test]
    fn keeps_the_authored_name_through_json_and_postcard() {
        let h: Holder = serde_json::from_str(r#"{"screen":"pause","any":["a"]}"#).unwrap();
        assert_eq!(h.screen, "pause");
        assert_eq!(h.any[0].as_str(), "a");
        assert_eq!(
            serde_json::to_value(&h).unwrap(),
            serde_json::json!({"screen": "pause", "any": ["a"]})
        );
        let bytes = postcard::to_allocvec(&h).unwrap();
        let back: Holder = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.screen.as_str(), "pause");
    }

    #[test]
    fn an_empty_null_or_missing_name_names_nothing() {
        let h: Holder = serde_json::from_str("{}").unwrap();
        assert!(h.screen.is_empty());
        let h: Holder = serde_json::from_str(r#"{"screen":null,"any":[null]}"#).unwrap();
        assert!(h.screen.is_empty() && h.any[0].is_empty());
        assert!(NameRef::<Screen>::from("").is_empty());
    }

    #[test]
    fn is_a_reference_field_to_its_target() {
        let refs = Holder::ref_fields();
        assert_eq!(refs.len(), 2);
        assert_eq!(
            (refs[0].path.as_str(), refs[0].targets),
            ("screen", &["Screen"][..])
        );
        assert_eq!(refs[1].path, "any");
        assert!(refs[1].targets.is_empty());
    }
}
