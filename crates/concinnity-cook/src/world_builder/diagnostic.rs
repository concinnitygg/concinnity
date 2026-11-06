//! One problem a world failed validation for, and the asset it is about.

use std::fmt;

/// The asset a [`Diagnostic`] is about: its type as authored and its name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetRef {
    /// The asset's type, as the world names it.
    pub type_name: String,
    /// The asset's name.
    pub name: String,
}

/// One problem a world failed validation for: what is wrong, and the asset it
/// is about when it is about one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    asset: Option<AssetRef>,
    message: String,
}

impl Diagnostic {
    /// A problem with the world as a whole, or one no single asset owns.
    pub fn world(message: impl Into<String>) -> Self {
        Self {
            asset: None,
            message: message.into(),
        }
    }

    /// A problem with the `type_name` asset `name`.
    pub fn asset(
        type_name: impl Into<String>,
        name: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            asset: Some(AssetRef {
                type_name: type_name.into(),
                name: name.into(),
            }),
            message: message.into(),
        }
    }

    /// The asset the problem is about, if one.
    pub fn asset_ref(&self) -> Option<&AssetRef> {
        self.asset.as_ref()
    }

    /// What is wrong, without the asset it is about.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The same problem, found while expanding `context`: what the message
    /// names it by, in parentheses after it.
    #[must_use]
    pub(crate) fn within(mut self, context: &str) -> Self {
        self.message = format!("{} (in {context})", self.message);
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.asset {
            Some(a) => write!(f, "{} '{}': {}", a.type_name, a.name, self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for Diagnostic {}

impl From<String> for Diagnostic {
    fn from(message: String) -> Self {
        Self::world(message)
    }
}

impl From<&str> for Diagnostic {
    fn from(message: &str) -> Self {
        Self::world(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // An asset's problem reads as the hand-written prefixes did, so every
    // printed message stays as it was.
    #[test]
    fn an_asset_problem_names_its_asset_first() {
        let d = Diagnostic::asset("Prefab", "crate", "missing `entries`");
        assert_eq!(d.to_string(), "Prefab 'crate': missing `entries`");
        assert_eq!(
            d.asset_ref(),
            Some(&AssetRef {
                type_name: "Prefab".into(),
                name: "crate".into(),
            })
        );
        assert_eq!(d.message(), "missing `entries`");
    }

    #[test]
    fn a_world_problem_is_its_message() {
        let d = Diagnostic::from("Include cycle: a -> b -> a".to_string());
        assert_eq!(d.to_string(), "Include cycle: a -> b -> a");
        assert!(d.asset_ref().is_none());
    }

    #[test]
    fn context_follows_the_message() {
        let d = Diagnostic::asset("Prop", "rock", "unknown mesh").within("preset 'cliffs'");
        assert_eq!(
            d.to_string(),
            "Prop 'rock': unknown mesh (in preset 'cliffs')"
        );
    }
}
