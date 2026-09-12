// Closed authored vocabularies: the trait a fixed-word field's type carries,
// and the two macros that give one its name list and its serde impls.

/// The closed set of names an authored field of this type accepts.
///
/// A field typed by a `Vocabulary` is a picker in an authoring tool rather than
/// a text box: the asset registry names the type per field and reads
/// [`Vocabulary::VARIANTS`] off it, so a variant added to the enum reaches the
/// editor with no second list to update.
pub trait Vocabulary {
    /// Every canonical authored name, in the order a picker steps through
    /// them. Each is what serde writes for its variant.
    const VARIANTS: &'static [&'static str];
}

/// Give an enum its authored-name list: `ALL`, `NAMES`, `as_str`, and the
/// [`Vocabulary`] impl the asset registry reads.
///
/// The generated `as_str` matches exhaustively, so a variant added without a
/// name here fails to compile rather than going missing from the lists.
///
/// The names must be what serde writes for the variant: for a derived
/// `Serialize` that is the variant's `rename_all` spelling, and for one built
/// by [`vocabulary_synonyms`] it is this list by construction. One test in the
/// cook's registry holds every declared field to that.
macro_rules! vocabulary {
    ($ty:ident { $($variant:ident => $name:literal),+ $(,)? }) => {
        impl $ty {
            /// Every variant, in the order an editor picker steps through them.
            pub const ALL: &'static [$ty] = &[$($ty::$variant),+];

            /// Every variant's authored name, in [`Self::ALL`] order. The
            /// editor's picker list.
            pub const NAMES: &'static [&'static str] = &[$($name),+];

            /// This variant's canonical authored name: what serde writes for it.
            pub const fn as_str(self) -> &'static str {
                match self { $($ty::$variant => $name),+ }
            }
        }

        impl $crate::components::Vocabulary for $ty {
            const VARIANTS: &'static [&'static str] = <$ty>::NAMES;
        }
    };
}

/// Build the serde impls of a vocabulary that accepts authored synonyms, on top
/// of the `from_str_norm` the enum defines and the `as_str` [`vocabulary`]
/// generates.
///
/// A human-readable format carries the canonical name and accepts every synonym
/// `from_str_norm` knows, so authored JSON is unchanged by typing a field with
/// one of these. A binary format carries the variant's index in `ALL`, which is
/// what a derived impl writes: the baked bytes are a discriminant, not a string.
macro_rules! vocabulary_synonyms {
    ($ty:ident, $expecting:literal) => {
        impl serde::Serialize for $ty {
            fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
                if ser.is_human_readable() {
                    ser.serialize_str(self.as_str())
                } else {
                    let idx = <$ty>::ALL.iter().position(|v| v == self).unwrap_or(0);
                    ser.serialize_u32(idx as u32)
                }
            }
        }

        impl<'de> serde::Deserialize<'de> for $ty {
            fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
                use serde::de::Error as _;
                if de.is_human_readable() {
                    let name = alloc::borrow::Cow::<'de, str>::deserialize(de)?;
                    <$ty>::from_str_norm(&name).ok_or_else(|| {
                        D::Error::unknown_variant(
                            &name,
                            <$ty as $crate::components::Vocabulary>::VARIANTS,
                        )
                    })
                } else {
                    let idx = u32::deserialize(de)?;
                    <$ty>::ALL.get(idx as usize).copied().ok_or_else(|| {
                        D::Error::invalid_value(
                            serde::de::Unexpected::Unsigned(u64::from(idx)),
                            &$expecting,
                        )
                    })
                }
            }
        }
    };
}

pub(crate) use {vocabulary, vocabulary_synonyms};

#[cfg(test)]
mod tests {
    use crate::components::{SceneTransition, Vocabulary};
    use alloc::format;

    // The claim the binary arm rests on: a synonym vocabulary bakes to the same
    // bytes a plain derived enum of the same shape does.
    #[derive(serde::Serialize, serde::Deserialize)]
    enum DerivedShape {
        First,
        Second,
    }

    #[test]
    fn a_vocabulary_bakes_to_the_discriminant_a_derived_enum_writes() {
        for (transition, derived) in SceneTransition::ALL
            .iter()
            .zip([DerivedShape::First, DerivedShape::Second])
        {
            assert_eq!(
                postcard::to_allocvec(transition).expect("bakes"),
                postcard::to_allocvec(&derived).expect("bakes")
            );
            let bytes = postcard::to_allocvec(transition).expect("bakes");
            assert_eq!(
                postcard::from_bytes::<SceneTransition>(&bytes).expect("loads"),
                *transition
            );
            // The name, not the index, is what a world file carries.
            assert_eq!(
                serde_json::to_string(transition).expect("serializes"),
                format!("\"{}\"", transition.as_str())
            );
        }
    }

    #[test]
    fn a_baked_index_past_the_last_variant_is_refused() {
        assert!(postcard::from_bytes::<SceneTransition>(&[9]).is_err());
    }

    // Every vocabulary lists as many names as it has variants, and the trait
    // reports the same list the enum does.
    #[test]
    fn the_trait_reports_the_enums_own_names() {
        assert_eq!(SceneTransition::ALL.len(), SceneTransition::NAMES.len());
        assert_eq!(
            <SceneTransition as Vocabulary>::VARIANTS,
            SceneTransition::NAMES
        );
    }
}
