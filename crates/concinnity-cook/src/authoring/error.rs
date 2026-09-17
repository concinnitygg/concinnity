//! Why an authored asset did not resolve into a blob record.

use thiserror::Error;

/// Why an authored asset did not resolve into a blob record.
///
/// The authored medium is JSON, so a schema failure carries serde_json's own
/// report of what it found and what it expected, which is the part an author
/// can act on.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AuthoringError {
    /// The type name is not one the registry has.
    #[error("`{0}` is not an asset type this engine registers")]
    UnknownType(String),

    /// The type is registered but a world cannot declare it: it is minted by
    /// the runtime rather than authored.
    #[error("{asset} is created by the runtime, so a world cannot declare one")]
    NotAuthorable {
        /// The type that was asked for.
        asset: &'static str,
    },

    /// The type compiles into the resource stream rather than the component
    /// stream, so it has no component record to build.
    #[error("{asset} compiles into the resource stream, not a component record")]
    NotAComponent {
        /// The type that was asked for.
        asset: &'static str,
    },

    /// The args do not fit the type's schema.
    #[error("{asset} args: {source}")]
    Args {
        /// The type whose schema rejected them.
        asset: &'static str,
        /// What the schema reported: what it found, and what it expected.
        #[source]
        source: serde_json::Error,
    },

    /// The baked component could not be serialized into its record.
    #[error("serializing the baked {asset} failed")]
    Encode {
        /// The type being baked.
        asset: &'static str,
        /// What postcard reported.
        #[source]
        source: postcard::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    // The schema report is the whole value of typing this: the flat status it
    // replaces said only that an argument was invalid.
    #[test]
    fn a_schema_failure_reports_what_the_author_got_wrong() {
        let source = serde_json::from_value::<Vec<u32>>(serde_json::json!({ "generator": 42 }))
            .expect_err("an object is not a list");
        let error = AuthoringError::Args {
            asset: "ProceduralMesh",
            source,
        };

        assert!(error.to_string().contains("ProceduralMesh"), "{error}");
        assert!(error.to_string().contains("invalid type"), "{error}");
        assert!(error.source().is_some(), "the serde report stays reachable");
    }

    // The two refusals a registered type can meet read differently, since one
    // is a world declaring something only the runtime makes and the other is a
    // type that belongs in the resource stream.
    #[test]
    fn the_registered_but_unusable_refusals_say_which_one_they_are() {
        let runtime = AuthoringError::NotAuthorable { asset: "Transform" };
        assert!(runtime.to_string().contains("runtime"), "{runtime}");

        let resource = AuthoringError::NotAComponent { asset: "Texture" };
        assert!(
            resource.to_string().contains("resource stream"),
            "{resource}"
        );
    }
}
