//! The `#[asset(...)]` field attribute: `owned_file` marks a path field as a
//! file the asset owns.

use syn::Attribute;

/// Whether a field carries `#[asset(owned_file)]`. Any other `asset` key is
/// an error, so a misspelling cannot silently drop the mark.
pub(crate) fn is_owned_file(attrs: &[Attribute]) -> syn::Result<bool> {
    let mut owned = false;
    for attr in attrs.iter().filter(|a| a.path().is_ident("asset")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("owned_file") {
                owned = true;
                Ok(())
            } else {
                Err(meta.error("expected `owned_file`"))
            }
        })?;
    }
    Ok(owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attrs(field: syn::Field) -> Vec<Attribute> {
        field.attrs
    }

    #[test]
    fn owned_file_is_read_among_other_attributes() {
        let field: syn::Field = syn::parse_quote! {
            #[serde(default)]
            #[asset(owned_file)]
            source: String
        };
        assert!(is_owned_file(&attrs(field)).unwrap());
        let plain: syn::Field = syn::parse_quote! { #[serde(default)] source: String };
        assert!(!is_owned_file(&attrs(plain)).unwrap());
    }

    #[test]
    fn an_unknown_key_is_an_error() {
        let field: syn::Field = syn::parse_quote! { #[asset(owned)] source: String };
        assert!(is_owned_file(&attrs(field)).is_err());
    }
}
