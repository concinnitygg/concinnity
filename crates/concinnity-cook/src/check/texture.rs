use crate::world_builder::Diagnostic;
pub(crate) fn check(name: &str, args: &serde_json::Value) -> Result<(), Diagnostic> {
    crate::compile::texture::validate_texture_generator(args)
        .map_err(|e| Diagnostic::asset("Texture", name, e))
}
