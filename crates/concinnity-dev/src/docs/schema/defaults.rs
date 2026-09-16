// Default values, read off the `impl Default` blocks in the file that declares
// the struct. Only literals are recoverable; anything computed renders as
// `UNKNOWN` and is dropped rather than guessed at.

use std::collections::HashMap;

// Stands in for a value that is not a literal. A field whose whole default is
// this is reported as having none; one that merely embeds it (an array with a
// computed element) keeps the partial rendering, which reads truer than
// dropping the whole default.
pub(super) const UNKNOWN: &str = "\u{2014}";

// struct ident -> field ident -> rendered default, for every `impl Default` in
// the file. Built once per file: an asset's default lives beside its struct.
pub(super) fn per_struct(file: &syn::File) -> HashMap<String, HashMap<String, String>> {
    let mut out: HashMap<String, HashMap<String, String>> = HashMap::new();
    for item in &file.items {
        let syn::Item::Impl(imp) = item else {
            continue;
        };
        if !implements_default(imp) {
            continue;
        }
        let syn::Type::Path(tp) = imp.self_ty.as_ref() else {
            continue;
        };
        let Some(name) = tp.path.segments.last().map(|s| s.ident.to_string()) else {
            continue;
        };
        let entry = out.entry(name).or_default();
        for impl_item in &imp.items {
            if let syn::ImplItem::Fn(f) = impl_item
                && f.sig.ident == "default"
            {
                for stmt in &f.block.stmts {
                    fields_from_stmt(stmt, entry);
                }
            }
        }
    }
    out
}

fn implements_default(imp: &syn::ItemImpl) -> bool {
    imp.trait_
        .as_ref()
        .is_some_and(|(_, path, _)| path.segments.last().is_some_and(|s| s.ident == "Default"))
}

fn fields_from_stmt(stmt: &syn::Stmt, out: &mut HashMap<String, String>) {
    if let syn::Stmt::Expr(e, _) = stmt {
        fields_from_expr(e, out);
    }
}

fn fields_from_expr(expr: &syn::Expr, out: &mut HashMap<String, String>) {
    match expr {
        syn::Expr::Struct(es) => {
            for fv in &es.fields {
                if let syn::Member::Named(n) = &fv.member {
                    out.insert(n.to_string(), render_expr(&fv.expr));
                }
            }
        }
        syn::Expr::Block(eb) => {
            for stmt in &eb.block.stmts {
                fields_from_stmt(stmt, out);
            }
        }
        _ => {}
    }
}

// Render a default's initializer as the value a JSON author would write.
fn render_expr(expr: &syn::Expr) -> String {
    match expr {
        syn::Expr::Lit(l) => match &l.lit {
            syn::Lit::Float(f) => f.to_string(),
            syn::Lit::Int(i) => i.base10_digits().to_string(),
            syn::Lit::Bool(b) => b.value.to_string(),
            syn::Lit::Str(s) => format!("\"{}\"", s.value()),
            _ => UNKNOWN.to_string(),
        },
        syn::Expr::Array(arr) => {
            let items: Vec<String> = arr.elems.iter().map(render_expr).collect();
            format!("[{}]", items.join(", "))
        }
        syn::Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => {
            let inner = render_expr(&u.expr);
            if inner == UNKNOWN {
                UNKNOWN.to_string()
            } else {
                format!("-{inner}")
            }
        }
        syn::Expr::MethodCall(mc) if mc.method == "to_string" => render_expr(&mc.receiver),
        syn::Expr::Path(p) => {
            let last = p
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            if last == "None" {
                "null".to_string()
            } else {
                UNKNOWN.to_string()
            }
        }
        _ => UNKNOWN.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults_for(src: &str, struct_name: &str) -> HashMap<String, String> {
        let file: syn::File = syn::parse_str(src).expect("parse");
        per_struct(&file).remove(struct_name).unwrap_or_default()
    }

    #[test]
    fn literal_defaults_render_as_json_values() {
        let d = defaults_for(
            r#"
            impl Default for A {
                fn default() -> Self {
                    Self {
                        count: 2,
                        scale: 1.5,
                        on: true,
                        name: "metal".to_string(),
                        offset: -3.0,
                        pair: [0.0, 1.0],
                        missing: None,
                    }
                }
            }
            "#,
            "A",
        );
        assert_eq!(d["count"], "2");
        assert_eq!(d["scale"], "1.5");
        assert_eq!(d["on"], "true");
        assert_eq!(d["name"], "\"metal\"");
        assert_eq!(d["offset"], "-3.0");
        assert_eq!(d["pair"], "[0.0, 1.0]");
        assert_eq!(d["missing"], "null");
    }

    #[test]
    fn computed_defaults_render_as_unknown() {
        let d = defaults_for(
            "impl Default for A { fn default() -> Self { Self { size: compute() } } }",
            "A",
        );
        assert_eq!(d["size"], UNKNOWN);
    }

    #[test]
    fn a_trailing_block_expression_is_followed() {
        let d = defaults_for(
            "impl Default for A { fn default() -> Self { { Self { n: 1 } } } }",
            "A",
        );
        assert_eq!(d["n"], "1");
    }

    #[test]
    fn only_the_default_impl_contributes() {
        let d = defaults_for(
            "impl A { fn default() -> Self { Self { n: 1 } } }
             impl Default for B { fn default() -> Self { Self { n: 2 } } }",
            "A",
        );
        assert!(d.is_empty());
    }
}
