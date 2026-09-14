//! The one forwarding macro the three backends' RenderBackend impls share.
//!
//! Almost every method of the trait families is a mechanical 1:1 call into the
//! inherent method of the same name, so each family's impl block states the
//! signatures and this macro writes the bodies. Inherent methods shadow trait
//! methods in resolution, so `self.$name(...)` binds the inherent one and there
//! is no recursion. Forwarders that rename, drop args, or need a custom body
//! stay hand-written beside the invocation.
//!
//! The main-thread assertion is a parameter rather than a name resolved at the
//! expansion site, so the backend that owns the invariant names the function
//! that proves it. The `&mut self` arms assert first: every mutation reached
//! through the boxed trait object proves the main-thread invariant the
//! `unsafe impl Send` on each context rests on. The `&self` arms are read-only
//! and skip it.
//!
//! Metal's window entry points live on the shared AppKit layer rather than on
//! the context, which is what `via` is for: `via = self.window().appkit,
//! via_mut = self.window_mut().appkit` forwards the `&self` arms of the block to
//! the first receiver and the `&mut self` arms to the second.

// Recurses token-by-token over the signature list, threading the assert and the
// shared and mutable receiver prefixes through each step.
macro_rules! forward {
    (assert = $assert:path; $($sigs:tt)*) => {
        $crate::forward::forward!(@each $assert, [] [], $($sigs)*);
    };
    (assert = $assert:path,
     via = self . $get:ident () $(. $field:ident)*,
     via_mut = self . $get_mut:ident () $(. $field_mut:ident)*;
     $($sigs:tt)*) => {
        $crate::forward::forward!(
            @each $assert, [.$get() $(. $field)*] [.$get_mut() $(. $field_mut)*], $($sigs)*
        );
    };
    (@each $assert:path, [$($recv:tt)*] [$($recv_mut:tt)*],) => {};
    (@each $assert:path, [$($recv:tt)*] [$($recv_mut:tt)*],
     fn $name:ident(&self $(, $arg:ident: $ty:ty)* $(,)?) -> $ret:ty; $($rest:tt)*) => {
        fn $name(&self $(, $arg: $ty)*) -> $ret { self $($recv)* .$name($($arg),*) }
        $crate::forward::forward!(@each $assert, [$($recv)*] [$($recv_mut)*], $($rest)*);
    };
    (@each $assert:path, [$($recv:tt)*] [$($recv_mut:tt)*],
     fn $name:ident(&self $(, $arg:ident: $ty:ty)* $(,)?); $($rest:tt)*) => {
        fn $name(&self $(, $arg: $ty)*) { self $($recv)* .$name($($arg),*) }
        $crate::forward::forward!(@each $assert, [$($recv)*] [$($recv_mut)*], $($rest)*);
    };
    // A `RenderResult` return lifts the inherent method's error through `?`, so
    // an inherent method still reporting `String` lands as `RenderError::Other`.
    (@each $assert:path, [$($recv:tt)*] [$($recv_mut:tt)*],
     fn $name:ident(&mut self $(, $arg:ident: $ty:ty)* $(,)?) -> RenderResult<$ok:ty>; $($rest:tt)*) => {
        fn $name(&mut self $(, $arg: $ty)*) -> concinnity_core::render::error::RenderResult<$ok> {
            $assert(stringify!($name));
            Ok(self $($recv_mut)* .$name($($arg),*)?)
        }
        $crate::forward::forward!(@each $assert, [$($recv)*] [$($recv_mut)*], $($rest)*);
    };
    (@each $assert:path, [$($recv:tt)*] [$($recv_mut:tt)*],
     fn $name:ident(&mut self $(, $arg:ident: $ty:ty)* $(,)?) -> $ret:ty; $($rest:tt)*) => {
        fn $name(&mut self $(, $arg: $ty)*) -> $ret {
            $assert(stringify!($name));
            self $($recv_mut)* .$name($($arg),*)
        }
        $crate::forward::forward!(@each $assert, [$($recv)*] [$($recv_mut)*], $($rest)*);
    };
    (@each $assert:path, [$($recv:tt)*] [$($recv_mut:tt)*],
     fn $name:ident(&mut self $(, $arg:ident: $ty:ty)* $(,)?); $($rest:tt)*) => {
        fn $name(&mut self $(, $arg: $ty)*) {
            $assert(stringify!($name));
            self $($recv_mut)* .$name($($arg),*)
        }
        $crate::forward::forward!(@each $assert, [$($recv)*] [$($recv_mut)*], $($rest)*);
    };
}

pub(crate) use forward;
