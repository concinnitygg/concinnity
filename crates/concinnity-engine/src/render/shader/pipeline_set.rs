//! The pipelines a backend rebuilds when its shader sources change, one set
//! per feature: how the set is built from the current sources and where it is
//! installed. A rebuild builds every live set before installing any, so a
//! failed build leaves every running pipeline in place.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::render::backend::error::RenderResult;

/// One feature's pipelines in a backend context `C`.
pub trait PipelineSet<C> {
    /// Build the feature's pipelines without installing them, reading shader
    /// sources from disk first when `hot_reload` is set, or `None` when the
    /// feature is not live.
    ///
    /// # Errors
    ///
    /// Returns the build's error; nothing is installed.
    fn stage(&self, ctx: &C, hot_reload: bool) -> RenderResult<Option<Staged<C>>>;
}

/// Pipelines built and waiting to be installed in a `C`.
pub struct Staged<C>(Box<dyn FnOnce(&mut C)>);

impl<C> Staged<C> {
    /// Pipelines that `install` puts in place.
    pub fn new(install: impl FnOnce(&mut C) + 'static) -> Self {
        Self(Box::new(install))
    }

    /// Put the pipelines in place in `ctx`.
    pub fn install(self, ctx: &mut C) {
        (self.0)(ctx);
    }
}

/// A feature's pipelines as a pair of functions: `build` makes them for a live
/// feature (`None` when it is not), and `install` puts them in place.
pub struct Pipelines<C, P> {
    /// The pipelines built from the sources `hot_reload` selects, or `None`
    /// when the feature is not live.
    pub build: fn(&C, bool) -> Option<RenderResult<P>>,
    /// Put built pipelines in place.
    pub install: fn(&mut C, P),
}

impl<C: 'static, P: 'static> PipelineSet<C> for Pipelines<C, P> {
    fn stage(&self, ctx: &C, hot_reload: bool) -> RenderResult<Option<Staged<C>>> {
        let install = self.install;
        Ok(match (self.build)(ctx, hot_reload) {
            Some(built) => {
                let pipelines = built?;
                Some(Staged::new(move |ctx| install(ctx, pipelines)))
            }
            None => None,
        })
    }
}

/// A feature held in the context as optional state `S`: live while the state
/// exists, its pipelines built from that state and installed into it.
pub struct Gated<C, S, P> {
    /// The feature's state, or `None` when it is not live.
    pub state: fn(&C) -> Option<&S>,
    /// The same state, to install into.
    pub state_mut: fn(&mut C) -> Option<&mut S>,
    /// The pipelines built from the sources `hot_reload` selects.
    pub build: fn(&C, &S, bool) -> RenderResult<P>,
    /// Put built pipelines in place.
    pub install: fn(&mut S, P),
}

impl<C: 'static, S: 'static, P: 'static> PipelineSet<C> for Gated<C, S, P> {
    fn stage(&self, ctx: &C, hot_reload: bool) -> RenderResult<Option<Staged<C>>> {
        let Some(state) = (self.state)(ctx) else {
            return Ok(None);
        };
        let pipelines = (self.build)(ctx, state, hot_reload)?;
        let (state_mut, install) = (self.state_mut, self.install);
        Ok(Some(Staged::new(move |ctx| {
            if let Some(state) = state_mut(ctx) {
                install(state, pipelines);
            }
        })))
    }
}

/// Rebuild every live set in `sets`, in order, from the sources `hot_reload`
/// selects, installing them only once every one has built.
///
/// # Errors
///
/// Returns the first build error, with nothing installed.
pub fn rebuild<C>(ctx: &mut C, sets: &[&dyn PipelineSet<C>], hot_reload: bool) -> RenderResult<()> {
    let staged = sets
        .iter()
        .filter_map(|set| set.stage(ctx, hot_reload).transpose())
        .collect::<RenderResult<Vec<_>>>()?;
    for set in staged {
        set.install(ctx);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::backend::error::RenderError;
    use alloc::string::ToString;
    use alloc::vec;

    // A context with two features: `a` always live, `b` live while its state
    // exists. A feature whose source is broken fails to build, and a build
    // reads the hot-reload flag into the pipeline it makes.
    struct Ctx {
        a: u32,
        b: Option<u32>,
        broken: &'static str,
    }

    fn compile(ctx: &Ctx, name: &str, hot_reload: bool) -> RenderResult<u32> {
        if ctx.broken == name {
            Err(RenderError::Other("broken source".to_string()))
        } else {
            Ok(1 + u32::from(hot_reload) * 10)
        }
    }

    const A: Pipelines<Ctx, u32> = Pipelines {
        build: |ctx, hr| Some(compile(ctx, "a", hr)),
        install: |ctx, p| ctx.a = p,
    };
    const B: Gated<Ctx, u32, u32> = Gated {
        state: |ctx| ctx.b.as_ref(),
        state_mut: |ctx| ctx.b.as_mut(),
        build: |ctx, _, hr| compile(ctx, "b", hr).map(|p| p + 100),
        install: |b, p| *b = p,
    };

    fn ctx(b: Option<u32>) -> Ctx {
        Ctx {
            a: 0,
            b,
            broken: "",
        }
    }

    #[test]
    fn a_rebuild_installs_every_live_set() {
        let mut live = ctx(Some(0));
        rebuild(&mut live, &[&A, &B], false).expect("builds");
        assert_eq!((live.a, live.b), (1, Some(101)));
    }

    #[test]
    fn a_rebuild_builds_from_the_sources_hot_reload_selects() {
        let mut live = ctx(Some(0));
        rebuild(&mut live, &[&A, &B], true).expect("builds");
        assert_eq!((live.a, live.b), (11, Some(111)));
    }

    #[test]
    fn a_set_that_is_not_live_is_neither_built_nor_installed() {
        let mut dormant = Ctx {
            broken: "b",
            ..ctx(None)
        };
        rebuild(&mut dormant, &[&A, &B], false).expect("builds");
        assert_eq!((dormant.a, dormant.b), (1, None));
    }

    #[test]
    fn one_failed_build_installs_nothing() {
        let mut broken = Ctx {
            broken: "b",
            ..ctx(Some(0))
        };
        assert!(rebuild(&mut broken, &[&A, &B], false).is_err());
        assert_eq!((broken.a, broken.b), (0, Some(0)));
    }

    #[test]
    fn staged_pipelines_wait_for_their_install() {
        let mut live = ctx(Some(0));
        let staged: Vec<_> = vec![A.stage(&live, false), B.stage(&live, false)]
            .into_iter()
            .map(|s| s.expect("builds").expect("live"))
            .collect();
        assert_eq!(live.a, 0);
        for s in staged {
            s.install(&mut live);
        }
        assert_eq!((live.a, live.b), (1, Some(101)));
    }
}
