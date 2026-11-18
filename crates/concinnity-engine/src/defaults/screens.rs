// Every overlay element belongs to a screen. One the world leaves on none, an
// injected HUD chip or a label added in code, joins a persistent window-space
// screen minted for them, which draws in the HUD band and takes input while no
// capturing screen is open.

use crate::components::{
    Button, Checkbox, Grid, HitRegion, Layout, List, Menu, Modal, Screen, ScreenInput, ScreenSpace,
    ScrollPanel, Sprite, Tabs, TextArea, TextField, TextLabel, Toasts, ValueSlider,
};
use crate::ecs::{PipelineContext, Ref};
use crate::error::WorldError;

use super::Minter;

pub(super) fn adopt_screenless(
    ctx: &mut PipelineContext<'_>,
    minter: &mut Minter,
) -> Result<(), WorldError> {
    // Every overlay element type, by its `screen` field.
    macro_rules! each_element {
        ($apply:ident) => {
            $apply!(
                Sprite,
                TextLabel,
                TextField,
                HitRegion,
                ScrollPanel,
                Layout,
                List,
                Grid,
                ValueSlider,
                Tabs,
                Checkbox,
                Button,
                TextArea,
                Menu,
                Modal,
                Toasts
            )
        };
    }
    macro_rules! any_screenless {
        ($($ty:ty),*) => {
            false $(|| ctx.query::<$ty>().any(|e| e.screen.is_none()))*
        };
    }
    if !each_element!(any_screenless) {
        return Ok(());
    }
    let id = minter.id()?;
    ctx.push_identified(
        id,
        Screen {
            persistent: true,
            space: ScreenSpace::Window,
            input: ScreenInput::Passthrough,
            pauses_world: false,
            ..Default::default()
        },
    );
    let hud = Some(Ref::new(id));
    macro_rules! adopt {
        ($($ty:ty),*) => {
            $(for e in ctx.query_mut::<$ty>() {
                e.screen = e.screen.or(hud);
            })*
        };
    }
    each_element!(adopt);
    Ok(())
}
