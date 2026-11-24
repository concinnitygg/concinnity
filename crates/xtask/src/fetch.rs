//! `vendor fetch`: download a pinned release and unpack it.

use concinnity_vendor::{Component, Host, Result, fail};

use crate::{status, tools, vendor_dir};

pub(crate) fn fetch(
    component: &Component,
    current: Option<Host>,
    named: bool,
    force: bool,
) -> Result {
    let name = component.name;
    let Some(host) = component.host_for(current, named)? else {
        println!("{name}: no release for this host, skipped");
        return Ok(());
    };
    if component.is_built(host) {
        if named {
            return fail(format!(
                "{name} is built here, not downloaded -- run `cargo xtask vendor build {name}`"
            ));
        }
        println!("{name}: built here, skipped -- `cargo xtask vendor build {name}` builds it");
        return Ok(());
    }
    let vendor = vendor_dir();
    let fresh = force || !component.installed(&vendor, host);
    let target = concinnity_vendor::fetch::fetch(component, host, &vendor, force)?;
    if fresh {
        tools::touch_watchers(component);
        status::print_report(component, &target, host);
    }
    Ok(())
}
