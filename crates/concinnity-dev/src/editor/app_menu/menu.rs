//! Building the NSMenu tree, and carrying choices back to the editor.
//!
//! The tree is built once. A state change pushes into the items already there
//! (`sync`), so the marks are right the moment the menu drops down. A choice
//! lands in the target's queue and is drained on the next frame rather than
//! acted on where it arrives, which is inside the nested run loop AppKit
//! tracks an open menu with.

#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::{NSObject, NSObjectProtocol, Sel};
use objc2::{DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::{
    NSApplication, NSControlStateValue, NSControlStateValueOff, NSControlStateValueOn,
    NSEventModifierFlags, NSMenu, NSMenuItem,
};
use objc2_foundation::NSString;
use std::cell::RefCell;

use super::spec::{self, Entry, ItemKind, ItemSpec, KeyEquivalent, MenuCommand, PanelMarks};

struct Chosen {
    queue: RefCell<Vec<MenuCommand>>,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "ConcinnityEditorMenuTarget"]
    #[ivars = Chosen]
    struct MenuTarget;

    unsafe impl NSObjectProtocol for MenuTarget {}

    impl MenuTarget {
        // Queued, not acted on: the editor's own frame is parked partway
        // through while AppKit tracks the open menu.
        #[unsafe(method(menuItemChosen:))]
        fn menu_item_chosen(&self, sender: &NSMenuItem) {
            if let Some(command) = MenuCommand::from_tag(sender.tag()) {
                self.ivars().queue.borrow_mut().push(command);
            }
        }
    }
);

impl MenuTarget {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(Chosen {
            queue: RefCell::new(Vec::new()),
        });
        // SAFETY: `this` is a freshly allocated instance with its ivars set,
        // and NSObject's `init` is the superclass designated initializer,
        // which consumes the allocation and returns the same instance.
        unsafe { msg_send![super(this), init] }
    }
}

// The menu bar belongs to the application, not to any one value, so it is held
// where AppKit's own is: once, on the main thread. The editor's per-frame drive
// is `Send` (it rides the run loop's hook seam) and AppKit's types are not, so
// it could not hold this even if the bar were its to own.
thread_local! {
    static BAR: RefCell<Option<AppMenu>> = const { RefCell::new(None) };
}

/// Install the session's menu bar, replacing whatever the application carries.
/// Does nothing off the main thread, which is where AppKit refuses to be used
/// at all; the session then simply runs without a menu bar.
pub(crate) fn install(state: PanelMarks) {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    BAR.with(|bar| *bar.borrow_mut() = Some(AppMenu::build(mtm, state)));
}

/// Take the choices made since the last call, in the order they were made.
pub(crate) fn take_chosen() -> Vec<MenuCommand> {
    BAR.with(|bar| {
        bar.borrow()
            .as_ref()
            .map(AppMenu::take_chosen)
            .unwrap_or_default()
    })
}

/// Bring the checkmarks in line with which panels are open. Cheap to call
/// every frame: unchanged state does nothing.
pub(crate) fn sync(state: PanelMarks) {
    BAR.with(|bar| {
        if let Some(menu) = bar.borrow_mut().as_mut() {
            menu.sync(state);
        }
    });
}

// The installed bar's live pieces.
struct AppMenu {
    // Kept alive for the session: NSMenuItem holds its target weakly.
    target: Retained<MenuTarget>,
    // The View menu's items in entry order, held so a state change can be
    // pushed into them without rebuilding the tree.
    view_items: Vec<Retained<NSMenuItem>>,
    // What those items were last set from. A frame whose state is unchanged
    // touches nothing.
    synced: PanelMarks,
}

impl AppMenu {
    fn build(mtm: MainThreadMarker, state: PanelMarks) -> Self {
        let target = MenuTarget::new(mtm);

        let bar = NSMenu::new(mtm);
        // Whatever it is titled, the first item's submenu is the application
        // menu: AppKit draws it under the bundle's name.
        let (app, _) = submenu(mtm, "", &spec::app_entries(), &target);
        bar.addItem(&app);
        let (view, view_items) = submenu(mtm, "View", &spec::view_entries(state), &target);
        bar.addItem(&view);

        NSApplication::sharedApplication(mtm).setMainMenu(Some(&bar));
        Self {
            target,
            view_items,
            synced: state,
        }
    }

    fn take_chosen(&self) -> Vec<MenuCommand> {
        std::mem::take(&mut self.target.ivars().queue.borrow_mut())
    }

    fn sync(&mut self, state: PanelMarks) {
        if self.synced == state {
            return;
        }
        self.synced = state;
        let entries = spec::view_entries(state);
        let marks = entries.iter().filter_map(Entry::item);
        for (item, spec) in self.view_items.iter().zip(marks) {
            item.setState(check_state(spec.checked));
        }
    }
}

// Build one submenu and the bar item that carries it, returning the submenu's
// items in entry order (dividers excluded) so their marks can be pushed later.
fn submenu(
    mtm: MainThreadMarker,
    title: &str,
    entries: &[Entry],
    target: &MenuTarget,
) -> (Retained<NSMenuItem>, Vec<Retained<NSMenuItem>>) {
    let menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), &NSString::from_str(title));
    // Every item has a target that answers its action, so AppKit's own
    // enabling pass has nothing to add.
    menu.setAutoenablesItems(false);

    let mut items = Vec::new();
    for entry in entries {
        match entry {
            Entry::Separator => menu.addItem(&NSMenuItem::separatorItem(mtm)),
            Entry::Item(spec) => {
                let item = build_item(mtm, spec, target);
                menu.addItem(&item);
                items.push(item);
            }
        }
    }

    let holder = NSMenuItem::new(mtm);
    holder.setTitle(&NSString::from_str(title));
    holder.setSubmenu(Some(&menu));
    (holder, items)
}

fn build_item(mtm: MainThreadMarker, spec: &ItemSpec, target: &MenuTarget) -> Retained<NSMenuItem> {
    let item = NSMenuItem::new(mtm);
    item.setTitle(&NSString::from_str(&spec.title));
    item.setState(check_state(spec.checked));

    match spec.kind {
        ItemKind::Standard(selector) => {
            // SAFETY: every selector the spec names is one the application
            // object implements, and it is set with no target, so AppKit sends
            // it down the responder chain to that object rather than to us.
            unsafe { item.setAction(Some(standard_selector(selector))) };
        }
        ItemKind::Command(command) => {
            item.setTag(command.tag());
            // SAFETY: `menuItemChosen:` is defined on MenuTarget above, taking
            // the one NSMenuItem argument AppKit sends a menu action with, and
            // the target outlives the menu: the AppMenu holding it owns both
            // for the session.
            unsafe {
                item.setAction(Some(sel!(menuItemChosen:)));
                item.setTarget(Some(target));
            }
        }
    }

    if let Some(key) = spec.key {
        item.setKeyEquivalent(&NSString::from_str(key.key));
        item.setKeyEquivalentModifierMask(modifiers(key));
    }
    item
}

// The application-menu selectors, matched by name so the spec stays free of
// AppKit types.
fn standard_selector(name: &str) -> Sel {
    match name {
        "orderFrontStandardAboutPanel:" => sel!(orderFrontStandardAboutPanel:),
        "hide:" => sel!(hide:),
        "hideOtherApplications:" => sel!(hideOtherApplications:),
        "unhideAllApplications:" => sel!(unhideAllApplications:),
        other => unreachable!("{other} is not one of the spec's standard selectors"),
    }
}

fn modifiers(key: KeyEquivalent) -> NSEventModifierFlags {
    let mut flags = NSEventModifierFlags::Command;
    if key.option {
        flags |= NSEventModifierFlags::Option;
    }
    flags
}

fn check_state(checked: bool) -> NSControlStateValue {
    if checked {
        NSControlStateValueOn
    } else {
        NSControlStateValueOff
    }
}
