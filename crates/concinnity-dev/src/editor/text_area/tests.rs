//! The text area's editing model, driven the way a frame drives it: key events
//! through `handle_events`, pointer presses through `press`.

use concinnity_core::components::{InputKey, KeyEvent, KeyMods, KeyPress};
use concinnity_core::window::clipboard::Clipboard;

use super::clipboard::InternalClipboard;
use super::editing::Response;
use super::keys::Platform;
use super::{Pos, Scroll, TextArea, ViewSize};

const OTHER: Platform = Platform::Other;

fn area(text: &str) -> TextArea {
    let mut a = TextArea::from_text(text);
    a.set_view(ViewSize { rows: 10, cols: 40 });
    a
}

fn key(k: InputKey, mods: KeyMods) -> KeyEvent {
    KeyEvent::Press(KeyPress::new(k, mods))
}

fn press(a: &mut TextArea, k: InputKey, mods: KeyMods) -> Response {
    press_on(a, k, mods, OTHER)
}

fn press_on(a: &mut TextArea, k: InputKey, mods: KeyMods, p: Platform) -> Response {
    a.handle_events(&[key(k, mods)], p, &mut InternalClipboard::default())
}

fn tap(a: &mut TextArea, k: InputKey) {
    press(a, k, KeyMods::NONE);
}

fn type_str(a: &mut TextArea, s: &str) {
    let events: Vec<KeyEvent> = s.chars().map(KeyEvent::Text).collect();
    a.handle_events(&events, OTHER, &mut InternalClipboard::default());
}

fn at(line: usize, col: usize) -> Pos {
    Pos::new(line, col)
}

fn select(a: &mut TextArea, from: Pos, to: Pos) {
    a.press(from, false, 0.0);
    a.drag_to(to);
    a.release();
}

// Typing

#[test]
fn a_frame_of_events_types_every_character_in_order() {
    let mut a = area("");
    a.handle_events(
        &[
            key(InputKey::A, KeyMods::NONE),
            KeyEvent::Text('a'),
            KeyEvent::Text('b'),
            key(InputKey::Backspace, KeyMods::NONE),
            KeyEvent::Text('c'),
        ],
        OTHER,
        &mut InternalClipboard::default(),
    );
    assert_eq!(a.text(), "ac");
    assert_eq!(a.caret(), at(0, 2));
}

#[test]
fn typing_replaces_the_selection() {
    let mut a = area("hello world");
    select(&mut a, at(0, 0), at(0, 5));
    type_str(&mut a, "bye");
    assert_eq!(a.text(), "bye world");
    assert_eq!(a.selection(), None);
}

#[test]
fn typing_is_char_indexed() {
    let mut a = area("é");
    a.go_to(0, 1);
    type_str(&mut a, "ü");
    assert_eq!(a.text(), "éü");
    assert_eq!(a.caret(), at(0, 2));
}

// Deletion

#[test]
fn backspace_and_delete_remove_one_character() {
    let mut a = area("abc");
    a.go_to(0, 2);
    tap(&mut a, InputKey::Backspace);
    assert_eq!((a.text().as_str(), a.caret()), ("ac", at(0, 1)));
    tap(&mut a, InputKey::Delete);
    assert_eq!((a.text().as_str(), a.caret()), ("a", at(0, 1)));
}

#[test]
fn backspace_at_a_line_start_joins_the_lines() {
    let mut a = area("ab\ncd");
    a.go_to(1, 0);
    tap(&mut a, InputKey::Backspace);
    assert_eq!(a.text(), "abcd");
    assert_eq!(a.caret(), at(0, 2));
    tap(&mut a, InputKey::Delete);
    assert_eq!(a.text(), "abd");
}

#[test]
fn deletion_stops_at_the_text_edges() {
    let mut a = area("x");
    tap(&mut a, InputKey::Backspace);
    a.go_to(0, 1);
    tap(&mut a, InputKey::Delete);
    assert_eq!(a.text(), "x");
    assert!(!a.is_dirty(), "a no-op edit records nothing");
}

#[test]
fn word_deletion_uses_option_on_mac_and_ctrl_elsewhere() {
    let mut a = area("let foo_bar = 1;");
    a.go_to(0, 11);
    press_on(&mut a, InputKey::Backspace, KeyMods::ALT, Platform::Mac);
    assert_eq!(a.text(), "let  = 1;");
    a.go_to(0, 0);
    press_on(&mut a, InputKey::Delete, KeyMods::CTRL, Platform::Other);
    assert_eq!(a.text(), "  = 1;");
}

#[test]
fn deleting_a_selection_removes_exactly_it() {
    let mut a = area("one\ntwo\nthree");
    select(&mut a, at(0, 2), at(2, 1));
    tap(&mut a, InputKey::Delete);
    assert_eq!(a.text(), "onhree");
    assert_eq!(a.caret(), at(0, 2));
}

// Line breaks and indentation

#[test]
fn enter_keeps_the_leading_whitespace() {
    let mut a = area("    let x;");
    a.go_to(0, 10);
    tap(&mut a, InputKey::Enter);
    assert_eq!(a.text(), "    let x;\n    ");
    assert_eq!(a.caret(), at(1, 4));
}

#[test]
fn enter_inside_the_indentation_carries_only_what_precedes_the_caret() {
    let mut a = area("\t\tx");
    a.go_to(0, 1);
    tap(&mut a, InputKey::Enter);
    assert_eq!(a.text(), "\t\n\t\tx");
}

#[test]
fn tab_inserts_four_spaces_and_replaces_a_one_line_selection() {
    let mut a = area("ab");
    a.go_to(0, 1);
    tap(&mut a, InputKey::Tab);
    assert_eq!(a.text(), "a    b");
    select(&mut a, at(0, 0), at(0, 1));
    tap(&mut a, InputKey::Tab);
    assert_eq!(a.text(), "        b");
}

#[test]
fn tab_indents_every_line_a_selection_covers() {
    let mut a = area("a\nb\nc\nd");
    select(&mut a, at(0, 1), at(2, 0));
    tap(&mut a, InputKey::Tab);
    assert_eq!(
        a.text(),
        "    a\n    b\nc\nd",
        "the line the selection ends at column 0 of stays"
    );
    assert_eq!(a.selection(), Some((at(0, 5), at(2, 0))));
    a.undo();
    assert_eq!(
        a.text(),
        "a\nb\nc\nd",
        "one undo takes the whole indent back"
    );
}

#[test]
fn shift_tab_outdents_spaces_and_tabs() {
    let mut a = area("      six\n\ttab\n  two\nnone");
    select(&mut a, at(0, 8), at(3, 2));
    press(&mut a, InputKey::Tab, KeyMods::SHIFT);
    assert_eq!(a.text(), "  six\ntab\ntwo\nnone");
    assert_eq!(a.selection(), Some((at(0, 4), at(3, 2))));
}

#[test]
fn shift_tab_without_a_selection_outdents_the_caret_line() {
    let mut a = area("        x\ny");
    a.go_to(0, 9);
    press(&mut a, InputKey::Tab, KeyMods::SHIFT);
    assert_eq!(a.text(), "    x\ny");
    assert_eq!(a.caret(), at(0, 5));
}

// Caret motion

#[test]
fn vertical_moves_remember_the_goal_column() {
    let mut a = area("long line here\nab\nanother long line");
    a.go_to(0, 10);
    tap(&mut a, InputKey::Down);
    assert_eq!(a.caret(), at(1, 2), "clamped to the short line");
    tap(&mut a, InputKey::Down);
    assert_eq!(a.caret(), at(2, 10), "back at the goal column");
    tap(&mut a, InputKey::Left);
    tap(&mut a, InputKey::Up);
    assert_eq!(a.caret(), at(1, 2));
}

#[test]
fn vertical_moves_past_the_ends_reach_the_ends() {
    let mut a = area("abc\ndef");
    a.go_to(0, 2);
    tap(&mut a, InputKey::Up);
    assert_eq!(a.caret(), at(0, 0));
    a.go_to(1, 1);
    tap(&mut a, InputKey::Down);
    assert_eq!(a.caret(), at(1, 3));
}

#[test]
fn vertical_moves_keep_the_visual_column_across_tabs() {
    let mut a = area("\tx\nabcdef");
    a.go_to(0, 2);
    tap(&mut a, InputKey::Down);
    assert_eq!(a.caret(), at(1, 5), "cell 5 on both lines");
}

#[test]
fn home_is_smart_and_end_reaches_the_line_end() {
    let mut a = area("    code;");
    a.go_to(0, 7);
    tap(&mut a, InputKey::Home);
    assert_eq!(a.caret(), at(0, 4));
    tap(&mut a, InputKey::Home);
    assert_eq!(a.caret(), at(0, 0));
    tap(&mut a, InputKey::End);
    assert_eq!(a.caret(), at(0, 9));
}

#[test]
fn shortcut_home_and_end_reach_the_document_ends() {
    let mut a = area("a\nb\nlast");
    a.go_to(1, 1);
    press(&mut a, InputKey::End, KeyMods::CTRL);
    assert_eq!(a.caret(), at(2, 4));
    press_on(&mut a, InputKey::Up, KeyMods::CMD, Platform::Mac);
    assert_eq!(a.caret(), at(0, 0));
}

#[test]
fn page_keys_move_by_the_window() {
    let text: Vec<String> = (0..50).map(|i| format!("line {i}")).collect();
    let mut a = area(&text.join("\n"));
    tap(&mut a, InputKey::PageDown);
    assert_eq!(a.caret().line, 9, "a page is the window less one line");
    assert!(a.scroll().top > 0, "the window follows");
    tap(&mut a, InputKey::PageUp);
    assert_eq!(a.caret().line, 0);
}

#[test]
fn shift_extends_and_a_plain_arrow_collapses() {
    let mut a = area("abcdef");
    a.go_to(0, 2);
    press(&mut a, InputKey::Right, KeyMods::SHIFT);
    press(&mut a, InputKey::Right, KeyMods::SHIFT);
    assert_eq!(a.selection(), Some((at(0, 2), at(0, 4))));
    tap(&mut a, InputKey::Left);
    assert_eq!(
        (a.selection(), a.caret()),
        (None, at(0, 2)),
        "Left lands at the start"
    );
    press(&mut a, InputKey::End, KeyMods::SHIFT);
    tap(&mut a, InputKey::Right);
    assert_eq!(a.caret(), at(0, 6), "Right lands at the end");
}

#[test]
fn word_moves_follow_the_platform_modifier() {
    let mut a = area("alpha beta");
    press_on(&mut a, InputKey::Right, KeyMods::ALT, Platform::Mac);
    assert_eq!(a.caret(), at(0, 5));
    press_on(&mut a, InputKey::Right, KeyMods::CTRL, Platform::Other);
    assert_eq!(a.caret(), at(0, 10));
}

// Clipboard

#[test]
fn select_all_copy_and_paste_round_trip() {
    let mut a = area("one\ntwo");
    let mut clip = InternalClipboard::default();
    let chord = |k| key(k, KeyMods::CTRL);
    a.handle_events(&[chord(InputKey::A), chord(InputKey::C)], OTHER, &mut clip);
    assert_eq!(clip.text().as_deref(), Some("one\ntwo"));
    a.handle_events(
        &[key(InputKey::End, KeyMods::CTRL), chord(InputKey::V)],
        OTHER,
        &mut clip,
    );
    assert_eq!(a.text(), "one\ntwoone\ntwo");
}

#[test]
fn cut_removes_and_paste_normalizes_line_endings() {
    let mut a = area("keep cut keep");
    let mut clip = InternalClipboard::default();
    select(&mut a, at(0, 5), at(0, 9));
    a.handle_events(&[key(InputKey::X, KeyMods::CTRL)], OTHER, &mut clip);
    assert_eq!(a.text(), "keep keep");
    assert_eq!(clip.text().as_deref(), Some("cut "));
    clip.set_text("a\r\nb\rc");
    a.handle_events(&[key(InputKey::V, KeyMods::CMD)], Platform::Mac, &mut clip);
    assert_eq!(a.text(), "keep a\nb\nckeep");
    assert_eq!(a.caret(), at(2, 1));
}

#[test]
fn copy_without_a_selection_leaves_the_clipboard_alone() {
    let mut a = area("text");
    let mut clip = InternalClipboard::default();
    clip.set_text("before");
    a.handle_events(&[key(InputKey::C, KeyMods::CTRL)], OTHER, &mut clip);
    assert_eq!(clip.text().as_deref(), Some("before"));
}

// Undo, redo, and the dirty flag

#[test]
fn undo_takes_typing_back_a_word_at_a_time() {
    let mut a = area("");
    type_str(&mut a, "hello world");
    press(&mut a, InputKey::Z, KeyMods::CTRL);
    assert_eq!(a.text(), "hello");
    press(&mut a, InputKey::Z, KeyMods::CTRL);
    assert_eq!(a.text(), "");
    press(&mut a, InputKey::Y, KeyMods::CTRL);
    assert_eq!(a.text(), "hello");
    press_on(
        &mut a,
        InputKey::Z,
        KeyMods::CMD.with_shift(),
        Platform::Mac,
    );
    assert_eq!(a.text(), "hello world");
}

#[test]
fn undo_restores_the_selection_it_replaced() {
    let mut a = area("abc");
    select(&mut a, at(0, 1), at(0, 3));
    type_str(&mut a, "X");
    a.undo();
    assert_eq!(a.text(), "abc");
    assert_eq!(a.selection(), Some((at(0, 1), at(0, 3))));
    a.redo();
    assert_eq!((a.text().as_str(), a.caret()), ("aX", at(0, 2)));
}

#[test]
fn a_caret_move_splits_the_undo_step() {
    let mut a = area("");
    type_str(&mut a, "ab");
    tap(&mut a, InputKey::Left);
    type_str(&mut a, "c");
    a.undo();
    assert_eq!(a.text(), "ab");
}

#[test]
fn undo_of_a_multi_line_paste_and_a_join() {
    let mut a = area("x\ny");
    a.go_to(1, 0);
    tap(&mut a, InputKey::Backspace);
    let mut clip = InternalClipboard::default();
    clip.set_text("1\n2\n");
    a.handle_events(&[key(InputKey::V, KeyMods::CTRL)], OTHER, &mut clip);
    assert_eq!(a.text(), "x1\n2\ny");
    a.undo();
    assert_eq!(a.text(), "xy");
    a.undo();
    assert_eq!(a.text(), "x\ny");
    assert_eq!(a.caret(), at(1, 0));
}

#[test]
fn dirty_tracks_the_saved_text_across_undo() {
    let mut a = area("base");
    assert!(!a.is_dirty());
    a.go_to(0, 4);
    type_str(&mut a, "!");
    assert!(a.is_dirty());
    a.undo();
    assert!(!a.is_dirty(), "undo back to the loaded text is clean");
    a.redo();
    a.mark_saved();
    assert!(!a.is_dirty());
    type_str(&mut a, "?");
    assert!(
        a.is_dirty(),
        "typing after a save does not join the saved step"
    );
    a.undo();
    assert!(!a.is_dirty());
    assert_eq!(a.text(), "base!");
}

#[test]
fn the_save_shortcut_is_reported_not_applied() {
    let mut a = area("x");
    let r = press(&mut a, InputKey::S, KeyMods::CTRL);
    assert!(r.save);
    assert_eq!(a.text(), "x");
    assert!(!press(&mut a, InputKey::Left, KeyMods::NONE).save);
}

#[test]
fn crlf_text_saves_back_as_crlf() {
    let mut a = area("a\r\nb");
    a.go_to(1, 1);
    tap(&mut a, InputKey::Enter);
    type_str(&mut a, "c");
    assert_eq!(a.text(), "a\r\nb\r\nc");
}

// Scrolling and navigation

#[test]
fn moving_the_caret_keeps_it_in_view() {
    let text: Vec<String> = (0..30).map(|i| i.to_string()).collect();
    let mut a = area(&text.join("\n"));
    for _ in 0..15 {
        tap(&mut a, InputKey::Down);
    }
    assert_eq!(a.scroll().top, 6, "the caret line sits on the last row");
    press(&mut a, InputKey::Home, KeyMods::CTRL);
    assert_eq!(a.scroll().top, 0);
}

#[test]
fn a_long_line_scrolls_across() {
    let mut a = area(&"x".repeat(100));
    tap(&mut a, InputKey::End);
    let s = a.scroll();
    assert!(s.left > 0 && a.caret_cell() < s.left + 40);
    tap(&mut a, InputKey::Home);
    assert_eq!(a.scroll().left, 0);
}

#[test]
fn go_to_centers_a_distant_line_and_clamps() {
    let text: Vec<String> = (0..100).map(|i| i.to_string()).collect();
    let mut a = area(&text.join("\n"));
    a.go_to(60, 1);
    assert_eq!(a.caret(), at(60, 1));
    assert_eq!(a.scroll().top, 55);
    a.go_to(58, 0);
    assert_eq!(a.scroll().top, 55, "a visible line does not recenter");
    a.go_to(500, 500);
    assert_eq!(a.caret(), at(99, 2));
    assert_eq!(a.scroll().top, 90, "the window stays over the text");
}

#[test]
fn the_wheel_scrolls_by_lines_and_carries_fractions() {
    let text: Vec<String> = (0..100).map(|i| i.to_string()).collect();
    let mut a = area(&text.join("\n"));
    a.wheel(20.0, false);
    assert_eq!(a.scroll().top, 3, "a notch is three lines");
    a.wheel(3.0, false);
    a.wheel(3.0, false);
    assert_eq!(a.scroll().top, 3);
    a.wheel(1.0, false);
    assert_eq!(a.scroll().top, 4, "the carried fraction completes a line");
    a.wheel(-1000.0, false);
    assert_eq!(a.scroll().top, 0);
    a.wheel(1000.0, false);
    assert_eq!(a.scroll().top, 90, "clamped to the last page");
}

#[test]
fn set_scroll_clamps_to_the_text() {
    let mut a = area("short");
    a.set_scroll(Scroll { top: 9, left: 9 });
    assert_eq!(a.scroll(), Scroll::default());
}

// Pointer

#[test]
fn clicks_place_select_words_and_select_lines() {
    let mut a = area("foo bar\nnext");
    a.press(at(0, 5), false, 1.0);
    assert_eq!((a.caret(), a.selection()), (at(0, 5), None));
    a.press(at(0, 5), false, 1.2);
    assert_eq!(
        a.selection(),
        Some((at(0, 4), at(0, 7))),
        "double-click takes the word"
    );
    a.press(at(0, 5), false, 1.4);
    assert_eq!(
        a.selection(),
        Some((at(0, 0), at(1, 0))),
        "triple-click takes the line"
    );
    a.press(at(0, 5), false, 3.0);
    assert_eq!(a.selection(), None, "a slow press starts over");
}

#[test]
fn shift_press_and_drag_extend_the_selection() {
    let mut a = area("abcdef\nghij");
    a.press(at(0, 1), false, 0.0);
    a.drag_to(at(1, 2));
    assert_eq!(a.selection(), Some((at(0, 1), at(1, 2))));
    a.release();
    a.drag_to(at(0, 0));
    assert_eq!(a.caret(), at(1, 2), "a released button no longer drags");
    a.press(at(0, 4), true, 5.0);
    assert_eq!(a.selection(), Some((at(0, 1), at(0, 4))));
}
