use super::*;

fn fed(feed: &mut ToastFeed, n: &Notifier, now: Instant) -> Vec<Toast> {
    let mut out = Vec::new();
    feed.feed(n, now, &mut out);
    out
}

#[test]
fn info_counts_down_and_errors_hold() {
    let n = Notifier::default();
    let mut feed = ToastFeed::default();
    let now = Instant::now();
    n.info("done");
    n.error_with("broke", Action::OpenConsole);
    let out = fed(&mut feed, &n, now);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].seconds, 4.0);
    assert_eq!(out[0].tone, ToastTone::Info);
    assert_eq!(out[1].seconds, f32::INFINITY, "errors hold until clicked");
    assert!(
        fed(&mut feed, &n, now).is_empty(),
        "nothing new, nothing sent"
    );
    fed(&mut feed, &n, now + Duration::from_secs(10));
    assert_eq!(feed.messages().collect::<Vec<_>>(), ["broke"]);
}

#[test]
fn a_repeat_refreshes_its_card_instead_of_stacking() {
    let n = Notifier::default();
    let mut feed = ToastFeed::default();
    let now = Instant::now();
    n.info("saved");
    let first = fed(&mut feed, &n, now);
    n.info("saved");
    let again = fed(&mut feed, &n, now + Duration::from_secs(3));
    assert_eq!(again[0].key, first[0].key, "the same card");
    assert_eq!(again[0].seconds, 4.0, "its time starts over");
    assert_eq!(feed.messages().count(), 1);
}

#[test]
fn a_dismissed_card_hands_back_its_action() {
    let n = Notifier::default();
    let mut feed = ToastFeed::default();
    n.error_with("cook failed", Action::SelectAsset("tree".into()));
    let key = fed(&mut feed, &n, Instant::now())[0].key.clone();
    assert_eq!(feed.dismiss(&key), Some(Action::SelectAsset("tree".into())));
    assert!(feed.is_empty());
    assert_eq!(feed.dismiss(&key), None);
}

#[test]
fn ops_report_progress_and_fade_on_finish_or_drop() {
    let n = Notifier::default();
    let mut feed = ToastFeed::default();
    let now = Instant::now();
    let op = n.begin_op("Cooking");
    let out = fed(&mut feed, &n, now);
    assert_eq!(out[0].progress, Some(ToastProgress::Ongoing));
    assert_eq!(out[0].seconds, f32::INFINITY);
    assert!(fed(&mut feed, &n, now).is_empty(), "unchanged progress");
    op.set(1, 4);
    let out = fed(&mut feed, &n, now);
    assert_eq!(out[0].progress, Some(ToastProgress::Fraction(0.25)));
    op.finish();
    let out = fed(&mut feed, &n, now);
    assert_eq!(out[0].progress, None);
    assert_eq!(out[0].seconds, OP_FADE_S, "a finished op fades away");
    assert!(feed.is_empty());

    let dropped = n.begin_op("Exporting");
    fed(&mut feed, &n, now);
    drop(dropped);
    assert_eq!(fed(&mut feed, &n, now).len(), 1, "a dropped handle retires");
    assert!(feed.is_empty());
}

#[test]
fn a_replay_shows_every_live_card_again() {
    let n = Notifier::default();
    let mut feed = ToastFeed::default();
    let now = Instant::now();
    n.success("saved");
    let _op = n.begin_op("Cooking");
    fed(&mut feed, &n, now);
    let mut out = Vec::new();
    feed.replay(now + Duration::from_secs(1), &mut out);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].seconds, 3.0, "the time it has left");
    out.clear();
    feed.replay(now + Duration::from_secs(4), &mut out);
    assert_eq!(out.len(), 1, "a note out of time is not shown again");
}

#[test]
fn notifier_is_shared_across_clones() {
    let n = Notifier::default();
    let worker = n.clone();
    std::thread::spawn(move || worker.success("from a worker"))
        .join()
        .unwrap();
    let mut feed = ToastFeed::default();
    assert_eq!(fed(&mut feed, &n, Instant::now()).len(), 1);
}
