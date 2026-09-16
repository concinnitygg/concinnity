use super::ActiveSceneFlow;

#[test]
fn the_first_read_anchors_the_clock_at_zero() {
    let mut flow = ActiveSceneFlow::new(None);
    assert_eq!(flow.elapsed(100.0), 0.0);
    assert_eq!(flow.elapsed(101.5), 1.5);
}

#[test]
fn the_clock_never_reads_negative() {
    let mut flow = ActiveSceneFlow::new(None);
    flow.elapsed(10.0);
    assert_eq!(flow.elapsed(4.0), 0.0);
}
