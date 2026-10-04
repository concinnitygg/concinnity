use super::*;
use alloc::string::ToString;
use alloc::vec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Start(usize),
    Face(usize, usize),
    Begin(usize),
    Mip(usize, u32),
    Finish(usize),
    Abandon {
        capture: Option<usize>,
        prefilter: Option<usize>,
    },
}

// What a failing step is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    Start,
    Face(usize),
    Begin,
    Mip(u32),
}

struct Mock {
    book: ProbeBook,
    events: Vec<Event>,
    supported: bool,
    ready: bool,
    // Whether a capture may start while a convolution is in flight.
    overlaps: bool,
    faces_retire: bool,
    mips_retire: bool,
    reserve_fails: bool,
    fault: Option<Fault>,
}

impl Mock {
    fn new() -> Mock {
        Mock {
            book: ProbeBook::new(),
            events: Vec::new(),
            supported: true,
            ready: true,
            overlaps: true,
            faces_retire: true,
            mips_retire: true,
            reserve_fails: false,
            fault: None,
        }
    }

    fn fails(&self, fault: Fault) -> RenderResult<()> {
        if self.fault == Some(fault) {
            Err(RenderError::Other("injected".to_string()))
        } else {
            Ok(())
        }
    }

    // The events since the last call.
    fn drain(&mut self) -> Vec<Event> {
        core::mem::take(&mut self.events)
    }
}

impl ProbeBakeDevice for Mock {
    type Capture = usize;
    type Prefilter = usize;
    type Frame<'f> = ();

    fn book(&mut self) -> &mut ProbeBook {
        &mut self.book
    }

    fn capture_supported(&self) -> bool {
        self.supported
    }

    fn capture_ready(&self, prefilter_in_flight: bool) -> bool {
        self.ready && (self.overlaps || !prefilter_in_flight)
    }

    fn reserve_cubes(&mut self, count: usize) -> RenderResult<()> {
        if self.reserve_fails {
            Err(RenderError::Other(alloc::format!("no room for {count}")))
        } else {
            Ok(())
        }
    }

    fn start_capture(&mut self, _: &(), index: usize, _: ProbePlacement) -> RenderResult<usize> {
        self.fails(Fault::Start)?;
        self.events.push(Event::Start(index));
        Ok(index)
    }

    fn render_face(&mut self, _: &(), capture: &mut usize, face: usize) -> RenderResult<()> {
        self.fails(Fault::Face(face))?;
        self.events.push(Event::Face(*capture, face));
        Ok(())
    }

    fn capture_retired(&self, _: &usize) -> bool {
        self.faces_retire
    }

    fn begin_prefilter(&mut self, index: usize, capture: usize) -> RenderResult<usize> {
        assert_eq!(index, capture);
        self.fails(Fault::Begin)?;
        self.events.push(Event::Begin(index));
        Ok(index)
    }

    fn prefilter_mip(&mut self, prefilter: &mut usize, mip: u32) -> RenderResult<()> {
        self.fails(Fault::Mip(mip))?;
        self.events.push(Event::Mip(*prefilter, mip));
        Ok(())
    }

    fn prefilter_retired(&self, _: &usize) -> bool {
        self.mips_retire
    }

    fn finish_prefilter(&mut self, prefilter: usize) {
        self.events.push(Event::Finish(prefilter));
    }

    fn abandon(&mut self, capture: Option<usize>, prefilter: Option<usize>) {
        self.events.push(Event::Abandon { capture, prefilter });
    }
}

type Bake = ProbeBake<usize, usize>;

const MIPS: u32 = PrefilterPlan::RUNTIME.mips();

fn placements(n: usize) -> Vec<ProbePlacement> {
    (0..n)
        .map(|i| ProbePlacement::from_center_extents([i as f32, 0.0, 0.0], [1.0; 3]))
        .collect()
}

fn placed(n: usize) -> (Bake, Mock) {
    let mut bake = Bake::default();
    let mut mock = Mock::new();
    bake.place(&mut mock, placements(n)).unwrap();
    (bake, mock)
}

// Advance until the queue drains and nothing is in flight, failing a test that
// never gets there.
fn run_to_idle(bake: &mut Bake, mock: &mut Mock) -> Vec<BakeReport> {
    let mut reports = Vec::new();
    for _ in 0..1000 {
        if !mock.book.pending() && bake.capture().is_none() && bake.prefilter().is_none() {
            return reports;
        }
        reports.push(bake.advance(mock, &()));
    }
    panic!("the bake never drained");
}

fn one_probe_events(index: usize) -> Vec<Event> {
    let mut events = vec![Event::Start(index)];
    events.extend((0..CAPTURE_FACES).map(|f| Event::Face(index, f)));
    events.push(Event::Begin(index));
    events.extend((0..MIPS).map(|m| Event::Mip(index, m)));
    events.push(Event::Finish(index));
    events
}

#[test]
fn one_probe_renders_every_face_then_convolves_every_mip_then_installs() {
    let (mut bake, mut mock) = placed(1);
    let reports = run_to_idle(&mut bake, &mut mock);
    assert_eq!(mock.drain(), one_probe_events(0));
    assert_eq!(mock.book.count(), 1);
    let installs: Vec<ProbeProgress> = reports.iter().filter_map(|r| r.installed).collect();
    assert_eq!(
        installs,
        [ProbeProgress {
            installed: 1,
            placed: 1
        }]
    );
    assert!(reports.iter().all(|r| r.failed.is_none()));
}

#[test]
fn each_frame_submits_at_most_one_face_and_one_mip() {
    let (mut bake, mut mock) = placed(3);
    for _ in 0..200 {
        let _ = bake.advance(&mut mock, &());
        let events = mock.drain();
        let faces = events
            .iter()
            .filter(|e| matches!(e, Event::Face(..)))
            .count();
        let mips = events
            .iter()
            .filter(|e| matches!(e, Event::Mip(..)))
            .count();
        let starts = events
            .iter()
            .filter(|e| matches!(e, Event::Start(_)))
            .count();
        assert!(faces <= 1 && mips <= 1 && starts <= 1, "{events:?}");
    }
    assert_eq!(mock.book.count(), 3);
}

// The first frame only builds the capture; mip 0 rides the frame that hands
// the capture over.
#[test]
fn the_budget_spreads_one_probe_over_its_faces_and_mips() {
    let (mut bake, mut mock) = placed(1);
    let frames = run_to_idle(&mut bake, &mut mock).len();
    let expected = 1 + CAPTURE_FACES + MIPS as usize + 1;
    assert_eq!(frames, expected);
}

#[test]
fn the_next_capture_overlaps_the_previous_convolution() {
    let (mut bake, mut mock) = placed(2);
    run_to_idle(&mut bake, &mut mock);
    let events = mock.drain();
    let second_start = events.iter().position(|e| *e == Event::Start(1)).unwrap();
    let first_finish = events.iter().position(|e| *e == Event::Finish(0)).unwrap();
    assert!(second_start < first_finish, "{events:?}");
    assert_eq!(mock.book.count(), 2);
}

#[test]
fn a_device_that_cannot_overlap_starts_after_the_install() {
    let (mut bake, mut mock) = placed(2);
    mock.overlaps = false;
    run_to_idle(&mut bake, &mut mock);
    let events = mock.drain();
    let mut expected = one_probe_events(0);
    expected.extend(one_probe_events(1));
    assert_eq!(events, expected);
}

#[test]
fn installs_follow_placement_order() {
    let (mut bake, mut mock) = placed(4);
    run_to_idle(&mut bake, &mut mock);
    let finished: Vec<usize> = mock
        .drain()
        .into_iter()
        .filter_map(|e| match e {
            Event::Finish(i) => Some(i),
            _ => None,
        })
        .collect();
    assert_eq!(finished, [0, 1, 2, 3]);
    for (i, record) in mock.book.records().iter().enumerate() {
        assert_eq!(record.probe_pos[0], i as f32);
    }
}

#[test]
fn the_convolution_waits_for_the_faces_to_retire() {
    let (mut bake, mut mock) = placed(1);
    mock.faces_retire = false;
    for _ in 0..20 {
        let _ = bake.advance(&mut mock, &());
    }
    let events = mock.drain();
    assert_eq!(events.len(), 1 + CAPTURE_FACES, "{events:?}");
    mock.faces_retire = true;
    let _ = bake.advance(&mut mock, &());
    assert_eq!(mock.drain(), [Event::Begin(0), Event::Mip(0, 0)]);
}

#[test]
fn the_install_waits_for_the_convolution_to_retire() {
    let (mut bake, mut mock) = placed(1);
    mock.mips_retire = false;
    for _ in 0..50 {
        let _ = bake.advance(&mut mock, &());
    }
    assert_eq!(mock.book.count(), 0);
    assert!(bake.prefilter().is_some());
    mock.mips_retire = true;
    let report = bake.advance(&mut mock, &());
    assert_eq!(report.installed.map(|p| p.installed), Some(1));
    assert_eq!(mock.drain().last(), Some(&Event::Finish(0)));
}

#[test]
fn a_capture_waits_for_a_free_convolution_slot() {
    let (mut bake, mut mock) = placed(2);
    mock.mips_retire = false;
    for _ in 0..50 {
        let _ = bake.advance(&mut mock, &());
    }
    // The second probe captured every face but cannot hand over while the
    // first still holds the convolution slot.
    let begins = mock
        .drain()
        .into_iter()
        .filter(|e| matches!(e, Event::Begin(_)))
        .count();
    assert_eq!(begins, 1);
    assert!(bake.capture().is_some());
}

#[test]
fn a_scene_that_is_not_ready_keeps_the_queue() {
    let (mut bake, mut mock) = placed(2);
    mock.ready = false;
    for _ in 0..10 {
        let _ = bake.advance(&mut mock, &());
    }
    assert!(mock.drain().is_empty());
    assert!(mock.book.pending());
    mock.ready = true;
    run_to_idle(&mut bake, &mut mock);
    assert_eq!(mock.book.count(), 2);
}

#[test]
fn an_unsupported_device_abandons_the_queue_and_keeps_what_installed() {
    let (mut bake, mut mock) = placed(3);
    while mock.book.count() < 1 {
        let _ = bake.advance(&mut mock, &());
    }
    // The install frame handed probe 1 to the convolution; the next one starts
    // probe 2's capture beside it.
    let _ = bake.advance(&mut mock, &());
    mock.drain();
    mock.supported = false;
    let report = bake.advance(&mut mock, &());
    assert!(report.failed.is_none() && report.installed.is_none());
    assert_eq!(
        mock.drain(),
        [Event::Abandon {
            capture: Some(2),
            prefilter: Some(1)
        }]
    );
    assert!(!mock.book.pending());
    assert_eq!(mock.book.count(), 1);
    assert!(bake.capture().is_none() && bake.prefilter().is_none());
}

#[test]
fn an_idle_bake_asks_the_device_nothing() {
    let (mut bake, mut mock) = placed(0);
    mock.supported = false;
    let _ = bake.advance(&mut mock, &());
    assert!(mock.drain().is_empty());
}

fn fail_at(fault: Fault) -> (Bake, Mock, Vec<BakeReport>) {
    let (mut bake, mut mock) = placed(3);
    // Let the first probe install so the failure has something to keep.
    while mock.book.count() < 1 {
        let _ = bake.advance(&mut mock, &());
    }
    mock.fault = Some(fault);
    let reports = run_to_idle(&mut bake, &mut mock);
    (bake, mock, reports)
}

#[test]
fn a_failed_step_abandons_both_slots_and_the_queue() {
    for fault in [
        Fault::Start,
        Fault::Face(3),
        Fault::Begin,
        Fault::Mip(0),
        Fault::Mip(4),
    ] {
        let (bake, mock, reports) = fail_at(fault);
        let failures: Vec<&BakeFailure> =
            reports.iter().filter_map(|r| r.failed.as_ref()).collect();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].kept, mock.book.count());
        assert!(!mock.book.pending());
        assert!(bake.capture().is_none() && bake.prefilter().is_none());
        assert!(mock.book.count() >= 1);
    }
}

#[test]
fn a_failed_face_releases_the_capture_through_the_device() {
    let (mut bake, mut mock) = placed(1);
    mock.fault = Some(Fault::Face(2));
    run_to_idle(&mut bake, &mut mock);
    assert_eq!(
        mock.drain(),
        [
            Event::Start(0),
            Event::Face(0, 0),
            Event::Face(0, 1),
            Event::Abandon {
                capture: Some(0),
                prefilter: None
            },
        ]
    );
    assert_eq!(mock.book.count(), 0);
}

#[test]
fn a_failed_first_mip_releases_the_new_convolution() {
    let (mut bake, mut mock) = placed(1);
    mock.fault = Some(Fault::Mip(0));
    run_to_idle(&mut bake, &mut mock);
    assert_eq!(
        mock.drain().last(),
        Some(&Event::Abandon {
            capture: None,
            prefilter: Some(0)
        })
    );
}

#[test]
fn placing_again_abandons_the_bake_in_flight_and_requeues() {
    let (mut bake, mut mock) = placed(2);
    for _ in 0..3 {
        let _ = bake.advance(&mut mock, &());
    }
    mock.drain();
    bake.place(&mut mock, placements(1)).unwrap();
    assert_eq!(
        mock.drain(),
        [Event::Abandon {
            capture: Some(0),
            prefilter: None
        }]
    );
    run_to_idle(&mut bake, &mut mock);
    assert_eq!(mock.drain(), one_probe_events(0));
    assert_eq!(mock.book.count(), 1);
}

#[test]
fn a_cube_array_that_cannot_grow_keeps_the_sky() {
    let mut bake = Bake::default();
    let mut mock = Mock::new();
    mock.reserve_fails = true;
    assert!(bake.place(&mut mock, placements(3)).is_err());
    assert!(!mock.book.pending());
    assert_eq!(mock.book.header().count, 0);
}

#[test]
fn the_capture_ring_slot_is_never_one_the_frame_cycles_through() {
    for frames_in_flight in 1..=4 {
        let slot = capture_ring_slot(frames_in_flight);
        assert!((0..64).all(|frame| frame % frames_in_flight != slot));
        assert!(slot < frames_in_flight + 1, "a ring of frames + 1 holds it");
    }
}
