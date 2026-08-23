// src/ecs/waves.rs
//
// The executable schedule, built once per world start (after system init, so
// data-dependent access declarations are final) and rebuilt only when the
// system set changes. It validates the registry's declared before/after edges
// against table order, derives conflict edges from each system's declared
// `Access` (conflicting pairs keep their table order), and groups systems
// into waves: sets whose members neither conflict nor have an ordering edge
// between them. Execution walks waves in level order, members in table order,
// so the serial walk is byte-identical to the plain table iteration.

use crate::ecs::{Access, SYSTEMS, SystemAsset};

pub(crate) struct ExecSchedule {
    // Wave membership: indices into the world's system list, grouped by
    // level, members in table order within each wave. Derived and asserted by
    // this module's tests; `World::step` still walks systems in table order,
    // so nothing reads it yet.
    waves: Vec<Vec<usize>>,
    accesses: Vec<Access>,
}

impl ExecSchedule {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "read by this module's tests until `World::step` walks waves"
        )
    )]
    pub(crate) fn waves(&self) -> &[Vec<usize>] {
        &self.waves
    }

    pub(crate) fn access(&self, system: usize) -> Access {
        self.accesses[system]
    }

    // Total systems scheduled (across all waves).
    pub(crate) fn len(&self) -> usize {
        self.accesses.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.accesses.is_empty()
    }
}

// Build the schedule for the world's gated systems, in their table order.
pub(crate) fn build(systems: &[SystemAsset]) -> ExecSchedule {
    let names: Vec<&'static str> = systems.iter().map(|s| s.name()).collect();
    let accesses: Vec<Access> = systems.iter().map(|s| s.access()).collect();
    build_from(&names, &accesses, |name| {
        SYSTEMS
            .iter()
            .find(|e| e.name == name)
            .map(|e| (e.after, e.before))
            .unwrap_or((&[], &[]))
    })
}

// The generic core, separated so tests can drive synthetic tables.
fn build_from(
    names: &[&'static str],
    accesses: &[Access],
    edges_of: impl Fn(&str) -> (&'static [&'static str], &'static [&'static str]),
) -> ExecSchedule {
    let position = |name: &str| names.iter().position(|n| *n == name);

    // Predecessor sets: declared edges first, validated against table order.
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); names.len()];
    for (i, name) in names.iter().enumerate() {
        let (after, before) = edges_of(name);
        for a in after {
            if let Some(j) = position(a) {
                assert!(
                    j < i,
                    "schedule edge violated: {a} is declared before {name} but the table runs it later",
                );
                preds[i].push(j);
            }
        }
        for b in before {
            if let Some(j) = position(b) {
                assert!(
                    i < j,
                    "schedule edge violated: {name} is declared before {b} but the table runs it later",
                );
                preds[j].push(i);
            }
        }
    }

    // Conflict edges: a conflicting pair keeps its table order, which is what
    // makes the parallel schedule's observable state identical to serial.
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            if accesses[i].conflicts_with(accesses[j]) {
                preds[j].push(i);
            }
        }
    }

    // Level assignment: every predecessor is earlier in table order (declared
    // edges validated, conflict edges forward), so one pass suffices.
    let mut level = vec![0usize; names.len()];
    for i in 0..names.len() {
        level[i] = preds[i].iter().map(|&p| level[p] + 1).max().unwrap_or(0);
    }
    let wave_count = level.iter().map(|&l| l + 1).max().unwrap_or(0);
    let mut waves: Vec<Vec<usize>> = vec![Vec::new(); wave_count];
    for (i, &l) in level.iter().enumerate() {
        waves[l].push(i);
    }

    ExecSchedule {
        waves,
        accesses: accesses.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::{ComponentId, ComponentMask};

    fn mask(ids: &[u8]) -> ComponentMask {
        let mut m = ComponentMask::EMPTY;
        for &id in ids {
            m.insert(ComponentId::new(id));
        }
        m
    }

    fn no_edges(_: &str) -> (&'static [&'static str], &'static [&'static str]) {
        (&[], &[])
    }

    #[test]
    fn exclusive_systems_are_singleton_waves_in_table_order() {
        let names = ["A", "B", "C"];
        let accesses = vec![Access::new().exclusive(); 3];
        let s = build_from(&names, &accesses, no_edges);
        let flat: Vec<usize> = s.waves().iter().flatten().copied().collect();
        assert_eq!(flat, vec![0, 1, 2]);
        assert!(s.waves().iter().all(|w| w.len() == 1));
    }

    #[test]
    fn non_conflicting_systems_share_a_wave() {
        let names = ["A", "B", "C"];
        let accesses = vec![
            Access::new().writes_components(mask(&[1])),
            Access::new().writes_components(mask(&[2])),
            Access::new().reads_components(mask(&[1])),
        ];
        let s = build_from(&names, &accesses, no_edges);
        // A and B are disjoint; C reads what A writes, so C waits for A.
        assert_eq!(s.waves(), &[vec![0, 1], vec![2]]);
    }

    #[test]
    fn declared_edges_order_non_conflicting_systems() {
        let names = ["A", "B"];
        let accesses = vec![Access::new(); 2];
        let s = build_from(&names, &accesses, |n| {
            if n == "B" {
                (&["A"][..], &[][..])
            } else {
                (&[], &[])
            }
        });
        assert_eq!(s.waves(), &[vec![0], vec![1]]);
    }

    #[test]
    #[should_panic(expected = "schedule edge violated")]
    fn edge_contradicting_table_order_panics() {
        let names = ["A", "B"];
        let accesses = vec![Access::new(); 2];
        // A declares it runs after B, but the table has A first.
        let _ = build_from(&names, &accesses, |n| {
            if n == "A" {
                (&["B"][..], &[][..])
            } else {
                (&[], &[])
            }
        });
    }

    #[test]
    fn absent_edge_targets_are_ignored() {
        // A gated-out system named in an edge simply contributes no edge.
        let names = ["B"];
        let accesses = vec![Access::new()];
        let s = build_from(&names, &accesses, |_| (&["A"][..], &["Z"][..]));
        assert_eq!(s.waves(), &[vec![0]]);
    }

    #[test]
    fn chains_stack_levels() {
        let names = ["A", "B", "C", "D"];
        let w = |id| Access::new().writes_components(mask(&[id]));
        // A->B (conflict on 1), B->C (conflict on 2), D independent.
        let accesses = vec![
            w(1),
            Access::new()
                .reads_components(mask(&[1]))
                .writes_components(mask(&[2])),
            Access::new().reads_components(mask(&[2])),
            w(9),
        ];
        let s = build_from(&names, &accesses, no_edges);
        assert_eq!(s.waves(), &[vec![0, 3], vec![1], vec![2]]);
    }

    // The real registry's declared edges must all hold under table order for
    // a fully-populated system list.
    #[test]
    fn registry_edges_respect_table_order() {
        let names: Vec<&'static str> = SYSTEMS.iter().map(|e| e.name).collect();
        let accesses = vec![Access::new().exclusive(); names.len()];
        let s = build_from(&names, &accesses, |name| {
            SYSTEMS
                .iter()
                .find(|e| e.name == name)
                .map(|e| (e.after, e.before))
                .unwrap_or((&[], &[]))
        });
        assert_eq!(s.len(), SYSTEMS.len());
    }

    // Every name in a declared edge is a real table entry: a typo would
    // silently drop the constraint.
    #[test]
    fn registry_edge_names_exist() {
        for entry in SYSTEMS {
            for name in entry.after.iter().chain(entry.before) {
                assert!(
                    SYSTEMS.iter().any(|e| e.name == *name),
                    "{} names unknown system {name}",
                    entry.name
                );
            }
        }
    }
}
