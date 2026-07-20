// Per-rule firing state and the fire decision.

use super::vars::Variables;
use crate::assets::{Condition, InteractSignal, Reaction, ReactionSource, VolumeEvent};

// One declared reaction plus its runtime firing state. The component stays
// pure data; every clock and flag lives here.
#[derive(Debug)]
pub(super) struct Rule {
    pub(super) def: Reaction,
    // Content hash of `def`, pairing persisted fired flags to this rule.
    def_hash: u64,
    started: bool,
    fired_once: bool,
    cooldown_left: f32,
    timer_accum: f32,
    timer_done: bool,
    last_value: i32,
}

impl Rule {
    pub(super) fn new(def: Reaction) -> Self {
        Self {
            def_hash: super::save::def_hash(&def),
            def,
            started: false,
            fired_once: false,
            cooldown_left: 0.0,
            timer_accum: 0.0,
            timer_done: false,
            last_value: 0,
        }
    }

    pub(super) fn def_hash(&self) -> u64 {
        self.def_hash
    }

    pub(super) fn fired(&self) -> bool {
        self.fired_once
    }

    // Mark this rule as already fired (a persisted flag from a prior run).
    pub(super) fn restore_fired(&mut self) {
        self.fired_once = true;
    }

    // Re-baseline a variable source against restored values, so restoring a
    // variable does not read as a change on the first tick.
    pub(super) fn sync_variable_baseline(&mut self, vars: &Variables) {
        if let ReactionSource::Variable(name) = &self.def.on {
            self.last_value = vars.get(name);
        }
    }

    // Advance this rule's clocks by dt and decide whether it fires this tick.
    // The decision applies `once`, `cooldown`, and the conditions; a source
    // event suppressed by any of them is dropped, not queued. `crossings` and
    // `presses` are the volume boundary events and interact presses that
    // arrived since the last tick.
    pub(super) fn due(
        &mut self,
        vars: &Variables,
        dt: f32,
        crossings: &[VolumeEvent],
        presses: &[InteractSignal],
    ) -> bool {
        self.cooldown_left = (self.cooldown_left - dt).max(0.0);
        let sourced = match &self.def.on {
            ReactionSource::Start => !std::mem::replace(&mut self.started, true),
            ReactionSource::Timer { interval, repeat } => self.timer_due(*interval, *repeat, dt),
            ReactionSource::Variable(name) => {
                let current = vars.get(name);
                let changed = current != self.last_value;
                self.last_value = current;
                changed
            }
            ReactionSource::Enter(volume) => crossing_matches(crossings, *volume, true),
            ReactionSource::Exit(volume) => crossing_matches(crossings, *volume, false),
            ReactionSource::Interact(target) => {
                target.is_some_and(|target| presses.iter().any(|p| p.target == target))
            }
        };
        if !sourced
            || (self.def.once && self.fired_once)
            || self.cooldown_left > 0.0
            || !conditions_pass(vars, &self.def.conditions)
        {
            return false;
        }
        self.fired_once = true;
        self.cooldown_left = self.def.cooldown;
        true
    }

    fn timer_due(&mut self, interval: f32, repeat: bool, dt: f32) -> bool {
        if self.timer_done {
            return false;
        }
        self.timer_accum += dt;
        if self.timer_accum < interval {
            return false;
        }
        if repeat {
            // At most one firing per tick; dropping whole elapsed intervals
            // keeps a long frame from queueing a burst.
            self.timer_accum %= interval.max(f32::EPSILON);
        } else {
            self.timer_done = true;
        }
        true
    }
}

pub(super) fn conditions_pass(vars: &Variables, conditions: &[Condition]) -> bool {
    conditions
        .iter()
        .all(|c| c.op.eval(vars.get(&c.name), c.value))
}

fn crossing_matches(
    crossings: &[VolumeEvent],
    volume: Option<crate::ecs::asset_id::AssetId>,
    entered: bool,
) -> bool {
    let Some(volume) = volume else { return false };
    crossings
        .iter()
        .any(|e| e.entered == entered && e.volume == volume)
}
