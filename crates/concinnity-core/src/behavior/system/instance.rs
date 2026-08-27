// One behavior's firing state, and the clocks that decide whether it fires.
//
// A world-scoped behavior has a single instance; a scoped one has an instance
// per matching entity, each with its own locals and its own clocks.

use alloc::vec::Vec;

use crate::behavior::Val;
use crate::components::{Behavior, BehaviorSource, InteractEvent, VolumeEvent};
use crate::ecs::{Entity, asset_id::AssetId};

#[derive(Debug)]
pub(super) struct Instance {
    pub(super) entity: Option<Entity>,
    pub(super) locals: Vec<Val>,
    pub(super) started: bool,
    pub(super) spawned_pending: bool,
    pub(super) fired_once: bool,
    pub(super) cooldown_left: f32,
    pub(super) timer_accum: f32,
    pub(super) timer_done: bool,
    pub(super) last_value: Val,
}

impl Instance {
    pub(super) fn new(entity: Option<Entity>, locals: Vec<Val>, spawned_pending: bool) -> Self {
        Self {
            entity,
            locals,
            started: false,
            spawned_pending,
            fired_once: false,
            cooldown_left: 0.0,
            timer_accum: 0.0,
            timer_done: false,
            last_value: Val::Int(0),
        }
    }

    // Advance this instance's clocks by dt and decide whether it fires.
    pub(super) fn due(
        &mut self,
        def: &Behavior,
        vars: &[Val],
        var_slot: Option<u16>,
        dt: f32,
        crossings: &[VolumeEvent],
        presses: &[InteractEvent],
    ) -> bool {
        self.cooldown_left = (self.cooldown_left - dt).max(0.0);
        let sourced = match &def.on {
            BehaviorSource::Start => !core::mem::replace(&mut self.started, true),
            BehaviorSource::Tick => true,
            BehaviorSource::Spawned => core::mem::replace(&mut self.spawned_pending, false),
            BehaviorSource::Timer { interval, repeat } => self.timer_due(*interval, *repeat, dt),
            BehaviorSource::Variable(_) => {
                let current = var_slot
                    .and_then(|s| vars.get(s as usize))
                    .copied()
                    .unwrap_or(Val::Int(0));
                let changed = current != self.last_value;
                self.last_value = current;
                changed
            }
            BehaviorSource::Enter(volume) => crossing_matches(crossings, *volume, true),
            BehaviorSource::Exit(volume) => crossing_matches(crossings, *volume, false),
            BehaviorSource::Interact(target) => {
                target.is_some_and(|target| presses.iter().any(|p| p.target == target))
            }
        };
        if !sourced || (def.once && self.fired_once) || self.cooldown_left > 0.0 {
            return false;
        }
        self.fired_once = true;
        self.cooldown_left = def.cooldown;
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
            // At most one firing per tick. Fixed ticks make the subtraction
            // exact for any interval above the tick length; a sub-tick
            // interval fires every tick without accumulating unbounded debt.
            self.timer_accum = (self.timer_accum - interval).min(interval);
        } else {
            self.timer_done = true;
        }
        true
    }
}

fn crossing_matches(crossings: &[VolumeEvent], volume: Option<AssetId>, entered: bool) -> bool {
    let Some(volume) = volume else { return false };
    crossings
        .iter()
        .any(|e| e.entered == entered && e.volume == volume)
}
