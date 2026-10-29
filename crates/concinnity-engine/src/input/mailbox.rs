//! The sampled window input waiting for the next tick.

use crate::ecs::Resources;
use crate::input::snapshot::InputPacket;

/// The sampled window input and when it was sampled on the host clock, taken by
/// the next tick's input system. A host deposits each packet as its event pump
/// produces it; a missed take merges into the next deposit so no edge is lost.
#[derive(Debug, Default)]
pub struct InputMailbox {
    packet: Option<InputPacket>,
    sampled_us: u64,
}

impl InputMailbox {
    /// Deposit a packet sampled at `sampled_us`, merging onto an unconsumed
    /// one. The merged input is as fresh as its latest sample.
    pub fn deposit(&mut self, packet: InputPacket, sampled_us: u64) {
        match &mut self.packet {
            Some(pending) => pending.merge_from(packet),
            None => self.packet = Some(packet),
        }
        self.sampled_us = sampled_us;
    }

    /// Deposit into the mailbox in `resources`, publishing one on the first
    /// deposit.
    pub fn deposit_in(resources: &mut Resources, packet: InputPacket, sampled_us: u64) {
        if let Some(mailbox) = resources.get_mut::<Self>() {
            mailbox.deposit(packet, sampled_us);
        } else {
            let mut mailbox = Self::default();
            mailbox.deposit(packet, sampled_us);
            resources.insert(mailbox);
        }
    }

    /// Take the pending packet and its sample time, if one was deposited.
    pub fn take(&mut self) -> Option<(InputPacket, u64)> {
        self.packet.take().map(|packet| (packet, self.sampled_us))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A take empties the mailbox; deposits before it merge and keep the
    // latest sample time.
    #[test]
    fn deposits_merge_until_taken() {
        let mut mailbox = InputMailbox::default();
        assert!(mailbox.take().is_none());
        let mut first = InputPacket::default();
        first.raw.interact = true;
        mailbox.deposit(first, 10);
        mailbox.deposit(InputPacket::default(), 25);
        let (merged, at) = mailbox.take().expect("a packet");
        assert!(merged.raw.interact, "the older edge survives the merge");
        assert_eq!(at, 25);
        assert!(mailbox.take().is_none());
    }

    #[test]
    fn the_first_deposit_publishes_the_mailbox() {
        let mut resources = Resources::new();
        InputMailbox::deposit_in(&mut resources, InputPacket::default(), 3);
        InputMailbox::deposit_in(&mut resources, InputPacket::default(), 4);
        let mailbox = resources.get_mut::<InputMailbox>().expect("published");
        assert_eq!(mailbox.take().map(|(_, at)| at), Some(4));
    }
}
