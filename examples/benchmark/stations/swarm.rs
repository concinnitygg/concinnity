//! A drift of tokens that wakes when the camera arrives: the station that loads
//! the behavior tick.
//!
//! Three declared rules run against every token once per simulated tick, and
//! each one searches the world for what is near it. The search is over every
//! placement in the corridor, so the work is the token count times the world's,
//! and it lands entirely on the CPU. While the camera is elsewhere the rules
//! read the clock and stop.

use concinnity::components::{
    Behavior, BehaviorExpr, BehaviorLiteral, BehaviorLocal, BehaviorNode, BehaviorQuery,
    BehaviorSource, ProceduralMesh, Prop,
};
use concinnity::cook::WorldBuilder;

use crate::palette;
use crate::stations::spread;

/// The stretch of path this station is measured under.
pub(crate) const SEGMENT: &str = "swarm";

// The drift: how many tokens along each axis, and how far apart.
const DRIFT: [usize; 3] = [12, 6, 10];
const DRIFT_SPACING: f32 = 1.5;
const DRIFT_BASE_HEIGHT: f32 = 2.2;
const TOKEN_RADIUS: f32 = 0.30;

// How far a token looks for company, how close counts as touching, how many
// neighbours count as a crowd, and how fast a rule moves it.
const NEIGHBOUR_RADIUS: f32 = 4.5;
const TOUCHING_RADIUS: f32 = 1.1;
const CROWDED: i32 = 6;
const DRIFT_SPEED: f32 = 2.4;

// How far the tokens are drawn from. Short, so a drift big enough to load the
// tick does not also draw into every other segment.
const CULL_DISTANCE: f32 = 55.0;

// The query every rule searches: every placement in the corridor, not just the
// drift. A token gives way to the world around it, and the search is what the
// station costs.
const NEARBY: &str = "nearby";

/// Declare the tokens and the rules that move them.
pub(crate) fn declare(world: &mut WorldBuilder, centre: [f32; 3]) {
    world.add(
        "swarm_plinth_mesh",
        ProceduralMesh {
            generator: "cylinder".to_string(),
            radius: Some(6.0),
            height: Some(1.2),
            segments: Some(32),
            ..Default::default()
        },
    );
    world
        .add(
            "swarm_plinth",
            Prop {
                position: [centre[0], 0.6, centre[2]],
                ..Default::default()
            },
        )
        .reference("mesh", "swarm_plinth_mesh")
        .reference("material", palette::STONE);

    world.add(
        "swarm_token_mesh",
        ProceduralMesh {
            generator: "sphere".to_string(),
            radius: Some(TOKEN_RADIUS),
            rings: Some(10),
            segments: Some(14),
            ..Default::default()
        },
    );
    for x in 0..DRIFT[0] {
        for y in 0..DRIFT[1] {
            for z in 0..DRIFT[2] {
                let index = (x * DRIFT[1] + y) * DRIFT[2] + z;
                world
                    .add(
                        format!("swarm_token_{index}"),
                        Prop {
                            position: [
                                centre[0] + spread(x, DRIFT[0], DRIFT_SPACING),
                                DRIFT_BASE_HEIGHT + y as f32 * DRIFT_SPACING,
                                centre[2] + spread(z, DRIFT[2], DRIFT_SPACING),
                            ],
                            // A token is something the player could reach for,
                            // and the tag is also what marks a prop the engine
                            // has to re-upload every frame. The rules below
                            // scope on it, so nothing else in the corridor
                            // drifts.
                            interactable: true,
                            cull_distance: CULL_DISTANCE,
                            ..Default::default()
                        },
                    )
                    .reference("mesh", "swarm_token_mesh")
                    .reference("material", palette::GLOW);
            }
        }
    }

    // Three rules, each searching the world for itself. Authored apart because
    // they are three decisions, and because a rule that decides to stand still
    // has already paid for its search.
    world.add("swarm_spread_out", rule(SEPARATE));
    world.add("swarm_close_up", rule(GATHER));
    world.add("swarm_give_way", rule(GIVE_WAY));
}

// One rule: how far it looks, the count that makes it act, and the step it
// takes when it does.
#[derive(Clone, Copy)]
struct Rule {
    // How far out the crowd is counted.
    radius: f32,
    // The count either side of which the rule decides.
    threshold: i32,
    // Whether it acts above that count or at or below it.
    above: bool,
    // Whether the step closes on the nearest thing or backs away from it.
    closes: bool,
    // How fast the step is, in units per second.
    pace: f32,
}

// Back off when the crowd is thick.
const SEPARATE: Rule = Rule {
    radius: NEIGHBOUR_RADIUS,
    threshold: CROWDED,
    above: true,
    closes: false,
    pace: DRIFT_SPEED,
};

// Close up when it is not.
const GATHER: Rule = Rule {
    radius: NEIGHBOUR_RADIUS,
    threshold: CROWDED,
    above: false,
    closes: true,
    pace: DRIFT_SPEED,
};

// Give way to anything actually touching, whatever the wider crowd is doing.
const GIVE_WAY: Rule = Rule {
    radius: TOUCHING_RADIUS,
    threshold: 0,
    above: true,
    closes: false,
    pace: DRIFT_SPEED * 0.5,
};

// One rule as a declared behavior: find what is nearest, count the crowd, and
// step if the count says to.
fn rule(kind: Rule) -> Behavior {
    Behavior {
        on: BehaviorSource::Tick,
        scope: vec!["Interactable".to_string()],
        locals: vec![BehaviorLocal {
            name: "speed".to_string(),
            value: BehaviorLiteral::Float(kind.pace),
        }],
        queries: vec![BehaviorQuery {
            name: NEARBY.to_string(),
            has: vec!["Prop".to_string()],
        }],
        body: vec![BehaviorNode::If {
            cond: awake(),
            then: vec![
                BehaviorNode::Let {
                    name: "closest".to_string(),
                    value: BehaviorExpr::Nearest {
                        query: NEARBY.to_string(),
                        of: Box::new(here()),
                    },
                },
                BehaviorNode::Let {
                    name: "crowd".to_string(),
                    value: BehaviorExpr::CountWithin {
                        query: NEARBY.to_string(),
                        of: Box::new(here()),
                        radius: Box::new(BehaviorExpr::Float(kind.radius)),
                    },
                },
                BehaviorNode::If {
                    cond: crowd_test(kind),
                    then: vec![step(kind.closes)],
                    otherwise: Vec::new(),
                },
            ],
            otherwise: Vec::new(),
        }],
        ..Default::default()
    }
}

// True while the camera is on the leg that ends at this station.
//
// The drift asks the clock rather than the camera because the camera carries no
// position a behavior can read. The two agree exactly: the track advances on
// the same fixed simulation clock this reads, so the drift wakes on the frame
// the segment opens on every machine.
fn awake() -> BehaviorExpr {
    let (from, until) = crate::track::segment_window(SEGMENT);
    BehaviorExpr::All(vec![
        BehaviorExpr::Gt(
            Box::new(BehaviorExpr::Elapsed),
            Box::new(BehaviorExpr::Float(from)),
        ),
        BehaviorExpr::Lt(
            Box::new(BehaviorExpr::Elapsed),
            Box::new(BehaviorExpr::Float(until)),
        ),
    ])
}

fn crowd_test(kind: Rule) -> BehaviorExpr {
    let crowd = Box::new(BehaviorExpr::Bind("crowd".to_string()));
    let threshold = Box::new(BehaviorExpr::Int(kind.threshold));
    if kind.above {
        BehaviorExpr::Gt(crowd, threshold)
    } else {
        BehaviorExpr::Le(crowd, threshold)
    }
}

fn here() -> BehaviorExpr {
    BehaviorExpr::Position(Box::new(BehaviorExpr::SelfEntity))
}

fn there() -> BehaviorExpr {
    BehaviorExpr::Position(Box::new(BehaviorExpr::Bind("closest".to_string())))
}

// A step of `speed * dt` along the line joining this token and the nearest
// thing to it: toward it when `closer`, away from it otherwise.
fn step(closer: bool) -> BehaviorNode {
    let line = if closer {
        BehaviorExpr::Sub(Box::new(there()), Box::new(here()))
    } else {
        BehaviorExpr::Sub(Box::new(here()), Box::new(there()))
    };
    BehaviorNode::SetTransform {
        entity: BehaviorExpr::SelfEntity,
        position: Some(BehaviorExpr::Add(
            Box::new(here()),
            Box::new(BehaviorExpr::Mul(
                Box::new(BehaviorExpr::Normalize(Box::new(line))),
                Box::new(BehaviorExpr::Mul(
                    Box::new(BehaviorExpr::Local("speed".to_string())),
                    Box::new(BehaviorExpr::Dt),
                )),
            )),
        )),
        rotation_deg: None,
        scale: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stations::STATIONS;

    // How many tokens the drift holds.
    const fn token_count() -> usize {
        DRIFT[0] * DRIFT[1] * DRIFT[2]
    }

    #[test]
    fn the_drift_holds_what_its_dimensions_say() {
        assert_eq!(token_count(), 720);
    }

    // The drift has to be awake for exactly the frames the report attributes to
    // it, or its cost lands in a segment it does not belong to.
    #[test]
    fn the_drift_is_awake_for_its_own_segment_and_no_other() {
        let (from, until) = crate::track::segment_window(SEGMENT);
        assert!(until > from, "the segment has no frames in it");
        let names: Vec<&str> = STATIONS.iter().map(|s| s.segment).collect();
        assert!(names.contains(&SEGMENT), "no station reports as {SEGMENT}");
    }

    // Crowded and lonely have to be reachable on both sides, or every rule
    // takes the same branch and the drift runs off in one direction.
    #[test]
    fn a_token_can_be_both_crowded_and_alone() {
        let in_range = (NEIGHBOUR_RADIUS / DRIFT_SPACING).floor() as i32;
        assert!(in_range > 0, "no neighbour is ever in range");
        assert!(CROWDED > 0, "every token starts crowded");
    }

    // The two wide rules have to partition the count between them, or a token
    // either stands still or is pushed both ways at once.
    #[test]
    fn the_wide_rules_cover_every_count_exactly_once() {
        assert_eq!(SEPARATE.threshold, GATHER.threshold);
        assert_eq!(SEPARATE.radius, GATHER.radius);
        assert!(SEPARATE.above != GATHER.above);
        assert!(SEPARATE.closes != GATHER.closes);
    }

    // The close rule has to answer a question the wide ones do not, or it is a
    // third search for a decision already taken.
    #[test]
    fn the_close_rule_looks_closer_than_the_wide_ones() {
        assert!(GIVE_WAY.radius < SEPARATE.radius);
        assert!(GIVE_WAY.threshold < SEPARATE.threshold);
    }
}
