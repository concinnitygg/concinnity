// Scenario tests for the simulation: whole scenes stepped to a verdict, as
// opposed to the arithmetic checks that sit beside the code they exercise.
// In-crate rather than under `tests/` so the sim API they drive does not have
// to be public to be testable.

mod ccd;
mod character;
mod contact_events;
mod heightfield;
mod joints;
mod parallel;
mod sensor;
mod settling;
mod surface;
