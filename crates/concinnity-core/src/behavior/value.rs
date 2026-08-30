// The values a behavior body computes over, and the two operator vocabularies
// its arithmetic and comparisons are expressed in.

use crate::components::BehaviorLiteral;
use crate::ecs::{Entity, TraceVal};

/// A value flowing through a behavior body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Val {
    /// A boolean.
    Bool(bool),
    /// A signed integer.
    Int(i32),
    /// A float.
    Float(f32),
    /// A 3-component vector.
    Vec3([f32; 3]),
    /// An entity handle.
    Entity(Entity),
}

impl Val {
    pub(crate) fn as_bool(self) -> Option<bool> {
        match self {
            Val::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub(crate) fn as_entity(self) -> Option<Entity> {
        match self {
            Val::Entity(e) => Some(e),
            _ => None,
        }
    }

    pub(crate) fn as_vec3(self) -> Option<[f32; 3]> {
        match self {
            Val::Vec3(v) => Some(v),
            _ => None,
        }
    }

    /// The scalar reading of a numeric value, for mixed vector/scalar forms.
    pub fn as_f32(self) -> Option<f32> {
        match self {
            Val::Int(i) => Some(i as f32),
            Val::Float(f) => Some(f),
            _ => None,
        }
    }

    /// The runtime value an authored literal denotes.
    pub fn from_literal(lit: &BehaviorLiteral) -> Val {
        match *lit {
            BehaviorLiteral::Bool(b) => Val::Bool(b),
            BehaviorLiteral::Int(i) => Val::Int(i),
            BehaviorLiteral::Float(f) => Val::Float(f),
            BehaviorLiteral::Vec3(v) => Val::Vec3(v),
        }
    }

    /// The authored form, for persistence. Entities have no authored form and
    /// are never persisted, so they save as their declared-away default.
    pub fn to_literal(self) -> BehaviorLiteral {
        match self {
            Val::Bool(b) => BehaviorLiteral::Bool(b),
            Val::Int(i) => BehaviorLiteral::Int(i),
            Val::Float(f) => BehaviorLiteral::Float(f),
            Val::Vec3(v) => BehaviorLiteral::Vec3(v),
            Val::Entity(_) => BehaviorLiteral::Int(0),
        }
    }

    /// The cross-boundary form execution tracing publishes.
    pub fn to_trace(self) -> TraceVal {
        match self {
            Val::Bool(b) => TraceVal::Bool(b),
            Val::Int(i) => TraceVal::Int(i),
            Val::Float(f) => TraceVal::Float(f),
            Val::Vec3(v) => TraceVal::Vec3(v),
            Val::Entity(e) => TraceVal::Entity(e.to_bits()),
        }
    }

    /// Whether two values are the same shape, so a restored save can be
    /// rejected when the world's declaration changed type under it.
    pub fn same_type(self, other: Val) -> bool {
        core::mem::discriminant(&self) == core::mem::discriminant(&other)
    }
}

/// How two numbers combine.
#[derive(Clone, Copy, Debug)]
pub enum Arith {
    /// Sum.
    Add,
    /// Difference.
    Sub,
    /// Product.
    Mul,
    /// Quotient.
    Div,
}

/// How two values compare.
#[derive(Clone, Copy, Debug)]
pub enum Cmp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroU32;

    fn entity() -> Entity {
        Entity::new(7, NonZeroU32::MIN)
    }

    // Each reader answers for its own shape and declines every other, so a
    // mistyped expression yields nothing rather than a coerced guess.
    #[test]
    fn each_reader_answers_only_for_its_own_shape() {
        assert_eq!(Val::Bool(true).as_bool(), Some(true));
        assert_eq!(Val::Int(1).as_bool(), None);

        assert_eq!(Val::Entity(entity()).as_entity(), Some(entity()));
        assert_eq!(Val::Int(1).as_entity(), None);

        assert_eq!(Val::Vec3([1.0; 3]).as_vec3(), Some([1.0; 3]));
        assert_eq!(Val::Int(1).as_vec3(), None);
    }

    // The scalar reading is the one that widens: an int and a float both read
    // as numbers, and nothing else does.
    #[test]
    fn the_scalar_reading_covers_both_numeric_shapes() {
        assert_eq!(Val::Int(3).as_f32(), Some(3.0));
        assert_eq!(Val::Float(0.5).as_f32(), Some(0.5));
        assert_eq!(Val::Bool(true).as_f32(), None);
        assert_eq!(Val::Vec3([1.0; 3]).as_f32(), None);
        assert_eq!(Val::Entity(entity()).as_f32(), None);
    }

    #[test]
    fn a_literal_round_trips_through_its_runtime_value() {
        for lit in [
            BehaviorLiteral::Bool(true),
            BehaviorLiteral::Int(-2),
            BehaviorLiteral::Float(1.5),
            BehaviorLiteral::Vec3([1.0, 2.0, 3.0]),
        ] {
            assert_eq!(Val::from_literal(&lit).to_literal(), lit);
        }
    }

    // An entity is a runtime-only identity with no authored form, so it saves
    // as the declared-away default rather than as a handle a later run would
    // misread.
    #[test]
    fn an_entity_persists_as_the_declared_away_default() {
        assert_eq!(Val::Entity(entity()).to_literal(), BehaviorLiteral::Int(0));
    }

    #[test]
    fn every_value_has_a_trace_form() {
        assert_eq!(Val::Bool(true).to_trace(), TraceVal::Bool(true));
        assert_eq!(Val::Int(-2).to_trace(), TraceVal::Int(-2));
        assert_eq!(Val::Float(1.5).to_trace(), TraceVal::Float(1.5));
        assert_eq!(
            Val::Vec3([1.0, 2.0, 3.0]).to_trace(),
            TraceVal::Vec3([1.0, 2.0, 3.0])
        );
        assert_eq!(
            Val::Entity(entity()).to_trace(),
            TraceVal::Entity(entity().to_bits())
        );
    }

    // Shape, not value: a restored save whose declaration changed type under
    // it is rejected, while one that merely changed value is not.
    #[test]
    fn same_type_compares_shape_rather_than_value() {
        assert!(Val::Int(1).same_type(Val::Int(9)));
        assert!(!Val::Int(1).same_type(Val::Float(1.0)));
        assert!(!Val::Bool(true).same_type(Val::Int(1)));
    }
}
