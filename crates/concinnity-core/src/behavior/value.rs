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
