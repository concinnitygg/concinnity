// Turning a world's `Variables` and `Behavior` columns into what the system
// runs: the variable table and one compiled program per behavior.
//
// This happens at init and again whenever a write moves either column, so an
// authoring tool holding a live world can rewrite a body in place instead of
// reloading the world to apply it. A reseed keeps what the edit left
// meaningful: a variable whose declaration is untouched keeps the value the
// run put in it, and a behavior whose definition is unchanged keeps its
// instances, their locals, and their clocks.

use alloc::vec::Vec;

use super::instance::Instance;
use super::state::def_hash;
use crate::behavior::{Program, Val, VarTable, compile};
use crate::components::{Behavior, Variables};
use crate::ecs::{PipelineContext, Tick, asset_id::AssetId};

/// The compiled state one resolution produces. What each variable holds is not
/// here: that is the previous resolution's to hand over (see [`carry_vars`]),
/// and a world starting is the case where there is none.
pub(super) struct Resolved {
    pub(super) programs: Vec<Program>,
    pub(super) var_table: VarTable,
}

/// The source columns' change ticks as of the resolution that read them. A
/// write to either moves one, which is the whole of how an edit is noticed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct SourceTicks {
    behaviors: Tick,
    variables: Tick,
}

impl SourceTicks {
    pub(super) fn of(ctx: &PipelineContext) -> Self {
        Self {
            behaviors: ctx.changed_tick::<Behavior>(),
            variables: ctx.changed_tick::<Variables>(),
        }
    }
}

/// Compile `behaviors` against the table `variables` declares. Declared
/// variables get their slots first so each carries its authored type and
/// starting value; a name a body mentions without a declaration is interned as
/// an integer starting at zero while it compiles.
pub(super) fn resolve(variables: &[Variables], behaviors: &[Behavior]) -> Resolved {
    let mut var_table = VarTable::default();
    for declared in variables {
        for decl in &declared.vars {
            var_table.declare(&decl.name, Val::from_literal(&decl.value));
        }
    }
    let programs: Vec<Program> = behaviors
        .iter()
        .cloned()
        .map(|def| compile(def, &mut var_table))
        .collect();
    Resolved {
        programs,
        var_table,
    }
}

/// What a reseed keeps of the previous resolution.
pub(super) struct Carried {
    /// One instance list per new program, in program order.
    pub(super) instances: Vec<Vec<Instance>>,
    /// Where each previous program landed in the new list, for the ones whose
    /// definition is unchanged. `None` for a definition the edit moved, so
    /// anything still keyed on that program is dropped rather than aimed at a
    /// body that no longer matches it.
    pub(super) moved: Vec<Option<usize>>,
}

/// Move each unchanged behavior's instances onto its new program. Identity and
/// content both have to match: a behavior whose body was edited starts fresh,
/// because its locals and its firing history describe a program that no longer
/// exists.
pub(super) fn carry_instances(
    prev: &[Program],
    prev_instances: Vec<Vec<Instance>>,
    next: &[Program],
) -> Carried {
    let keys: Vec<(AssetId, u64)> = prev
        .iter()
        .map(|p| (p.def.asset_id, def_hash(&p.def)))
        .collect();
    let mut held: Vec<Option<Vec<Instance>>> = prev_instances.into_iter().map(Some).collect();
    held.resize_with(keys.len(), || None);
    let mut moved = Vec::new();
    moved.resize(keys.len(), None);

    let mut instances = Vec::with_capacity(next.len());
    for (i, program) in next.iter().enumerate() {
        let key = (program.def.asset_id, def_hash(&program.def));
        let mut kept = Vec::new();
        if let Some(j) = keys.iter().position(|k| *k == key)
            && let Some(held) = held[j].take()
        {
            moved[j] = Some(i);
            kept = held;
        }
        instances.push(kept);
    }
    Carried { instances, moved }
}

/// The values the new table starts holding: the declared ones, with whatever
/// the run has put in a slot carried over where the declaration behind it is
/// untouched. One whose declared value changed takes that new value, which is
/// what makes editing a starting value show in a world already running; a name
/// the edit introduced starts where it was declared, and one it removed is
/// gone. A world starting has no previous table, so every slot starts where it
/// was declared.
pub(super) fn carry_vars(prev: &VarTable, prev_vals: &[Val], next: &VarTable) -> Vec<Val> {
    let mut vals = next.initial();
    for (slot, name) in next.names().iter().enumerate() {
        if prev.init_of(name) != Some(vals[slot]) {
            continue;
        }
        let Some(current) = prev
            .slot_of(name)
            .and_then(|s| prev_vals.get(s as usize))
            .copied()
        else {
            continue;
        };
        if current.same_type(vals[slot]) {
            vals[slot] = current;
        }
    }
    vals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{BehaviorExpr, BehaviorLiteral, BehaviorNode, VariableDecl};
    use alloc::string::ToString;
    use alloc::vec;

    fn declared(name: &str, value: BehaviorLiteral) -> Variables {
        Variables {
            vars: vec![VariableDecl {
                name: name.to_string(),
                value,
            }],
            ..Default::default()
        }
    }

    fn setter(id: u32, var: &str, value: i32) -> Behavior {
        Behavior {
            asset_id: AssetId(id),
            body: vec![BehaviorNode::Set {
                var: var.to_string(),
                value: BehaviorExpr::Int(value),
                add: false,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn declared_variables_keep_their_authored_type_and_start() {
        let resolved = resolve(&[declared("score", BehaviorLiteral::Int(7))], &[]);
        assert_eq!(resolved.var_table.slot_of("score"), Some(0));
        assert_eq!(resolved.var_table.initial(), vec![Val::Int(7)]);
    }

    // A name only a body mentions still gets a slot, so the program compiles
    // against the same table a declaration would have given it.
    #[test]
    fn an_undeclared_name_is_interned_while_the_body_compiles() {
        let resolved = resolve(&[], &[setter(1, "hits", 3)]);
        assert_eq!(resolved.programs.len(), 1);
        assert_eq!(resolved.var_table.names(), &["hits".to_string()]);
        assert_eq!(resolved.var_table.initial(), vec![Val::Int(0)]);
    }

    // A world starting is this same path with nothing to carry over.
    #[test]
    fn a_first_resolution_starts_every_slot_where_it_was_declared() {
        let resolved = resolve(&[declared("score", BehaviorLiteral::Int(7))], &[]);
        let vars = carry_vars(&VarTable::default(), &[], &resolved.var_table);
        assert_eq!(vars, vec![Val::Int(7)]);
    }

    #[test]
    fn an_untouched_declaration_keeps_the_running_value() {
        let before = resolve(&[declared("score", BehaviorLiteral::Int(0))], &[]);
        let after = resolve(
            &[declared("score", BehaviorLiteral::Int(0))],
            &[setter(1, "score", 5)],
        );
        let carried = carry_vars(&before.var_table, &[Val::Int(42)], &after.var_table);
        assert_eq!(carried, vec![Val::Int(42)]);
    }

    // Editing the starting value is the edit; a world already running takes it,
    // rather than showing the author a field that appears not to apply.
    #[test]
    fn an_edited_declaration_takes_its_new_value() {
        let before = resolve(&[declared("score", BehaviorLiteral::Int(0))], &[]);
        let after = resolve(&[declared("score", BehaviorLiteral::Int(9))], &[]);
        let carried = carry_vars(&before.var_table, &[Val::Int(42)], &after.var_table);
        assert_eq!(carried, vec![Val::Int(9)]);
    }

    #[test]
    fn a_retyped_declaration_takes_its_new_value() {
        let before = resolve(&[declared("flag", BehaviorLiteral::Int(0))], &[]);
        let after = resolve(&[declared("flag", BehaviorLiteral::Bool(true))], &[]);
        let carried = carry_vars(&before.var_table, &[Val::Int(3)], &after.var_table);
        assert_eq!(carried, vec![Val::Bool(true)]);
    }

    #[test]
    fn a_new_declaration_starts_where_it_was_declared() {
        let before = resolve(&[], &[]);
        let after = resolve(&[declared("score", BehaviorLiteral::Int(4))], &[]);
        assert_eq!(
            carry_vars(&before.var_table, &[], &after.var_table),
            vec![Val::Int(4)]
        );
    }

    // Slots are assigned by the new table, so a value follows its name rather
    // than its old slot number.
    #[test]
    fn a_carried_value_follows_its_name_across_a_reordered_table() {
        let before = resolve(
            &[Variables {
                vars: vec![
                    VariableDecl {
                        name: "a".to_string(),
                        value: BehaviorLiteral::Int(0),
                    },
                    VariableDecl {
                        name: "b".to_string(),
                        value: BehaviorLiteral::Int(0),
                    },
                ],
                ..Default::default()
            }],
            &[],
        );
        let after = resolve(
            &[Variables {
                vars: vec![
                    VariableDecl {
                        name: "b".to_string(),
                        value: BehaviorLiteral::Int(0),
                    },
                    VariableDecl {
                        name: "a".to_string(),
                        value: BehaviorLiteral::Int(0),
                    },
                ],
                ..Default::default()
            }],
            &[],
        );
        let carried = carry_vars(
            &before.var_table,
            &[Val::Int(1), Val::Int(2)],
            &after.var_table,
        );
        assert_eq!(carried, vec![Val::Int(2), Val::Int(1)]);
    }

    #[test]
    fn an_unchanged_behavior_keeps_its_instances() {
        let prev = resolve(&[], &[setter(1, "a", 1), setter(2, "b", 1)]);
        let next = resolve(&[], &[setter(1, "a", 1), setter(2, "b", 1)]);
        let instances = vec![
            vec![Instance::new(None, Vec::new(), false)],
            vec![
                Instance::new(None, Vec::new(), false),
                Instance::new(None, Vec::new(), false),
            ],
        ];
        let carried = carry_instances(&prev.programs, instances, &next.programs);
        assert_eq!(carried.instances[0].len(), 1);
        assert_eq!(carried.instances[1].len(), 2);
        assert_eq!(carried.moved, vec![Some(0), Some(1)]);
    }

    // An edited body's instances describe a program that no longer exists, so
    // that one starts fresh while its neighbour carries.
    #[test]
    fn an_edited_behavior_starts_fresh() {
        let prev = resolve(&[], &[setter(1, "a", 1), setter(2, "b", 1)]);
        let next = resolve(&[], &[setter(1, "a", 99), setter(2, "b", 1)]);
        let instances = vec![
            vec![Instance::new(None, Vec::new(), false)],
            vec![Instance::new(None, Vec::new(), false)],
        ];
        let carried = carry_instances(&prev.programs, instances, &next.programs);
        assert!(carried.instances[0].is_empty());
        assert_eq!(carried.instances[1].len(), 1);
        assert_eq!(carried.moved, vec![None, Some(1)]);
    }

    // Identity is half the key: a body copied onto another asset does not
    // inherit that asset's firing history.
    #[test]
    fn instances_never_move_between_assets() {
        let prev = resolve(&[], &[setter(1, "a", 1)]);
        let next = resolve(&[], &[setter(2, "a", 1)]);
        let instances = vec![vec![Instance::new(None, Vec::new(), false)]];
        let carried = carry_instances(&prev.programs, instances, &next.programs);
        assert!(carried.instances[0].is_empty());
        assert_eq!(carried.moved, vec![None]);
    }
}
