// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::sync::LazyLock;

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr as unbound;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::expr::bound::pack;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

pub type FieldIndex = usize;
pub type BuildFn = Box<dyn FnOnce(&[Expression]) -> VortexResult<Expression>>;

pub enum NodeTouches {
    One(Option<FieldIndex>, BoundExpression),
    Many {
        slots: Vec<(FieldIndex, BoundExpression)>,
        build: BuildFn,
    },
}

pub trait Demand: Any + Debug + Send + Sync {}

#[derive(Clone, Copy, Debug, Default)]
pub struct DemandAll;

impl Demand for DemandAll {}

pub trait PartitionRule: Debug + Send + Sync {
    fn apply(
        &self,
        node: &BoundExpression,
        demand: &dyn Demand,
    ) -> VortexResult<Option<NodeTouches>>;
}

#[derive(Debug)]
pub struct PartitionRules {
    pub scope: &'static dyn PartitionRule,
    pub by_fn: HashMap<ScalarFnId, &'static dyn PartitionRule>,
}

#[derive(Debug, Clone)]
pub struct Partition {
    pub slot: FieldIndex,
    pub expr: BoundExpression,
}

#[derive(Debug, Clone)]
pub struct PartitionedExpr {
    pub root: BoundExpression,
    pub partitions: Vec<Partition>,
}

pub fn partition_expr(
    expr: &BoundExpression,
    rules: &PartitionRules,
    demand: &dyn Demand,
) -> VortexResult<Partitioning> {
    let mut parts = Parts::default();
    let root = match analyze(expr, rules, demand, &mut parts)? {
        Analyzed::Kept(slot, expr) => return Ok(Partitioning::Mono(slot, expr.clone())),
        Analyzed::New(slot, expr) => return Ok(Partitioning::Mono(slot, expr)),
        Analyzed::Root(root) => root,
    };

    let mut partitions = Vec::with_capacity(parts.parts.len());
    let mut partition_dtypes = Vec::with_capacity(parts.parts.len());
    let mut flattened = HashSet::default();
    for (position, part) in parts.parts.into_iter().enumerate() {
        let mut subs: Vec<Option<BoundExpression>> = vec![None; part.positions.len()];
        for (expr, sub) in part.positions {
            subs[sub] = Some(expr);
        }
        let mut subs = subs
            .into_iter()
            .map(|sub| sub.vortex_expect("partition positions are dense"));
        let expr = if subs.len() == 1 {
            flattened.insert(positional_name(position));
            subs.next().vortex_expect("a single sub-expression")
        } else {
            pack(
                subs.enumerate()
                    .map(|(sub, expr)| (positional_name(sub), expr)),
                Nullability::NonNullable,
            )
        };
        partition_dtypes.push(expr.dtype().clone());
        partitions.push(Partition {
            slot: part.slot,
            expr,
        });
    }

    let scope_names: FieldNames = (0..partitions.len()).map(positional_name).collect();
    let scope_dtype = DType::Struct(
        StructFields::new(scope_names, partition_dtypes),
        Nullability::NonNullable,
    );

    let root = flatten_refs(&root, &flattened)?.bind(&scope_dtype)?;
    Ok(Partitioning::Many(PartitionedExpr { root, partitions }))
}

const INTERNED_NAMES: usize = 100;
static POSITIONAL_NAMES: LazyLock<[FieldName; INTERNED_NAMES]> =
    LazyLock::new(|| std::array::from_fn(|position| FieldName::from(format!("{position}"))));

pub(crate) fn positional_name(position: usize) -> FieldName {
    POSITIONAL_NAMES
        .get(position)
        .cloned()
        .unwrap_or_else(|| format!("{position}").into())
}

#[derive(Debug, Clone)]
pub enum Partitioning {
    Mono(Option<FieldIndex>, BoundExpression),
    Many(PartitionedExpr),
}

enum Analyzed<'a> {
    Kept(Option<FieldIndex>, &'a BoundExpression),
    New(Option<FieldIndex>, BoundExpression),
    Root(Expression),
}

struct PartBuild {
    slot: FieldIndex,
    positions: HashMap<BoundExpression, usize>,
}

impl PartBuild {
    fn push_dedup(&mut self, expr: BoundExpression) -> usize {
        let next = self.positions.len();
        *self.positions.entry(expr).or_insert(next)
    }
}

#[derive(Default)]
struct Parts {
    index: HashMap<FieldIndex, usize>,
    parts: Vec<PartBuild>,
}

impl Parts {
    fn cut(&mut self, slot: FieldIndex, expr: BoundExpression) -> Expression {
        let part = *self.index.entry(slot).or_insert_with(|| {
            self.parts.push(PartBuild {
                slot,
                positions: HashMap::default(),
            });
            self.parts.len() - 1
        });
        let sub = self.parts[part].push_dedup(expr);
        unbound::get_item(
            positional_name(sub),
            unbound::get_item(positional_name(part), unbound::root()),
        )
    }
}

fn analyze<'a>(
    node: &'a BoundExpression,
    rules: &PartitionRules,
    demand: &dyn Demand,
    parts: &mut Parts,
) -> VortexResult<Analyzed<'a>> {
    if let Some(rule) = rule_for(rules, node)
        && let Some(applied) = rule.apply(node, demand)?
    {
        return Ok(match applied {
            NodeTouches::One(slot, expr) => Analyzed::New(slot, expr),
            NodeTouches::Many { slots, build } => {
                let refs = slots
                    .into_iter()
                    .map(|(slot, expr)| parts.cut(slot, expr))
                    .collect::<Vec<_>>();
                Analyzed::Root(build(&refs)?)
            }
        });
    }
    vortex_ensure!(!node.is_root());
    let scalar_fn = node
        .as_scalar()
        .vortex_expect("non-root expressions are scalar functions");

    let mut slot: Option<FieldIndex> = None;
    let mut many = false;
    let mut changed = false;
    let mut children = Vec::with_capacity(node.children().len());
    for child in node.children() {
        let analyzed = analyze(child, rules, demand, parts)?;
        let child_slot = match &analyzed {
            Analyzed::Kept(child_slot, _) => child_slot,
            Analyzed::New(child_slot, _) => {
                changed = true;
                child_slot
            }
            Analyzed::Root(_) => {
                many = true;
                &None
            }
        };
        if let Some(child_slot) = child_slot {
            match slot {
                None => slot = Some(*child_slot),
                Some(slot) if slot == *child_slot => {}
                Some(_) => many = true,
            }
        }
        children.push(analyzed);
    }

    Ok(if many {
        let children = children
            .into_iter()
            .map(|child| match child {
                Analyzed::Kept(None, expr) => to_unbound(expr),
                Analyzed::New(None, expr) => to_unbound(&expr),
                Analyzed::Kept(Some(slot), expr) => Ok(parts.cut(slot, expr.clone())),
                Analyzed::New(Some(slot), expr) => Ok(parts.cut(slot, expr)),
                Analyzed::Root(expr) => Ok(expr),
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Analyzed::Root(Expression::try_new(scalar_fn.clone(), children)?)
    } else if !changed {
        Analyzed::Kept(slot, node)
    } else {
        let exprs = children
            .into_iter()
            .map(|child| match child {
                Analyzed::Kept(_, expr) => Ok(expr.clone()),
                Analyzed::New(_, expr) => Ok(expr),
                Analyzed::Root(_) => {
                    Err(vortex_err!("A single-coordinate node has a spanning child"))
                }
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Analyzed::New(slot, BoundExpression::try_new(scalar_fn.clone(), exprs)?)
    })
}

fn rule_for<'a>(
    rules: &'a PartitionRules,
    node: &BoundExpression,
) -> Option<&'a dyn PartitionRule> {
    if node.is_root() {
        return Some(rules.scope);
    }
    let scalar_fn = node.as_scalar()?;
    rules.by_fn.get(&scalar_fn.id()).copied()
}

fn to_unbound(expr: &BoundExpression) -> VortexResult<Expression> {
    let Some(scalar_fn) = expr.as_scalar() else {
        vortex_bail!("scope reference");
    };
    let children = expr
        .children()
        .iter()
        .map(to_unbound)
        .collect::<VortexResult<Vec<_>>>()?;
    Expression::try_new(scalar_fn.clone(), children)
}

fn flatten_refs(node: &Expression, flattened: &HashSet<FieldName>) -> VortexResult<Expression> {
    if node.is::<GetItem>() {
        // TODO what if someone registers getitem for other subplan which conflicts
        let inner = node.child(0);
        if let Some(part_name) = inner.as_opt::<GetItem>()
            && inner.child(0).is_root()
            && flattened.contains(part_name)
        {
            return Ok(inner.clone());
        }
    }
    if node.children().is_empty() {
        return Ok(node.clone());
    }
    let children = node
        .children()
        .iter()
        .map(|child| flatten_refs(child, flattened))
        .collect::<VortexResult<Vec<_>>>()?;
    node.clone().with_children(children)
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::PType;
    use vortex_array::expr::bound::get_item;
    use vortex_array::expr::bound::is_not_null;
    use vortex_array::expr::bound::lit;

    use super::*;
    use crate::plan::physical::struct_partition_rules;

    fn two_field_struct() -> DType {
        let i32 = DType::Primitive(PType::I32, Nullability::NonNullable);
        DType::Struct(
            StructFields::from_iter([("a", i32.clone()), ("b", i32)]),
            Nullability::NonNullable,
        )
    }

    #[test]
    fn is_not_null_of_non_nullable() -> VortexResult<()> {
        let scope = two_field_struct();
        let root = BoundExpression::new_root(scope);
        let expr = pack(
            [
                ("v", is_not_null(root.clone())),
                ("a", get_item("a", root.clone())),
                ("b", get_item("b", root)),
            ],
            Nullability::NonNullable,
        );

        let rules = struct_partition_rules();
        let Partitioning::Many(partitioned) = partition_expr(&expr, rules, &DemandAll)? else {
            vortex_bail!("Expected a multi-coordinate partitioning");
        };

        assert_eq!(partitioned.partitions.len(), 2);
        let synth_root = BoundExpression::new_root(DType::Struct(
            StructFields::from_iter([
                ("0", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("1", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ]),
            Nullability::NonNullable,
        ));
        let expected = pack(
            [
                ("v", lit(true)),
                ("a", get_item("0", synth_root.clone())),
                ("b", get_item("1", synth_root)),
            ],
            Nullability::NonNullable,
        );
        assert_eq!(partitioned.root, expected);
        Ok(())
    }

    #[test]
    fn equal_cuts() -> VortexResult<()> {
        let scope = two_field_struct();
        let root = BoundExpression::new_root(scope);
        let a = get_item("a", root.clone());
        let b = get_item("b", root);
        let expr = pack(
            [("x", a.clone()), ("y", b), ("z", a)],
            Nullability::NonNullable,
        );

        let rules = struct_partition_rules();
        let Partitioning::Many(partitioned) = partition_expr(&expr, rules, &DemandAll)? else {
            vortex_bail!("Expected a multi-coordinate partitioning");
        };

        assert_eq!(partitioned.partitions.len(), 2);
        for partition in &partitioned.partitions {
            assert!(partition.expr.is_root());
        }
        Ok(())
    }

    fn i32() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    fn struct_of(fields: &[(&str, DType)], nullability: Nullability) -> DType {
        DType::Struct(
            StructFields::from_iter(fields.iter().map(|(name, dtype)| (*name, dtype.clone()))),
            nullability,
        )
    }

    fn synth(dtypes: &[DType]) -> DType {
        DType::Struct(
            StructFields::new(
                (0..dtypes.len()).map(positional_name).collect(),
                dtypes.to_vec(),
            ),
            Nullability::NonNullable,
        )
    }

    fn assert_partitions(
        partitioned: &PartitionedExpr,
        expected: &[(FieldIndex, BoundExpression)],
    ) {
        assert_eq!(partitioned.partitions.len(), expected.len());
        for (partition, (slot, expr)) in partitioned.partitions.iter().zip(expected) {
            assert_eq!(partition.slot, *slot);
            assert_eq!(&partition.expr, expr);
        }
    }

    fn partition(expr: Expression, scope: &DType) -> VortexResult<Partitioning> {
        partition_expr(&expr.bind(scope)?, struct_partition_rules(), &DemandAll)
    }

    #[test]
    fn is_not_null_one_partition() -> VortexResult<()> {
        let scope = two_field_struct();
        let expr = unbound::is_not_null(unbound::get_item("a", unbound::root()));

        let Partitioning::Mono(Some(0), expr) = partition(expr, &scope)? else {
            vortex_bail!("Expected the coordinate of field a");
        };
        assert_eq!(expr, unbound::is_not_null(unbound::root()).bind(&i32())?);
        Ok(())
    }

    #[test]
    fn is_not_null_two_fields() -> VortexResult<()> {
        let scope = struct_of(&[("a", i32()), ("b", i32())], Nullability::Nullable);
        let expr = unbound::is_not_null(unbound::get_item("a", unbound::root()));

        let Partitioning::Many(partitioned) = partition(expr, &scope)? else {
            vortex_bail!("Expected a recombination");
        };

        let bool_ = DType::Bool(Nullability::NonNullable);
        assert_partitions(
            &partitioned,
            &[
                (FieldIndex::MAX, BoundExpression::new_root(bool_.clone())),
                (0, BoundExpression::new_root(i32())),
            ],
        );
        let expected = unbound::is_not_null(unbound::mask(
            unbound::get_item("1", unbound::root()),
            unbound::get_item("0", unbound::root()),
        ))
        .bind(&synth(&[bool_, i32()]))?;
        assert_eq!(partitioned.root, expected);
        Ok(())
    }

    #[test]
    fn test_pack() -> VortexResult<()> {
        let scope = two_field_struct();
        let expr = unbound::get_item(
            "a",
            unbound::get_item(
                "a",
                unbound::pack([("a", unbound::root())], Nullability::NonNullable),
            ),
        );

        let Partitioning::Many(partitioned) = partition(expr, &scope)? else {
            vortex_bail!("Expected a recombination");
        };

        assert_partitions(
            &partitioned,
            &[
                (0, BoundExpression::new_root(i32())),
                (1, BoundExpression::new_root(i32())),
            ],
        );
        let expected = unbound::get_item(
            "a",
            unbound::get_item(
                "a",
                unbound::pack(
                    [(
                        "a",
                        unbound::pack(
                            [
                                ("a", unbound::get_item("0", unbound::root())),
                                ("b", unbound::get_item("1", unbound::root())),
                            ],
                            Nullability::NonNullable,
                        ),
                    )],
                    Nullability::NonNullable,
                ),
            ),
        )
        .bind(&synth(&[i32(), i32()]))?;
        assert_eq!(partitioned.root, expected);
        Ok(())
    }

    #[test]
    fn test_ab_greater_c() -> VortexResult<()> {
        let inner = struct_of(&[("b", i32())], Nullability::NonNullable);
        let scope = struct_of(
            &[("a", inner.clone()), ("c", i32())],
            Nullability::NonNullable,
        );
        let expr = unbound::gt(
            unbound::get_item("b", unbound::get_item("a", unbound::root())),
            unbound::get_item("c", unbound::root()),
        );

        let Partitioning::Many(partitioned) = partition(expr, &scope)? else {
            vortex_bail!("Expected a recombination");
        };

        assert_partitions(
            &partitioned,
            &[
                (0, unbound::get_item("b", unbound::root()).bind(&inner)?),
                (1, BoundExpression::new_root(i32())),
            ],
        );
        let expected = unbound::gt(
            unbound::get_item("0", unbound::root()),
            unbound::get_item("1", unbound::root()),
        )
        .bind(&synth(&[i32(), i32()]))?;
        assert_eq!(partitioned.root, expected);
        Ok(())
    }

    #[test]
    fn test_x_plus_1() -> VortexResult<()> {
        let scope = struct_of(&[("x", i32()), ("y", i32())], Nullability::NonNullable);
        let expr = unbound::gt(
            unbound::checked_add(unbound::get_item("x", unbound::root()), unbound::lit(1i32)),
            unbound::get_item("y", unbound::root()),
        );

        let Partitioning::Many(partitioned) = partition(expr, &scope)? else {
            vortex_bail!("Expected a recombination");
        };

        assert_partitions(
            &partitioned,
            &[
                (
                    0,
                    unbound::checked_add(unbound::root(), unbound::lit(1i32)).bind(&i32())?,
                ),
                (1, BoundExpression::new_root(i32())),
            ],
        );
        let expected = unbound::gt(
            unbound::get_item("0", unbound::root()),
            unbound::get_item("1", unbound::root()),
        )
        .bind(&synth(&[i32(), i32()]))?;
        assert_eq!(partitioned.root, expected);
        Ok(())
    }

    #[test]
    fn test_aa_greater_ba() -> VortexResult<()> {
        let inner = struct_of(&[("a", i32())], Nullability::NonNullable);
        let scope = struct_of(
            &[("a", inner.clone()), ("b", inner.clone())],
            Nullability::NonNullable,
        );
        let expr = unbound::gt(
            unbound::get_item("a", unbound::get_item("a", unbound::root())),
            unbound::get_item("a", unbound::get_item("b", unbound::root())),
        );

        let Partitioning::Many(partitioned) = partition(expr, &scope)? else {
            vortex_bail!("Expected a recombination");
        };

        let read = unbound::get_item("a", unbound::root()).bind(&inner)?;
        assert_partitions(&partitioned, &[(0, read.clone()), (1, read)]);
        Ok(())
    }

    #[test]
    fn test_nested_and() -> VortexResult<()> {
        let scope = two_field_struct();
        let a = unbound::get_item("a", unbound::root());
        let b = unbound::get_item("b", unbound::root());
        let expr = unbound::and(
            unbound::and(
                unbound::gt(a.clone(), unbound::lit(5i32)),
                unbound::gt(b, unbound::lit(5i32)),
            ),
            unbound::lt(a, unbound::lit(1i32)),
        );

        let Partitioning::Many(partitioned) = partition(expr, &scope)? else {
            vortex_bail!("Expected a recombination");
        };

        let bool_ = DType::Bool(Nullability::NonNullable);
        let gt5 = unbound::gt(unbound::root(), unbound::lit(5i32));
        let lt1 = unbound::lt(unbound::root(), unbound::lit(1i32));
        assert_partitions(
            &partitioned,
            &[
                (
                    0,
                    unbound::pack([("0", gt5.clone()), ("1", lt1)], Nullability::NonNullable)
                        .bind(&i32())?,
                ),
                (1, gt5.bind(&i32())?),
            ],
        );
        let a_scope = synth(&[bool_.clone(), bool_.clone()]);
        let expected = unbound::and(
            unbound::and(
                unbound::get_item("0", unbound::get_item("0", unbound::root())),
                unbound::get_item("1", unbound::root()),
            ),
            unbound::get_item("1", unbound::get_item("0", unbound::root())),
        )
        .bind(&synth(&[a_scope, bool_]))?;
        assert_eq!(partitioned.root, expected);
        Ok(())
    }

    #[test]
    fn test_duplicate_predicate() -> VortexResult<()> {
        let scope = two_field_struct();
        let a = unbound::get_item("a", unbound::root());
        let b = unbound::get_item("b", unbound::root());
        let gt5_a = unbound::gt(a, unbound::lit(5i32));
        let expr = unbound::and(
            unbound::and(gt5_a.clone(), unbound::gt(b, unbound::lit(5i32))),
            gt5_a,
        );

        let Partitioning::Many(partitioned) = partition(expr, &scope)? else {
            vortex_bail!("Expected a recombination");
        };

        let bool_ = DType::Bool(Nullability::NonNullable);
        let gt5 = unbound::gt(unbound::root(), unbound::lit(5i32)).bind(&i32())?;
        assert_partitions(&partitioned, &[(0, gt5.clone()), (1, gt5)]);
        let a_ref = unbound::get_item("0", unbound::root());
        let expected = unbound::and(
            unbound::and(a_ref.clone(), unbound::get_item("1", unbound::root())),
            a_ref,
        )
        .bind(&synth(&[bool_.clone(), bool_]))?;
        assert_eq!(partitioned.root, expected);
        Ok(())
    }

    #[test]
    fn test_abc_greater_abd() -> VortexResult<()> {
        let leaf = struct_of(&[("c", i32()), ("d", i32())], Nullability::NonNullable);
        let middle = struct_of(&[("b", leaf)], Nullability::NonNullable);
        let scope = struct_of(&[("a", middle.clone())], Nullability::NonNullable);
        let b = || unbound::get_item("b", unbound::root());
        let expr = unbound::gt(
            unbound::get_item(
                "c",
                unbound::get_item("b", unbound::get_item("a", unbound::root())),
            ),
            unbound::get_item(
                "d",
                unbound::get_item("b", unbound::get_item("a", unbound::root())),
            ),
        );

        let Partitioning::Mono(Some(0), expr) = partition(expr, &scope)? else {
            vortex_bail!("Expected the coordinate of field a");
        };
        let expected =
            unbound::gt(unbound::get_item("c", b()), unbound::get_item("d", b())).bind(&middle)?;
        assert_eq!(expr, expected);
        Ok(())
    }
}
