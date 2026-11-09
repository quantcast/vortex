// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A compact membership falsifier over the minimum and maximum of each zone.

use std::cmp::Ordering;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::arrays::BoolArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

/// Sorted non-null members, shared by clones of the pruning expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct SortedValues {
    dtype: DType,
    values: Arc<[Scalar]>,
}

impl SortedValues {
    pub(super) fn new(elements: Vec<Scalar>) -> Option<Self> {
        let dtype = elements.first()?.dtype().clone();
        if !matches!(
            dtype,
            DType::Bool(_)
                | DType::Primitive(..)
                | DType::Decimal(..)
                | DType::Utf8(_)
                | DType::Binary(_)
        ) || elements
            .iter()
            .any(|value| !value.dtype().eq_ignore_nullability(&dtype))
        {
            return None;
        }
        let mut values = elements
            .into_iter()
            .filter(|value| !value.is_null())
            .collect::<Vec<_>>();
        // These scalar types have the same total order as their min/max aggregates. In
        // particular, floating-point order distinguishes signed zero and NaN bit patterns.
        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        values.dedup();
        Some(Self {
            dtype,
            values: values.into(),
        })
    }

    fn disjoint(&self, min: &Scalar, max: &Scalar) -> bool {
        // Missing bounds prove nothing, including for all-null zones. Float callers also
        // retain the existing non-NaN proof because min/max may omit matching NaN needles.
        if min.is_null() || max.is_null() || min.partial_cmp(max).is_none() || min > max {
            return false;
        }
        let first = self.values.partition_point(|value| value < min);
        self.values
            .get(first)
            .is_none_or(|value| value.partial_cmp(max) == Some(Ordering::Greater))
    }
}

impl Display for SortedValues {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} members", self.values.len())
    }
}

/// An internal stats expression: true only if no set member falls in `[min, max]`.
#[derive(Clone, Debug)]
pub(super) struct SetDisjoint;

impl ScalarFnVTable for SetDisjoint {
    type Options = SortedValues;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.stats.set_disjoint");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => "min".into(),
            1 => "max".into(),
            _ => unreachable!(),
        }
    }

    fn return_dtype(&self, values: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(args[0].eq_ignore_nullability(&args[1]));
        vortex_ensure!(
            values.dtype.eq_ignore_nullability(&args[0]),
            "Set members must have the dtype of the zone bounds"
        );
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn execute(
        &self,
        values: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Materialize once and retain probe state, rather than executing each bound per zone.
        let mut min = args
            .get(0)?
            .execute::<RecursiveCanonical>(ctx)?
            .0
            .into_array()
            .repeated_probe();
        let mut max = args
            .get(1)?
            .execute::<RecursiveCanonical>(ctx)?
            .0
            .into_array()
            .repeated_probe();
        let results = (0..args.row_count())
            .map(|idx| {
                Ok(values.disjoint(
                    &min.execute_scalar(idx, ctx)?,
                    &max.execute_scalar(idx, ctx)?,
                ))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(BoolArray::from_iter(results).into_array())
    }
}
