// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::MinMaxPartial;
use super::MinMaxResult;
use super::min_max;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

pub(super) fn accumulate_extension(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    partial: &mut MinMaxPartial,
    array: &ExtensionArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let dtype = DType::Extension(array.ext_dtype().with_nullability(Nullability::NonNullable));
    // A NaN in float storage comes back as NaN extrema, which `merge` skips or poisons with.
    let local = min_max(array.storage_array(), ctx, *args.options)?
        .map(|MinMaxResult { min, max }| {
            // Storage extremes the extension type rejects are an error, not an invariant violation.
            VortexResult::Ok(MinMaxResult {
                min: Scalar::try_new(dtype.clone(), min.into_value())?,
                max: Scalar::try_new(dtype, max.into_value())?,
            })
        })
        .transpose()?;
    partial.merge(args, local);
    Ok(())
}
