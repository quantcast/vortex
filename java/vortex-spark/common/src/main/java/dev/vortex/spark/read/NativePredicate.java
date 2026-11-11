// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.Expression;
import java.io.Serializable;
import java.util.Optional;
import org.apache.spark.sql.types.StructType;

/** A serializable predicate whose native handles are built separately on each executor. */
public interface NativePredicate extends Serializable {
    /** Field paths needed to validate the predicate against a file's physical schema. */
    String[][] references();

    /** Returns empty if this file cannot safely evaluate the predicate. */
    Optional<Expression> convert(StructType schema);
}
