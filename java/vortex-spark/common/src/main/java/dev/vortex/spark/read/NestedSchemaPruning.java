// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.Expression;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * Pushes Spark's nested schema pruning down to the native scan.
 *
 * <p>Only struct fields are pruned. Fields inside arrays and maps keep their full table type, since projecting each
 * element needs a per-element expression the native scan does not expose yet.
 */
public final class NestedSchemaPruning {
    private NestedSchemaPruning() {}

    /**
     * Returns each field of {@code requiredSchema} with its {@code tableSchema} type {@link #prune pruned} to the
     * struct fields it requires. A field {@code tableSchema} does not carry keeps its required type.
     */
    public static StructType pruneFields(StructType requiredSchema, StructType tableSchema) {
        StructField[] fields = requiredSchema.fields().clone();
        for (int i = 0; i < fields.length; i++) {
            StructField field = fields[i];
            if (tableSchema.getFieldIndex(field.name()).isDefined()) {
                DataType tableType = tableSchema.apply(field.name()).dataType();
                fields[i] = new StructField(
                        field.name(), prune(tableType, field.dataType()), field.nullable(), field.metadata());
            }
        }
        return new StructType(fields);
    }

    /**
     * Returns {@code tableType} pruned to the struct fields present in {@code requiredType}, in table order. Any type
     * that is not a struct on both sides keeps its table type.
     */
    static DataType prune(DataType tableType, DataType requiredType) {
        if (!(tableType instanceof StructType table) || !(requiredType instanceof StructType required)) {
            return tableType;
        }
        List<StructField> fields = new ArrayList<>();
        for (StructField field : table.fields()) {
            if (required.getFieldIndex(field.name()).isDefined()) {
                DataType prunedType =
                        prune(field.dataType(), required.apply(field.name()).dataType());
                fields.add(new StructField(field.name(), prunedType, field.nullable(), field.metadata()));
            }
        }
        return fields.isEmpty() ? tableType : new StructType(fields.toArray(new StructField[0]));
    }

    /**
     * Builds the native projection that returns the {@code fileNames} columns with the types of {@code readSchema},
     * where each field is either its type in {@code tableSchema} or a {@link #prune pruned} version of it.
     *
     * <p>When nothing is pruned this is a plain {@code select} of the columns.
     */
    static Expression projection(String[] fileNames, StructType readSchema, StructType tableSchema) {
        return structProjection(Expression.root(), false, fileNames, readSchema, tableSchema);
    }

    private static Expression structProjection(
            Expression struct, boolean nullable, String[] names, StructType read, StructType table) {
        StructField[] fields = read.fields();
        Expression[] children = new Expression[fields.length];
        boolean pruned = false;
        for (int i = 0; i < fields.length; i++) {
            DataType tableType = table.apply(fields[i].name()).dataType();
            if (fields[i].dataType() instanceof StructType readStruct
                    && tableType instanceof StructType tableStruct
                    && !readStruct.equals(tableStruct)) {
                children[i] = structProjection(
                        Expression.getItem(names[i], struct),
                        fields[i].nullable(),
                        readStruct.fieldNames(),
                        readStruct,
                        tableStruct);
                pruned = true;
            }
        }
        // `select` keeps the struct's own nulls, so use it when no child needs pruning.
        if (!pruned) {
            return Expression.select(names, struct);
        }
        for (int i = 0; i < fields.length; i++) {
            if (children[i] == null) {
                children[i] = Expression.getItem(names[i], struct);
            }
        }
        Expression packed = Expression.pack(names, children, false);
        // `pack` builds a struct with no nulls, so carry over the source struct's nulls.
        return nullable ? Expression.mask(packed, Expression.isNotNull(struct)) : packed;
    }
}
