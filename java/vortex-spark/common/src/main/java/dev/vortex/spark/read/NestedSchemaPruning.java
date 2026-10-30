// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.Expression;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * Pushes Spark's nested schema pruning down to the native scan.
 *
 * <p>Struct fields are pruned, through arrays too: an array of structs is read through a {@code list_transform} that
 * keeps only the element fields Spark reads, which a file in the list layout reads without the other fields. Fields
 * inside maps keep their full table type.
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
     * Returns {@code tableType} pruned to the struct fields present in {@code requiredType}, in table order, through
     * arrays too. Any other type that is not a struct on both sides keeps its table type.
     */
    static DataType prune(DataType tableType, DataType requiredType) {
        if (tableType instanceof ArrayType table && requiredType instanceof ArrayType required) {
            return new ArrayType(prune(table.elementType(), required.elementType()), table.containsNull());
        }
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
     * For a struct the file stores, where each field Spark reads comes from: its position in the projected struct, or
     * {@code -1} for a field the file does not carry. {@code children} holds the shape of each struct field that needs
     * one, and is {@code null} elsewhere. For an array, {@code element} is the shape of its elements instead.
     */
    static final class Shape {
        private final int[] slots;
        private final Shape[] children;
        private final Shape element;

        private Shape(int[] slots, Shape[] children, Shape element) {
            this.slots = slots;
            this.children = children;
            this.element = element;
        }

        int slot(int field) {
            return slots[field];
        }

        Shape child(int field) {
            return children[field];
        }

        Shape element() {
            return element;
        }
    }

    /** A native projection, and how each column Spark reads comes out of it. */
    static final class Projection {
        private final Expression expression;
        private final Shape columns;

        private Projection(Expression expression, Shape columns) {
            this.expression = expression;
            this.columns = columns;
        }

        Expression expression() {
            return expression;
        }

        /** The shape of read column {@code column}, or {@code null} if it needs no nulls filled in. */
        Shape shape(int column) {
            return columns == null ? null : columns.child(column);
        }
    }

    /**
     * Builds the native projection that reads {@code readSchema} from a file whose columns are {@code fileSchema}.
     *
     * <p>Fields are matched by name under Spark's case sensitivity, at every level. The file's struct fields are read
     * under their own names, in the order Spark reads them, and only as far down as Spark reads them, through arrays
     * too. A nullable struct field the file does not carry is left out and filled in with nulls through
     * {@link Projection#shape}; a struct keeps its own nulls even when the file carries none of its read fields. Every
     * column of {@code readSchema} must be in the file, which {@code fileName} names in errors.
     */
    static Projection projection(StructType readSchema, StructType fileSchema, boolean caseSensitive, String fileName) {
        Expression root = Expression.root();
        Node node = struct(root, false, readSchema, fileSchema, new Resolution(caseSensitive, fileName), "", 0);
        Expression expression =
                node.expression() != null ? node.expression() : Expression.select(fileSchema.fieldNames(), root);
        return new Projection(expression, node.shape());
    }

    /** The projection of a struct, {@code null} when the file stores it as read, and its shape if it needs one. */
    private record Node(Expression expression, Shape shape) {}

    private record Resolution(boolean caseSensitive, String fileName) {}

    private static Node struct(
            Expression struct,
            boolean nullable,
            StructType read,
            StructType file,
            Resolution resolution,
            String path,
            int arrays) {
        StructField[] fields = read.fields();
        List<String> names = new ArrayList<>(fields.length);
        List<Expression> children = new ArrayList<>(fields.length);
        int[] slots = new int[fields.length];
        Shape[] shapes = new Shape[fields.length];
        boolean reshaped = fields.length != file.length();
        boolean nested = false;
        boolean filled = false;
        for (int i = 0; i < fields.length; i++) {
            String fieldPath = path.isEmpty() ? fields[i].name() : path + "." + fields[i].name();
            int index = fieldIndex(file, fields[i].name(), resolution, fieldPath);
            if (index < 0) {
                if (!fields[i].nullable()) {
                    throw new IllegalArgumentException(String.format(
                            Locale.ROOT,
                            "%s does not carry the non-nullable field %s that the query requires",
                            resolution.fileName(),
                            fieldPath));
                }
                slots[i] = -1;
                reshaped = true;
                filled = true;
                continue;
            }
            reshaped |= index != names.size();
            slots[i] = names.size();
            StructField fileField = file.fields()[index];
            Expression child = Expression.getItem(fileField.name(), struct);
            Node node = value(
                    child,
                    fileField.nullable(),
                    fields[i].dataType(),
                    fileField.dataType(),
                    resolution,
                    fieldPath,
                    arrays);
            if (node.expression() != null) {
                child = node.expression();
                nested = true;
            }
            shapes[i] = node.shape();
            filled |= node.shape() != null;
            names.add(fileField.name());
            children.add(child);
        }
        Shape shape = filled ? new Shape(slots, shapes, null) : null;
        String[] fileNames = names.toArray(new String[0]);
        // An empty `select` loses the struct's own nulls, so an empty struct goes through the masked `pack` below.
        if (!nested && !(nullable && names.isEmpty())) {
            // `select` keeps the struct's own nulls, so use it when no child needs reshaping.
            return new Node(reshaped ? Expression.select(fileNames, struct) : null, shape);
        }
        Expression packed = Expression.pack(fileNames, children.toArray(new Expression[0]), false);
        // `pack` builds a struct with no nulls, so carry over the source struct's nulls.
        return new Node(nullable ? Expression.mask(packed, Expression.isNotNull(struct)) : packed, shape);
    }

    /** The projection of a value the file stores as {@code file}, read as {@code read}. */
    private static Node value(
            Expression value,
            boolean nullable,
            DataType read,
            DataType file,
            Resolution resolution,
            String path,
            int arrays) {
        if (read instanceof StructType readStruct && file instanceof StructType fileStruct) {
            return struct(value, nullable, readStruct, fileStruct, resolution, path, arrays);
        }
        if (read instanceof ArrayType readArray && file instanceof ArrayType fileArray) {
            // A lambda body reads only its own parameter, so each level of arrays names its own.
            String element = "e" + arrays;
            Node node = value(
                    Expression.variable(element),
                    fileArray.containsNull(),
                    readArray.elementType(),
                    fileArray.elementType(),
                    resolution,
                    path,
                    arrays + 1);
            return new Node(
                    node.expression() == null ? null : Expression.listTransform(value, element, node.expression()),
                    node.shape() == null ? null : new Shape(null, null, node.shape()));
        }
        return new Node(null, null);
    }

    private static int fieldIndex(StructType struct, String name, Resolution resolution, String path) {
        int found = -1;
        StructField[] fields = struct.fields();
        for (int i = 0; i < fields.length; i++) {
            if (resolution.caseSensitive()
                    ? fields[i].name().equals(name)
                    : fields[i].name().equalsIgnoreCase(name)) {
                if (found >= 0) {
                    throw new IllegalArgumentException(String.format(
                            Locale.ROOT,
                            "Ambiguous fields %s and %s for %s in %s under Spark's case sensitivity setting",
                            fields[found].name(),
                            fields[i].name(),
                            path,
                            resolution.fileName()));
                }
                found = i;
            }
        }
        return found;
    }
}
