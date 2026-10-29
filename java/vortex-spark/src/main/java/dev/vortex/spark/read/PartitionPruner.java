// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.Literal;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.filter.AlwaysFalse;
import org.apache.spark.sql.connector.expressions.filter.AlwaysTrue;
import org.apache.spark.sql.connector.expressions.filter.And;
import org.apache.spark.sql.connector.expressions.filter.Not;
import org.apache.spark.sql.connector.expressions.filter.Or;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.FloatType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.TimestampNTZType;
import org.apache.spark.sql.types.TimestampType;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * Skips Vortex files whose Hive-style partition values cannot satisfy the scan's partition predicates.
 *
 * <p>Every row of a file shares the partition values parsed from its path, and {@link VortexPartitionReader} fills the
 * matching columns from those values. A predicate that evaluates to false or null for them therefore rejects every row
 * of the file. Evaluation uses three-valued logic plus an "unknown" state for anything this class cannot evaluate (data
 * columns, unsupported operators or types, unparsable values); a file is kept unless a predicate is definitely false or
 * null. Spark still evaluates the predicates after the scan, so pruning never decides which rows are returned.
 */
final class PartitionPruner {
    private final Predicate[] filters;
    private final Map<String, DataType> columnTypes;

    /**
     * @param filters predicates that reference partition columns
     * @param tableSchema the table schema, which supplies the type each partition value is parsed as
     */
    PartitionPruner(Predicate[] filters, StructType tableSchema) {
        this.filters = filters == null ? new Predicate[0] : filters.clone();
        this.columnTypes = new HashMap<>();
        for (StructField field : tableSchema.fields()) {
            columnTypes.put(field.name(), field.dataType());
        }
    }

    /** Returns the paths, in order, whose partition values may satisfy every filter. */
    List<String> prune(List<String> paths) {
        if (filters.length == 0) {
            return paths;
        }
        return paths.stream()
                .filter(path -> mayMatch(PartitionPathUtils.parsePartitionValues(path)))
                .collect(Collectors.toList());
    }

    /** Returns false only when some filter is definitely false or null for these partition values. */
    boolean mayMatch(Map<String, String> partitionValues) {
        for (Predicate filter : filters) {
            Truth result = evaluate(filter, partitionValues);
            if (result == Truth.FALSE || result == Truth.NULL) {
                return false;
            }
        }
        return true;
    }

    private enum Truth {
        TRUE,
        FALSE,
        NULL,
        UNKNOWN;

        static Truth of(boolean value) {
            return value ? TRUE : FALSE;
        }
    }

    /** A typed operand; {@code value} is null for SQL NULL. */
    private record Operand(DataType type, Object value) {}

    private Truth evaluate(Predicate predicate, Map<String, String> partitionValues) {
        if (predicate instanceof AlwaysTrue) {
            return Truth.TRUE;
        }
        if (predicate instanceof AlwaysFalse) {
            return Truth.FALSE;
        }
        if (predicate instanceof And and) {
            Truth left = evaluate(and.left(), partitionValues);
            Truth right = evaluate(and.right(), partitionValues);
            if (left == Truth.FALSE || right == Truth.FALSE) {
                return Truth.FALSE;
            }
            if (left == Truth.UNKNOWN || right == Truth.UNKNOWN) {
                return Truth.UNKNOWN;
            }
            return left == Truth.NULL || right == Truth.NULL ? Truth.NULL : Truth.TRUE;
        }
        if (predicate instanceof Or or) {
            Truth left = evaluate(or.left(), partitionValues);
            Truth right = evaluate(or.right(), partitionValues);
            if (left == Truth.TRUE || right == Truth.TRUE) {
                return Truth.TRUE;
            }
            if (left == Truth.UNKNOWN || right == Truth.UNKNOWN) {
                return Truth.UNKNOWN;
            }
            return left == Truth.NULL || right == Truth.NULL ? Truth.NULL : Truth.FALSE;
        }
        if (predicate instanceof Not not) {
            return switch (evaluate(not.child(), partitionValues)) {
                case TRUE -> Truth.FALSE;
                case FALSE -> Truth.TRUE;
                case NULL -> Truth.NULL;
                case UNKNOWN -> Truth.UNKNOWN;
            };
        }

        Expression[] children = predicate.children();
        return switch (predicate.name()) {
            case "=", "<>", "!=", ">", ">=", "<", "<=", "<=>" ->
                children.length == 2 ? compare(predicate.name(), children, partitionValues) : Truth.UNKNOWN;
            case "IS_NULL", "IS_NOT_NULL" -> {
                if (children.length != 1) {
                    yield Truth.UNKNOWN;
                }
                Optional<Operand> operand = operand(children[0], partitionValues);
                if (operand.isEmpty()) {
                    yield Truth.UNKNOWN;
                }
                boolean isNull = operand.get().value() == null;
                yield Truth.of(predicate.name().equals("IS_NULL") == isNull);
            }
            case "IN" -> in(children, partitionValues);
            case "STARTS_WITH", "ENDS_WITH", "CONTAINS" ->
                children.length == 2 ? stringMatch(predicate.name(), children, partitionValues) : Truth.UNKNOWN;
            case "BOOLEAN_EXPRESSION" -> {
                if (children.length != 1) {
                    yield Truth.UNKNOWN;
                }
                Optional<Operand> operand = operand(children[0], partitionValues);
                if (operand.isEmpty() || !(operand.get().type() instanceof BooleanType)) {
                    yield Truth.UNKNOWN;
                }
                Object value = operand.get().value();
                yield value == null ? Truth.NULL : Truth.of((Boolean) value);
            }
            default -> Truth.UNKNOWN;
        };
    }

    private Truth compare(String op, Expression[] children, Map<String, String> partitionValues) {
        Optional<Operand> left = operand(children[0], partitionValues);
        Optional<Operand> right = operand(children[1], partitionValues);
        if (left.isEmpty()
                || right.isEmpty()
                || !left.get().type().equals(right.get().type())) {
            return Truth.UNKNOWN;
        }
        Object l = left.get().value();
        Object r = right.get().value();
        if (op.equals("<=>")) {
            return Truth.of(l == null ? r == null : r != null && compareValues(l, r) == 0);
        }
        if (l == null || r == null) {
            return Truth.NULL;
        }
        int cmp = compareValues(l, r);
        return Truth.of(
                switch (op) {
                    case "=" -> cmp == 0;
                    case "<>", "!=" -> cmp != 0;
                    case ">" -> cmp > 0;
                    case ">=" -> cmp >= 0;
                    case "<" -> cmp < 0;
                    default -> cmp <= 0;
                });
    }

    /** SQL {@code IN}: true on a match, else null if the value or any list item is null, else false. */
    private Truth in(Expression[] children, Map<String, String> partitionValues) {
        if (children.length < 2) {
            return Truth.UNKNOWN;
        }
        Optional<Operand> value = operand(children[0], partitionValues);
        if (value.isEmpty()) {
            return Truth.UNKNOWN;
        }
        Object needle = value.get().value();
        boolean sawNull = needle == null;
        boolean matched = false;
        for (int i = 1; i < children.length; i++) {
            Optional<Operand> item = operand(children[i], partitionValues);
            if (item.isEmpty() || !item.get().type().equals(value.get().type())) {
                return Truth.UNKNOWN;
            }
            Object candidate = item.get().value();
            if (candidate == null) {
                sawNull = true;
            } else if (needle != null && compareValues(needle, candidate) == 0) {
                matched = true;
            }
        }
        if (matched) {
            return Truth.TRUE;
        }
        return sawNull ? Truth.NULL : Truth.FALSE;
    }

    private Truth stringMatch(String op, Expression[] children, Map<String, String> partitionValues) {
        Optional<Operand> left = operand(children[0], partitionValues);
        Optional<Operand> right = operand(children[1], partitionValues);
        if (left.isEmpty()
                || right.isEmpty()
                || !DataTypes.StringType.equals(left.get().type())
                || !DataTypes.StringType.equals(right.get().type())) {
            return Truth.UNKNOWN;
        }
        UTF8String l = (UTF8String) left.get().value();
        UTF8String r = (UTF8String) right.get().value();
        if (l == null || r == null) {
            return Truth.NULL;
        }
        return Truth.of(
                switch (op) {
                    case "STARTS_WITH" -> l.startsWith(r);
                    case "ENDS_WITH" -> l.endsWith(r);
                    default -> l.contains(r);
                });
    }

    /**
     * Resolves a column reference to the file's partition value, or a literal to its value. Empty when the operand is
     * neither, references a column that is not in this file's path, or has an unsupported type or unparsable value.
     */
    private Optional<Operand> operand(Expression expression, Map<String, String> partitionValues) {
        if (expression instanceof NamedReference ref) {
            String[] parts = ref.fieldNames();
            if (parts.length != 1) {
                return Optional.empty();
            }
            DataType type = columnTypes.get(parts[0]);
            String raw = partitionValues.get(parts[0]);
            if (type == null || raw == null || valueClass(type) == null) {
                return Optional.empty();
            }
            if (PartitionPathUtils.HIVE_DEFAULT_PARTITION.equals(raw)) {
                return Optional.of(new Operand(type, null));
            }
            try {
                return Optional.of(new Operand(type, parse(type, raw)));
            } catch (NumberFormatException e) {
                return Optional.empty();
            }
        }
        if (expression instanceof Literal<?> literal) {
            DataType type = literal.dataType();
            Class<?> valueClass = valueClass(type);
            Object value = literal.value();
            if (valueClass == null || (value != null && !valueClass.isInstance(value))) {
                return Optional.empty();
            }
            return Optional.of(new Operand(type, value));
        }
        return Optional.empty();
    }

    /**
     * The class of Spark's internal value for each supported type. Collated strings are excluded because they do not
     * compare bytewise.
     */
    private static Class<?> valueClass(DataType type) {
        if (DataTypes.StringType.equals(type)) {
            return UTF8String.class;
        } else if (type instanceof IntegerType || type instanceof DateType) {
            return Integer.class;
        } else if (type instanceof LongType || type instanceof TimestampType || type instanceof TimestampNTZType) {
            return Long.class;
        } else if (type instanceof ShortType) {
            return Short.class;
        } else if (type instanceof ByteType) {
            return Byte.class;
        } else if (type instanceof BooleanType) {
            return Boolean.class;
        } else if (type instanceof FloatType) {
            return Float.class;
        } else if (type instanceof DoubleType) {
            return Double.class;
        }
        return null;
    }

    /** Parses a partition value exactly as {@link PartitionPathUtils#createConstantVector} does. */
    private static Object parse(DataType type, String value) {
        if (DataTypes.StringType.equals(type)) {
            return UTF8String.fromString(value);
        } else if (type instanceof IntegerType || type instanceof DateType) {
            return Integer.parseInt(value);
        } else if (type instanceof LongType || type instanceof TimestampType || type instanceof TimestampNTZType) {
            return Long.parseLong(value);
        } else if (type instanceof ShortType) {
            return Short.parseShort(value);
        } else if (type instanceof ByteType) {
            return Byte.parseByte(value);
        } else if (type instanceof BooleanType) {
            return Boolean.parseBoolean(value);
        } else if (type instanceof FloatType) {
            return Float.parseFloat(value);
        } else {
            return Double.parseDouble(value);
        }
    }

    /** Compares two non-null values of the same type. Floating point follows Spark: NaN = NaN and -0.0 = 0.0. */
    @SuppressWarnings("unchecked")
    private static int compareValues(Object left, Object right) {
        if (left instanceof Float || left instanceof Double) {
            double l = ((Number) left).doubleValue();
            double r = ((Number) right).doubleValue();
            return l == r ? 0 : Double.compare(l, r);
        }
        return ((Comparable<Object>) left).compareTo(right);
    }
}
