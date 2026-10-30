// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.spark.ArrowUtils;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.MapType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * Matches the types Spark reads against the types a file stores.
 *
 * <p>A column vector takes its type from the file and converts nothing, so a column read as another type would return
 * its stored values under that type: a count of microseconds read as a timestamp, for one. Only conversions Spark's
 * Parquet source also makes, and that the vectors get right as they are, are allowed.
 */
final class ReadTypes {
    // Spark 4 lets Parquet widen decimal scales and read timestamps with a time zone as timestamps without one.
    private static final boolean SPARK_4 =
            !org.apache.spark.package$.MODULE$.SPARK_VERSION().startsWith("3.");

    private ReadTypes() {}

    /**
     * Fails unless every field of {@code readSchema} that the file carries can be read as its type.
     *
     * <p>Fields are matched by name under Spark's case sensitivity, at every level. Only the fields Spark reads are
     * converted, so a column Spark cannot represent fails only a query that reads it. Structs inside arrays and maps
     * are left unchecked.
     *
     * @throws IllegalArgumentException naming {@code fileName}, the field and both types
     */
    static void check(StructType readSchema, List<Field> fileFields, boolean caseSensitive, String fileName) {
        for (StructField field : readSchema.fields()) {
            Field fileField = find(fileFields, field.name(), caseSensitive);
            if (fileField != null) {
                check(field.dataType(), fileField, field.name(), caseSensitive, fileName);
            }
        }
    }

    private static void check(DataType read, Field file, String path, boolean caseSensitive, String fileName) {
        ArrowType type = file.getType();
        if (type instanceof ArrowType.Null) {
            // Every value is null, which reads the same as any type.
            return;
        }
        if (read instanceof StructType struct) {
            if (!(type instanceof ArrowType.Struct)) {
                throw mismatch(read, file, path, fileName);
            }
            for (StructField field : struct.fields()) {
                Field child = find(file.getChildren(), field.name(), caseSensitive);
                if (child != null) {
                    check(field.dataType(), child, path + "." + field.name(), caseSensitive, fileName);
                }
            }
        } else if (read instanceof ArrayType array) {
            if (!(type instanceof ArrowType.List || type instanceof ArrowType.ListView)) {
                throw mismatch(read, file, path, fileName);
            }
            element(array.elementType(), file.getChildren().get(0), path + ".element", caseSensitive, fileName);
        } else if (read instanceof MapType map) {
            if (!(type instanceof ArrowType.Map)) {
                throw mismatch(read, file, path, fileName);
            }
            List<Field> entry = file.getChildren().get(0).getChildren();
            element(map.keyType(), entry.get(0), path + ".key", caseSensitive, fileName);
            element(map.valueType(), entry.get(1), path + ".value", caseSensitive, fileName);
        } else if (!readable(file, read)) {
            throw mismatch(read, file, path, fileName);
        }
    }

    private static void element(DataType read, Field file, String path, boolean caseSensitive, String fileName) {
        // The fields of a struct inside an array or map are matched by position, not by name.
        if (!(read instanceof StructType)) {
            check(read, file, path, caseSensitive, fileName);
        }
    }

    /** Whether a vector of {@code fileField} values reads correctly as {@code read}. */
    private static boolean readable(Field fileField, DataType read) {
        DataType file = leafType(fileField);
        if (file == null) {
            return false;
        }
        // A collation only changes how strings compare.
        if (file.equals(read) || (file instanceof StringType && read instanceof StringType)) {
            return true;
        }
        // Dates are days since the epoch and timestamps microseconds since it. Parquet reads a timestamp of
        // another unit as BIGINT in that unit, while the vector normalizes it to microseconds.
        boolean micros = fileField.getType() instanceof ArrowType.Timestamp timestamp
                && timestamp.getUnit() == TimeUnit.MICROSECOND;
        if ((file.equals(DataTypes.IntegerType) && read.equals(DataTypes.DateType))
                || (file.equals(DataTypes.DateType) && read.equals(DataTypes.IntegerType))
                || (micros && read.equals(DataTypes.LongType))
                || (file.equals(DataTypes.TimestampNTZType) && read.equals(DataTypes.TimestampType))
                || (SPARK_4 && file.equals(DataTypes.TimestampType) && read.equals(DataTypes.TimestampNTZType))) {
            return true;
        }
        // Decimal values are rescaled to the type read, so it only has to keep every digit.
        if (file instanceof DecimalType stored && read instanceof DecimalType wanted) {
            int scaleIncrease = wanted.scale() - stored.scale();
            return (scaleIncrease == 0 || (SPARK_4 && scaleIncrease > 0))
                    && wanted.precision() - stored.precision() >= scaleIncrease;
        }
        return false;
    }

    /** The Spark type of a file leaf, or {@code null} for a nested or unsupported one. */
    private static DataType leafType(Field file) {
        ArrowType type = file.getType();
        if (type instanceof ArrowType.Struct
                || type instanceof ArrowType.List
                || type instanceof ArrowType.ListView
                || type instanceof ArrowType.Map) {
            return null;
        }
        return sparkType(file);
    }

    private static DataType sparkType(Field file) {
        try {
            return ArrowUtils.fromArrowField(file);
        } catch (UnsupportedOperationException e) {
            return null;
        }
    }

    private static IllegalArgumentException mismatch(DataType read, Field file, String path, String fileName) {
        DataType stored = sparkType(file);
        return new IllegalArgumentException(String.format(
                Locale.ROOT,
                "%s stores column %s as %s, which cannot be read as %s",
                fileName,
                path,
                stored != null ? stored.catalogString() : "Arrow " + file.getType(),
                read.catalogString()));
    }

    /**
     * Returns the fields of {@code schema} that filters can be pushed down on for this file: those it carries under the
     * same name, at every level, with the type filter literals are built in.
     */
    static StructType filterable(StructType schema, List<Field> fileFields) {
        List<StructField> present = new ArrayList<>(schema.length());
        for (StructField field : schema.fields()) {
            Field fileField = find(fileFields, field.name(), true);
            if (fileField == null) {
                continue;
            }
            if (field.dataType() instanceof StructType struct) {
                if (fileField.getType() instanceof ArrowType.Struct) {
                    present.add(new StructField(
                            field.name(),
                            filterable(struct, fileField.getChildren()),
                            field.nullable(),
                            field.metadata()));
                }
            } else if (field.dataType() instanceof ArrayType
                    || field.dataType() instanceof MapType
                    || literalType(fileField, field.dataType())) {
                present.add(field);
            }
        }
        return new StructType(present.toArray(new StructField[0]));
    }

    /** Whether the file stores {@code read} values as the native scan compares filter literals of that type. */
    private static boolean literalType(Field file, DataType read) {
        if (!read.equals(leafType(file))) {
            return false;
        }
        // Timestamp literals are microseconds in UTC, or without a time zone, and the scan does not convert units.
        return !(file.getType() instanceof ArrowType.Timestamp timestamp)
                || (timestamp.getUnit() == TimeUnit.MICROSECOND
                        && (timestamp.getTimezone() == null
                                || timestamp.getTimezone().equals("UTC")));
    }

    private static Field find(List<Field> fields, String name, boolean caseSensitive) {
        for (Field field : fields) {
            if (caseSensitive ? field.getName().equals(name) : field.getName().equalsIgnoreCase(name)) {
                return field;
            }
        }
        return null;
    }
}
