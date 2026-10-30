// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.execution.vectorized.ConstantColumnVector;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.apache.spark.sql.vectorized.ColumnarMap;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * A struct column whose fields the file only partly carries: the fields it lacks read as null, the rest come from the
 * struct the file returned.
 */
final class NullFilledStructVector extends ColumnVector {
    private final ColumnVector struct;
    private final ColumnVector[] children;
    private final List<ColumnVector> owned;

    private NullFilledStructVector(
            StructType type, ColumnVector struct, ColumnVector[] children, List<ColumnVector> owned) {
        super(type);
        this.struct = struct;
        this.children = children;
        this.owned = owned;
    }

    /** Lays {@code struct}, which the file returned, out as {@code type} by {@code shape}. */
    static ColumnVector of(ColumnVector struct, StructType type, NestedSchemaPruning.Shape shape, int rowCount) {
        List<ColumnVector> owned = new ArrayList<>();
        ColumnVector[] children = new ColumnVector[type.length()];
        for (int i = 0; i < children.length; i++) {
            int slot = shape.slot(i);
            if (slot < 0) {
                ConstantColumnVector nulls = new ConstantColumnVector(rowCount, type.fields()[i].dataType());
                nulls.setNull();
                owned.add(nulls);
                children[i] = nulls;
            } else if (shape.child(i) != null) {
                ColumnVector child =
                        of(struct.getChild(slot), (StructType) type.fields()[i].dataType(), shape.child(i), rowCount);
                owned.add(child);
                children[i] = child;
            } else {
                children[i] = struct.getChild(slot);
            }
        }
        return new NullFilledStructVector(type, struct, children, owned);
    }

    @Override
    public void close() {
        owned.forEach(ColumnVector::close);
        struct.close();
    }

    @Override
    public boolean hasNull() {
        return struct.hasNull();
    }

    @Override
    public int numNulls() {
        return struct.numNulls();
    }

    @Override
    public boolean isNullAt(int rowId) {
        return struct.isNullAt(rowId);
    }

    @Override
    public ColumnVector getChild(int ordinal) {
        return children[ordinal];
    }

    @Override
    public boolean getBoolean(int rowId) {
        throw notAStruct();
    }

    @Override
    public byte getByte(int rowId) {
        throw notAStruct();
    }

    @Override
    public short getShort(int rowId) {
        throw notAStruct();
    }

    @Override
    public int getInt(int rowId) {
        throw notAStruct();
    }

    @Override
    public long getLong(int rowId) {
        throw notAStruct();
    }

    @Override
    public float getFloat(int rowId) {
        throw notAStruct();
    }

    @Override
    public double getDouble(int rowId) {
        throw notAStruct();
    }

    @Override
    public ColumnarArray getArray(int rowId) {
        throw notAStruct();
    }

    @Override
    public ColumnarMap getMap(int rowId) {
        throw notAStruct();
    }

    @Override
    public Decimal getDecimal(int rowId, int precision, int scale) {
        throw notAStruct();
    }

    @Override
    public UTF8String getUTF8String(int rowId) {
        throw notAStruct();
    }

    @Override
    public byte[] getBinary(int rowId) {
        throw notAStruct();
    }

    private static UnsupportedOperationException notAStruct() {
        return new UnsupportedOperationException("a struct column has only fields");
    }
}
