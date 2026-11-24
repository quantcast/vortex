// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.apache.spark.sql.vectorized.ColumnarMap;
import org.apache.spark.unsafe.types.UTF8String;

/** An array column whose elements the file only partly carries, laid out as Spark reads them. */
final class NullFilledArrayVector extends ColumnVector {
    private final VortexArrowColumnVector array;
    private final ColumnVector elements;

    private NullFilledArrayVector(ArrayType type, VortexArrowColumnVector array, ColumnVector elements) {
        super(type);
        this.array = array;
        this.elements = elements;
    }

    /**
     * Lays {@code array}, which the file returned, out as {@code type}, with its elements laid out by {@code shape}.
     */
    static ColumnVector of(ColumnVector array, ArrayType type, NestedSchemaPruning.Shape shape) {
        VortexArrowColumnVector arrow = (VortexArrowColumnVector) array;
        ColumnVector elements = NullFilledStructVector.fill(
                arrow.arrayElements(), type.elementType(), shape, arrow.arrayElementCount());
        return new NullFilledArrayVector(type, arrow, elements);
    }

    @Override
    public void close() {
        elements.close();
        array.close();
    }

    @Override
    public boolean hasNull() {
        return array.hasNull();
    }

    @Override
    public int numNulls() {
        return array.numNulls();
    }

    @Override
    public boolean isNullAt(int rowId) {
        return array.isNullAt(rowId);
    }

    @Override
    public ColumnarArray getArray(int rowId) {
        if (array.isNullAt(rowId)) return null;
        return new ColumnarArray(
                elements, array.arrayOffset(rowId), array.getArray(rowId).numElements());
    }

    @Override
    public boolean getBoolean(int rowId) {
        throw notAnArray();
    }

    @Override
    public byte getByte(int rowId) {
        throw notAnArray();
    }

    @Override
    public short getShort(int rowId) {
        throw notAnArray();
    }

    @Override
    public int getInt(int rowId) {
        throw notAnArray();
    }

    @Override
    public long getLong(int rowId) {
        throw notAnArray();
    }

    @Override
    public float getFloat(int rowId) {
        throw notAnArray();
    }

    @Override
    public double getDouble(int rowId) {
        throw notAnArray();
    }

    @Override
    public ColumnarMap getMap(int rowId) {
        throw notAnArray();
    }

    @Override
    public Decimal getDecimal(int rowId, int precision, int scale) {
        throw notAnArray();
    }

    @Override
    public UTF8String getUTF8String(int rowId) {
        throw notAnArray();
    }

    @Override
    public byte[] getBinary(int rowId) {
        throw notAnArray();
    }

    @Override
    public ColumnVector getChild(int ordinal) {
        throw notAnArray();
    }

    private static UnsupportedOperationException notAnArray() {
        return new UnsupportedOperationException("an array column has only elements");
    }
}
