// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import java.util.Arrays

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types._
import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.Test

import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.{
  ArrowType,
  Field,
  FieldType
}

final class VortexListFilterTest {
  private def arrayField(name: String, element: Field): Field =
    new Field(
      name,
      FieldType.nullable(ArrowType.List.INSTANCE),
      Arrays.asList(element)
    )

  private def pushed(
      filter: NativePredicate,
      schema: StructType,
      fields: java.util.List[Field]
  ): Boolean =
    VortexPartitionReader
      .buildFilterExpression(Array.empty[Filter], Array(filter), schema, fields)
      .isPresent

  @Test
  def executorBuildsListPredicate(): Unit = {
    // A dot here is part of the field name, not a nested path.
    val arrayType = ArrayType(StringType, true)
    val schema = new StructType().add("tag.values", arrayType)
    val attribute = AttributeReference("tag.values", arrayType)()
    val filter = VortexListFilter(ArrayContains(attribute, Literal("sports")))
    val fields = Arrays.asList(
      arrayField(
        "tag.values",
        Field.nullable("element", ArrowType.Utf8.INSTANCE)
      )
    )
    assertTrue(
      pushed(filter, schema, fields)
    )
  }

  @Test
  def listInsideStructIsCheckedByItsOwnPath(): Unit = {
    val tags = ArrayType(StringType, true)
    val struct = new StructType().add("tags", tags).add("other", IntegerType)
    val schema = new StructType().add("s", struct)
    val attribute = AttributeReference("s", struct)()
    val filter = VortexListFilter(
      ArrayContains(GetStructField(attribute, 0), Literal("sports"))
    )
    assertArrayEquals(
      Array[AnyRef](Array("s", "tags")),
      filter.references().asInstanceOf[Array[AnyRef]]
    )
    // The file lacks s.other, which the predicate never reads.
    val fields = Arrays.asList(
      new Field(
        "s",
        FieldType.nullable(ArrowType.Struct.INSTANCE),
        Arrays.asList(
          arrayField("tags", Field.nullable("element", ArrowType.Utf8.INSTANCE))
        )
      )
    )
    assertTrue(
      pushed(filter, schema, fields)
    )
  }

  @Test
  def executorChecksNestedPhysicalTypes(): Unit = {
    val element = new StructType().add("price", IntegerType)
    val arrayType = ArrayType(element, true)
    val schema = new StructType().add("items", arrayType)
    val attribute = AttributeReference("items", arrayType)()
    val parameter = NamedLambdaVariable("x", element, true)
    val predicate = ArrayExists(
      attribute,
      LambdaFunction(
        GreaterThan(GetStructField(parameter, 0), Literal(100)),
        Seq(parameter)
      )
    )
    val filter = VortexListFilter(predicate)
    def fields(price: Field) = Arrays.asList(
      arrayField(
        "items",
        new Field(
          "element",
          FieldType.nullable(ArrowType.Struct.INSTANCE),
          Arrays.asList(price)
        )
      )
    )
    assertTrue(
      pushed(
        filter,
        schema,
        fields(Field.nullable("price", new ArrowType.Int(32, true)))
      )
    )
    assertFalse(
      pushed(
        filter,
        schema,
        fields(Field.nullable("price", new ArrowType.Int(64, true)))
      )
    )
  }

  @Test
  def timestampUnitsRemainCheckedInsideLists(): Unit = {
    val arrayType = ArrayType(TimestampType, true)
    val schema = new StructType().add("times", arrayType)
    val fields = Arrays.asList(
      arrayField(
        "times",
        Field.nullable(
          "element",
          new ArrowType.Timestamp(TimeUnit.MILLISECOND, "UTC")
        )
      )
    )
    assertEquals(
      NullType,
      ReadTypes
        .filterable(schema, fields, Arrays.asList(Array("times")))
        .apply("times")
        .dataType
    )
  }

  @Test
  def floatingPointComparisonsStayInSpark(): Unit = {
    val arrayType = ArrayType(DoubleType, true)
    val schema =
      new StructType().add("values", arrayType).add("needle", DoubleType)
    val values = AttributeReference("values", arrayType)()
    val needle = AttributeReference("needle", DoubleType)()
    assertTrue(
      VortexListFilter.from(ArrayContains(values, needle), schema).isEmpty
    )
    val parameter = NamedLambdaVariable("x", DoubleType, true)
    assertTrue(
      VortexListFilter
        .from(
          ArrayExists(
            values,
            LambdaFunction(EqualTo(parameter, parameter), Seq(parameter))
          ),
          schema
        )
        .isEmpty
    )
  }

  @Test
  def correlatedLambdasStayInSpark(): Unit = {
    val arrayType = ArrayType(IntegerType, true)
    val schema =
      new StructType().add("values", arrayType).add("threshold", IntegerType)
    val values = AttributeReference("values", arrayType)()
    val threshold = AttributeReference("threshold", IntegerType)()
    val parameter = NamedLambdaVariable("x", IntegerType, true)
    assertTrue(
      VortexListFilter
        .from(
          ArrayExists(
            values,
            LambdaFunction(GreaterThan(parameter, threshold), Seq(parameter))
          ),
          schema
        )
        .isEmpty
    )
  }
}
