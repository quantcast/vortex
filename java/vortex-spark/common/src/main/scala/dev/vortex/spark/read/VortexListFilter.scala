// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import java.util.Optional

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.types.{DataType, StringType, StructType}

import dev.vortex.api.{Expression => VortexExpression}
import dev.vortex.api.Expression.BinaryOp

/** Carries a Catalyst predicate to executors without serializing native
  * handles. Spark retains the original predicate as a residual filter.
  */
final case class VortexListFilter(predicate: Expression)
    extends NativePredicate {
  override def references(): Array[Array[String]] =
    VortexListFilter.columnPaths(predicate).map(_.toArray).toArray

  def convert(schema: StructType): Optional[VortexExpression] =
    Optional.ofNullable(VortexListFilter.translate(predicate, schema).orNull)
}

object VortexListFilter {
  def from(expression: Expression, schema: StructType): Seq[NativePredicate] =
    expression match {
      case And(left, right) => from(left, schema) ++ from(right, schema)
      case _: ArrayContains | _: ArrayExists =>
        if (translate(expression, schema).isDefined)
          Seq(VortexListFilter(expression))
        else Seq.empty
      case _ => Seq.empty
    }

  /** Nested field names of a column read, e.g. `s.tags` as `Seq("s", "tags")`.
    */
  private def columnPath(expression: Expression): Option[Seq[String]] =
    expression match {
      case attribute: AttributeReference => Some(Seq(attribute.name))
      case field: GetStructField         =>
        columnPath(field.child).map(_ :+ field.childSchema(field.ordinal).name)
      case _ => None
    }

  private def columnPaths(expression: Expression): Seq[Seq[String]] =
    columnPath(expression) match {
      case Some(path) => Seq(path)
      case None       => expression.children.flatMap(columnPaths)
    }

  private object ColumnPath {
    def unapply(expression: Expression): Option[Seq[String]] =
      columnPath(expression)
  }

  private def typeAt(schema: StructType, path: Seq[String]): Option[DataType] =
    path.foldLeft(Option[DataType](schema)) {
      case (Some(struct: StructType), name) =>
        struct.fields.find(_.name == name).map(_.dataType)
      case _ => None
    }

  /** Non-binary string collations compare differently in the native scan. */
  private def comparable(dataType: DataType): Boolean =
    SparkFilterToVortexExpression.comparableType(dataType) &&
      (!dataType.isInstanceOf[StringType] || dataType == StringType)

  private def translate(
      expression: Expression,
      schema: StructType,
      variable: Option[NamedLambdaVariable] = None
  ): Option[VortexExpression] = {
    def child(e: Expression): Option[VortexExpression] =
      translate(e, schema, variable)
    def binary(left: Expression, right: Expression, op: BinaryOp) =
      if (!comparable(left.dataType) || !comparable(right.dataType)) None
      else
        for { l <- child(left); r <- child(right) } yield VortexExpression
          .binary(op, l, r)

    expression match {
      case column @ ColumnPath(path) =>
        // Lambdas may read only their own parameter.
        if (variable.isEmpty && typeAt(schema, path).contains(column.dataType))
          Some(VortexExpression.column(path.toArray))
        else None
      case parameter: NamedLambdaVariable
          if variable.exists(_.exprId == parameter.exprId) =>
        Some(VortexExpression.variable(parameter.exprId.toString))
      case field: GetStructField =>
        child(field.child).map(
          VortexExpression.getItem(field.childSchema(field.ordinal).name, _)
        )
      case literal: Literal =>
        Option(
          SparkFilterToVortexExpression
            .literal(literal.value, literal.dataType)
            .orElse(null)
        )
      case ArrayContains(list, needle) if comparable(needle.dataType) =>
        for {
          l <- child(list)
          n <- child(needle)
        } yield VortexExpression.listContains(l, n, true)
      case exists: ArrayExists if variable.isEmpty =>
        exists.function match {
          case LambdaFunction(body, Seq(parameter: NamedLambdaVariable), _) =>
            for {
              list <- child(exists.argument)
              predicate <- translate(body, schema, Some(parameter))
            } yield VortexExpression.listContains(
              VortexExpression
                .listTransform(list, parameter.exprId.toString, predicate),
              VortexExpression.literal(true),
              exists.followThreeValuedLogic
            )
          case _ => None
        }
      case EqualTo(left, right)            => binary(left, right, BinaryOp.EQ)
      case GreaterThan(left, right)        => binary(left, right, BinaryOp.GT)
      case GreaterThanOrEqual(left, right) => binary(left, right, BinaryOp.GTE)
      case LessThan(left, right)           => binary(left, right, BinaryOp.LT)
      case LessThanOrEqual(left, right)    => binary(left, right, BinaryOp.LTE)
      case And(left, right)                =>
        for { l <- child(left); r <- child(right) } yield VortexExpression
          .and(l, r)
      case Or(left, right) =>
        for { l <- child(left); r <- child(right) } yield VortexExpression
          .or(l, r)
      case Not(value)       => child(value).map(VortexExpression.not)
      case IsNull(value)    => child(value).map(VortexExpression.isNull)
      case IsNotNull(value) => child(value).map(VortexExpression.isNotNull)
      case _                => None
    }
  }
}
