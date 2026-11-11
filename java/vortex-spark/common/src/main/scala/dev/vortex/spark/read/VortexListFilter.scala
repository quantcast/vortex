// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import java.util.Optional

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.types.StructType

import dev.vortex.api.{Expression => VortexExpression}
import dev.vortex.api.Expression.BinaryOp

/** Carries a Catalyst predicate to executors without serializing native
  * handles. Spark retains the original predicate as a residual filter.
  */
final case class VortexListFilter(predicate: Expression)
    extends NativePredicate {
  override def references(): Array[Array[String]] =
    predicate.references.toSeq.map(attribute => Array(attribute.name)).toArray

  def convert(schema: StructType): Optional[VortexExpression] =
    VortexListFilter.translate(predicate, schema) match {
      case Some(expression) => Optional.of(expression)
      case None             => Optional.empty()
    }
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

  private def translate(
      expression: Expression,
      schema: StructType,
      variable: Option[NamedLambdaVariable] = None
  ): Option[VortexExpression] = {
    def child(e: Expression): Option[VortexExpression] =
      translate(e, schema, variable)
    def comparable(e: Expression): Boolean =
      SparkFilterToVortexExpression.literal(null, e.dataType).isPresent &&
        (!e.dataType.isInstanceOf[org.apache.spark.sql.types.StringType] ||
          e.dataType == org.apache.spark.sql.types.DataTypes.StringType)
    def binary(left: Expression, right: Expression, op: BinaryOp) =
      for {
        _ <- if (comparable(left) && comparable(right)) Some(()) else None
        l <- child(left)
        r <- child(right)
      } yield VortexExpression.binary(op, l, r)

    expression match {
      case attribute: AttributeReference
          if variable.isEmpty && schema.fields.exists(field =>
            field.name == attribute.name && field.dataType == attribute.dataType
          ) =>
        Some(VortexExpression.column(attribute.name))
      case parameter: NamedLambdaVariable
          if variable.exists(_.exprId == parameter.exprId) =>
        Some(VortexExpression.variable(parameter.exprId.toString))
      case field: GetStructField =>
        child(field.child).map(
          VortexExpression.getItem(
            field.child.dataType
              .asInstanceOf[StructType]
              .fields(field.ordinal)
              .name,
            _
          )
        )
      case literal: Literal =>
        val converted =
          SparkFilterToVortexExpression.literal(literal.value, literal.dataType)
        if (converted.isPresent) Some(converted.get()) else None
      case ArrayContains(list, needle) if comparable(needle) =>
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
        for { l <- child(left); r <- child(right) } yield VortexExpression.and(
          l,
          r
        )
      case Or(left, right) =>
        for { l <- child(left); r <- child(right) } yield VortexExpression.or(
          l,
          r
        )
      case Not(value)       => child(value).map(VortexExpression.not)
      case IsNull(value)    => child(value).map(VortexExpression.isNull)
      case IsNotNull(value) => child(value).map(VortexExpression.isNotNull)
      case _                => None
    }
  }
}
