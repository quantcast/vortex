// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import org.apache.spark.sql.catalyst.expressions.{
  Alias,
  And,
  Attribute,
  AttributeReference,
  AttributeSet,
  ExprId,
  Expression,
  NamedExpression,
  ProjectionOverSchema
}
import org.apache.spark.sql.catalyst.expressions.SchemaPruning.{
  identifyRootFields,
  pruneSchema
}
import org.apache.spark.sql.catalyst.planning.ScanOperation
import org.apache.spark.sql.catalyst.plans.logical.{
  Filter,
  LogicalPlan,
  Project
}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.datasources.{
  HadoopFsRelation,
  LogicalRelation
}
import org.apache.spark.sql.types.{ArrayType, DataType, MapType, StructType}

import dev.vortex.spark.VortexFileFormat

/** Spark's `SchemaPruning` for V1 Vortex relations, such as catalog tables,
  * which Spark prunes only for Parquet and ORC.
  *
  * It runs just before Spark's rule, which still prunes `_metadata`.
  */
object VortexSchemaPruning extends Rule[LogicalPlan] {

  override def apply(plan: LogicalPlan): LogicalPlan =
    if (!conf.nestedSchemaPruningEnabled) {
      plan
    } else {
      plan.transformDown {
        case op @ ScanOperation(
              projects,
              filtersStayUp,
              filtersPushDown,
              relation: LogicalRelation
            ) =>
          relation.relation match {
            case fs: HadoopFsRelation
                if fs.fileFormat.isInstanceOf[VortexFileFormat] =>
              val filters =
                filtersPushDown.reduceOption(And).toSeq ++ filtersStayUp
              prune(relation, fs, projects, filters).getOrElse(op)
            case _ => op
          }
      }
    }

  private def prune(
      relation: LogicalRelation,
      fs: HadoopFsRelation,
      projects: Seq[NamedExpression],
      filters: Seq[Expression]
  ): Option[LogicalPlan] = {
    val names = relation.output.map(a => a.exprId -> a.name).toMap
    val normalizedProjects =
      normalize(names, projects).map(_.asInstanceOf[NamedExpression])
    val normalizedFilters = normalize(names, filters)
    val rootFields = identifyRootFields(normalizedProjects, normalizedFilters)
    if (rootFields.forall(_.derivedFromAtt)) {
      return None
    }
    // The reader returns maps whole, so only that type keeps the field ordinals right.
    val supported = NestedSchemaPruning.pruneFields(
      pruneSchema(fs.dataSchema, rootFields),
      fs.dataSchema
    )
    val prunedSchema = keepBucketColumns(fs, supported)
    if (countLeaves(prunedSchema) >= countLeaves(fs.dataSchema)) {
      return None
    }

    // Only columns whose type changed are rewritten, so a partition column that shares a data column's name keeps its
    // own type.
    val changed = StructType(
      prunedSchema.filter(f => f.dataType != fs.dataSchema(f.name).dataType)
    )
    val prunedFs = fs.copy(dataSchema = prunedSchema)(fs.sparkSession)
    val output = relation.output.map { a =>
      if (changed.fieldNames.contains(a.name)) {
        a.withDataType(changed(a.name).dataType)
      } else {
        a
      }
    }
    val prunedRelation = relation.copy(relation = prunedFs, output = output)
    prunedRelation.copyTagsFrom(relation)

    val projection =
      ProjectionOverSchema(changed, AttributeSet(relation.output))
    def rewrite(e: Expression): Expression = e.transformDown {
      case projection(projected) => projected
    }
    val child = normalizedFilters
      .map(rewrite)
      .foldLeft[LogicalPlan](prunedRelation)((plan, cond) => Filter(cond, plan))
    val newProjects =
      normalizedProjects.map(rewrite(_).asInstanceOf[NamedExpression])
    Some(Project(restoreNames(newProjects, projects.map(_.name)), child))
  }

  /** Bucket and sort columns keep their table type, which is what the files are
    * bucketed and sorted by.
    */
  private def keepBucketColumns(
      fs: HadoopFsRelation,
      schema: StructType
  ): StructType =
    fs.bucketSpec match {
      case Some(spec) =>
        val keys = spec.bucketColumnNames ++ spec.sortColumnNames
        StructType(schema.map { f =>
          if (keys.exists(conf.resolver(_, f.name))) fs.dataSchema(f.name)
          else f
        })
      case None => schema
    }

  private def normalize(
      names: Map[ExprId, String],
      expressions: Seq[Expression]
  ): Seq[Expression] =
    expressions.map(_.transform {
      case a: AttributeReference if names.contains(a.exprId) =>
        a.withName(names(a.exprId))
    })

  private def restoreNames(
      projects: Seq[NamedExpression],
      names: Seq[String]
  ): Seq[NamedExpression] =
    projects.zip(names).map {
      case (a: Attribute, name) => a.withName(name)
      case (a: Alias, name)     => a.withName(name)
      case (other, _)           => other
    }

  private def countLeaves(dataType: DataType): Int =
    dataType match {
      case array: ArrayType => countLeaves(array.elementType)
      case map: MapType => countLeaves(map.keyType) + countLeaves(map.valueType)
      case struct: StructType => struct.map(f => countLeaves(f.dataType)).sum
      case _                  => 1
    }
}
