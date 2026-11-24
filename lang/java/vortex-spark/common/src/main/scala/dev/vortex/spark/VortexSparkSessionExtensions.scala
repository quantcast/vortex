// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark

import java.util.{Collections, WeakHashMap}

import org.apache.spark.sql.{
  SparkSessionExtensions,
  SparkSessionExtensionsProvider
}
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.catalyst.rules.Rule

import dev.vortex.spark.read.{NestedDynamicPruning, VortexSchemaPruning}

/** Registers Vortex's optimizer rules. Spark loads it from the classpath, and
  * it can also be listed in `spark.sql.extensions`.
  *
  * The extensions add `VortexSchemaPruning`, and the `NestedDynamicPruning`
  * rule, which stops dynamic pruning filters from nesting into each other.
  */
final class VortexSparkSessionExtensions
    extends SparkSessionExtensionsProvider {
  override def apply(extensions: SparkSessionExtensions): Unit =
    if (VortexSparkSessionExtensions.firstRegistration(extensions)) {
      extensions.injectPreCBORule(_ => VortexSchemaPruning)
      extensions.injectPostHocResolutionRule { session =>
        // No injection point runs after Spark's own dynamic pruning, but the
        // user provided optimizations do. Spark builds the analyzer of a
        // session before it optimizes a query of that session, so the rule is
        // in time.
        NestedDynamicPruning.install(session)
        VortexSparkSessionExtensions.InstallNestedDynamicPruning
      }
    }
}

object VortexSparkSessionExtensions {
  private val registered = Collections.synchronizedSet(
    Collections.newSetFromMap(
      new WeakHashMap[SparkSessionExtensions, java.lang.Boolean]()
    )
  )

  /** Spark applies both the classpath provider and `spark.sql.extensions` to
    * the same extensions, so a session listing this class would run the rules
    * twice.
    */
  private def firstRegistration(extensions: SparkSessionExtensions): Boolean =
    registered.add(extensions)

  /** The rule Spark asks the extension for. It changes no plan. */
  private object InstallNestedDynamicPruning extends Rule[LogicalPlan] {
    override def apply(plan: LogicalPlan): LogicalPlan = plan
  }
}
