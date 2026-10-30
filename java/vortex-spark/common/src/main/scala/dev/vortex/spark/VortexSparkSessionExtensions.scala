// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark

import java.util.{Collections, WeakHashMap}

import org.apache.spark.sql.{
  SparkSessionExtensions,
  SparkSessionExtensionsProvider
}

import dev.vortex.spark.read.VortexSchemaPruning

/** Registers Vortex's optimizer rules. Spark loads it from the classpath, and
  * it can also be listed in `spark.sql.extensions`.
  */
final class VortexSparkSessionExtensions
    extends SparkSessionExtensionsProvider {
  override def apply(extensions: SparkSessionExtensions): Unit =
    if (VortexSparkSessionExtensions.firstRegistration(extensions)) {
      extensions.injectPreCBORule(_ => VortexSchemaPruning)
    }
}

object VortexSparkSessionExtensions {
  private val registered = Collections.synchronizedSet(
    Collections.newSetFromMap(
      new WeakHashMap[SparkSessionExtensions, java.lang.Boolean]()
    )
  )

  /** Spark applies both the classpath provider and `spark.sql.extensions` to
    * the same extensions, so a session listing this class would run the rule
    * twice.
    */
  private def firstRegistration(extensions: SparkSessionExtensions): Boolean =
    registered.add(extensions)
}
