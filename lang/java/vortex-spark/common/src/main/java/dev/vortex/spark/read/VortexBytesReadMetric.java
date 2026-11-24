// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import org.apache.spark.sql.connector.metric.CustomSumMetric;

/** Source bytes successfully read by Vortex, including footers and read-ahead. */
public final class VortexBytesReadMetric extends CustomSumMetric {
    static final String NAME = "vortexBytesRead";

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public String description() {
        return "Vortex source bytes read";
    }
}
