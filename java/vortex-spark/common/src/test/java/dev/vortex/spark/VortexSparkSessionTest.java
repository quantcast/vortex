// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

final class VortexSparkSessionTest {
    @Test
    void everyWorkerThreadCountIsCheckedAfterThePoolIsSized() {
        VortexSparkSession.get(VortexOptions.empty());

        for (String value : new String[] {"many", "-1"}) {
            VortexOptions options = VortexOptions.of(Map.of(VortexSparkSession.WORKER_THREADS_OPTION, value));
            IllegalArgumentException failure =
                    assertThrows(IllegalArgumentException.class, () -> VortexSparkSession.get(options));
            assertTrue(failure.getMessage().contains(VortexSparkSession.WORKER_THREADS_OPTION), failure::getMessage);
        }
    }
}
