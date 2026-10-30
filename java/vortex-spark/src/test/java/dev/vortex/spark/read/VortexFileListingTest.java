// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

final class VortexFileListingTest {

    @ParameterizedTest
    @CsvSource({
        "_SUCCESS, true",
        "_temporary, true",
        ".part-0.vortex.crc, true",
        "part-0.vortex._COPYING_, true",
        "part-0.vortex, false",
        "_k=v, false",
        "_metadata, false",
        "_common_metadata.vortex, false",
    })
    void matchesSparkHiddenFileRule(String name, boolean hidden) {
        assertEquals(hidden, VortexFileListing.isHiddenName(name));
    }
}
