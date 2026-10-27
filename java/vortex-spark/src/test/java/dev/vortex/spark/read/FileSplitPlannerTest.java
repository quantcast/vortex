// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.spark.VortexFilePartition.RowRange;
import dev.vortex.spark.read.FileSplitPlanner.FileSize;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

final class FileSplitPlannerTest {

    @Test
    @DisplayName("Row ranges tile the file once on row-block boundaries; malformed options fail naming the option")
    void rangesTileTheFile() {
        assertFalse(FileSplitPlanner.enabled(Map.of()));
        assertTrue(FileSplitPlanner.enabled(Map.of(FileSplitPlanner.SPLIT_FILES_OPTION, "TRUE")));
        assertNamesOption(
                FileSplitPlanner.SPLIT_FILES_OPTION,
                () -> FileSplitPlanner.enabled(Map.of(FileSplitPlanner.SPLIT_FILES_OPTION, "yes")));
        List<FileSize> files = List.of(new FileSize(100_000, 1_000_000));
        assertEquals(
                FileSplitPlanner.MIN_SPLIT_BYTES,
                FileSplitPlanner.maxSplitBytes(Map.of(FileSplitPlanner.MAX_SPLIT_BYTES_OPTION, "1048576"), files));
        for (String bad : List.of("abc", "", "1048575", "-1")) {
            assertNamesOption(
                    FileSplitPlanner.MAX_SPLIT_BYTES_OPTION,
                    () -> FileSplitPlanner.maxSplitBytes(Map.of(FileSplitPlanner.MAX_SPLIT_BYTES_OPTION, bad), files));
        }

        // 300 KB of a 1 MB, 100k-row file is 30k rows, rounded down to three 8192-row blocks.
        assertEquals(
                List.of(
                        new RowRange(0, 24_576),
                        new RowRange(24_576, 49_152),
                        new RowRange(49_152, 73_728),
                        new RowRange(73_728, 98_304),
                        new RowRange(98_304, 100_000)),
                FileSplitPlanner.splitRows(new FileSize(100_000, 1_000_000), 300_000));

        assertEquals(List.of(), FileSplitPlanner.splitRows(new FileSize(100_000, 1_000_000), 1_000_000));

        // A budget below one row block still yields whole blocks rather than tiny ranges.
        assertEquals(
                FileSplitPlanner.ROW_BLOCK_SIZE,
                FileSplitPlanner.splitRows(new FileSize(20_000, 1_000_000), 10)
                        .get(0)
                        .rows());
    }

    private static void assertNamesOption(String option, Executable call) {
        String message = assertThrows(IllegalArgumentException.class, call).getMessage();
        assertTrue(message.contains(option), message);
    }
}
