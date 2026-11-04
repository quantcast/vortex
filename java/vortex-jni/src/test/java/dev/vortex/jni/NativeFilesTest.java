// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.arrow.ArrowAllocation;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

public final class NativeFilesTest {
    @TempDir
    Path tempDir;

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    public void listedUrisAddressTheWrittenFiles(boolean useUri) throws IOException {
        Path directory = Files.createDirectories(tempDir.resolve("list delete %20 Côte+"));
        Path literalPercent = directory.resolve("literal%20name.vortex");
        Path decodedSibling = directory.resolve("literal name.vortex");
        Set<Path> files = Set.of(literalPercent, decodedSibling, directory.resolve("Côte+%?#.vortex"));
        Session session = Session.create();
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Schema schema = new Schema(List.of(Field.notNullable("value", new ArrowType.Int(64, true))));
        for (Path file : files) {
            VortexWriter.builder(session, pathOrUri(file, useUri), schema, allocator)
                    .build()
                    .close();
        }

        Map<Path, String> listed = NativeFiles.listFiles(session, pathOrUri(directory, useUri), Map.of()).stream()
                .collect(Collectors.toMap(uri -> Path.of(URI.create(uri)), uri -> uri));
        assertEquals(files, listed.keySet());

        NativeFiles.delete(session, new String[] {listed.get(literalPercent)}, Map.of());
        assertFalse(Files.exists(literalPercent));
        assertTrue(Files.exists(decodedSibling), "deleting a percent-literal key must preserve its decoded sibling");
        assertTrue(Files.exists(directory.resolve("Côte+%?#.vortex")));

        String[] remaining = files.stream()
                .filter(file -> !file.equals(literalPercent))
                .map(file -> pathOrUri(file, useUri))
                .toArray(String[]::new);
        NativeFiles.delete(session, remaining, Map.of());
        for (Path file : files) {
            assertFalse(Files.exists(file), "delete must address the same file as the writer");
        }
    }

    private static String pathOrUri(Path path, boolean useUri) {
        return useUri ? path.toUri().toString() : path.toString();
    }
}
