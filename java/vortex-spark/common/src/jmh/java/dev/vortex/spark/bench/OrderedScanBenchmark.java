// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.bench;

import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.io.VortexIo;
import dev.vortex.spark.read.VortexFooterReader;
import dev.vortex.spark.read.VortexPartitionReader;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.spark.paths.SparkPath;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.types.StructType;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/** Compares ordered and unordered decoding of every column of an existing, unsplit Vortex file. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
@Fork(2)
public class OrderedScanBenchmark {
    /** The same representative store file must be supplied for both modes with {@code -p path=...}. */
    @Param({""})
    private String path;

    @Param({"false", "true"})
    private boolean ordered;

    private VortexOptions options;
    private VortexIo io;
    private StructType schema;
    private PartitionedFile file;

    @Setup(Level.Trial)
    public void setUp() throws IOException {
        if (path.isBlank()) {
            throw new IllegalArgumentException("Supply a representative Vortex file with -p path=...");
        }
        Configuration conf = new Configuration();
        Path input = new Path(path);
        var status = input.getFileSystem(conf).getFileStatus(input);
        if (!status.isFile()) {
            throw new IllegalArgumentException("path must name one Vortex file: " + path);
        }
        options = VortexOptions.of(Map.of("vortex.read.ordered", Boolean.toString(ordered)));
        io = VortexIo.create(options, conf);
        schema = VortexFooterReader.inferSchema(List.of(status), io, options, true);
        file = new PartitionedFile(
                InternalRow.empty(),
                SparkPath.fromPathString(status.getPath().toString()),
                0,
                status.getLen(),
                new String[0],
                status.getModificationTime(),
                status.getLen(),
                scala.collection.immutable.Map$.MODULE$.empty());
        System.out.printf("Input: %s, bytes: %d, columns: %d%n", path, status.getLen(), schema.length());
    }

    /** Loading each batch materializes all columns as Arrow; no row-count or projection shortcut is used. */
    @Benchmark
    public long fullScan(Blackhole blackhole) throws IOException {
        long rows = 0;
        try (VortexPartitionReader reader =
                new VortexPartitionReader(file, schema, schema, new StructType(), io, options, new Filter[0], true)) {
            while (reader.next()) {
                var batch = reader.get();
                rows += batch.numRows();
                blackhole.consume(batch);
            }
        }
        return rows;
    }
}
