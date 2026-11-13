# vortex-spark

A Spark file data source for reading and writing [Vortex](https://vortex.dev) files. It
registers itself as `vortex` and supports the DataFrame API and Spark SQL without session
extensions or custom catalogs.

Choose the artifact matching both Spark and Scala:

| Artifact                           | Spark          | Scala |
|------------------------------------|----------------|-------|
| `dev.vortex:vortex-spark-3.5_2.12` | 3.5.x          | 2.12  |
| `dev.vortex:vortex-spark-3.5_2.13` | 3.5.x          | 2.13  |
| `dev.vortex:vortex-spark-4.0_2.13` | 4.0.x to 4.3.x | 2.13  |

Use the `all` classifier JAR. It bundles the Vortex JNI bindings, native libraries, and
relocated Arrow, Guava, and Jackson dependencies. The unclassified thin JAR is not usable by
itself.

For example, with version `VERSION`:

```shell
spark-shell --jars https://repo1.maven.org/maven2/dev/vortex/vortex-spark-4.0_2.13/VERSION/vortex-spark-4.0_2.13-VERSION-all.jar
```

Or configure it on the session builder, for example in PySpark:

```python
spark = (
    SparkSession.builder
    .config("spark.jars", "/path/to/vortex-spark-4.0_2.13-VERSION-all.jar")
    .getOrCreate()
)
```

Or add the classified artifact to a JVM project:

```kotlin
implementation("dev.vortex:vortex-spark-4.0_2.13:VERSION:all")
```

Spark's `--packages` flag cannot select the `all` classifier, so pass the classified JAR with
`--jars` instead.

## Usage

```java
df.write()
    .format("vortex")
    .mode(SaveMode.Overwrite)
    .save("/path/to/output");

Dataset<Row> result = spark.read()
    .format("vortex")
    .load("/path/to/output");
```

Spark discovers files, Hive-style partitions, and storage credentials through its Hadoop
configuration. This also enables standard file-source options such as `pathGlobFilter` and
`recursiveFileLookup`. All file content is read and written through Hadoop streams.

Only files that end with `.vortex` belong to a Vortex dataset. Writes produce that extension, and
reads skip every other file.

```sql
CREATE TABLE student (id INT, name STRING) USING vortex;
INSERT INTO student VALUES (1, 'Alice'), (2, 'Bob');
SELECT * FROM student;

SELECT * FROM vortex.`/path/to/output`;
```

`CREATE TABLE ... USING vortex` and direct path queries work on Spark 3.5 and 4.x without
catalog configuration. Direct path queries are read-only, as with Spark's built-in formats.
Catalog tables read and write through the V1 file format;
`spark.sql.sources.useV1SourceList=vortex` puts every path on it, which is what the `_metadata`
column and dynamic partition pruning need.

## Read ordering

Set `.option("vortex.read.ordered", true)` to preserve a file's row order within each
Spark file split. The default is `false`, allowing batches to arrive as decoding finishes.
This does not establish ordering across Spark partitions or files.

The option is case-insensitive and works with both V1 and V2 reads. Session defaults use
`vortex.read.ordered`; cluster defaults can use `spark.datasource.vortex.read.ordered`.

## Read coalescing

Nearby storage reads can be merged to reduce requests. Set the maximum gap and merged span in bytes:

```java
Dataset<Row> result = spark.read()
    .format("vortex")
    .option("vortex.read.coalesce.distance", 1048576)
    .option("vortex.read.coalesce.maxSize", 16777216)
    .load("s3a://bucket/path");
```

The defaults are a 1 MiB gap and a 16 MiB maximum merged span. The gap must be non-negative
(zero merges only adjacent or overlapping reads); the maximum span must be positive. Both accept
integers up to 2147483647 bytes. The native reader caps maximums near that limit to leave room
for alignment padding (up to 64 KiB minus one byte). Individual reads can exceed the maximum
merged span, and merged reads may include leading alignment padding.
A wider gap trades fewer storage requests for more bytes read, including segments already held
in the file's segment cache.

Options are case-insensitive and work with both V1 and V2 reads. Session defaults use the same
names; cluster defaults can use `spark.datasource.vortex.read.coalesce.distance` and
`spark.datasource.vortex.read.coalesce.maxSize`.

Java callers supplying their own `NativeReadable` can override `coalesceDistance()` and
`coalesceMaxSize()`. Native code captures these limits once when registering each source.

## Direct writer summaries

Callers using `VortexOutputWriter` directly with their own `NativeWritable` can obtain an
immutable, serializable summary after a successful close, without reopening the file:

```java
writer.close();
VortexOutputWriterSummary summary = writer.summary();
long bytes = summary.fileSize();
long rows = summary.rowCount();
```

The summary also reports the number of batches submitted, the largest batch's Arrow buffer
capacities (including nested buffers), and the maximum native `bufferedBytes()` sampled after
each batch. These memory measurements exclude some retained allocations and can miss native
peaks between samples; they are not a bound on process memory. No summary is available before
close or after a failed write or close. Repeated closes retain the same summary.

## Benchmarks

JMH benchmarks live in `common/src/jmh/java`. They measure the per-file open cost of the
framework's file-splitting model and the footer reads behind statistics and `COUNT(*)` pushdown.
Run them for one variant and pass JMH arguments through `-PjmhArgs`:

```bash
cd java
./gradlew :vortex-spark-4.0_2.13:jmh -PjmhArgs="SparkScanBenchmark -p fileCount=1,128"
./gradlew :vortex-spark-4.0_2.13:jmh -PjmhArgs="FooterReadBenchmark -f 1 -wi 2 -i 5"
```

Compare ordered and unordered decoding on an existing Vortex file, running once for a
current bid-hour event file and once for a current featoid bucket file:

For performance measurements, first stage an optimized JNI library from the same revision
under `vortex-jni/src/main/resources/native/<platform>/`. The environment setting below
keeps Gradle from replacing it with its default debug build.

```bash
VORTEX_SKIP_MAKE_TEST_FILES=true ./gradlew :vortex-spark-4.0_2.13:jmh -PjmhArgs="OrderedScanBenchmark -p path=/data/events.vortex -p ordered=false,true -rf json -rff /tmp/events-ordering.json"
VORTEX_SKIP_MAKE_TEST_FILES=true ./gradlew :vortex-spark-4.0_2.13:jmh -PjmhArgs="OrderedScanBenchmark -p path=/data/featoids.vortex -p ordered=false,true -rf json -rff /tmp/featoids-ordering.json"
```

Each invocation opens and fully decodes every column through `VortexPartitionReader`,
without Spark planning, aggregation, filtering, or file splitting. Schema inference runs
outside the measured region. Use the same files and machine for both modes; repeated scans
can use warm storage caches. The setup prints the file size and column count alongside the
JMH parameters. Ordered overhead is `(ordered / unordered - 1) * 100%`; compare the error
intervals before deciding whether the cost is small enough for a caller to opt in.

See the [Spark user guide](https://docs.vortex.dev/user-guide/spark.html) for supported types,
options, partitioning, and remote storage details. The
[Java quickstart](https://docs.vortex.dev/getting-started/java.html) covers the underlying
`vortex-jni` API.
