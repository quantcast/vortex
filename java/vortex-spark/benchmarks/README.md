# Arrow writer capacity policy benchmark

`vortex.write.reset.capacity=auto` targets an allocation problem that depends on the schema:
Arrow remembers allocation sizes after exporting and clearing a batch. A large string or
list can inflate subsequent batches while native code still retains earlier buffers. The
number of elements in a top-level fixed-width vector is limited by the row batch size, so
its retained hint has a bounded cost.

| Mode | After each exported batch |
|------|---------------------------|
| `false` | Keep all learned sizing hints, including after byte-triggered flushes. |
| `auto` (default) | Reset variable-width and nested root vectors, including their children; keep top-level fixed-width hints. Reset every vector after a byte-triggered flush. |
| `true` | Reset every vector after every flush. |

All modes start with the existing small initial capacity. Resetting affects future allocations;
it does not overwrite buffers already exported to native code.

## Why `auto` is the default

Measured on QC `develop` base `769938bf9`, macOS arm64, Corretto 17.0.16, Spark 4.0.4
schema/row classes, Arrow 19, and a release JNI build. Each entry is the median of three fresh
JVMs. The matrix has 15 size/shape/batch-size cases and 135 measured writes, rotating mode order
between repetitions. Complete medians, timing ranges and build identifiers are in
[results-2026-11-02.json](results-2026-11-02.json).

MiB means 1,048,576 bytes. Retained Arrow allocation and cumulative JVM allocation are different
quantities: cumulative allocation is **not** peak live memory.

| Case | Metric (MiB) | `false` | `auto` | `true` |
|------|--------------|--------:|-------:|-------:|
| Nested outlier, 16,384 rows, batch 64 | Sampled peak retained Arrow | 320.13 | 2.62 | 2.62 |
| Small outlier, 65 rows, batch 1 | Sampled peak retained Arrow | 73.13 | 1.13 | 1.13 |
| 8 long columns, 4,194,304 rows, batch 2,048 | Cumulative JVM allocation | 83.55 | 84.17 | 123.63 |
| 64 long columns, 262,144 rows, batch 2,048 | Cumulative JVM allocation | 39.36 | 39.34 | 60.09 |
| 64 long columns, 262,144 rows, batch 2,048 | Sampled peak retained Arrow | 10.00 | 10.00 | 8.12 |
| Steady 4 KiB strings, 8,192 rows, batch 2,048 | Sampled peak retained Arrow | 128.25 | 128.25 | 128.25 |
| Steady 4 KiB strings, 8,192 rows, batch 2,048 | Cumulative JVM allocation | 3.02 | 3.17 | 3.18 |

`auto` preserves the outlier memory reduction while avoiding the extra allocation/copying
of repeatedly regrowing bounded fixed-width vectors. Forced reset had a lower sampled peak
in the 64-column case (8.12 versus 10.00 MiB); aggregate samples alone do not identify its cause.
`true` remains available for that tradeoff. `auto` is a memory-protection default, not a claim
that it wins every workload:
steady variable-width data can have unchanged peaks and extra growth work. Such workloads
can use `false`.

Elapsed times for the longer steady cases, in milliseconds, as median (min–max):

| Case | `false` | `auto` | `true` |
|------|--------:|-------:|-------:|
| 8 long columns, 4,194,304 rows | 446 (443–454) | 437 (396–458) | 531 (499–552) |
| 64 long columns, 262,144 rows | 209 (209–213) | 208 (202–224) | 262 (258–270) |
| Steady 4 KiB strings, 8,192 rows | 115 (115–115) | 118 (117–119) | 119 (119–119) |

These are short local runs, not production throughput claims. The allocation counts and
retained bytes are the primary evidence for the default; more repetitions and production
storage backpressure are needed for a throughput conclusion.

## Capacity reset and byte-limited batching

The options are complementary. `vortex.write.batch.bytes` limits the current batch's Arrow
allocations after each complete row; capacity reset prevents an outlier inflating future
allocations. Neither bounds total process memory or a single oversized row.

A separate case writes the same 65 outlier/small rows with a 256-KiB batch threshold:

| Metric | `false` | `auto` | `true` |
|--------|--------:|-------:|-------:|
| Sampled peak retained Arrow (MiB) | 73.19 | 1.13 | 1.13 |
| Exported batches | 65 | 2 | 2 |

The current QC writer already resets all hints after a byte-triggered flush. `auto` preserves
that behavior. This row demonstrates the interaction and the explicit opt-out, not a new
improvement over the existing byte-budget implementation: `false` disables that reset too,
so later one-row batches can repeatedly reach the byte threshold from their initial allocation.

## Reproduce

From the repository root, on macOS or Linux:

```shell
cargo build --release -p vortex-jni --locked
python3 java/vortex-spark/benchmarks/run.py --runs 3
```

The runner reuses the existing benchmark source set for dependencies, resolves the configured
Java toolchain through Gradle, and compiles the standalone fixture driver. A benchmark-only
loader selects the supplied release library instead of the jar's bundled library. The runner
copies that library into the result directory to pin it between samples. It does not rebuild
or stage a debug native library.

The byte sink counts and discards every native output write. It opens no destination file,
so cleanup cannot remove any device or data file. The count is checked against the completed
writer summary for every sample.

Use `--offline` with cached dependencies, `--native /path/to/library` for another build, or
`--output-dir /new/path` for another result destination. To run a smaller selection:

```shell
python3 java/vortex-spark/benchmarks/run.py --runs 5 \
    --cases narrow-long,wide-long,nested-outlier,long-text,byte-limited-outlier
```

Each new result directory contains logs, `results.jsonl`, medians/ranges in `summary.json`,
and a manifest with JVM version, writer source hash and native library hash. All cases must
produce the same serialized byte count across modes and repetitions. Value correctness is
checked separately by `VortexOutputWriterTest`, including outlier readback, byte-triggered
flushes, reused rows, nested/list/map/null values, decimals and failure cleanup.

## Workloads and measurement limits

Fixtures cycle through 256 deterministic rows generated with seed 481516. Cases cover 8/64
long columns, nullable 32-byte/4-KiB strings, nullable nested structs/lists, nullable maps,
empty lists, and an outlier containing 65,536 longs plus a 128-KiB string. Following outlier
rows contain two longs and a 16-byte string. Row counts range from 65 to 4,194,304; batch sizes
cover 1, 64, 512, 2,048, 8,192 and 65,536. List-layout writing is left at the QC default, off.

Each JVM performs a same-shape warm-up write of `min(rows, 8192)`, then explicit GC. Measured
time includes writer creation, all writes and close. This isolates conversion, compression
and JNI output from Parquet decoding, Spark scheduling and storage. The repeated-row pool
favors dictionary compression compared with high-cardinality data. JVMs use G1 GC, a 2-GiB
maximum heap and an 8-GiB direct-memory allowance.

Arrow allocations are sampled every 16 rows and at the last row, so brief growth/copy
transients inside `write()` can be missed. Native progress can vary the sampled peak between
runs. RSS is sampled every 20 ms and includes JVM/native/pool retention from warm-up. JVM
allocation includes background threads and the heap sampler. These counters should not be
added to estimate live memory. Cases shorter than 100 ms are useful for allocation comparisons,
not timing claims. Equal byte counts alone do not prove value equality.
