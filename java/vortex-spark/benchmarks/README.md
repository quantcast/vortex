# Arrow writer capacity policy benchmark

`vortex.write.reset.capacity=auto` targets an allocation problem that depends on the schema:
Arrow remembers allocation sizes after exporting and clearing a batch. A large string or
list can inflate subsequent batches while native code still retains earlier buffers. The
number of elements in a fixed-width vector at the root or inside structs is limited by the
row batch size, so its retained hint has a bounded cost. Lists and maps can multiply element
counts, so their entire subtrees need resets.

| Mode | After each exported batch |
|------|---------------------------|
| `false` | Keep learned sizing hints across row-limit flushes. Reset every vector after a byte-triggered flush. |
| `auto` (default) | Recurse through structs, keeping their own and fixed-width children's hints. Reset variable-width leaves and entire list/map subtrees. Reset every vector after a byte-triggered flush. |
| `true` | Reset every vector after every flush. |

All modes start with the existing small initial capacity. Resetting affects future allocations;
it does not overwrite buffers already exported to native code.

## Before and after the review fixes

Compared the original PR writer (`ef9bc8bc7`) with the updated writer on macOS arm64,
Corretto 17.0.16, Spark 4.0.4 schema/row classes and Arrow 19. Both use the same release JNI
library as the original benchmarks and identical fixtures. Each revision has 17 cases, three
modes and three fresh JVMs per mode: 306 measured writes in total, rotating mode order between
repetitions. Output byte counts match across all modes, repetitions and revisions.

Complete medians, timing ranges and source/library hashes are in
[results-2026-11-03.json](results-2026-11-03.json). The original measurements remain in
[results-2026-11-02.json](results-2026-11-02.json).

| Case and metric | Original PR | Updated |
|-----------------|------------:|--------:|
| Struct with 64 long fields, `auto`: JVM allocation (MiB) | 62.02 | 41.48 |
| Steady long, 4-KiB threshold, `false`: exported batches | 7,940 | 33 |
| Steady long, 4-KiB threshold, `false`: retained Arrow (MiB) | 31.02 | 0.13 |
| Outlier, 256-KiB threshold, `false`: exported batches | 65 | 2 |
| Outlier, 256-KiB threshold, `false`: retained Arrow (MiB) | 73.19 | 1.13 |

Recursing through structs lets `auto` retain row-bounded fixed-width hints, reducing JVM
allocation on the wide struct by about 33%. `true` still regrows those vectors, allocating
62.16 MiB in this case. Retaining hints also keeps larger per-batch allocations:
`auto` sampled 10.00 MiB of retained Arrow versus 8.13 MiB for `true`.

The steady byte-limited case has one long column, 8,192 rows and an 8,192-row batch limit.
The original `false` policy retained a budget-sized allocation after its first byte flush,
then flushed every following row. Every policy now resets all hints after a byte-triggered
flush, preserving QC `develop` behavior and giving 33 batches in all modes. The outlier case
likewise gives two batches in all modes. `false` retains hints only across row-limit flushes.

**Timing limitation:** a separate Rust release build overlapped part of the updated run.
Elapsed times are recorded for transparency but do not support a before/after speed claim.
Native progress also affects sampled retained Arrow peaks. JVM allocation and batch counts
are the primary evidence for these behavior changes.

## Why `auto` is the default

Selected results from the updated writer, as medians of three JVMs. MiB means 1,048,576 bytes;
cumulative allocation is **not** peak live memory.

| Case | Metric (MiB) | `false` | `auto` | `true` |
|------|--------------|--------:|-------:|-------:|
| Nested outlier, 16,384 rows, batch 64 | Sampled retained Arrow | 320.13 | 2.62 | 2.62 |
| Small outlier, 65 rows, batch 1 | Sampled retained Arrow | 73.13 | 1.13 | 1.13 |
| 8 long columns, 4,194,304 rows | Cumulative JVM allocation | 81.30 | 81.69 | 121.82 |
| 64 long columns, 262,144 rows | Cumulative JVM allocation | 39.35 | 39.38 | 60.07 |
| Struct with 64 long fields, 262,144 rows | Cumulative JVM allocation | 41.03 | 41.48 | 62.16 |
| Steady 32-byte strings, 65,536 rows | Cumulative JVM allocation | 1.71 | 2.86 | 2.87 |
| Steady structs/lists, 16,384 rows | Cumulative JVM allocation | 2.21 | 2.72 | 2.79 |
| Steady 4-KiB strings, 8,192 rows | Sampled retained Arrow | 128.25 | 128.25 | 128.25 |

`auto` preserves the outlier memory reduction while avoiding repeated growth of fixed-width
vectors whose counts are bounded by the row limit, including fields inside structs. `true`
remains available when the smaller retained allocation is preferable.

Resetting strings, lists and maps regrows buffers from a small capacity each batch. This adds
JVM allocation on steady data: short text uses 2.86 MiB under `auto` versus
1.71 MiB under `false`; steady nested data uses 2.72 versus 2.21 MiB.
Steady variable-width data can also have unchanged peaks. `false` offers that tradeoff for
row-limit flushes, while byte-triggered flushes always reset.

## Capacity reset and byte-limited batching

The options are complementary. `vortex.write.batch.bytes` limits the current batch's Arrow
allocations after each complete row; capacity reset prevents an outlier inflating future
allocations. Neither bounds total process memory or a single oversized row.

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

For a before/after comparison, use the same fixtures and native library and compile only the
writer from the original PR revision. Run these commands sequentially:

```shell
python3 java/vortex-spark/benchmarks/run.py --runs 3 --offline \
    --native /path/to/libvortex_jni.dylib \
    --writer-ref ef9bc8bc707fccbd46a6a4704e7ba9023d6a955f --output-dir /tmp/capacity-before
python3 java/vortex-spark/benchmarks/run.py --runs 3 --offline \
    --native /path/to/libvortex_jni.dylib --output-dir /tmp/capacity-after
```

Use the `.so` library on Linux. `--writer-ref` changes only `VortexOutputWriter.java`; it is
intended for revisions compatible with the current Java API. The manifest records the actual
writer, fixture and native library hashes.

Each new result directory contains logs, `results.jsonl`, medians/ranges in `summary.json`,
and a manifest with JVM version, writer source hash and native library hash. All cases must
produce the same serialized byte count across modes and repetitions. Value correctness is
checked separately by `VortexOutputWriterTest`, including outlier readback, byte-triggered
flushes, reused rows, nested/list/map/null values, decimals and failure cleanup.

## Workloads and measurement limits

Fixtures cycle through 256 deterministic rows generated with seed 481516. Cases cover 8/64
long columns, a struct containing 64 long fields, nullable 32-byte/4-KiB strings, nullable
nested structs/lists, nullable maps, empty lists, and an outlier containing 65,536 longs plus a 128-KiB string. Following outlier
rows contain two longs and a 16-byte string. Row counts range from 65 to 4,194,304; batch sizes
cover 1, 64, 512, 2,048, 8,192 and 65,536. A steady single-long-column case uses 8,192 rows,
an 8,192-row batch limit and a 4-KiB byte threshold. List-layout writing is left at the QC
default, off.

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
