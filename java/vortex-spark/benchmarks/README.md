# Arrow writer capacity policy benchmark

Arrow remembers allocation sizes after exporting and clearing a batch. Reusing those sizing
hints avoids regrowth on steady data, but a large string or list can inflate every later
batch while native code still retains earlier buffers.

| Mode | After a row-limit flush |
|------|-------------------------|
| `false` | Keep all sizing hints. |
| `auto` (default) | Keep useful hints. Reset a variable-width leaf or list/map subtree only when used bytes are less than one quarter of buffer capacity. Recurse through structs, preserving their own and fixed-width fields' row-bounded hints. |
| `true` | Reset every vector to a small starting capacity. |

Every mode resets every hint after a byte-triggered flush. All modes start small; changing
hints affects future allocations and leaves exported buffers intact.

## Adaptive heuristic

Arrow rounds allocations upward and grows buffers geometrically. A four-times threshold
leaves headroom for this ordinary slack, so steady variable-width batches retain their useful
sizes. After a drop in usage, `auto` resets only the oversized leaf or entire list/map subtree.
Struct fields are evaluated separately, so unrelated fixed-width data cannot hide a string's
slack. The check reads completed buffer lengths once per batch and does not scan values.

The first smaller batch can inherit the previous large allocation; shrinking takes effect
for the following batch. This can retain one extra inflated batch compared with the previous
always-reset variable-width policy. A byte-triggered flush still resets immediately in all modes.
The threshold is a heuristic, not a total-process memory bound.

## Assessment of the default

`false` remains a strong choice for known steady data. The previous `auto` paid regrowth
cost on every variable-width batch even when its capacity was useful. Adaptive `auto` keeps
those hints and is close to `false` in JVM allocation on the longer steady cases:

| Case | false JVM MiB | Previous auto | Adaptive auto |
|------|--------------:|--------------:|--------------:|
| short-text-long | 6.08 | 11.08 | 6.12 |
| long-text-long | 3.85 | 5.44 | 3.77 |
| nested-long | 17.14 | 20.92 | 17.33 |
| map-null-long | 3.22 | 4.51 | 3.28 |

The remaining per-batch utilization check costs some metadata work. `false` can still be faster:
for example, `nested-long` measured 121 ms versus 135 ms for adaptive `auto`.
These are medians of three short local runs, so use the recorded ranges rather than treating
small differences as reliable speedups. The byte-limited cases do not establish a default:
all modes reset after byte-triggered flushes and show the same batch counts and retained peaks.

The reason to retain `auto` as the default is protection when variable-width values change
size unexpectedly and the byte threshold is unset. In `nested-outlier`, retained Arrow is
320.13 MiB with `false`, 2.62 MiB with previous `auto`, and
3.87 MiB with adaptive `auto`. With eight recurring outliers it is
320.13, 11.34 and 21.30 MiB respectively.
The extra retained batch is the explicit tradeoff for preserving useful hints. Adaptive `auto`
also tolerates up to four times the observed used bytes before shrinking, rather than trying
to minimize every allocation. `true` keeps the more aggressive reset behavior.

## Full matrix

Compared the previous `auto` at `8aadab199` with the adaptive writer, using identical fixtures
and the same release JNI library. The current writer runs `false`, adaptive `auto`, and `true`;
the previous writer runs `auto`. Each has three fresh JVMs per mode and case, for
23 workloads × 4 comparisons × 3 repetitions = **276 measured writes**. Current modes rotate
order between repetitions. All output byte counts match across policies and revisions.

Measured on macOS arm64, Corretto 17.0.16, Spark 4.0.4 schema/row classes and Arrow 19.
Results and timing ranges: [results-2026-11-03-adaptive.json](results-2026-11-03-adaptive.json).
Earlier review-fix results remain in [results-2026-11-03.json](results-2026-11-03.json), whose
elapsed times were affected by a concurrent release build. This rerun was started after that
build finished. These remain short local runs with a discarding sink, not storage throughput
measurements.

### Cumulative JVM allocation (MiB)

Allocation is cumulative across the write; it is **not** peak live heap.

| Case | false | Previous auto | Adaptive auto | true |
|------|------:|--------------:|--------------:|-----:|
| tiny-flat | 3.31 | 3.33 | 3.31 | 3.32 |
| narrow-flat | 2.73 | 2.73 | 2.74 | 4.01 |
| narrow-largebatch | 1.51 | 1.52 | 1.51 | 1.93 |
| wide-flat | 5.40 | 5.39 | 5.40 | 7.85 |
| short-text | 1.72 | 2.87 | 1.74 | 2.86 |
| long-text | 3.02 | 3.19 | 3.03 | 3.18 |
| long-text-largebatch | 2.98 | 2.97 | 2.97 | 2.97 |
| nested | 2.20 | 2.71 | 2.23 | 2.80 |
| nested-outlier | 4.33 | 4.01 | 4.00 | 4.03 |
| map-null | 0.47 | 0.63 | 0.48 | 0.63 |
| small-outlier | 1.26 | 1.17 | 1.17 | 1.17 |
| narrow-long | 81.45 | 81.64 | 81.59 | 121.97 |
| wide-long | 39.39 | 39.40 | 39.40 | 59.69 |
| wide-struct | 41.36 | 41.47 | 41.20 | 62.32 |
| max-batch | 8.65 | 8.66 | 8.65 | 9.27 |
| byte-limited-outlier | 0.28 | 0.28 | 0.28 | 0.28 |
| byte-limited-steady | 0.32 | 0.32 | 0.32 | 0.32 |
| short-text-long | 6.08 | 11.08 | 6.12 | 11.08 |
| long-text-long | 3.85 | 5.44 | 3.77 | 5.45 |
| nested-long | 17.14 | 20.92 | 17.33 | 21.54 |
| map-null-long | 3.22 | 4.51 | 3.28 | 4.61 |
| periodic-outlier | 5.21 | 4.94 | 4.94 | 4.96 |
| byte-limited-periodic | 4.88 | 5.05 | 4.98 | 5.03 |

### Sampled peak retained Arrow (MiB)

Peaks depend on native progress and sampling, so small differences do not establish savings.

| Case | false | Previous auto | Adaptive auto | true |
|------|------:|--------------:|--------------:|-----:|
| tiny-flat | 0.02 | 0.02 | 0.02 | 0.02 |
| narrow-flat | 1.25 | 1.25 | 1.25 | 1.12 |
| narrow-largebatch | 2.88 | 2.50 | 3.00 | 2.02 |
| wide-flat | 10.00 | 10.00 | 10.00 | 8.41 |
| short-text | 8.27 | 6.94 | 8.03 | 8.11 |
| long-text | 128.25 | 128.25 | 128.25 | 128.25 |
| long-text-largebatch | 128.25 | 128.25 | 128.25 | 128.25 |
| nested | 14.25 | 14.25 | 14.25 | 14.25 |
| nested-outlier | 320.13 | 2.62 | 3.87 | 2.62 |
| map-null | 1.07 | 1.07 | 1.07 | 1.07 |
| small-outlier | 73.13 | 1.13 | 2.26 | 1.13 |
| narrow-long | 2.25 | 2.69 | 2.50 | 2.50 |
| wide-long | 10.00 | 10.00 | 10.00 | 9.00 |
| wide-struct | 10.00 | 10.00 | 10.00 | 8.06 |
| max-batch | 23.00 | 24.00 | 24.00 | 16.12 |
| byte-limited-outlier | 1.13 | 1.13 | 1.13 | 1.13 |
| byte-limited-steady | 0.13 | 0.13 | 0.13 | 0.13 |
| short-text-long | 7.69 | 7.00 | 7.66 | 7.28 |
| long-text-long | 448.69 | 464.42 | 448.69 | 464.41 |
| nested-long | 23.36 | 21.30 | 22.03 | 21.30 |
| map-null-long | 2.18 | 2.06 | 2.18 | 2.06 |
| periodic-outlier | 320.13 | 11.34 | 21.30 | 11.34 |
| byte-limited-periodic | 11.22 | 11.22 | 11.22 | 11.22 |

### Elapsed time (ms)

Medians of three JVMs; min/max ranges are in the JSON. Cases below 100 ms are primarily useful
for allocation comparisons. The `wide-struct` adaptive timing spans
243–350 ms, illustrating the noise even in a longer case.

| Case | false | Previous auto | Adaptive auto | true |
|------|------:|--------------:|--------------:|-----:|
| tiny-flat | 15.2 | 16.1 | 14.8 | 15.1 |
| narrow-flat | 33.0 | 33.4 | 33.0 | 39.6 |
| narrow-largebatch | 24.4 | 25.1 | 25.2 | 31.6 |
| wide-flat | 42.4 | 42.7 | 40.8 | 52.0 |
| short-text | 34.4 | 36.7 | 34.0 | 38.7 |
| long-text | 114.3 | 116.9 | 119.8 | 115.4 |
| long-text-largebatch | 119.1 | 121.7 | 120.5 | 123.5 |
| nested | 31.2 | 31.2 | 30.6 | 32.7 |
| nested-outlier | 64.2 | 45.5 | 45.4 | 47.9 |
| map-null | 15.8 | 16.2 | 16.1 | 17.1 |
| small-outlier | 27.3 | 25.6 | 25.6 | 25.5 |
| narrow-long | 454.1 | 449.5 | 449.5 | 513.3 |
| wide-long | 249.5 | 218.7 | 226.7 | 279.9 |
| wide-struct | 232.9 | 248.0 | 319.3 | 387.2 |
| max-batch | 91.2 | 90.9 | 94.7 | 102.7 |
| byte-limited-outlier | 17.9 | 19.5 | 18.4 | 17.9 |
| byte-limited-steady | 8.2 | 8.5 | 8.5 | 8.1 |
| short-text-long | 73.3 | 95.9 | 76.7 | 101.6 |
| long-text-long | 168.3 | 201.9 | 163.4 | 186.7 |
| nested-long | 121.4 | 138.5 | 134.6 | 138.5 |
| map-null-long | 44.0 | 48.9 | 46.2 | 50.3 |
| periodic-outlier | 64.1 | 46.9 | 46.4 | 48.2 |
| byte-limited-periodic | 47.6 | 50.9 | 47.3 | 46.8 |

### Workload sizes

A zero byte threshold means unset.

| Case | Rows | Row limit | Byte threshold |
|------|-----:|----------:|---------------:|
| tiny-flat | 128 | 1 | 0 |
| narrow-flat | 131,072 | 2,048 | 0 |
| narrow-largebatch | 131,072 | 8,192 | 0 |
| wide-flat | 32,768 | 2,048 | 0 |
| short-text | 65,536 | 2,048 | 0 |
| long-text | 8,192 | 2,048 | 0 |
| long-text-largebatch | 8,192 | 8,192 | 0 |
| nested | 16,384 | 512 | 0 |
| nested-outlier | 16,384 | 64 | 0 |
| map-null | 16,384 | 2,048 | 0 |
| small-outlier | 65 | 1 | 0 |
| narrow-long | 4,194,304 | 2,048 | 0 |
| wide-long | 262,144 | 2,048 | 0 |
| wide-struct | 262,144 | 2,048 | 0 |
| max-batch | 1,048,576 | 65,536 | 0 |
| byte-limited-outlier | 65 | 2,048 | 262,144 |
| byte-limited-steady | 8,192 | 8,192 | 4,096 |
| short-text-long | 262,144 | 2,048 | 0 |
| long-text-long | 65,536 | 2,048 | 0 |
| nested-long | 131,072 | 512 | 0 |
| map-null-long | 131,072 | 2,048 | 0 |
| periodic-outlier | 16,384 | 64 | 0 |
| byte-limited-periodic | 16,384 | 64 | 262,144 |

## Reproduce

Build the release native library once, then run the comparisons sequentially with the same
library. On Linux, use `.so` instead of `.dylib`.

```shell
cargo build --release -p vortex-jni --locked
python3 java/vortex-spark/benchmarks/run.py --runs 3 --offline \
    --native /path/to/libvortex_jni.dylib --modes auto \
    --writer-ref 8aadab199ee6229809d2130fcc6c746ad8bac547 --output-dir /tmp/auto-previous
python3 java/vortex-spark/benchmarks/run.py --runs 3 --offline \
    --native /path/to/libvortex_jni.dylib --output-dir /tmp/auto-adaptive
```

Use `--cases short-text-long,nested-long,periodic-outlier` for a smaller selection and
`--modes false,auto` to compare only those modes. Omit `--offline` if dependencies are not cached.
`--writer-ref` compiles only the writer from that Git ref; it requires a compatible Java API.

The runner resolves dependencies and the configured Java toolchain through Gradle. It compiles
the fixture, selected writer and a benchmark-only native loader, then copies the supplied
release library into the result directory to pin it between samples. It does not rebuild or
stage a debug library. Each result directory contains logs, raw `results.jsonl`, median/range
`summary.json`, and a manifest recording JVM version and writer/fixture/native library hashes.

The sink counts and discards output; it opens no destination file. Every sample checks native
row counts and output counts against the completed writer summary. The runner checks serialized
byte counts across modes and repetitions. Value correctness and buffer release are checked
separately by `VortexOutputWriterTest`, including steady-to-small transitions for strings,
binary, lists, maps and struct fields, and byte-triggered flushes in every mode.

## Workloads and measurement limits

Fixtures cycle through 256 deterministic rows generated with seed 481516. They cover 8/64 long
columns, a struct with 64 long fields, nullable 32-byte/4-KiB strings, nullable structs/lists,
maps, empty lists, and outliers containing 65,536 longs plus a 128-KiB string. Normal outlier-case
rows contain two longs and a 16-byte string. The periodic cases inject that outlier every 2,048
rows, eight times per measured write. List-layout writing is off.

Each JVM warms up with the same shape for `min(rows, 8192)` rows, then runs explicit GC.
Measured time includes writer creation, writing and close. The pool of repeated rows favors
dictionary compression compared with high-cardinality data. JVMs use G1, a 2-GiB maximum heap
and an 8-GiB direct-memory allowance.

Arrow allocation is sampled every 16 rows and at the last row, so brief growth/copy transients
can be missed. Native scheduling changes sampled retained peaks. RSS is sampled every 20 ms
and includes JVM/native/pool retention from warm-up. JVM allocation includes background threads
and the heap sampler. These counters should not be added to estimate live memory. Matching
serialized byte counts alone do not prove value equality.
