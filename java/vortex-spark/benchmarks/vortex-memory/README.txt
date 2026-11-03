SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors

Vortex Spark peak-memory investigation (2026-11-03)

The candidate bounds the default JNI scan's completed-output handoff to one
item, retaining a background producer for overlap with Java consumption. The
baseline queues one completed item per host CPU. Native decode concurrency,
conversion lookahead, read batch boundaries and compression settings are fixed.
Explicit splitConcurrency limits retain their existing direct-iterator path.

Results

Baseline: QC develop 29348ee3956597c1609a380a2226bb47c23d2c93.
Candidate: that snapshot with the bounded handoff in vortex-jni/src/scan.rs.
Sanitized individual measurements, native SHA-256 hashes and fixture identifiers
are in results-2026-11-03.json. Private Vortex fixtures and Spark logs are omitted.
Fixture labels describe their Vortex file layout, not a flat versus nested schema.

Medians, with min..max in brackets; memory is macOS physical footprint in GiB.

Workload        n/variant  Baseline footprint   Candidate footprint  Change
Flat-layout read    3     1.46 [1.42..1.59]     1.05 [1.03..1.05]    -28.2%
List-layout read    2     3.25 [3.03..3.47]     2.22 [2.18..2.27]    -31.6%
Ordered rewrite    2     1.86 [1.85..1.87]     1.43 [1.39..1.46]    -23.3%

Workload                 Baseline seconds     Candidate seconds    Change
Flat-layout read         10.59 [7.11..12.45]    10.33 [6.22..12.39]    -2.4%
List-layout read         10.01 [7.70..12.32]     8.94 [7.59..10.29]   -10.7%
Ordered rewrite           9.49 [9.48..9.49]      8.59 [7.81..9.38]     -9.4%

Timing varies substantially on this shared host. These runs show consistent
memory reductions and no observed slowdown, but do not establish a speedup or
rule out regressions on other workloads.

Compression and value checks

All inputs are existing Vortex files. Rewrites retain 8,192-row write batches,
the default capacity policy, no write-byte threshold, and unchanged native write
strategy and compression settings. Both rewrite variants use a benchmark-only
reader overlay enabling ordered scans, holding row order fixed as well. Default
unordered scans can vary row order with scheduling and change compressed size.
The overlay does not change the production Spark reader.

All four timed rewrite runs and the separate actual-file verification produced
395,835 rows, 49 batches (48 full batches plus a 2,619-row tail), and exactly
111,065,312 serialized bytes. Baseline and candidate full-row read digests match
on the 774,057-row list-layout fixture. A candidate rewrite to a real Vortex file
matches its input's count and digest. These checks use count plus the sum of
64-bit xxhash64 hashes of complete-row JSON, cast to decimal(38,0). They are
order-independent probabilistic checks, not bytewise value comparisons.

Reproduction

Build a release JNI library at the baseline revision and at the candidate
revision with separate target directories, preserving both libraries:

  cargo build --release -p vortex-jni --locked --target-dir /absolute/target-dir

Use the candidate checkout's Java connector. Generate a Java classpath from its
java directory (omit --offline if dependencies are not cached):

  cd java
  VORTEX_SKIP_MAKE_TEST_FILES=true ./gradlew --offline --no-daemon \
    --init-script vortex-spark/benchmarks/vortex-memory/classpath.gradle \
    :vortex-spark-4.0_2.13:vortexMemoryClasspath

The task writes classpath.txt and java-home.txt beneath
java/vortex-spark/v4.0/build/vortex-spark-4.0_2.13/vortex-memory/.
Run from the repository root, supplying absolute paths:

  python3 java/vortex-spark/benchmarks/vortex-memory/run.py \
    --input /absolute/input.vortex \
    --native /absolute/baseline/libvortex_jni.dylib \
    --candidate-native /absolute/candidate/libvortex_jni.dylib \
    --classpath-file /absolute/classpath.txt --java-home /absolute/java-home \
    --output /absolute/new-results --runs 3

Use the appropriate native-library extension on other platforms. The macOS
footprint metric is unavailable on other platforms; the RSS fallback uses ps.

Variants:
* read-default,read-bridge (default): baseline versus bounded handoff, full schema.
* ordered-rewrite-baseline,ordered-rewrite-bridge: Vortex-to-Vortex rewrite with
  the identical ordered source control for both libraries.

Pass --variants ordered-rewrite-baseline,ordered-rewrite-bridge for the rewrite
comparison. The default synchronous sink discards serialized bytes; --sink FILE
writes a local Vortex file and verifies the row digest after timing. Each run
replaces that file. --verify on reads reports the digest after timing; compare
it between variants. Libraries, harness sources, local logs and SHA-256 manifests
are preserved in the new output directory. Local manifests/logs can include
private paths. The benchmark-only NativeLoader selects the requested library.

Method and limits

The measurements used macOS arm64, Mac15,7, 12 logical/physical CPUs, 36 GiB RAM,
Java 17.0.16 Amazon, Spark 4.0.4 and Scala 2.13.16. Each case uses a fresh JVM,
2 GiB G1 heap, an 8 GiB direct-memory ceiling, four native workers, one local
Spark task, default scan concurrency and disabled Spark file splitting. Order
rotates between repetitions. Timing includes action planning, cold JIT, scan
consumption and writer finish; it excludes JVM startup/schema inference.
No compiler jobs overlapped the timed confirmation runs.

Metrics:
* physicalFootprintPeakBytes: macOS charged physical footprint, including
  compressed-memory accounting, sampled every 50 ms through proc_pid_rusage.
* rssPeakBytes: resident bytes sampled simultaneously. Host memory compression
  can reduce RSS independently of retained allocations.
* arrowPeakBytes: exact writer RootAllocator high-water mark, including imported
  Java buffers retained by Rust. Reader Arrow memory is not counted here.
* heapPeakBytes: sampled used JVM heap every 10 ms, not post-GC live heap.
* bufferedPeakBytes: partial native compressed-buffer accounting.
* jvmAllocatedBytes: cumulative JVM allocation, not live or peak memory.

Independent peaks cannot be added/subtracted to infer another memory category.
Multi-task executors, object-store latency/backpressure, partition-writer fanout
and a broad query mix remain unmeasured.
