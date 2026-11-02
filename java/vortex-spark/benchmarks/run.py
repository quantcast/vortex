# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Compare Arrow capacity policies with isolated JVMs and a prebuilt release JNI library."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parent
JAVA = BENCHMARK.parents[1]
REPOSITORY = JAVA.parent
BUILD = JAVA / "vortex-spark/v4.0/build/vortex-spark-4.0_2.13/capacity-benchmark"
CASES = {
    "tiny-flat": ("flat", 128, 1),
    "narrow-flat": ("flat", 131072, 2048),
    "narrow-largebatch": ("flat", 131072, 8192),
    "wide-flat": ("wide", 32768, 2048),
    "short-text": ("short-text", 65536, 2048),
    "long-text": ("long-text", 8192, 2048),
    "long-text-largebatch": ("long-text", 8192, 8192),
    "nested": ("nested", 16384, 512),
    "nested-outlier": ("outlier", 16384, 64),
    "map-null": ("map-null", 16384, 2048),
    "small-outlier": ("outlier", 65, 1),
    "narrow-long": ("flat", 4194304, 2048),
    "wide-long": ("wide", 262144, 2048),
    "max-batch": ("flat", 1048576, 65536),
    "byte-limited-outlier": ("outlier", 65, 2048),
}
MODES = ("false", "auto", "true")

# Load the requested release library instead of the JNI jar's bundled debug library.
LOADER = """// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
package dev.vortex.jni;
public final class NativeLoader {
    private static boolean loaded;
    private NativeLoader() {}
    public static synchronized void loadJni() {
        if (!loaded) {
            System.load(System.getProperty("capacity.native"));
            loaded = true;
        }
    }
}
"""


def prepare(offline: bool) -> tuple[Path, str]:
    command = [
        str(JAVA / "gradlew"),
        "--init-script",
        str(BENCHMARK / "classpath.gradle"),
        ":vortex-spark-4.0_2.13:capacityBenchmarkClasspath",
        "--no-daemon",
    ]
    if offline:
        command.append("--offline")
    # The benchmark-only loader supplies the already built native library.
    environment = dict(os.environ, VORTEX_SKIP_MAKE_TEST_FILES="true")
    subprocess.run(command, cwd=JAVA, env=environment, check=True)
    java_home = Path((BUILD / "java-home.txt").read_text())
    classpath = (BUILD / "classpath.txt").read_text()
    classes = BUILD / "classes"
    classes.mkdir(exist_ok=True)
    loader = BUILD / "src/dev/vortex/jni/NativeLoader.java"
    loader.parent.mkdir(parents=True, exist_ok=True)
    loader.write_text(LOADER)
    subprocess.run(
        [
            str(java_home / "bin/javac"),
            "-proc:none",
            "-cp",
            classpath,
            "-d",
            str(classes),
            str(BENCHMARK / "CapacityBench.java"),
            str(loader),
        ],
        check=True,
    )
    return java_home, str(classes) + os.pathsep + classpath


def run_case(command: list[str], log_path: Path) -> dict:
    active = threading.Event()
    peak_rss = 0
    sampling_errors = []
    result = None
    with log_path.open("w") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True) as process:

            def sample():
                nonlocal peak_rss
                try:
                    while process.poll() is None:
                        if active.is_set():
                            snapshot = subprocess.run(
                                ["ps", "-o", "rss=", "-p", str(process.pid)],
                                capture_output=True,
                                text=True,
                                check=False,
                            ).stdout.strip()
                            if snapshot:
                                peak_rss = max(peak_rss, int(snapshot) * 1024)
                        time.sleep(0.02)
                except (OSError, ValueError) as error:
                    sampling_errors.append(error)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                if line.strip() == "BENCH_BEGIN":
                    active.set()
                elif line.startswith("BENCH "):
                    active.clear()
                    result = json.loads(line[6:])
            code = process.wait()
            sampler.join()
    if code or result is None:
        raise RuntimeError(f"Benchmark failed (exit {code}); see {log_path}")
    if sampling_errors:
        raise RuntimeError("RSS sampling failed") from sampling_errors[0]
    result["rssPeakBytes"] = peak_rss or None
    return result


def summarize(results: list[dict]) -> list[dict]:
    summary = []
    for case in dict.fromkeys(row["case"] for row in results):
        selected = [row for row in results if row["case"] == case]
        if len({row["outputBytes"] for row in selected}) != 1:
            raise RuntimeError(f"Output sizes differ for {case}")
        modes = {}
        for mode in MODES:
            samples = [row for row in selected if row["reset"] == mode]
            modes[mode] = {
                "samples": len(samples),
                "rows": samples[0]["rows"],
                "batch": samples[0]["batch"],
                "batchBytes": samples[0]["batchBytes"],
                **{
                    key: statistics.median(row[key] for row in samples)
                    for key in (
                        "seconds",
                        "arrowPeakBytes",
                        "jvmAllocatedBytes",
                        "heapPeakBytes",
                        "outputBytes",
                        "batchCount",
                    )
                },
                "secondsRange": [min(row["seconds"] for row in samples), max(row["seconds"] for row in samples)],
            }
        summary.append({"case": case, "modes": modes})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--native", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    arguments = parser.parse_args()
    if platform.system() not in ("Darwin", "Linux"):
        parser.error("This runner requires macOS or Linux (ps).")
    if arguments.runs < 1:
        parser.error("--runs must be positive")
    cases = arguments.cases.split(",")
    if any(case not in CASES for case in cases) or len(cases) != len(set(cases)):
        parser.error("--cases must name distinct cases from: " + ",".join(CASES))
    extension = "dylib" if platform.system() == "Darwin" else "so"
    native = (arguments.native or REPOSITORY / f"target/release/libvortex_jni.{extension}").resolve()
    if not native.is_file():
        parser.error("Build the release library first: cargo build --release -p vortex-jni")
    output = arguments.output_dir or BUILD / datetime.now().strftime("results-%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    pinned_native = output / native.name
    shutil.copyfile(native, pinned_native)
    java_home, classpath = prepare(arguments.offline)
    manifest = {
        "command": vars(arguments) | {"native": str(native), "output_dir": str(output)},
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "nativeSha256": hashlib.sha256(pinned_native.read_bytes()).hexdigest(),
        "writerSha256": hashlib.sha256(
            (BENCHMARK.parent / "common/src/main/java/dev/vortex/spark/write/VortexOutputWriter.java").read_bytes()
        ).hexdigest(),
        "javaVersion": subprocess.check_output(
            [str(java_home / "bin/java"), "-version"], stderr=subprocess.STDOUT, text=True
        ),
        "platform": platform.platform(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for name in cases:
        shape, rows, batch = CASES[name]
        for iteration in range(arguments.runs):
            offset = iteration % len(MODES)
            for mode in MODES[offset:] + MODES[:offset]:
                tag = f"{name}-{mode}-{iteration}"
                command = [
                    str(java_home / "bin/java"),
                    "-Xms128m",
                    "-Xmx2g",
                    "-XX:MaxDirectMemorySize=8g",
                    "-XX:+UseG1GC",
                    f"-Dcapacity.native={pinned_native.resolve()}",
                    *[
                        f"--add-opens=java.base/{package}=ALL-UNNAMED"
                        for package in ("sun.nio.ch", "java.nio", "java.lang.invoke", "java.util", "java.io")
                    ],
                    "-cp",
                    classpath,
                    "dev.vortex.spark.write.CapacityBench",
                    shape,
                    str(rows),
                    str(batch),
                    mode,
                    "262144" if name == "byte-limited-outlier" else "0",
                ]
                result = run_case(command, output / f"{tag}.log") | {"case": name, "iteration": iteration}
                results.append(result)
                with (output / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
                print(
                    f"{tag}: {result['seconds']:.3f}s, Arrow {result['arrowPeakBytes'] / 1048576:.2f} MiB", flush=True
                )
    (output / "summary.json").write_text(json.dumps(summarize(results), indent=2) + "\n")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
