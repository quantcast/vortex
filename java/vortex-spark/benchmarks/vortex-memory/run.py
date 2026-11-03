# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Measure Vortex scans and rewrites with fixed batch sizes in serial fresh JVMs."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from measurement import measure

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
VARIANTS = (
    "read-default",
    "read-bridge",
    "ordered-rewrite-baseline",
    "ordered-rewrite-bridge",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--candidate-native", type=Path)
    parser.add_argument("--classpath-file", type=Path, required=True)
    parser.add_argument("--java-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", default="read-default,read-bridge")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--sink", default="discard")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    variants = args.variants.split(",")
    if args.input.suffix != ".vortex":
        parser.error("The input must be a Vortex file")
    if args.runs < 1 or len(variants) != len(set(variants)) or any(v not in VARIANTS for v in variants):
        parser.error("Invalid runs or variants")
    if any(v.endswith("bridge") for v in variants) and not args.candidate_native:
        parser.error("Candidate variants require --candidate-native")
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.java_home = args.java_home.resolve()
    if args.sink != "discard":
        args.sink = str(Path(args.sink).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    classes = args.output / "classes"
    classes.mkdir()
    loader = args.output / "NativeLoader.java"
    loader.write_text('''// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
package dev.vortex.jni;
public final class NativeLoader {
    private static boolean loaded;
    public static synchronized void loadJni() {
        if (!loaded) { System.load(System.getProperty("vortex.memory.native")); loaded = true; }
    }
}
''')
    baseline = args.output / args.native.name
    candidate = args.output / ("candidate-" + args.candidate_native.name) if args.candidate_native else None
    shutil.copyfile(args.native, baseline)
    if args.candidate_native:
        shutil.copyfile(args.candidate_native, candidate)
    for name in ("VortexMemoryBench.java", "run.py", "measurement.py"):
        shutil.copyfile(HERE / name, args.output / name)
    classpath = args.classpath_file.read_text().strip()
    subprocess.run([str(args.java_home / "bin/javac"), "-proc:none", "-cp", classpath,
                    "-d", str(classes), str(HERE / "VortexMemoryBench.java"), str(loader)], check=True)
    ordered_classes = args.output / "ordered-classes"
    if any(v.startswith("ordered-rewrite") for v in variants):
        ordered_classes.mkdir()
        source = REPOSITORY / "java/vortex-spark/common/src/main/java/dev/vortex/spark/read/VortexPartitionReader.java"
        code = source.read_text()
        if code.count("var options = ScanOptions.builder();") != 1:
            raise RuntimeError("Reader source changed; review the ordered-scan overlay")
        ordered_source = args.output / "VortexPartitionReader.java"
        ordered_source.write_text(code.replace(
            "var options = ScanOptions.builder();",
            "var options = ScanOptions.builder().ordered(true);",
        ))
        subprocess.run([str(args.java_home / "bin/javac"), "-proc:none", "-cp", classpath,
                        "-d", str(ordered_classes), str(ordered_source)], check=True)
    manifest = {
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True,
        ).strip(),
        "baselineSha256": sha256(baseline),
        "candidateSha256": sha256(candidate) if candidate is not None else None,
        "writerBatchRows": 8192,
        "writerBatchBytes": 0,
        "workerThreads": 4,
        "sourceSplitConcurrency": "default",
        "orderedSource": {v: v.startswith("ordered-rewrite") for v in variants},
        "sourceBytes": args.input.stat().st_size,
        "sourceSha256": sha256(args.input),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for repetition in range(args.runs):
        offset = repetition % len(variants)
        for variant in variants[offset:] + variants[:offset]:
            selected = candidate if variant.endswith("bridge") else baseline
            command = [str(args.java_home / "bin/java"), "-Xms256m", "-Xmx2g", "-XX:MaxDirectMemorySize=8g",
                       "-XX:+UseG1GC", f"-Dvortex.memory.native={selected.resolve()}"]
            command += [f"--add-opens=java.base/{p}=ALL-UNNAMED" for p in
                        ["sun.nio.ch", "java.nio", "sun.util.calendar", "sun.security.action", "java.lang.invoke", "java.util", "java.io"]]
            selected_classpath = str(classes) + os.pathsep + classpath
            if variant.startswith("ordered-rewrite"):
                selected_classpath = str(ordered_classes) + os.pathsep + selected_classpath
            command += ["-cp", selected_classpath,
                        "dev.vortex.spark.write.VortexMemoryBench", str(args.input.resolve()),
                        "8192", "0", "auto", "4", args.sink,
                        str(variant.startswith("read")).lower(), "0",
                        "verify" if args.verify else ""]
            tag = f"{variant}-{repetition}"
            print("START", tag, flush=True)
            result = measure(command, args.output / f"{tag}.log", dict(os.environ, SPARK_LOCAL_IP="127.0.0.1"))
            result |= {"variant": variant, "repetition": repetition}
            with (args.output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
