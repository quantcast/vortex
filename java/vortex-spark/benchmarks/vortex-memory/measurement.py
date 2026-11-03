# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

import ctypes
import json
import platform
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path


def usage_reader() -> Callable[[int], tuple[int, int] | None]:
    """Return a function reading `(rss, physical footprint)` bytes for a pid, or None on failure."""
    if platform.system() != "Darwin":

        def read_ps(pid: int) -> tuple[int, int] | None:
            value = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False)
            if value.returncode != 0 or not value.stdout.strip():
                return None
            return int(value.stdout.strip()) * 1024, 0

        return read_ps

    # rusage_info_v0 from macOS sys/resource.h, including its leading UUID.
    class Usage(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
            (name, ctypes.c_uint64)
            for name in (
                "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size "
                "resident_size phys_footprint proc_start_abstime proc_exit_abstime"
            ).split()
        ]

    mac_proc = ctypes.CDLL("/usr/lib/libproc.dylib")
    mac_proc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    mac_proc.proc_pid_rusage.restype = ctypes.c_int

    def read_rusage(pid: int) -> tuple[int, int] | None:
        usage = Usage()
        if mac_proc.proc_pid_rusage(pid, 0, ctypes.byref(usage)) != 0:
            return None
        return usage.resident_size, usage.phys_footprint

    return read_rusage


def measure(command: list[str], log_path: Path, environment: dict[str, str]) -> dict[str, object]:
    active = threading.Event()
    peak_rss = 0
    peak_footprint = 0
    errors = []
    result = None
    read_usage = usage_reader()
    with log_path.open("w") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True, env=environment) as process:

            def sample():
                nonlocal peak_rss, peak_footprint
                try:
                    while process.poll() is None:
                        if active.is_set() and (usage := read_usage(process.pid)) is not None:
                            peak_rss = max(peak_rss, usage[0])
                            peak_footprint = max(peak_footprint, usage[1])
                        time.sleep(0.05)
                except Exception as error:
                    errors.append(str(error))

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.strip() == "BENCH_BEGIN":
                    active.set()
                elif line.startswith("BENCH "):
                    active.clear()
                    result = json.loads(line[6:])
            code = process.wait()
            sampler.join()
    if code or result is None:
        raise RuntimeError(f"Failed ({code}), see {log_path}")
    if errors or peak_rss == 0:
        raise RuntimeError(f"RSS measurement failed: {errors}")
    return result | {"rssPeakBytes": peak_rss, "physicalFootprintPeakBytes": peak_footprint or None}
