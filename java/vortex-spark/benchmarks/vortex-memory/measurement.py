# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import ctypes
import json
import platform
import subprocess
import threading
import time


def measure(command, log_path, environment):
    active = threading.Event()
    peak_rss = 0
    peak_footprint = 0
    errors = []
    result = None
    mac_proc = None
    if platform.system() == "Darwin":
        # rusage_info_v0 from macOS sys/resource.h, including its leading UUID.
        class Usage(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
                (name, ctypes.c_uint64) for name in (
                    "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size "
                    "resident_size phys_footprint proc_start_abstime proc_exit_abstime"
                ).split()
            ]
        mac_proc = ctypes.CDLL("/usr/lib/libproc.dylib")
        mac_proc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        mac_proc.proc_pid_rusage.restype = ctypes.c_int
    with log_path.open("w") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True, env=environment) as process:
            def sample():
                nonlocal peak_rss, peak_footprint
                try:
                    while process.poll() is None:
                        if active.is_set():
                            if mac_proc is not None:
                                usage = Usage()
                                if mac_proc.proc_pid_rusage(process.pid, 0, ctypes.byref(usage)) == 0:
                                    peak_rss = max(peak_rss, usage.resident_size)
                                    peak_footprint = max(peak_footprint, usage.phys_footprint)
                            else:
                                value = subprocess.run(
                                    ["ps", "-o", "rss=", "-p", str(process.pid)],
                                    capture_output=True, text=True, check=False,
                                )
                                if value.returncode == 0 and value.stdout.strip():
                                    peak_rss = max(peak_rss, int(value.stdout.strip()) * 1024)
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

