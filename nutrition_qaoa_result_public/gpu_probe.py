"""Read-only CUDA availability check with useful diagnostics for Jupyter jobs."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import socket
import subprocess


def _cuda_driver_details() -> dict:
    details = {
        "library": "libcuda.so.1",
        "load_error": None,
        "init_result": None,
        "device_count_result": None,
        "device_count": 0,
        "driver_version_result": None,
        "driver_version_raw": None,
        "driver_cuda_major": None,
    }
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        details["load_error"] = str(exc)
        return details

    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    details["init_result"] = int(driver.cuInit(0))
    if details["init_result"] != 0:
        return details

    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = ctypes.c_int
    count = ctypes.c_int()
    details["device_count_result"] = int(driver.cuDeviceGetCount(ctypes.byref(count)))
    if details["device_count_result"] == 0:
        details["device_count"] = int(count.value)

    driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDriverGetVersion.restype = ctypes.c_int
    version = ctypes.c_int()
    details["driver_version_result"] = int(driver.cuDriverGetVersion(ctypes.byref(version)))
    if details["driver_version_result"] == 0:
        details["driver_version_raw"] = int(version.value)
        details["driver_cuda_major"] = int(version.value) // 1000
    return details


def probe_gpu() -> dict:
    """Report devices available *inside this kernel*, not on another node."""
    system = platform.system()
    machine = platform.machine()
    linux_x64 = system == "Linux" and machine.lower() in {"x86_64", "amd64"}
    result = {
        "hostname": socket.gethostname(),
        "system": system,
        "machine": machine,
        "linux_x64": linux_x64,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "PBS_GPUFILE",
            )
        },
        "nvidia_smi_path": shutil.which("nvidia-smi"),
        "nvidia_smi_exit_code": None,
        "nvidia_smi_stdout": "",
        "nvidia_smi_stderr": "",
        "nvidia_smi_error": None,
        "cuda": None,
        "visible": False,
        "reason": "",
    }
    if result["nvidia_smi_path"]:
        try:
            completed = subprocess.run(
                [result["nvidia_smi_path"]], capture_output=True,
                text=True, errors="replace", timeout=15, check=False,
            )
            result["nvidia_smi_exit_code"] = completed.returncode
            result["nvidia_smi_stdout"] = completed.stdout
            result["nvidia_smi_stderr"] = completed.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"

    if not linux_x64:
        result["reason"] = "Aer GPU wheel requires x86-64 Linux"
        return result
    try:
        result["cuda"] = _cuda_driver_details()
    except (AttributeError, OSError) as exc:
        result["cuda"] = {"probe_error": f"{type(exc).__name__}: {exc}"}
        result["reason"] = "CUDA driver API probe failed"
        return result
    cuda = result["cuda"]
    result["visible"] = cuda.get("device_count", 0) > 0
    if result["visible"]:
        result["reason"] = f"CUDA driver reports {cuda['device_count']} usable device(s)"
    elif cuda.get("load_error"):
        result["reason"] = "CUDA driver library libcuda.so.1 is unavailable in this kernel"
    elif cuda.get("init_result") != 0:
        result["reason"] = f"CUDA driver initialization failed (code {cuda['init_result']})"
    elif cuda.get("device_count_result") != 0:
        result["reason"] = f"CUDA device count failed (code {cuda['device_count_result']})"
    else:
        result["reason"] = "CUDA driver reports zero usable devices in this kernel"
    return result
