"""The GPU probe must report unavailable devices without masking the reason."""

import unittest
from unittest.mock import patch

from gpu_probe import probe_gpu


class GpuProbeTests(unittest.TestCase):
    @patch("gpu_probe.shutil.which", return_value=None)
    @patch("gpu_probe.platform.machine", return_value="x86_64")
    @patch("gpu_probe.platform.system", return_value="Windows")
    def test_non_linux_is_not_reported_as_gpu(self, _system, _machine, _which):
        result = probe_gpu()
        self.assertFalse(result["visible"])
        self.assertIn("x86-64 Linux", result["reason"])

    @patch("gpu_probe._cuda_driver_details", return_value={"device_count": 0, "init_result": 0, "device_count_result": 0})
    @patch("gpu_probe.shutil.which", return_value=None)
    @patch("gpu_probe.platform.machine", return_value="x86_64")
    @patch("gpu_probe.platform.system", return_value="Linux")
    def test_zero_cuda_devices_is_not_silent(self, _system, _machine, _which, _cuda):
        result = probe_gpu()
        self.assertFalse(result["visible"])
        self.assertIn("zero usable devices", result["reason"])

    @patch("gpu_probe._cuda_driver_details", return_value={"device_count": 1, "driver_cuda_major": 12})
    @patch("gpu_probe.shutil.which", return_value=None)
    @patch("gpu_probe.platform.machine", return_value="x86_64")
    @patch("gpu_probe.platform.system", return_value="Linux")
    def test_cuda_device_is_found_without_nvidia_smi(self, _system, _machine, _which, _cuda):
        result = probe_gpu()
        self.assertTrue(result["visible"])
        self.assertEqual(result["cuda"]["driver_cuda_major"], 12)


if __name__ == "__main__":
    unittest.main()
