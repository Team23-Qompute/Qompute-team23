"""Checks for durable progress and heartbeat diagnostics."""

import json
import io
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from progress import ProgressLog


class ProgressTests(unittest.TestCase):
    def test_records_status_and_heartbeats(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            log = ProgressLog(Path(temporary), heartbeat_seconds=0.02)
            log.start(arguments={"device": "CPU"})
            log.event("estimator_eval_start", phase="estimator", console=False,
                      evaluation=3)
            time.sleep(0.07)
            during = json.loads(log.status_path.read_text(encoding="utf-8"))
            self.assertEqual(during["phase"], "estimator")
            self.assertEqual(during["current_operation"]["evaluation"], 3)
            self.assertGreaterEqual(during["seconds_since_progress"], 0)
            log.event("run_complete", phase="complete", state="complete")
            log.stop()
            final = json.loads(log.status_path.read_text(encoding="utf-8"))
            events = [json.loads(line) for line in log.events_path.read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual(final["state"], "complete")
            self.assertIn("heartbeat", [event["event"] for event in events])
            self.assertFalse(log.status_path.with_suffix(".json.tmp").exists())

    def test_rejects_nonpositive_heartbeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                ProgressLog(Path(temporary), heartbeat_seconds=0)

    def test_failure_preserves_last_active_step(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            log = ProgressLog(Path(temporary))
            log.start()
            log.event("estimator_eval_start", phase="estimator", console=False,
                      evaluation=7)
            log.event("run_failed", phase="failed", state="failed",
                      error_type="RuntimeError", error="GPU failed")
            log.stop()
            status = json.loads(log.status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["last_active_phase"], "estimator")
            self.assertEqual(status["last_active_operation"]["evaluation"], 7)


if __name__ == "__main__":
    unittest.main()
