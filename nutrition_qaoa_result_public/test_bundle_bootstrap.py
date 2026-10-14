"""The two uploaded files must reconstruct a complete runnable project."""

import json
import os
import tempfile
import unittest
from pathlib import Path
import tarfile

from build_bundle import PROJECT, SOURCE_FILES


class BundleBootstrapTests(unittest.TestCase):
    def test_notebook_extracts_archive_before_looking_at_partial_upload(self):
        notebook = json.loads((PROJECT / "start_here.ipynb").read_text(encoding="utf-8"))
        first_cell = "".join(notebook["cells"][1]["source"])
        with tempfile.TemporaryDirectory() as temporary:
            upload_dir = Path(temporary)
            with tarfile.open(upload_dir / "nutrition_qaoa_extended.tar.gz", "w:gz") as bundle:
                for name in SOURCE_FILES:
                    bundle.add(PROJECT / name, arcname=name, recursive=False)
            # Mirrors the user's partial upload, where Python files exist but data/ does not.
            (upload_dir / "run_experiments.py").write_text("# incomplete upload\n", encoding="utf-8")
            previous_dir = Path.cwd()
            try:
                os.chdir(upload_dir)
                scope = {}
                exec(compile(first_cell, "start_here.ipynb:first_cell", "exec"), scope)
                extracted = scope["project_dir"]
                self.assertNotEqual(extracted, upload_dir)
                self.assertTrue((extracted / "data" / "foods.csv").is_file())
                self.assertTrue((extracted / "config.json").is_file())
                self.assertTrue((extracted / "gpu_probe.py").is_file())
                self.assertEqual(scope["bundle_hash"], (extracted / ".bundle_sha256").read_text(encoding="utf-8"))
            finally:
                os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
