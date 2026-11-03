# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RewriteSinkTests(unittest.TestCase):
    def test_input_aliases_are_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.vortex"
            source.write_bytes(b"fixture must survive")
            symlink = root / "symlink.vortex"
            symlink.symlink_to(source)
            hardlink = root / "hardlink.vortex"
            os.link(source, hardlink)
            for sink in (source, symlink, hardlink):
                with self.subTest(sink=sink.name):
                    output = root / "results"
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("run.py")),
                            "--input",
                            str(source),
                            "--sink",
                            str(sink),
                            "--variants",
                            "ordered-rewrite-baseline",
                            "--native",
                            str(root / "unused-native"),
                            "--classpath-file",
                            str(root / "unused-classpath"),
                            "--java-home",
                            str(root / "unused-java"),
                            "--output",
                            str(output),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("must not overwrite the input file", result.stderr)
                    self.assertEqual(source.read_bytes(), b"fixture must survive")
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
