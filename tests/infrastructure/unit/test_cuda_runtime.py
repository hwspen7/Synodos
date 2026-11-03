"""Windows CUDA DLL search paths must be explicit and scoped to the current process."""
from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from speech.cuda_runtime import candidate_directories, prepare_windows_cuda_runtime


class CudaRuntimeTests(unittest.TestCase):
    def test_local_torch_lib_is_first_candidate(self):
        paths = candidate_directories(Path("D:/work/meeting/.venv"), "C:/CUDA/v12")
        self.assertEqual(paths[0], Path("D:/work/meeting/.venv") / "Lib/site-packages/torch/lib")
        self.assertEqual(paths[-1], Path("C:/CUDA/v12/bin"))

    def test_non_windows_does_not_mutate_path(self):
        before = os.environ.get("PATH", "")
        if os.name != "nt":
            with patch.dict(os.environ, {}, clear=False):
                self.assertEqual(prepare_windows_cuda_runtime(), ())
                self.assertEqual(os.environ.get("PATH", ""), before)


if __name__ == "__main__":
    unittest.main()
