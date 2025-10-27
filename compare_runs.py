# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
compare_runs.py — Compare signals from multiple runs of your Playwright+LLM browser agent.


Inputs (positional): one or more run directories.
Each run directory should contain:
- page_log.csv (columns: timestamp, step, action, description, url)
- screenshots/ (PNG files named like 001-<label>-<timestamp>.png)


Example:
python compare_runs.py runs/1 runs/2 runs/3 runs/4 runs/5


Outputs (written next to this script unless --out is provided):
- comparison_report.md
- url_overlap.csv
- selector_overlap.csv
- screenshot_similarity.csv
- artifacts/
- side_by_side/<step>/<run_name>.png (symlinks or copies)
- diffs/<step>/<pair>.png (optional; if pillow is available, visual diffs)


What is compared:
1) URLs — per-run sequences, pairwise Jaccard (set) + LCS (sequence) similarity and stepwise agreement.
2) Selectors — extracted from rows where action == 'click' (using description; falls back to selector if present there).
3) Screenshots — image perceptual hashes (phash) via imagehash (if available) and PIL; returns Hamming distances.


Notes:
- Works with any N>=2 runs (not hard-coded to 5), but your use-case likely uses 5.
- If imagehash/Pillow are not installed, screenshot comparison is skipped with a warning.
- The script tries to align steps by the numeric prefix in screenshot filenames (001, 002, ...).
- If a run lacks a screenshot for a given step index, that cell will be empty in the similarity CSV.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Optional deps for image comparison
try:
    from PIL import Image, ImageChops
    PIL_OK = True
except Exception:
    PIL_OK = False

try:
    import imagehash  # type: ignore
    IMAGEHASH_OK = True
except Exception:
    IMAGEHASH_OK = False

try:
    import pandas as pd  # type: ignore
    PD_OK = True
except Exception:
    PD_OK = False


# ------------------------------ Utilities ------------------------------


@dataclass
class RunData:
name: str
root: Path
log_path: Path
screenshots_dir: Path
urls: List[str]
selectors: List[str]
step_to_screenshot: Dict[int, Path] # step index -> path
