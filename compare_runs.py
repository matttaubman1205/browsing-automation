# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
compare_runs.py — Compare signals from multiple runs of your Playwright+LLM browser agent.

Inputs (positional): one or more run directories.
Each run directory should contain:
  - page_log.csv      (columns: timestamp, step, action, description, url)
  - screenshots/      (PNG files named like 001-<label>-<timestamp>.png)

Example:
    python compare_runs.py runs/1 runs/2 runs/3 runs/4 runs/5

Outputs (written next to this script unless --out is provided):
  - comparison_report.md
  - url_overlap.csv
  - selector_overlap.csv
  - screenshot_similarity.csv
  - artifacts/
      - side_by_side/<step>/<run_name>.png           (symlinks or copies)
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
from typing import Dict, List, Tuple, Optional

# Optional deps for image comparison
try:
    from PIL import Image, ImageChops
    PIL_OK = True
except Exception:  # pragma: no cover
    PIL_OK = False
try:
    import imagehash  # type: ignore
    IMAGEHASH_OK = True
except Exception:  # pragma: no cover
    IMAGEHASH_OK = False

# Optional deps for nicer tables
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
    step_to_screenshot: Dict[int, Path]  # step index -> path


FNAME_STEP_RE = re.compile(r"^(?P<idx>\d{3})-.*\.png$")


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_run(run_path: Path) -> RunData:
    log_path = run_path / "page_log.csv"
    screenshots_dir = run_path / "screenshots"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing page_log.csv in {run_path}")
    if not screenshots_dir.exists():
        # Allow empty; still create mapping from whatever exists
        screenshots_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(log_path)

    urls: List[str] = []
    selectors: List[str] = []
    # Map screenshot step index -> file (best-effort by filename prefix)
    step_to_screenshot: Dict[int, Path] = {}

    # Parse URLs and selectors from CSV
    for r in rows:
        url = (r.get("url") or "").strip()
        if url:
            urls.append(url)
        if (r.get("action") or "").strip() == "click":
            desc = (r.get("description") or "").strip()
            # In your logger, description holds the selector (or a human label).
            if desc:
                selectors.append(desc)

    # Map screenshots by leading numeric index
    if screenshots_dir.exists():
        for p in sorted(screenshots_dir.glob("*.png")):
            m = FNAME_STEP_RE.match(p.name)
            if not m:
                continue
            idx = int(m.group("idx"))
            # Keep the first encountered per index
            step_to_screenshot.setdefault(idx, p)

    return RunData(
        name=run_path.name,
        root=run_path,
        log_path=log_path,
        screenshots_dir=screenshots_dir,
        urls=urls,
        selectors=selectors,
        step_to_screenshot=step_to_screenshot,
    )


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def lcs_len(a: List[str], b: List[str]) -> int:
    # classic DP LCS length
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            if ai == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]


def phash_distance(img_a: Path, img_b: Path) -> Optional[int]:
    if not (PIL_OK and IMAGEHASH_OK):
        return None
    try:
        ha = imagehash.phash(Image.open(img_a))
        hb = imagehash.phash(Image.open(img_b))
        return ha - hb  # Hamming distance
    except Exception:
        return None


def make_visual_diff(img_a: Path, img_b: Path, out_path: Path) -> bool:
    if not PIL_OK:
        return False
    try:
        a = Image.open(img_a).convert("RGB")
        b = Image.open(img_b).convert("RGB")
        # Resize to the smaller common size
        w = min(a.width, b.width)
        h = min(a.height, b.height)
        a = a.resize((w, h))
        b = b.resize((w, h))
        diff = ImageChops.difference(a, b)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        diff.save(out_path)
        return True
    except Exception:
        return False


# ------------------------------ Main comparison ------------------------------

def compare_runs(runs: List[RunData], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # ===== URLs =====
    url_rows: List[Dict[str, str]] = []
    for i in range(len(runs)):
        for j in range(i+1, len(runs)):
            r1, r2 = runs[i], runs[j]
            jac = jaccard(r1.urls, r2.urls)
            lcs = lcs_len(r1.urls, r2.urls)
            denom = max(len(r1.urls), len(r2.urls)) or 1
            lcs_pct = lcs / denom
            url_rows.append({
                "run_a": r1.name,
                "run_b": r2.name,
                "jaccard": f"{jac:.3f}",
                "lcs_len": str(lcs),
                "lcs_pct": f"{lcs_pct:.3f}",
                "len_a": str(len(r1.urls)),
                "len_b": str(len(r2.urls)),
            })

    url_csv = outdir / "url_overlap.csv"
    with url_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_a","run_b","jaccard","lcs_len","lcs_pct","len_a","len_b"])
        writer.writeheader()
        writer.writerows(url_rows)

    # ===== Selectors =====
    sel_rows: List[Dict[str, str]] = []
    for i in range(len(runs)):
        for j in range(i+1, len(runs)):
            r1, r2 = runs[i], runs[j]
            jac = jaccard(r1.selectors, r2.selectors)
            sel_rows.append({
                "run_a": r1.name,
                "run_b": r2.name,
                "jaccard": f"{jac:.3f}",
                "count_a": str(len(r1.selectors)),
                "count_b": str(len(r2.selectors)),
            })

    sel_csv = outdir / "selector_overlap.csv"
    with sel_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_a","run_b","jaccard","count_a","count_b"])
        writer.writeheader()
        writer.writerows(sel_rows)

    # ===== Screenshots =====
    # Build union of step indices across all runs
    all_steps: List[int] = sorted({idx for r in runs for idx in r.step_to_screenshot.keys()})

    sim_rows: List[Dict[str, str]] = []
    for step in all_steps:
        row: Dict[str, str] = {"step": f"{step:03d}"}
        # For each run, note presence
        for r in runs:
            row[f"has_{r.name}"] = "1" if step in r.step_to_screenshot else "0"
        # Pairwise distances
        for i in range(len(runs)):
            for j in range(i+1, len(runs)):
                key = f"d({runs[i].name},{runs[j].name})"
                pa = runs[i].step_to_screenshot.get(step)
                pb = runs[j].step_to_screenshot.get(step)
                if pa and pb:
                    dist = phash_distance(pa, pb)
                    row[key] = str(dist) if dist is not None else "NA"
                else:
                    row[key] = ""
        sim_rows.append(row)

    ss_csv = outdir / "screenshot_similarity.csv"
    # Build header dynamically
    header = ["step"] + [f"has_{r.name}" for r in runs]
    for i in range(len(runs)):
        for j in range(i+1, len(runs)):
            header.append(f"d({runs[i].name},{runs[j].name})")
    with ss_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(sim_rows)

    # ===== Optional artifacts: side-by-side and diffs =====
    artifacts = outdir / "artifacts"
    sbs_root = artifacts / "side_by_side"
    diffs_root = artifacts / "diffs"
    for step in all_steps:
        step_dir = sbs_root / f"{step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for r in runs:
            p = r.step_to_screenshot.get(step)
            if not p:
                continue
            # Copy (or link) to a common folder for quick inspection
            target = step_dir / f"{r.name}.png"
            try:
                if target.exists():
                    target.unlink()
                # Prefer hardlink/symlink to save space
                try:
                    os.link(p, target)
                except Exception:
                    try:
                        os.symlink(p, target)
                    except Exception:
                        # Fallback: copy bytes
                        with open(p, "rb") as src, open(target, "wb") as dst:
                            dst.write(src.read())
            except Exception:
                pass
        # Make a visual diff for the first pair (if possible)
        if PIL_OK:
            present = [(r.name, r.step_to_screenshot.get(step)) for r in runs if step in r.step_to_screenshot]
            if len(present) >= 2:
                (n1, p1), (n2, p2) = present[0], present[1]
                if p1 and p2:
                    outp = diffs_root / f"{step:03d}" / f"{n1}_vs_{n2}.png"
                    make_visual_diff(p1, p2, outp)

    # ===== Report =====
    report = outdir / "comparison_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Run Comparison Report\n\n")
        f.write("## Inputs\n")
        for r in runs:
            f.write(f"- **{r.name}** — {r.root}\n")
        f.write("\n---\n")

        f.write("## URL Similarity\n")
        f.write(f"CSV: `{url_csv}`\n\n")
        if PD_OK:
            df = pd.DataFrame(url_rows)
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
        else:
            for row in url_rows:
                f.write(f"- {row['run_a']} vs {row['run_b']}: Jaccard {row['jaccard']}, LCS {row['lcs_len']} (pct {row['lcs_pct']})\n")
            f.write("\n")

        f.write("## Selector Overlap (click actions)\n")
        f.write(f"CSV: `{sel_csv}`\n\n")
        if PD_OK:
            df = pd.DataFrame(sel_rows)
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
        else:
            for row in sel_rows:
                f.write(f"- {row['run_a']} vs {row['run_b']}: Jaccard {row['jaccard']} (counts {row['count_a']} vs {row['count_b']})\n")
            f.write("\n")

        f.write("## Screenshot Similarity (phash Hamming distance)\n")
        if not (PIL_OK and IMAGEHASH_OK):
            f.write("_Screenshot hashing skipped: install Pillow and imagehash to enable._\n\n")
        f.write(f"CSV: `{ss_csv}`\n\n")
        f.write(f"Artifacts: `{artifacts}` — side-by-side images per step; simple visual diffs for first pair per step.\n\n")
        f.write("**Interpretation tip:** Hamming distance 0–4 ~ very similar; 5–10 ~ related; >10 ~ different (rule of thumb).\n\n")


# ------------------------------ CLI ------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Compare selectors, URLs, and screenshots across multiple runs.")
    ap.add_argument("runs", nargs="+", help="Run directories containing page_log.csv and screenshots/")
    ap.add_argument("--out", default="comparison_output", help="Output directory for reports and artifacts")
    args = ap.parse_args()

    run_dirs = [Path(r).resolve() for r in args.runs]
    if len(run_dirs) < 2:
        print("Please provide at least two run directories.", file=sys.stderr)
        return 2

    runs: List[RunData] = []
    for rd in run_dirs:
        try:
            runs.append(load_run(rd))
        except Exception as e:
            print(f"Error loading run {rd}: {e}", file=sys.stderr)
            return 1

    outdir = Path(args.out).resolve()
    compare_runs(runs, outdir)
    print(f"Wrote report to {outdir / 'comparison_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
