#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
import subprocess
from pathlib import Path

PAT_K = re.compile(r"Thermal conductivity\s*\(k\)\s*is:\s*([0-9.+\-Ee]+)")
PAT_KSPEC = re.compile(r"Spectral thermal conductivity\s*\(k_spec\)\s*is\s*([0-9.+\-Ee]+)")
PAT_KQC = re.compile(r"Quantum correlated spectral thermal conductivity\s*\(k_spec\)\s*is\s*([0-9.+\-Ee]+)")

def parse_values(text: str) -> tuple[str, str, str] | None:
    m1 = PAT_K.search(text)
    m2 = PAT_KSPEC.search(text)
    m3 = PAT_KQC.search(text)
    if not (m1 and m2 and m3):
        return None
    return (m1.group(1), m2.group(1), m3.group(1))

def safe_name(rel_dir: str) -> str:
    # turn path into a safe filename
    return re.sub(r"[^0-9A-Za-z._-]+", "_", rel_dir)

def main() -> int:
    base_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
    output_file = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else (base_dir / "qctc.data")
    err_dir = base_dir / "qctc_errors"
    err_dir.mkdir(parents=True, exist_ok=True)

    base_qctc = base_dir / "qctc.py"
    if not base_qctc.exists():
        print(f"ERROR: cannot find {base_qctc}", file=sys.stderr)
        return 2

    kappa_files = sorted(base_dir.rglob("kappa.out"))
    print(f"Found {len(kappa_files)} kappa.out under {base_dir}")

    ok = 0
    fail = 0

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Directory",
            "Thermal Conductivity (k)",
            "Spectral Thermal Conductivity (k_spec)",
            "Quantum Corrected Thermal Conductivity (k_qc)",
        ])

        for kappa in kappa_files:
            workdir = kappa.parent
            rel_dir = "." if workdir == base_dir else str(workdir.relative_to(base_dir))

            # IMPORTANT: prefer per-directory qctc.py if it exists (matches your manual run behavior)
            qctc_script = workdir / "qctc.py"
            if not qctc_script.exists():
                qctc_script = base_qctc

            print(f"Processing directory: {rel_dir}")

            try:
                p = subprocess.run(
                    [sys.executable, "-u", str(qctc_script)],
                    cwd=str(workdir),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception as e:
                log = err_dir / f"{safe_name(rel_dir)}.log"
                log.write_text(
                    f"FAILED TO RUN\ncwd={workdir}\nscript={qctc_script}\nexception={e}\n",
                    encoding="utf-8",
                )
                print(f"ERROR: run failed in {rel_dir}. Logged: {log}", file=sys.stderr)
                fail += 1
                continue

            out = (p.stdout or "") + ("\n" if p.stdout and p.stderr else "") + (p.stderr or "")

            if p.returncode != 0:
                log = err_dir / f"{safe_name(rel_dir)}.log"
                log.write_text(
                    f"qctc.py FAILED\nexit={p.returncode}\ncwd={workdir}\nscript={qctc_script}\n\n=== output ===\n{out}\n",
                    encoding="utf-8",
                )
                print(f"ERROR: qctc.py failed in {rel_dir} (exit={p.returncode}). Logged: {log}", file=sys.stderr)
                fail += 1
                continue

            vals = parse_values(out)
            if not vals:
                log = err_dir / f"{safe_name(rel_dir)}_parsefail.log"
                log.write_text(
                    f"PARSE FAILED\ncwd={workdir}\nscript={qctc_script}\n\n=== output ===\n{out}\n",
                    encoding="utf-8",
                )
                print(f"Warning: parse failed in {rel_dir}. Logged: {log}", file=sys.stderr)
                fail += 1
                continue

            k_val, kspec_val, kqc_val = vals
            writer.writerow([rel_dir, k_val, kspec_val, kqc_val])
            ok += 1

    print(f"Processing complete. OK={ok} FAIL={fail}")
    print(f"Results saved to: {output_file}")
    if fail:
        print(f"Failure logs in: {err_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
