"""
Batch OCR runner for ZIP files of newspaper PDFs.

What it does:
  1. Extracts a ZIP file containing PDFs.
  2. Runs run_hybrid_fast_reuse.py on every PDF.
  3. Optionally runs remove_page_furniture_preserve_blanks.py.
  4. Copies one final TXT transcript per PDF into final_txt/.
  5. Creates a ZIP file containing all TXT transcripts.

Required files in the same folder:
  run_batch_zip_ocr.py
  run_hybrid_fast_reuse.py
  hybrid_ppstructure_tesseract.py
  remove_page_furniture_preserve_blanks.py

Example on Windows:
  python run_batch_zip_ocr.py --zip newspapers20.zip --dpi 400 --workers 6 --device gpu:0

Example with explicit Tesseract path:
  python run_batch_zip_ocr.py --zip newspapers20.zip --dpi 400 --workers 6 --device gpu:0 --tesseract "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

Fast test on first 2 PDFs:
  python run_batch_zip_ocr.py --zip newspapers20.zip --limit 2 --dpi 300 --workers 4

Resume after interruption:
  python run_batch_zip_ocr.py --zip newspapers20.zip --dpi 400 --workers 6 --skip-existing

Main output:
  batch_ocr_output\final_txt_results.zip
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import fitz


def print_step(message: str) -> None:
    print(f"\n{'=' * 80}\n{message}\n{'=' * 80}", flush=True)


def run(cmd: List[str], cwd: Path, env: dict, stop_on_error: bool) -> Tuple[bool, str]:
    printable = " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd)
    print("\n>>> " + printable, flush=True)

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=None,
        stderr=None,
        text=True,
    )

    if proc.returncode == 0:
        return True, ""

    msg = f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}"
    if stop_on_error:
        raise RuntimeError(msg)

    print(f"\nWARNING: {msg}", flush=True)
    return False, msg


def count_pages(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    pages = len(doc)
    doc.close()
    return pages


def extract_zip(zip_path: Path, extract_dir: Path, overwrite: bool) -> None:
    if overwrite and extract_dir.exists():
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    existing_pdfs = list(extract_dir.rglob("*.pdf"))
    if existing_pdfs and not overwrite:
        print(f"Using already extracted PDFs: {len(existing_pdfs)} found in {extract_dir}")
        return

    print_step(f"Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)


def safe_stem(pdf_path: Path, used: Dict[str, int]) -> str:
    stem = pdf_path.stem.strip()
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)
    stem = stem.strip("._-") or "document"

    if len(stem) > 90:
        digest = hashlib.md5(str(pdf_path).encode("utf-8")).hexdigest()[:8]
        stem = stem[:80] + "_" + digest

    if stem not in used:
        used[stem] = 1
        return stem

    used[stem] += 1
    return f"{stem}_{used[stem]:03d}"


def find_tesseract(user_path: str) -> str:
    if user_path:
        p = Path(user_path)
        if p.exists():
            return str(p)
        found = shutil.which(user_path)
        if found:
            return found
        raise FileNotFoundError(f"Tesseract executable not found: {user_path}")

    found = shutil.which("tesseract")
    if found:
        return found

    win_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if win_default.exists():
        return str(win_default)

    raise FileNotFoundError(
        "Could not find Tesseract. Use --tesseract with the full path, for example "
        r'--tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"'
    )


def make_final_zip(final_txt_dir: Path, zip_output: Path) -> None:
    if zip_output.exists():
        zip_output.unlink()

    zip_output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for txt in sorted(final_txt_dir.glob("*.txt")):
            z.write(txt, arcname=txt.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="ZIP file containing PDF newspapers")
    parser.add_argument("--output-root", default="batch_ocr_output", help="Main output folder")
    parser.add_argument("--extract-dir", default="", help="Default: output-root/extracted_pdfs")
    parser.add_argument("--work-root", default="", help="Default: output-root/work")
    parser.add_argument("--final-txt-dir", default="", help="Default: output-root/final_txt")
    parser.add_argument("--zip-output", default="", help="Default: output-root/final_txt_results.zip")

    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--lang", default="kaz+eng")
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--min-conf", type=float, default=45.0)
    parser.add_argument("--min-area", type=int, default=5000)
    parser.add_argument("--tesseract", default="", help="Full path to tesseract.exe. Empty = auto-detect.")

    parser.add_argument("--limit", type=int, default=0, help="Process only first N PDFs. 0 = all.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based index in sorted PDF list")
    parser.add_argument("--skip-existing", action="store_true", help="Skip PDFs whose final TXT already exists")
    parser.add_argument("--overwrite-extract", action="store_true", help="Delete and re-extract ZIP contents")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop batch if one PDF fails")

    parser.add_argument("--no-furniture-clean", action="store_true", help="Skip remove_page_furniture_preserve_blanks.py")
    parser.add_argument("--no-remove-section-labels", action="store_true", help="Pass to furniture cleaner")
    parser.add_argument("--top-margin", type=float, default=0.13)
    parser.add_argument("--bottom-margin", type=float, default=0.88)
    parser.add_argument("--block-separator-lines", type=int, default=1)

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    zip_path = Path(args.zip).resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    output_root = Path(args.output_root).resolve()
    extract_dir = Path(args.extract_dir).resolve() if args.extract_dir else output_root / "extracted_pdfs"
    work_root = Path(args.work_root).resolve() if args.work_root else output_root / "work"
    final_txt_dir = Path(args.final_txt_dir).resolve() if args.final_txt_dir else output_root / "final_txt"
    zip_output = Path(args.zip_output).resolve() if args.zip_output else output_root / "final_txt_results.zip"

    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    final_txt_dir.mkdir(parents=True, exist_ok=True)

    required_scripts = [
        script_dir / "run_hybrid_fast_reuse.py",
        script_dir / "hybrid_ppstructure_tesseract.py",
    ]

    if not args.no_furniture_clean:
        required_scripts.append(script_dir / "remove_page_furniture_preserve_blanks.py")

    missing = [str(p) for p in required_scripts if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required script(s):\n" + "\n".join(missing))

    tesseract_path = find_tesseract(args.tesseract)
    print(f"Using Tesseract: {tesseract_path}")

    extract_zip(zip_path, extract_dir, overwrite=args.overwrite_extract)

    pdfs = sorted(extract_dir.rglob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDF files found after extraction: {extract_dir}")

    if args.start_index > 1:
        pdfs = pdfs[args.start_index - 1 :]

    if args.limit > 0:
        pdfs = pdfs[: args.limit]

    print_step(f"PDFs to process: {len(pdfs)}")

    used_names: Dict[str, int] = {}
    failures: List[str] = []

    env = os.environ.copy()
    env.setdefault("OMP_THREAD_LIMIT", "1")

    for i, pdf in enumerate(pdfs, start=1):
        out_name = safe_stem(pdf, used_names)
        pdf_work_dir = work_root / out_name
        final_txt = final_txt_dir / f"{out_name}.txt"

        if args.skip_existing and final_txt.exists() and final_txt.stat().st_size > 0:
            print(f"[{i}/{len(pdfs)}] Skipping existing: {final_txt}")
            continue

        try:
            pages = count_pages(pdf)
        except Exception as exc:
            msg = f"{pdf}\tPAGE_COUNT_ERROR\t{exc}"
            failures.append(msg)
            print(f"WARNING: {msg}")
            if args.stop_on_error:
                raise
            continue

        print_step(f"[{i}/{len(pdfs)}] {pdf.name} | pages={pages}")

        ocr_cmd = [
            sys.executable,
            str(script_dir / "run_hybrid_fast_reuse.py"),
            "--pdf", str(pdf),
            "--pages", str(pages),
            "--dpi", str(args.dpi),
            "--workers", str(args.workers),
            "--device", args.device,
            "--lang", args.lang,
            "--psm", str(args.psm),
            "--min-conf", str(args.min_conf),
            "--min-area", str(args.min_area),
            "--tesseract", tesseract_path,
            "--output-root", str(pdf_work_dir),
        ]

        ok, err = run(ocr_cmd, cwd=script_dir, env=env, stop_on_error=args.stop_on_error)
        if not ok:
            failures.append(f"{pdf}\tOCR_ERROR\t{err}")
            continue

        if args.no_furniture_clean:
            source_txt = pdf_work_dir / "all_pages_tsv_filtered.txt"
        else:
            clean_cmd = [
                sys.executable,
                str(script_dir / "remove_page_furniture_preserve_blanks.py"),
                "--root", str(pdf_work_dir),
                "--pages", str(pages),
                "--top-margin", str(args.top_margin),
                "--bottom-margin", str(args.bottom_margin),
                "--block-separator-lines", str(args.block_separator_lines),
            ]

            if args.no_remove_section_labels:
                clean_cmd.append("--no-remove-section-labels")

            ok, err = run(clean_cmd, cwd=script_dir, env=env, stop_on_error=args.stop_on_error)
            if not ok:
                failures.append(f"{pdf}\tFURNITURE_CLEAN_ERROR\t{err}")
                source_txt = pdf_work_dir / "all_pages_tsv_filtered.txt"
            else:
                source_txt = pdf_work_dir / "all_pages_no_furniture.txt"

        if not source_txt.exists():
            msg = f"{pdf}\tMISSING_OUTPUT\t{source_txt}"
            failures.append(msg)
            print(f"WARNING: {msg}")
            if args.stop_on_error:
                raise FileNotFoundError(msg)
            continue

        shutil.copyfile(source_txt, final_txt)
        print(f"Saved TXT: {final_txt}", flush=True)

        make_final_zip(final_txt_dir, zip_output)
        print(f"Updated ZIP: {zip_output}", flush=True)

    failure_path = output_root / "failed_pdfs.txt"
    failure_path.write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")

    make_final_zip(final_txt_dir, zip_output)

    print_step("DONE")
    print(f"TXT folder: {final_txt_dir}")
    print(f"Final ZIP:  {zip_output}")
    print(f"Failures:   {len(failures)}")
    print(f"Failure log: {failure_path}")


if __name__ == "__main__":
    main()
