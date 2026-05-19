"""
Fast full-PDF runner using your existing hybrid_ppstructure_tesseract.py helpers.

Main speedups:
  1. Loads PP-StructureV3 once for the whole PDF instead of once per page.
  2. Uses Tesseract TSV once per crop, so confidence filtering is included immediately.
  3. Runs Tesseract on crop blocks in parallel with --workers.
  4. Does not call clean_hybrid_text.py or reocr_filter_tsv_confidence.py.

Put this file in the same folder as:
    hybrid_ppstructure_tesseract.py

Run:
    python run_hybrid_fast_reuse.py --pdf input_newspaper.pdf --pages 44 --dpi 300 --workers 4

Higher quality, slower:
    python run_hybrid_fast_reuse.py --pdf input_newspaper.pdf --pages 44 --dpi 400 --workers 4

Output:
    hybrid_fast_full_pdf\all_pages_tsv_filtered.txt
    hybrid_fast_full_pdf\tsv_suspect_lines.txt
    hybrid_fast_full_pdf\page_001\clean_text_tsv_filtered.txt
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from paddleocr import PPStructureV3

from hybrid_ppstructure_tesseract import (
    render_pdf_page,
    load_json_files,
    extract_blocks,
    is_text_block,
    clip_bbox,
    draw_debug_image,
    find_tesseract,
)


GARBAGE_PATTERNS = [
    r"\bЕЕЕ\b",
    r"\bкк\b.*\bЧИН\b",
    r"\bЧИН\b",
    r"\bБЕРАЕРИИЕ\b",
    r"\bБЕРЕШЕК\b",
    r"\bШЕР\b.*\bЕЕЕ\b",
    r"\bКЕК\b",
    r"\bРЕКЕ\b",
    r"\bИИК\b",
    r"\bЧЛЕН\b",
    r"\bкЕЕЭ\b",
]

CYRILLIC_RE = re.compile(r"[А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІіЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"[A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІіЁё0-9]+")
KAZAKH_VOWELS = set("аәеёиіоөұүуыэюяАӘЕЁИІОӨҰҮУЫЭЮЯ")


def create_ppstructure(device: str) -> PPStructureV3:
    print("Loading PP-StructureV3 once...")
    return PPStructureV3(
        device=device,
        text_recognition_model_name="cyrillic_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_table_recognition=False,
        use_formula_recognition=False,
        use_seal_recognition=False,
        use_chart_recognition=False,
    )


def run_ppstructure_reuse(pipeline: PPStructureV3, image_path: Path, output_dir: Path) -> List[Path]:
    pp_dir = output_dir / "ppstructure_raw"
    if pp_dir.exists():
        shutil.rmtree(pp_dir)
    pp_dir.mkdir(parents=True, exist_ok=True)

    results = pipeline.predict(str(image_path))
    for result in results:
        # Do not call result.print() or save markdown. JSON only is faster.
        result.save_to_json(str(pp_dir))

    json_files = sorted(pp_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"PP-StructureV3 did not produce JSON files in {pp_dir}")
    return json_files


def run_tesseract_tsv(tesseract_path: str, image_path: Path, lang: str, psm: int, dpi: int) -> str:
    cmd = [
        tesseract_path,
        str(image_path),
        "stdout",
        "-l",
        lang,
        "--oem",
        "1",
        "--psm",
        str(psm),
        "--dpi",
        str(dpi),
        "tsv",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout


def parse_tsv_lines(tsv_text: str) -> List[Dict[str, object]]:
    reader = csv.DictReader(tsv_text.splitlines(), delimiter="\t")
    grouped: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            conf = float(row.get("conf", "-1"))
        except ValueError:
            conf = -1
        if conf < 0:
            continue

        key = (row.get("block_num", ""), row.get("par_num", ""), row.get("line_num", ""))
        grouped.setdefault(key, {"words": [], "confs": [], "tops": [], "lefts": []})
        grouped[key]["words"].append(text)
        grouped[key]["confs"].append(conf)
        try:
            grouped[key]["tops"].append(int(row.get("top", 0)))
            grouped[key]["lefts"].append(int(row.get("left", 0)))
        except ValueError:
            pass

    lines = []
    for key, item in grouped.items():
        words = item["words"]
        confs = item["confs"]
        lines.append(
            {
                "key": key,
                "text": " ".join(words).strip(),
                "avg_conf": sum(confs) / len(confs) if confs else 0.0,
                "top": min(item["tops"]) if item["tops"] else 0,
                "left": min(item["lefts"]) if item["lefts"] else 0,
            }
        )

    lines.sort(key=lambda x: (int(x["key"][0] or 0), int(x["key"][1] or 0), int(x["key"][2] or 0), x["top"], x["left"]))
    return lines


def token_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def uppercase_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch.upper() == ch and ch.lower() != ch) / len(letters)


def latin_ratio(text: str) -> float:
    cyr = len(CYRILLIC_RE.findall(text))
    lat = len(LATIN_RE.findall(text))
    return lat / max(1, cyr + lat)


def vowel_ratio(text: str) -> float:
    cyr = len(CYRILLIC_RE.findall(text))
    if not cyr:
        return 0.0
    return sum(1 for ch in text if ch in KAZAKH_VOWELS) / cyr




def is_garbage(text: str, avg_conf: float, min_conf: float) -> Tuple[bool, str]:
    s = text.strip()
    if not s:
        return True, "empty"

    n_tokens = token_count(s)
    up = uppercase_ratio(s)
    lat = latin_ratio(s)
    vow = vowel_ratio(s)

    if n_tokens <= 4 and len(s) <= 70 and lat > 0.25 and up > 0.60:
        return True, f"mixed_uppercase_running_header:latin={lat:.2f},upper={up:.2f},conf={avg_conf:.1f}"
    
    if avg_conf >= 90 and lat < 0.90:
        return False, ""

    for pat in GARBAGE_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            if len(s) <= 90 or avg_conf < 70:
                return True, f"garbage_pattern:{pat}"

    if avg_conf < min_conf:
        return True, f"low_conf:{avg_conf:.1f}"

    if n_tokens <= 5 and up > 0.70 and avg_conf < 75:
        return True, f"short_uppercase_noise:upper={up:.2f},conf={avg_conf:.1f}"

    if lat > 0.70 and len(s.strip()) <= 4 and avg_conf < 80:
        return True, f"tiny_latin_noise:latin={lat:.2f},conf={avg_conf:.1f}"

    if len(CYRILLIC_RE.findall(s)) >= 8 and vow < 0.15 and avg_conf < 80:
        return True, f"low_vowel_noise:vowel={vow:.2f},conf={avg_conf:.1f}"

    if re.search(r"\b([А-ЯӘҒҚҢӨҰҮҺІЁ]{2,})\b(?:\s+\1\b)+", s):
        return True, "repeated_upper_token"

    if re.search(r"([А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІіЁё])\1{4,}", s):
        return True, "repeated_char_run"

    return False, ""


def ocr_crop_job(idx: int, crop_path: Path, tesseract_path: str, lang: str, psm: int, dpi: int, min_conf: float) -> Dict[str, object]:
    try:
        tsv = run_tesseract_tsv(tesseract_path, crop_path, lang, psm, dpi)
        lines = parse_tsv_lines(tsv)
    except Exception as exc:
        return {"idx": idx, "text": "", "suspects": [f"BLOCK {idx}\tERROR\t{crop_path}\t{exc}"]}

    kept = []
    suspects = []
    for line in lines:
        text = str(line["text"]).strip()
        avg_conf = float(line["avg_conf"])
        remove, reason = is_garbage(text, avg_conf, min_conf)
        if remove:
            suspects.append(f"BLOCK {idx}\tconf={avg_conf:.1f}\t{reason}\t{text}")
        else:
            kept.append(text)

    return {"idx": idx, "text": "\n".join(kept).strip(), "suspects": suspects}


def process_page(page: int, pipeline: PPStructureV3, pdf_path: Path, output_root: Path, args: argparse.Namespace, tesseract_path: str) -> Tuple[str, List[str]]:
    page_dir = output_root / f"page_{page:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)

    image_path = page_dir / f"page_{page:03d}_{args.dpi}dpi.png"
    if args.reuse_images and image_path.exists():
        print(f"Using existing image: {image_path}")
    else:
        print(f"Rendering page {page} at {args.dpi} DPI -> {image_path}")
        render_pdf_page(pdf_path, page, args.dpi, image_path)

    print(f"Running PP-StructureV3 page {page}...")
    json_files = run_ppstructure_reuse(pipeline, image_path, page_dir)
    blocks = extract_blocks(load_json_files(json_files))

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    filtered = []
    for block in blocks:
        x1, y1, x2, y2 = clip_bbox(block["bbox"], width, height, args.padding)
        area = (x2 - x1) * (y2 - y1)
        block["bbox"] = (x1, y1, x2, y2)
        block["area"] = area
        if area < args.min_area:
            continue
        if not args.keep_images and not is_text_block(block):
            continue
        filtered.append(block)

    print(f"Detected blocks: {len(blocks)}")
    print(f"Text blocks to OCR: {len(filtered)}")

    if args.debug_image:
        draw_debug_image(image_path, filtered, page_dir / "hybrid_block_order.png")

    crops_dir = page_dir / "crops"
    if crops_dir.exists() and not args.reuse_crops:
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for idx, block in enumerate(filtered, start=1):
        x1, y1, x2, y2 = block["bbox"]
        label = block.get("label", "") or "text"
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)[:40]
        crop_path = crops_dir / f"block_{idx:03d}_{safe_label}.png"
        if not (args.reuse_crops and crop_path.exists()):
            image.crop((x1, y1, x2, y2)).save(crop_path)
        block["crop_path"] = str(crop_path)
        jobs.append((idx, block, crop_path))

    results = []
    print(f"Running Tesseract TSV with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(ocr_crop_job, idx, crop_path, tesseract_path, args.lang, args.psm, args.dpi, args.min_conf)
            for idx, _, crop_path in jobs
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda x: int(x["idx"]))

    block_by_idx = {idx: block for idx, block, _ in jobs}
    page_parts = []
    page_suspects = []
    jsonl_rows = []

    for result in results:
        idx = int(result["idx"])
        block = block_by_idx[idx]
        text = str(result["text"]).strip()
        label = block.get("label", "") or "text"
        x1, y1, x2, y2 = block["bbox"]

        for s in result["suspects"]:
            page_suspects.append(f"PAGE {page}\t{s}")

        if args.include_block_headers:
            page_parts.append(f"=== BLOCK {idx:03d} | label={label} | bbox={x1},{y1},{x2},{y2} ===")
        if text:
            page_parts.append(text)
            page_parts.append("")

        jsonl_rows.append(
            {
                "page": page,
                "block_order": idx,
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "crop_path": block["crop_path"],
                "text": text,
                "pp_text": block.get("pp_text", ""),
            }
        )

    page_text = "\n".join(page_parts).strip() + "\n"
    (page_dir / "clean_text_tsv_filtered.txt").write_text(page_text, encoding="utf-8")
    (page_dir / "hybrid_ordered_text.txt").write_text(page_text, encoding="utf-8")

    with (page_dir / "hybrid_chunks.jsonl").open("w", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return page_text, page_suspects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="input_newspaper.pdf")
    parser.add_argument("--pages", type=int, default=44)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--output-root", default="hybrid_fast_full_pdf")
    parser.add_argument("--tesseract", default=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    parser.add_argument("--lang", default="kaz+eng")
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--padding", type=int, default=10)
    parser.add_argument("--min-area", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-conf", type=float, default=45.0)
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument("--debug-image", action="store_true")
    parser.add_argument("--include-block-headers", action="store_true")
    parser.add_argument("--include-page-headers", action="store_true")
    parser.add_argument("--reuse-images", action="store_true")
    parser.add_argument("--reuse-crops", action="store_true")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path.resolve()}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tesseract_path = find_tesseract(args.tesseract)
    print(f"Using Tesseract: {tesseract_path}")

    pipeline = create_ppstructure(args.device)

    combined = []
    all_suspects = []

    for page in range(args.start_page, args.pages + 1):
        print(f"\n\n================ PAGE {page} / {args.pages} ================")
        page_text, suspects = process_page(page, pipeline, pdf_path, output_root, args, tesseract_path)
        if args.include_page_headers:
            combined.append(f"=== PAGE {page} ===\n{page_text.strip()}\n")
        else:
            combined.append(page_text.strip())
        all_suspects.extend(suspects)

    combined_path = output_root / "all_pages_tsv_filtered.txt"
    combined_path.write_text("\n\n".join(x for x in combined if x).strip() + "\n", encoding="utf-8")

    suspects_path = output_root / "tsv_suspect_lines.txt"
    suspects_path.write_text("\n".join(all_suspects) + ("\n" if all_suspects else ""), encoding="utf-8")

    print("\nDone.")
    print(f"Combined output: {combined_path.resolve()}")
    print(f"Suspect lines:   {suspects_path.resolve()}")
    print(f"Suspects count:  {len(all_suspects)}")


if __name__ == "__main__":
    main()
