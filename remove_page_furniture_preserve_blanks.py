"""
Generic page-furniture remover for hybrid OCR outputs.

Purpose:
    Remove running headers/footers/page numbers/date/section labels without
    hardcoding newspaper names such as "Egemen Qazaqstan".

Important update:
    This version preserves empty lines between OCR blocks by default.

Input:
    output folder created by run_hybrid_fast_reuse.py, e.g.
        news2_pdf_fast\\page_001\\hybrid_chunks.jsonl
        news2_pdf_fast\\page_002\\hybrid_chunks.jsonl

Output:
    news2_pdf_fast\\all_pages_no_furniture.txt
    news2_pdf_fast\\page_furniture_suspects.txt

Usage:
    python remove_page_furniture_preserve_blanks.py --root news2_pdf_fast --pages 16

Safer:
    python remove_page_furniture_preserve_blanks.py --root news2_pdf_fast --pages 16 --no-remove-section-labels

More aggressive:
    python remove_page_furniture_preserve_blanks.py --root news2_pdf_fast --pages 16 --top-margin 0.16 --bottom-margin 0.84

If you want to collapse blank lines:
    python remove_page_furniture_preserve_blanks.py --root news2_pdf_fast --pages 16 --collapse-blank-lines --max-blank-lines 1

Only produce suspects report, keep everything:
    python remove_page_furniture_preserve_blanks.py --root news2_pdf_fast --pages 16 --review-only
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


WORD_RE = re.compile(r"[A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІіЁё0-9]+", re.UNICODE)

MONTHS = {
    "қаңтар", "ақпан", "наурыз", "сәуір", "мамыр", "маусым",
    "шілде", "тамыз", "қыркүйек", "қазан", "қараша", "желтоқсан",
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
}


def normalize_line_for_repeat(line: str) -> str:
    s = line.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_count(line: str) -> int:
    return len(WORD_RE.findall(line))


def uppercase_ratio(line: str) -> float:
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for ch in letters if ch.upper() == ch and ch.lower() != ch)
    return upper / len(letters)


def is_standalone_page_number(line: str) -> bool:
    return bool(re.fullmatch(r"\d{1,4}", line.strip()))


def is_date_line(line: str) -> bool:
    low = line.lower()
    has_month = any(m in low for m in MONTHS)
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", low))
    has_year_word = "жыл" in low or "г." in low or "год" in low
    return has_month and (has_year or has_year_word)


def is_short_section_label(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > 45:
        return False
    if token_count(s) > 4:
        return False
    if uppercase_ratio(s) < 0.70:
        return False
    return True


def find_page_image_size(page_dir: Path) -> Tuple[int, int]:
    pngs = sorted(page_dir.glob("page_*dpi.png"))
    if not pngs:
        return 1, 1
    with Image.open(pngs[0]) as im:
        return im.size


def load_lines(root: Path, pages: int, block_separator_lines: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for page in range(1, pages + 1):
        page_dir = root / f"page_{page:03d}"
        jsonl_path = page_dir / "hybrid_chunks.jsonl"

        if not jsonl_path.exists():
            print(f"Warning: missing {jsonl_path}")
            continue

        width, height = find_page_image_size(page_dir)

        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                item = json.loads(raw)

                bbox = item.get("bbox", [0, 0, 0, 0])
                if not isinstance(bbox, list) or len(bbox) != 4:
                    bbox = [0, 0, 0, 0]

                text = str(item.get("text", "") or "")
                block_order = int(item.get("block_order", 0))

                # Keep blank lines inside the OCR block.
                for line_idx, line in enumerate(text.splitlines(), start=1):
                    rows.append(
                        {
                            "page": page,
                            "block_order": block_order,
                            "line_idx": line_idx,
                            "line": line.rstrip(),
                            "bbox": bbox,
                            "width": width,
                            "height": height,
                            "separator": False,
                        }
                    )

                # Add empty lines between blocks. These separators are never removed
                # by page-furniture rules.
                for sep_idx in range(block_separator_lines):
                    rows.append(
                        {
                            "page": page,
                            "block_order": block_order,
                            "line_idx": 10_000 + sep_idx,
                            "line": "",
                            "bbox": bbox,
                            "width": width,
                            "height": height,
                            "separator": True,
                        }
                    )

    rows.sort(key=lambda r: (int(r["page"]), int(r["block_order"]), int(r["line_idx"])))
    return rows


def is_in_margin(row: Dict[str, object], top_margin: float, bottom_margin: float) -> bool:
    bbox = row["bbox"]
    height = max(1, int(row["height"]))
    y1 = float(bbox[1])
    y2 = float(bbox[3])
    return y1 <= top_margin * height or y2 >= bottom_margin * height


def build_repeated_margin_lines(
    rows: List[Dict[str, object]],
    pages: int,
    top_margin: float,
    bottom_margin: float,
    min_repeat: int,
    repeat_ratio: float,
) -> set[str]:
    line_pages: Dict[str, set[int]] = defaultdict(set)

    for row in rows:
        if row.get("separator"):
            continue

        line = str(row["line"]).strip()
        if not line:
            continue
        if not is_in_margin(row, top_margin, bottom_margin):
            continue
        if len(line) > 100:
            continue

        norm = normalize_line_for_repeat(line)
        if not norm:
            continue
        if token_count(norm) > 8:
            continue

        line_pages[norm].add(int(row["page"]))

    needed = max(min_repeat, math.ceil(pages * repeat_ratio))
    return {norm for norm, pgset in line_pages.items() if len(pgset) >= needed}


def should_remove_line(
    row: Dict[str, object],
    repeated_norms: set[str],
    top_margin: float,
    bottom_margin: float,
    remove_section_labels: bool,
) -> Tuple[bool, str]:
    if row.get("separator"):
        return False, ""

    line = str(row["line"]).strip()
    if not line:
        return False, ""

    in_margin = is_in_margin(row, top_margin, bottom_margin)
    norm = normalize_line_for_repeat(line)

    if in_margin and norm in repeated_norms:
        return True, "repeated_margin_line"

    if in_margin and is_standalone_page_number(line):
        return True, "standalone_page_number"

    if in_margin and is_date_line(line):
        return True, "date_line_in_margin"

    if remove_section_labels and in_margin and is_short_section_label(line):
        return True, "short_section_label_in_margin"

    return False, ""


def collapse_blank_lines(lines: List[str], max_blank_lines: int) -> List[str]:
    out = []
    blanks = 0
    for line in lines:
        if line.strip():
            blanks = 0
            out.append(line.rstrip())
        else:
            blanks += 1
            if blanks <= max_blank_lines:
                out.append("")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Output folder from run_hybrid_fast_reuse.py")
    parser.add_argument("--pages", type=int, required=True)
    parser.add_argument("--output", default="", help="Default: root/all_pages_no_furniture.txt")
    parser.add_argument("--suspects", default="", help="Default: root/page_furniture_suspects.txt")
    parser.add_argument("--top-margin", type=float, default=0.13)
    parser.add_argument("--bottom-margin", type=float, default=0.88)
    parser.add_argument("--min-repeat", type=int, default=3)
    parser.add_argument("--repeat-ratio", type=float, default=0.15)

    # Blank-line behavior.
    parser.add_argument("--block-separator-lines", type=int, default=1, help="Empty lines to insert between OCR blocks")
    parser.add_argument("--collapse-blank-lines", action="store_true", help="Collapse repeated blank lines")
    parser.add_argument("--max-blank-lines", type=int, default=1)

    parser.add_argument("--no-remove-section-labels", action="store_true")
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root.resolve()}")

    rows = load_lines(
        root=root,
        pages=args.pages,
        block_separator_lines=args.block_separator_lines,
    )

    repeated_norms = build_repeated_margin_lines(
        rows=rows,
        pages=args.pages,
        top_margin=args.top_margin,
        bottom_margin=args.bottom_margin,
        min_repeat=args.min_repeat,
        repeat_ratio=args.repeat_ratio,
    )

    output_lines: List[str] = []
    suspects: List[str] = []
    removed = 0
    current_page = None

    remove_section_labels = not args.no_remove_section_labels

    for row in rows:
        page = int(row["page"])
        line = str(row["line"])

        if current_page is None:
            current_page = page
        elif page != current_page:
            # Keep a visible gap between pages.
            output_lines.append("")
            output_lines.append("")
            current_page = page

        remove, reason = should_remove_line(
            row=row,
            repeated_norms=repeated_norms,
            top_margin=args.top_margin,
            bottom_margin=args.bottom_margin,
            remove_section_labels=remove_section_labels,
        )

        if remove:
            removed += 1
            suspects.append(
                f"PAGE {page}\tBLOCK {row['block_order']}\t{reason}\t{line}"
            )
            if not args.review_only:
                continue

        output_lines.append(line)

    # By default, do not collapse blank lines.
    # Collapse only if the user explicitly requests it.
    if args.collapse_blank_lines:
        output_lines = collapse_blank_lines(output_lines, args.max_blank_lines)

    output_path = Path(args.output) if args.output else root / "all_pages_no_furniture.txt"
    suspects_path = Path(args.suspects) if args.suspects else root / "page_furniture_suspects.txt"

    output_path.write_text("\n".join(output_lines).strip() + "\n", encoding="utf-8")
    suspects_path.write_text("\n".join(suspects) + ("\n" if suspects else ""), encoding="utf-8")

    print(f"Repeated margin patterns found: {len(repeated_norms)}")
    print(f"Removed lines: {0 if args.review_only else removed}")
    print(f"Suspect lines: {len(suspects)}")
    print(f"Output saved: {output_path.resolve()}")
    print(f"Suspects saved: {suspects_path.resolve()}")


if __name__ == "__main__":
    main()
