"""
Hybrid newspaper OCR:
    PP-StructureV3 detects layout/order
    Tesseract kaz+eng recognizes text inside each detected text block
    Results are joined in PP-Structure order.

Usage with an already rendered page image:
    python hybrid_ppstructure_tesseract.py --input page_001_400dpi.png --output hybrid_page_001

Usage with PDF page rendering:
    python hybrid_ppstructure_tesseract.py --input input_newspaper.pdf --page 1 --dpi 400 --output hybrid_page_001

Requirements:
    pip install pillow pymupdf paddleocr
    Tesseract OCR installed with kaz.traineddata in tessdata.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PPStructureV3


EXCLUDE_LABEL_KEYWORDS = {
    "image",
    "figure",
    "fig",
    "table",
    "formula",
    "chart",
    "seal",
    "stamp",
}


def render_pdf_page(pdf_path: Path, page_number: int, dpi: int, out_image: Path) -> Path:
    doc = fitz.open(str(pdf_path))
    page_index = page_number - 1
    if page_index < 0 or page_index >= len(doc):
        raise ValueError(f"Page {page_number} is outside PDF range 1-{len(doc)}")
    pix = doc[page_index].get_pixmap(dpi=dpi)
    pix.save(str(out_image))
    return out_image


def find_tesseract(user_path: str) -> str:
    if user_path:
        p = Path(user_path)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Tesseract executable not found: {p}")

    found = shutil.which("tesseract")
    if found:
        return found

    default_windows = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if default_windows.exists():
        return str(default_windows)

    raise FileNotFoundError(
        r'Could not find tesseract.exe. Pass --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"'
    )


def run_ppstructure(image_path: Path, output_dir: Path, device: str) -> List[Path]:
    pp_dir = output_dir / "ppstructure_raw"
    pp_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PPStructureV3(
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

    print(f"Running PP-StructureV3 on {image_path} ...")
    results = pipeline.predict(str(image_path))

    for result in results:
        result.print()
        result.save_to_json(str(pp_dir))
        result.save_to_markdown(str(pp_dir))

    json_files = sorted(pp_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"PP-StructureV3 did not produce JSON files in {pp_dir}")

    return json_files


def load_json_files(json_files: Iterable[Path]) -> List[Any]:
    data = []
    for path in json_files:
        try:
            data.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"Warning: could not load JSON {path}: {exc}")
    return data


def as_number_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (int, float)) for v in value):
            return [float(v) for v in value]

        flat: List[float] = []
        ok = True
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                x, y = item[0], item[1]
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    flat.extend([float(x), float(y)])
                else:
                    ok = False
                    break
            else:
                ok = False
                break
        if ok and len(flat) >= 4:
            return flat

    return None


def bbox_from_value(value: Any) -> Optional[Tuple[int, int, int, int]]:
    nums = as_number_list(value)
    if not nums:
        return None

    if len(nums) == 4:
        x1, y1, x2, y2 = nums
    elif len(nums) >= 8:
        xs = nums[0::2]
        ys = nums[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return None

    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def find_bbox_in_dict(obj: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    bbox_keys = [
        "block_bbox",
        "bbox",
        "box",
        "coordinate",
        "coordinates",
        "poly",
        "polygon",
        "points",
        "dt_polys",
    ]
    for key in bbox_keys:
        if key in obj:
            bbox = bbox_from_value(obj[key])
            if bbox:
                return bbox
    return None


def find_label_in_dict(obj: Dict[str, Any]) -> str:
    label_keys = [
        "block_label",
        "label",
        "type",
        "category",
        "cls_name",
        "class_name",
        "layout_label",
    ]
    for key in label_keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def find_lists_by_key(obj: Any, key: str) -> List[List[Any]]:
    found: List[List[Any]] = []

    if isinstance(obj, dict):
        value = obj.get(key)
        if isinstance(value, list):
            found.append(value)
        for child in obj.values():
            found.extend(find_lists_by_key(child, key))

    elif isinstance(obj, list):
        for child in obj:
            found.extend(find_lists_by_key(child, key))

    return found


def extract_from_ordered_list(items: List[Any], source: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        bbox = find_bbox_in_dict(item)
        if not bbox:
            continue
        label = find_label_in_dict(item)
        text = ""
        for text_key in ["block_content", "text", "content", "rec_text"]:
            if isinstance(item.get(text_key), str):
                text = item[text_key]
                break
        blocks.append(
            {
                "bbox": bbox,
                "label": label,
                "source": source,
                "pp_text": text,
            }
        )
    return blocks


def extract_blocks(data_items: List[Any]) -> List[Dict[str, Any]]:
    all_blocks: List[Dict[str, Any]] = []

    for data in data_items:
        parsing_lists = find_lists_by_key(data, "parsing_res_list")
        for lst in parsing_lists:
            all_blocks.extend(extract_from_ordered_list(lst, "parsing_res_list"))

    if not all_blocks:
        for data in data_items:
            box_lists = find_lists_by_key(data, "boxes")
            for lst in box_lists:
                all_blocks.extend(extract_from_ordered_list(lst, "boxes"))

    if not all_blocks:
        raise RuntimeError(
            "Could not find layout blocks in PP-Structure JSON. "
            "Open the JSON and check which key contains bbox coordinates."
        )

    seen = set()
    unique: List[Dict[str, Any]] = []
    for block in all_blocks:
        bbox = block["bbox"]
        label = block.get("label", "")
        key = (bbox, label)
        if key in seen:
            continue
        seen.add(key)
        unique.append(block)

    return unique


def is_text_block(block: Dict[str, Any]) -> bool:
    label = str(block.get("label", "")).lower()
    if not label:
        return True

    for bad in EXCLUDE_LABEL_KEYWORDS:
        if bad in label:
            return False

    return True


def clip_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    padding: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    return x1, y1, x2, y2


def ocr_crop_with_tesseract(
    tesseract_path: str,
    crop_path: Path,
    lang: str,
    psm: int,
    dpi: int,
) -> str:
    cmd = [
        tesseract_path,
        str(crop_path),
        "stdout",
        "-l",
        lang,
        "--oem",
        "1",
        "--psm",
        str(psm),
        "--dpi",
        str(dpi),
        "-c",
        "preserve_interword_spaces=1",
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
        return f"[TESSERACT_ERROR]\n{proc.stderr.strip()}"

    return proc.stdout.strip()


def draw_debug_image(
    image_path: Path,
    blocks: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    for idx, block in enumerate(blocks, start=1):
        x1, y1, x2, y2 = block["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1 + 4, y1 + 4), str(idx), fill="red", font=font)

    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image or PDF path")
    parser.add_argument("--page", type=int, default=1, help="PDF page number, 1-based")
    parser.add_argument("--dpi", type=int, default=400, help="DPI for PDF rendering and Tesseract")
    parser.add_argument("--output", default="hybrid_output", help="Output folder")
    parser.add_argument("--device", default="gpu:0", help="Paddle device, for example gpu:0 or cpu")
    parser.add_argument("--tesseract", default="", help="Path to tesseract.exe")
    parser.add_argument("--lang", default="kaz+eng", help="Tesseract languages, e.g. kaz or kaz+eng")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract PSM for each cropped block")
    parser.add_argument("--padding", type=int, default=10, help="Padding around each crop in pixels")
    parser.add_argument("--min-area", type=int, default=5000, help="Ignore tiny boxes below this area")
    parser.add_argument("--keep-images", action="store_true", help="Also OCR image/figure/table blocks")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path.resolve()}")

    if input_path.suffix.lower() == ".pdf":
        image_path = output_dir / f"page_{args.page:03d}_{args.dpi}dpi.png"
        print(f"Rendering PDF page {args.page} at {args.dpi} DPI -> {image_path}")
        render_pdf_page(input_path, args.page, args.dpi, image_path)
    else:
        image_path = input_path

    tesseract_path = find_tesseract(args.tesseract)
    print(f"Using Tesseract: {tesseract_path}")

    json_files = run_ppstructure(image_path, output_dir, args.device)
    data_items = load_json_files(json_files)
    blocks = extract_blocks(data_items)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    filtered: List[Dict[str, Any]] = []
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

    if not filtered:
        raise RuntimeError("No text blocks left after filtering. Try --keep-images or lower --min-area.")

    draw_debug_image(image_path, filtered, output_dir / "hybrid_block_order.png")

    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    txt_parts: List[str] = []
    md_parts: List[str] = []
    jsonl_rows: List[Dict[str, Any]] = []

    for idx, block in enumerate(filtered, start=1):
        x1, y1, x2, y2 = block["bbox"]
        label = block.get("label", "") or "text"

        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)[:40]
        crop = image.crop((x1, y1, x2, y2))
        crop_path = crops_dir / f"block_{idx:03d}_{safe_label}.png"
        crop.save(crop_path)

        print(f"OCR block {idx:03d}/{len(filtered)} label={label} bbox={(x1, y1, x2, y2)}")
        text = ocr_crop_with_tesseract(
            tesseract_path=tesseract_path,
            crop_path=crop_path,
            lang=args.lang,
            psm=args.psm,
            dpi=args.dpi,
        )

        header = f"=== BLOCK {idx:03d} | label={label} | bbox={x1},{y1},{x2},{y2} ==="
        txt_parts.append(header)
        txt_parts.append(text)
        txt_parts.append("")

        md_parts.append(f"## Block {idx:03d} | {label}")
        md_parts.append(f"`bbox={x1},{y1},{x2},{y2}`")
        md_parts.append("")
        md_parts.append(text)
        md_parts.append("")

        jsonl_rows.append(
            {
                "page": args.page,
                "block_order": idx,
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "crop_path": str(crop_path),
                "text": text,
                "pp_text": block.get("pp_text", ""),
            }
        )

    (output_dir / "hybrid_ordered_text.txt").write_text("\n".join(txt_parts), encoding="utf-8")
    (output_dir / "hybrid_ordered_text.md").write_text("\n".join(md_parts), encoding="utf-8")

    with (output_dir / "hybrid_chunks.jsonl").open("w", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nDone.")
    print(f"Ordered TXT:  {output_dir / 'hybrid_ordered_text.txt'}")
    print(f"Ordered MD:   {output_dir / 'hybrid_ordered_text.md'}")
    print(f"JSONL chunks: {output_dir / 'hybrid_chunks.jsonl'}")
    print(f"Debug image:  {output_dir / 'hybrid_block_order.png'}")


if __name__ == "__main__":
    main()
