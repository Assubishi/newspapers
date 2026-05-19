# Local Batch OCR Setup and Run Guide

This guide explains how to set up the Python environment, install dependencies, and run the batch OCR pipeline locally on Windows.

The pipeline does this:

1. Extracts a ZIP file containing newspaper PDFs.
2. Runs `PP-StructureV3` to detect page layout and reading order.
3. Uses Tesseract OCR with `kaz+eng` to recognize text in each layout block.
4. Filters low-confidence OCR noise.
5. Removes repeated page furniture such as page numbers, headers, dates, and section labels.
6. Saves one `.txt` transcript per PDF.
7. Creates one final ZIP file containing all transcripts.

---

## 1. Required Files

Put these files in the same folder:

```text
run_batch_zip_ocr.py
run_hybrid_fast_reuse.py
hybrid_ppstructure_tesseract.py
remove_page_furniture_preserve_blanks.py
requirements.txt
```

Example folder:

```text
C:\Users\YourName\Desktop\ocr_project\
```

---

## 2. Install Python

Recommended:

```text
Python 3.11.x
```

Check Python:

```cmd
python --version
```

Expected example:

```text
Python 3.11.9
```

If `python` is not recognized, reinstall Python and enable:

```text
Add Python to PATH
```

---

## 3. Create Virtual Environment

Open CMD in your project folder:

cd C:\Users\YourName\Desktop\ocr_project


Create environment:

python -m venv ocr-env


Activate environment:

ocr-env\Scripts\activate.bat


After activation, you should see:

(ocr-env) C:\Users\YourName\Desktop\ocr_project>


Upgrade pip:

python -m pip install --upgrade pip


---

## 4. Install Python Requirements

Install everything from `requirements.txt`:


python -m pip install -r requirements.txt


This installs packages such as:

```text
paddleocr
paddlex
paddlepaddle-gpu
pymupdf
pillow
numpy
```

Important: `requirements.txt` does **not** install Tesseract OCR. Tesseract must be installed separately.

---

## 5. Install Tesseract OCR

Install Tesseract on Windows:

winget install --id UB-Mannheim.TesseractOCR -e


After installation, check:

"C:\Program Files\Tesseract-OCR\tesseract.exe" --version


Check available languages:

```cmd
"C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs
```

You must see:

```text
eng
kaz
```

If `kaz` is missing, install or copy `kaz.traineddata` into:

C:\Program Files\Tesseract-OCR\tessdata\


Then check again:

```cmd
"C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs
```

---

## 6. Check GPU / Paddle

Check NVIDIA GPU:

nvidia-smi


Check Paddle GPU:

python -c "import paddle; print(paddle.__version__); print(paddle.device.cuda.device_count())"


Expected output should show at least: 1


If it shows `0`, Paddle cannot see your GPU. You can still run with CPU, but it will be slower.

---

## 7. Test on First 2 PDFs

Run a small test first:

python run_batch_zip_ocr.py --zip <your_zip_file>.zip --limit 2 --dpi 400 --workers 6 --device gpu:0 --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"


This processes only the first 2 PDFs from the ZIP.

Output will be created here:

batch_ocr_output\final_txt\
batch_ocr_output\final_txt_results.zip
batch_ocr_output\failed_pdfs.txt


Check the `.txt` files in:

batch_ocr_output\final_txt\


---

## 8. Run All PDFs

When the test works, run the full ZIP:

python run_batch_zip_ocr.py --zip <your_zip_file>.zip --dpi 400 --workers 6 --device gpu:0 --skip-existing --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"


The final result will be:

batch_ocr_output\final_txt_results.zip


This ZIP contains one `.txt` transcript per newspaper PDF.

---

## 9. Resume After Interruption

If the process stops, crashes, or your computer restarts, run the same command again:


python run_batch_zip_ocr.py --zip <your_zip_file>.zip --dpi 400 --workers 6 --device gpu:0 --skip-existing --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"


Because of:  --skip-existing


the script skips already completed `.txt` files.

---

## 10. Faster Run

For faster processing, use 300 DPI:


python run_batch_zip_ocr.py --zip <your_zip_file>.zip --dpi 300 --workers 6 --device gpu:0 --skip-existing --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"


This is faster but may reduce OCR quality a little.

Recommended settings:


Best quality:  dpi 400
Faster:        dpi 300
Workers:       4 to 6 on a strong laptop ( this is cored that computer has, you can change it by yourself)
Device:        gpu:0 if Paddle GPU works


---

## 11. CPU Mode

If GPU does not work, use CPU:


python run_batch_zip_ocr.py --zip <your_zip_file>.zip --dpi 300 --workers 4 --device cpu --skip-existing --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"


CPU mode is slower.

---

## 12. Output Structure

After running, you will get:

```text
batch_ocr_output\
    extracted_pdfs\
        ...
    work\
        newspaper_001\
            page_001\
            page_002\
            all_pages_tsv_filtered.txt
            all_pages_no_furniture.txt
        newspaper_002\
            ...
    final_txt\
        newspaper_001.txt
        newspaper_002.txt
        ...
    final_txt_results.zip
    failed_pdfs.txt
```

Use this final file:

```text
batch_ocr_output\final_txt_results.zip
```

---

## 13. What Each Main Script Does

### `run_batch_zip_ocr.py`

Main batch script. It:

```text
extracts ZIP
finds PDFs
runs OCR for each PDF
runs page-furniture cleanup
copies final TXT files
creates final ZIP
```

### `run_hybrid_fast_reuse.py`

OCR engine for one PDF. It:

```text
renders PDF pages
uses PP-StructureV3 for layout
crops text blocks
runs Tesseract kaz+eng
filters obvious OCR garbage
saves page and document text
```

### `hybrid_ppstructure_tesseract.py`

Helper functions used by the OCR runner:

```text
PDF rendering
layout block extraction
bbox parsing
Tesseract path detection
debug image drawing
```

### `remove_page_furniture_preserve_blanks.py`

Removes repeated page furniture:

```text
page numbers
running headers
dates
section labels
repeated top/bottom lines
```

It preserves blank lines between OCR blocks.

---

## 14. Common Problems

### Problem: Tesseract not found

Error:

```text
Tesseract executable not found
```

Use explicit path:

```cmd
--tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Problem: Kazakh language missing

Error:

```text
Error opening data file ... kaz.traineddata
```

Check:

```cmd
"C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs
```

Make sure `kaz` appears.

### Problem: Paddle cannot use GPU

Check:

```cmd
python -c "import paddle; print(paddle.device.cuda.device_count())"
```

If it prints `0`, run with:

```cmd
--device cpu
```

or reinstall the correct Paddle GPU package.

### Problem: Some PDFs failed

Check:

```text
batch_ocr_output\failed_pdfs.txt
```

The script still creates transcripts for successful PDFs.

---

## 15. Recommended Final Command

For your laptop with NVIDIA GPU:

```cmd
python run_batch_zip_ocr.py --zip <your_zip_file>.zip --dpi 400 --workers 6 --device gpu:0 --skip-existing --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Final output:

```text
batch_ocr_output\final_txt_results.zip
```