"""
VTU RAG Ingest Script — Bulletproof Edition + OCR Support
==========================================================
Loads 3 data sources:
  1. MODULE PYQS/   -> subject subfolders with PDF question papers
  2. MODULE NOTES/  -> subject subfolders with module-wise PDF notes
  3. answer_guide.json -> VTU answer writing rules and system prompt

SAFETY MECHANISMS:
  1. SHA-256 file hashing  — tracks content, not just filenames
  2. ChromaDB ↔ Tracker sync — verifies data integrity at startup
  3. Deterministic chunk IDs — same file always produces same IDs
  4. Atomic per-file saves  — tracker saved immediately after each file
  5. Startup integrity check — detects and recovers from corruption

OCR SUPPORT (NEW):
  - Auto-detects scanned/image-only PDFs (text chars per page < OCR_TEXT_THRESHOLD)
  - Falls back to pytesseract OCR pipeline for those pages
  - Preprocesses images: grayscale + sharpen + contrast boost for handwritten notes
  - Requires: pip install pytesseract pdf2image pillow
  - Requires: Tesseract installed at TESSERACT_PATH
  - Requires: Poppler installed at POPPLER_PATH

Image pipeline:
  - During ingestion: scans PDFs for embedded images and records their
    references (xref IDs) in chunk metadata. NO images saved to disk.
  - During query: images are extracted from the original PDF on-demand.

Progress is saved per-PDF so nothing is lost if interrupted.
"""

import os
import sys
import json
import hashlib
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ─────────────────────────────────────────────
# OCR IMPORTS — graceful fallback if not installed
# ─────────────────────────────────────────────
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image, ImageFilter, ImageEnhance
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PAST_QUESTIONS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MODULE PYQS")
MODULES_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MODULE NOTES")
ANSWER_GUIDE_JSON     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "answer_guide.json")
CHROMA_DB_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
INGESTED_TRACKER_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ingested_files.json")

EMBED_MODEL    = "nomic-embed-text"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 100
MIN_IMAGE_SIZE = 50   # skip images smaller than 50x50 px

# ─────────────────────────────────────────────
# OCR CONFIG — update paths for your machine
# ─────────────────────────────────────────────
TESSERACT_PATH      = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH        = r"C:\Users\ADMIN\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
OCR_DPI             = 150     # 150 is fast & sufficient for printed text; use 200 for handwritten
OCR_TEXT_THRESHOLD  = 30      # chars per page below this → treat as scanned, use OCR
                               # tune: printed notes ~300+, scanned ~0-10
OCR_WORKER_COUNT    = 4       # parallel threads for tesseract OCR per PDF
TESSERACT_CONFIG    = "--oem 1 --psm 6"  # LSTM-only engine (fastest) + block text mode

if OCR_AVAILABLE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ═════════════════════════════════════════════
# SAFETY LAYER 1: SHA-256 File Hashing
# ═════════════════════════════════════════════
def compute_file_hash(file_path):
    """Compute SHA-256 hash of a file. Reads in 8KB blocks to handle large files."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while True:
                block = f.read(8192)
                if not block:
                    break
                sha256.update(block)
        return sha256.hexdigest()
    except Exception as e:
        print(f"  [ERROR] Could not hash {file_path}: {e}")
        return None


def generate_chunk_id(file_hash, chunk_index):
    """
    SAFETY LAYER 3: Deterministic chunk IDs.
    Same file + same chunk index = same ID every time.
    """
    raw = f"{file_hash}::chunk::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ═════════════════════════════════════════════
# SAFETY LAYER 4: Atomic Tracker I/O
# ═════════════════════════════════════════════
def load_tracker():
    if not os.path.exists(INGESTED_TRACKER_JSON):
        return {}
    try:
        with open(INGESTED_TRACKER_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  [WARN] Tracker file corrupted ({e}). Starting fresh tracker.")
        return {}

    if isinstance(data, list):
        print("  [MIGRATE] Converting old tracker format (list) to new format (dict)...")
        new_tracker = {}
        for key in data:
            new_tracker[key] = {
                "sha256": "MIGRATED_NO_HASH",
                "chunk_ids": [],
                "ingested_at": "MIGRATED",
                "num_chunks": 0,
                "num_pages": 0
            }
        return new_tracker
    return data


def save_tracker(tracker_dict):
    temp_path = INGESTED_TRACKER_JSON + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(tracker_dict, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(INGESTED_TRACKER_JSON):
            os.replace(temp_path, INGESTED_TRACKER_JSON)
        else:
            os.rename(temp_path, INGESTED_TRACKER_JSON)
    except Exception as e:
        print(f"  [ERROR] Failed to save tracker: {e}")
        try:
            with open(INGESTED_TRACKER_JSON, "w", encoding="utf-8") as f:
                json.dump(tracker_dict, f, indent=2, ensure_ascii=False)
        except Exception as e2:
            print(f"  [CRITICAL] Tracker save failed completely: {e2}")


# ═════════════════════════════════════════════
# SAFETY LAYER 2 & 5: Startup Integrity Check
# ═════════════════════════════════════════════
def verify_integrity(tracker, vectorstore):
    if not tracker:
        return tracker

    print("[*] Running integrity check...")
    collection = vectorstore._collection
    total_in_db = collection.count()
    print(f"    ChromaDB has {total_in_db} chunks total")

    if total_in_db == 0 and tracker:
        print("    [!] ChromaDB is EMPTY but tracker has entries!")
        print("    [!] Clearing tracker — all files will be re-ingested.")
        return {}

    bad_keys = []
    checked = 0
    for key, info in tracker.items():
        chunk_ids = info.get("chunk_ids", [])
        if not chunk_ids:
            if info.get("sha256") == "MIGRATED_NO_HASH":
                bad_keys.append(key)
            continue
        try:
            result = collection.get(ids=[chunk_ids[0]])
            if not result or not result["ids"]:
                print(f"    [!] Missing chunks for: {key}")
                bad_keys.append(key)
        except Exception:
            bad_keys.append(key)
        checked += 1

    if bad_keys:
        print(f"    [!] {len(bad_keys)} files have missing/invalid data — will re-ingest")
        for key in bad_keys:
            old_ids = tracker[key].get("chunk_ids", [])
            if old_ids:
                try:
                    collection.delete(ids=old_ids)
                except Exception:
                    pass
            del tracker[key]
        save_tracker(tracker)
    else:
        print(f"    [OK] All {checked} tracked files verified in ChromaDB")

    return tracker


# ═════════════════════════════════════════════
# OCR LAYER — Auto-detect + Fallback
# ═════════════════════════════════════════════
def is_scanned_pdf(pdf_path):
    """
    Check if a PDF is scanned/image-only by measuring average
    extractable text characters per page.
    Returns True if avg chars/page < OCR_TEXT_THRESHOLD.
    """
    try:
        doc = fitz.open(pdf_path)
        total_chars = 0
        num_pages = len(doc)
        if num_pages == 0:
            return False
        for page in doc:
            total_chars += len(page.get_text().strip())
        doc.close()
        avg_chars = total_chars / num_pages
        return avg_chars < OCR_TEXT_THRESHOLD
    except Exception:
        return False


def preprocess_image_for_ocr(image):
    """
    Preprocess PIL image for better OCR accuracy on handwritten/scanned notes.
    - Grayscale → Sharpen → Contrast boost
    """
    image = image.convert("L")
    image = image.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    return image


def _render_page_to_pil(doc_path, page_index, dpi):
    """
    Render a single PDF page to a PIL Image using PyMuPDF (fitz).
    Much faster than pdf2image/poppler — no external process spawn.
    """
    doc = fitz.open(doc_path)
    page = doc.load_page(page_index)
    # fitz uses a zoom matrix; 72 DPI is the base
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)  # grayscale directly
    img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def _ocr_single_page(args):
    """
    Render + OCR a single page. Designed to run in a thread pool.
    Uses PyMuPDF for rendering (no poppler needed).
    Returns (page_num, Document) or (page_num, None) on failure/empty.
    """
    page_num, pdf_path, subject, pdf_file, page_xrefs, dpi = args
    try:
        # Render page to grayscale PIL image via PyMuPDF
        image = _render_page_to_pil(pdf_path, page_num - 1, dpi)

        # Light preprocessing: sharpen + contrast boost
        image = image.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        text = pytesseract.image_to_string(
            image, lang="eng", config=TESSERACT_CONFIG
        )
        text = text.strip()
        if not text:
            return (page_num, None)

        doc = Document(
            page_content=text,
            metadata={
                "source_type" : "ocr",
                "subject"     : subject,
                "file_name"   : pdf_file,
                "pdf_path"    : pdf_path,
                "page"        : page_num - 1,   # 0-indexed to match PyMuPDF
                "image_xrefs" : page_xrefs.get(page_num, ""),
                "ocr"         : True
            }
        )
        return (page_num, doc)
    except Exception:
        return (page_num, None)


def ocr_pdf_to_documents(pdf_path, subject, pdf_file, page_xrefs):
    """
    Run pytesseract OCR on a scanned PDF — MAXIMUM SPEED.
    - Uses PyMuPDF (fitz) for rendering — NO poppler needed, ~3-5x faster
    - Renders directly to grayscale — skips color conversion overhead
    - Uses ThreadPoolExecutor for parallel page OCR
    - LSTM-only engine (--oem 1) + block text mode (--psm 6)
    Returns list of LangChain Document objects (one per page).
    """
    if not OCR_AVAILABLE:
        print("  [WARN] pytesseract/pdf2image not installed — cannot OCR.")
        print("  [TIP]  pip install pytesseract pillow")
        return []

    docs = []
    try:
        # Get page count using fitz (no full render yet)
        pdf_doc = fitz.open(pdf_path)
        num_pages = len(pdf_doc)
        pdf_doc.close()

        # Build args for parallel render + OCR
        ocr_args = [
            (page_num, pdf_path, subject, pdf_file, page_xrefs, OCR_DPI)
            for page_num in range(1, num_pages + 1)
        ]

        # Render + OCR pages in parallel
        with ThreadPoolExecutor(max_workers=OCR_WORKER_COUNT) as executor:
            futures = {executor.submit(_ocr_single_page, arg): arg[0] for arg in ocr_args}
            for future in as_completed(futures):
                page_num, doc = future.result()
                if doc is not None:
                    docs.append((page_num, doc))

        # Sort by page number to maintain order
        docs.sort(key=lambda x: x[0])
        docs = [doc for _, doc in docs]

    except Exception as e:
        print(f"  [ERROR] OCR failed for {pdf_file}: {e}")
        if "tesseract" in str(e).lower():
            print(f"  [TIP]  Check TESSERACT_PATH in CONFIG: {TESSERACT_PATH}")

    return docs


# ─────────────────────────────────────────────
# HELPER: Scan PDF pages for image references
# ─────────────────────────────────────────────
def scan_images_in_pdf(pdf_path):
    page_xrefs = {}
    try:
        doc = fitz.open(pdf_path)
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            page_num = page_index + 1
            xrefs = []
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.width >= MIN_IMAGE_SIZE and pix.height >= MIN_IMAGE_SIZE:
                        xrefs.append(str(xref))
                    pix = None
                except Exception:
                    pass
            if xrefs:
                page_xrefs[page_num] = ",".join(xrefs)
        doc.close()
    except Exception as e:
        print(f"    [WARN] Image scan failed: {e}")
    return page_xrefs


# ─────────────────────────────────────────────
# CORE: Process a single PDF file (atomic)
# ─────────────────────────────────────────────
def process_single_pdf(pdf_path, source_type, subject, pdf_file, file_hash, tracker, vectorstore):
    tracker_key = f"{source_type}|{subject}|{pdf_file}"

    try:
        # Step 1: Scan for image xrefs
        page_xrefs = scan_images_in_pdf(pdf_path)
        img_count = sum(len(v.split(",")) for v in page_xrefs.values())

        # Step 2: Detect scanned vs digital PDF
        scanned = is_scanned_pdf(pdf_path)

        if scanned:
            # ── OCR PATH
            print(f"  [OCR] Detected scanned PDF: {pdf_file}")
            pages = ocr_pdf_to_documents(pdf_path, subject, pdf_file, page_xrefs)
            ocr_used = True
        else:
            # ── NORMAL PATH
            loader = PyMuPDFLoader(pdf_path)
            pages = loader.load()
            for page in pages:
                page_num = page.metadata.get("page", 0) + 1
                page.metadata["source_type"] = source_type
                page.metadata["subject"]     = subject
                page.metadata["file_name"]   = pdf_file
                page.metadata["pdf_path"]    = pdf_path
                page.metadata["image_xrefs"] = page_xrefs.get(page_num, "")
                page.metadata["ocr"]         = False
            ocr_used = False

        # Step 3: Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(pages)

        if not chunks:
            print(f"  [WARN] No text chunks from {pdf_file} — skipping.")
            tracker[tracker_key] = {
                "sha256"      : file_hash,
                "chunk_ids"   : [],
                "ingested_at" : datetime.now(timezone.utc).isoformat(),
                "num_chunks"  : 0,
                "num_pages"   : len(pages),
                "ocr_used"    : ocr_used
            }
            save_tracker(tracker)
            return len(pages), img_count

        # Step 4: Deterministic IDs
        chunk_ids = [generate_chunk_id(file_hash, i) for i in range(len(chunks))]

        # Step 5: Add to ChromaDB
        texts     = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=chunk_ids)

        # Step 6: Save tracker immediately
        tracker[tracker_key] = {
            "sha256"      : file_hash,
            "chunk_ids"   : chunk_ids,
            "ingested_at" : datetime.now(timezone.utc).isoformat(),
            "num_chunks"  : len(chunks),
            "num_pages"   : len(pages),
            "ocr_used"    : ocr_used
        }
        save_tracker(tracker)

        ocr_tag = " [OCR]" if ocr_used else ""
        print(f"  [OK]{ocr_tag} {source_type}/{subject}/{pdf_file} — "
              f"{len(pages)} pages, {len(chunks)} chunks, {img_count} image refs")

        return len(pages), img_count

    except KeyboardInterrupt:
        print(f"\n  [!] Interrupted during {pdf_file}. Will retry next run.")
        save_tracker(tracker)
        raise
    except Exception as e:
        print(f"  [ERROR] {pdf_path}: {e}")
        return None


# ─────────────────────────────────────────────
# CORE: Process all PDFs from subject subfolders
# ─────────────────────────────────────────────
def load_pdfs_from_subfolders(root_dir, source_type, tracker, vectorstore):
    if not os.path.exists(root_dir):
        print(f"[WARN] Directory not found: {root_dir} — skipping.")
        return 0, 0, 0, 0

    total_pages     = 0
    total_img_refs  = 0
    total_skipped   = 0
    total_processed = 0

    for subject in sorted(os.listdir(root_dir)):
        subject_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_path):
            continue

        pdf_files = sorted(f for f in os.listdir(subject_path) if f.lower().endswith(".pdf"))
        if not pdf_files:
            print(f"  [WARN] No PDFs in {subject_path}")
            continue

        for pdf_file in pdf_files:
            pdf_path    = os.path.join(subject_path, pdf_file)
            tracker_key = f"{source_type}|{subject}|{pdf_file}"

            file_hash = compute_file_hash(pdf_path)
            if file_hash is None:
                continue

            existing = tracker.get(tracker_key)
            if existing and existing.get("sha256") == file_hash:
                total_skipped += 1
                continue

            if existing and existing.get("sha256") != file_hash:
                print(f"  [UPDATE] {tracker_key} — content changed, re-ingesting")
                old_ids = existing.get("chunk_ids", [])
                if old_ids:
                    try:
                        vectorstore._collection.delete(ids=old_ids)
                        print(f"    Removed {len(old_ids)} old chunks")
                    except Exception as e:
                        print(f"    [WARN] Could not remove old chunks: {e}")

            result = process_single_pdf(
                pdf_path, source_type, subject, pdf_file,
                file_hash, tracker, vectorstore
            )

            if result:
                pages, imgs = result
                total_pages    += pages
                total_img_refs += imgs
                total_processed += 1

    return total_pages, total_img_refs, total_processed, total_skipped


# ─────────────────────────────────────────────
# CORE: Process answer_guide.json
# ─────────────────────────────────────────────
def load_answer_guide(json_path, tracker, vectorstore):
    tracker_key = "answer_guide.json"

    if not os.path.exists(json_path):
        print(f"  [WARN] JSON not found: {json_path} — skipping.")
        return 0

    file_hash = compute_file_hash(json_path)
    if file_hash is None:
        return 0

    existing = tracker.get(tracker_key)
    if existing and existing.get("sha256") == file_hash:
        print(f"  [SKIP] answer_guide.json — unchanged")
        return 0

    if existing:
        old_ids = existing.get("chunk_ids", [])
        if old_ids:
            try:
                vectorstore._collection.delete(ids=old_ids)
            except Exception:
                pass

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    system_prompt = data.get("system_prompt", {})
    if system_prompt:
        docs.append(Document(
            page_content=system_prompt.get("content", ""),
            metadata={
                "source_type" : "answer_guide",
                "subject"     : "general",
                "section"     : "system_prompt",
                "rule"        : "system_prompt",
                "file_name"   : os.path.basename(json_path),
                "pdf_path"    : "",
                "image_xrefs" : "",
                "ocr"         : False
            }
        ))

    for chunk in data.get("chunks", []):
        docs.append(Document(
            page_content=chunk.get("content", ""),
            metadata={
                "source_type" : "answer_guide",
                "subject"     : "general",
                "section"     : chunk.get("section", ""),
                "rule"        : chunk.get("rule", ""),
                "file_name"   : os.path.basename(json_path),
                "pdf_path"    : "",
                "image_xrefs" : "",
                "ocr"         : False
            }
        ))

    if not docs:
        return 0

    chunk_ids = [generate_chunk_id(file_hash, i) for i in range(len(docs))]
    texts     = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=chunk_ids)

    tracker[tracker_key] = {
        "sha256"      : file_hash,
        "chunk_ids"   : chunk_ids,
        "ingested_at" : datetime.now(timezone.utc).isoformat(),
        "num_chunks"  : len(docs),
        "num_pages"   : 0,
        "ocr_used"    : False
    }
    save_tracker(tracker)

    print(f"  [OK] answer_guide.json — {len(docs)} entries ingested")
    return len(docs)


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════
def main():
    start_time = time.time()

    print("\n" + "=" * 55)
    print("  VTU RAG INGEST — BULLETPROOF EDITION + OCR")
    print("=" * 55)
    print()
    print("[*] Safety features active:")
    print("    [+] SHA-256 file hashing (detect changes)")
    print("    [+] ChromaDB <-> tracker sync verification")
    print("    [+] Deterministic chunk IDs (no duplicates)")
    print("    [+] Atomic per-file saves (interrupt-safe)")
    print("    [+] Startup integrity check")

    if OCR_AVAILABLE:
        print(f"    [+] OCR auto-fallback (pytesseract, DPI={OCR_DPI}, threshold={OCR_TEXT_THRESHOLD} chars/page)")
    else:
        print("    [!] OCR NOT available — install pytesseract + pdf2image + pillow")
        print("        Scanned PDFs will produce 0 chunks until OCR is set up.")
    print()

    # Load tracker
    tracker = load_tracker()
    if tracker:
        real_entries = sum(1 for v in tracker.values() if v.get("sha256") != "MIGRATED_NO_HASH")
        migrated = len(tracker) - real_entries
        ocr_entries = sum(1 for v in tracker.values() if v.get("ocr_used"))
        print(f"[*] Tracker: {real_entries} verified files ({ocr_entries} via OCR)", end="")
        if migrated:
            print(f", {migrated} migrated (will re-verify)")
        else:
            print()

    # Initialize ChromaDB
    print(f"[*] Initializing ChromaDB with '{EMBED_MODEL}'...")
    try:
        embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    except Exception as e:
        print(f"\n[FATAL] Could not initialize ChromaDB or Ollama: {e}")
        print("[TIP] Make sure Ollama is running: 'ollama serve'")
        print(f"[TIP] Make sure model is pulled: 'ollama pull {EMBED_MODEL}'")
        sys.exit(1)

    # Integrity check
    tracker = verify_integrity(tracker, vectorstore)

    try:
        # ── 1. Past question papers
        print(f"\n{'-' * 55}")
        print("[1/3] Past Question PDFs...")
        print(f"{'-' * 55}")
        p1, i1, processed1, skipped1 = load_pdfs_from_subfolders(
            PAST_QUESTIONS_DIR, "past_questions", tracker, vectorstore
        )
        print(f"      Result: {processed1} new, {skipped1} skipped, "
              f"{p1} pages, {i1} image refs")

        # ── 2. Module notes
        print(f"\n{'-' * 55}")
        print("[2/3] Module Note PDFs...")
        print(f"{'-' * 55}")
        p2, i2, processed2, skipped2 = load_pdfs_from_subfolders(
            MODULES_DIR, "module_notes", tracker, vectorstore
        )
        print(f"      Result: {processed2} new, {skipped2} skipped, "
              f"{p2} pages, {i2} image refs")

        # ── 3. Answer guide
        print(f"\n{'-' * 55}")
        print("[3/3] Answer Guide JSON...")
        print(f"{'-' * 55}")
        guide_count = load_answer_guide(ANSWER_GUIDE_JSON, tracker, vectorstore)

    except KeyboardInterrupt:
        print("\n\n[!] INTERRUPTED -- all completed files are SAFE.")
        print(f"[!] {len(tracker)} files saved in tracker.")
        print("[!] Run again to resume.\n")
        sys.exit(0)

    # Summary
    elapsed         = time.time() - start_time
    total_processed = processed1 + processed2 + (1 if guide_count else 0)
    total_skipped   = skipped1 + skipped2
    total_ocr       = sum(1 for v in tracker.values() if v.get("ocr_used"))

    print(f"\n{'=' * 55}")
    print("  INGEST COMPLETE — SUMMARY")
    print(f"{'=' * 55}")
    print(f"  DB location    : {CHROMA_DB_PATH}")
    print(f"  Total tracked  : {len(tracker)} files")
    print(f"  This run       : {total_processed} ingested, {total_skipped} skipped")
    print(f"  OCR processed  : {total_ocr} files total (all runs)")
    print(f"  Time elapsed   : {elapsed:.1f}s")

    # Per-subject breakdown
    subjects = {}
    for key, info in tracker.items():
        if "|" in key:
            parts = key.split("|")
            subj = parts[1] if len(parts) >= 3 else "general"
        else:
            subj = "general"
        if subj not in subjects:
            subjects[subj] = {"files": 0, "chunks": 0, "ocr": 0}
        subjects[subj]["files"]  += 1
        subjects[subj]["chunks"] += info.get("num_chunks", 0)
        if info.get("ocr_used"):
            subjects[subj]["ocr"] += 1

    print(f"\n  {'Subject':<15} {'Files':>6} {'Chunks':>8} {'OCR':>5}")
    print(f"  {'-' * 38}")
    for subj in sorted(subjects):
        s = subjects[subj]
        print(f"  {subj:<15} {s['files']:>6} {s['chunks']:>8} {s['ocr']:>5}")

    total_chunks = sum(s["chunks"] for s in subjects.values())
    total_ocr_files = sum(s["ocr"] for s in subjects.values())
    print(f"  {'-' * 38}")
    print(f"  {'TOTAL':<15} {len(tracker):>6} {total_chunks:>8} {total_ocr_files:>5}")

    print(f"\n{'=' * 55}")
    print("  Run: python query.py")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()