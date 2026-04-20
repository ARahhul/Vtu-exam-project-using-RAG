"""
VTU RAG — FastAPI Server
========================
Exposes the query logic from query.py as an HTTP API so the React
frontend can call it.

Start with:
    uvicorn api:app --reload --port 8000

Endpoint:
    POST /query
    Body:  { "question": "...", "subject": "BAD401" | null }
    Returns: { "answer": "...", "sources": [...], "images": [...] }

    POST /query/stream
    Body:  { "question": "...", "subject": "BAD401" | null }
    Returns: Server-Sent Events stream of answer tokens + final metadata
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import os
import io
import csv
import json
import base64
import asyncio
import time
import logging

import fitz                          # PyMuPDF — top-level import
from PIL import Image

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH    = "./chroma_db"
IMAGES_OUTPUT_DIR = "./extracted_images"
EMBED_MODEL       = "nomic-embed-text"
LLM_MODEL         = "glm-5:cloud"
TOP_K             = 3               # Reduced from 6 → 3 for faster retrieval + less LLM context

PROMPT_TEMPLATE = """
You are a VTU exam answer writing assistant.

Follow the VTU Answer Writing Guide strictly:
- Structure every answer: Definition -> Explanation -> Mechanism/Steps -> Diagram mention -> Example -> Advantages/Disadvantages (if applicable) -> Conclusion
- Identify the mark value from the question and set answer length accordingly:
    2-mark  -> 2-3 lines
    5-mark  -> half a page
    10-mark -> 3-4 pages equivalent (detailed)
    15-mark -> 5-6 pages equivalent (exhaustive)
- Always bold or underline key technical terms
- For numerical questions: state given data -> formulas -> step-by-step -> box final answer with units
- Never give a vague or unstructured answer
- Treat every answer as a model answer intended to score full marks in VTU external examination

Use the following retrieved context (from past question papers, module notes, and answer guide):
-----
{context}
-----

Question: {question}

VTU Model Answer:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="VTU RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

os.makedirs(IMAGES_OUTPUT_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES_OUTPUT_DIR), name="images")

# ── Load vectorstore + chain ONCE at startup (not per request) ────────────────
logger.info("[*] Loading vectorstore...")
embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)
logger.info("[*] Vectorstore loaded.")

# Build a single default chain (no subject filter) reused across requests.
# Subject-filtered chains are built on-demand but still cached below.
_chain_cache: dict = {}

def get_chain(subject_filter: Optional[str] = None):
    """Return a cached RetrievalQA chain for the given subject filter."""
    cache_key = subject_filter or "__all__"
    if cache_key not in _chain_cache:
        # Build filter conditions
        filters = []

        # Always restrict to module notes + answer guide (exclude past_questions)
        filters.append({
            "source_type": {"$in": ["module_notes", "ocr", "answer_guide"]}
        })

        # Add subject filter if a specific subject is selected
        if subject_filter:
            filters.append({
                "subject": {"$eq": subject_filter}
            })

        # Combine filters
        if len(filters) == 2:
            where_filter = {"$and": filters}
        else:
            where_filter = filters[0]

        search_kwargs: dict = {"k": TOP_K, "filter": where_filter}
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        llm = OllamaLLM(model=LLM_MODEL, temperature=0.2, num_predict=800)
        _chain_cache[cache_key] = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        logger.info("[*] Built and cached chain for filter: %s", cache_key)
    return _chain_cache[cache_key]


# ── Answer cache — instant responses for repeated questions ───────────────────
_answer_cache: dict = {}
MAX_CACHE_SIZE = 200


def extract_images_on_demand(source_docs) -> List[str]:
    """Extract relevant images from source PDFs and return as WebP data URIs."""
    saved_images: List[str] = []
    seen_keys: set = set()

    for doc in source_docs:
        m         = doc.metadata
        xrefs_str = m.get("image_xrefs", "")
        pdf_path  = m.get("pdf_path", "")

        if not xrefs_str or not pdf_path or not os.path.exists(pdf_path):
            continue

        xrefs     = [int(x.strip()) for x in xrefs_str.split(",") if x.strip()]
        subject   = m.get("subject", "unknown")
        file_name = m.get("file_name", "unknown")
        pdf_stem  = os.path.splitext(file_name)[0]
        page_num  = m.get("page", 0)
        if isinstance(page_num, str):
            page_num = int(page_num) if page_num.isdigit() else 0
        page_num += 1

        save_dir = os.path.join(IMAGES_OUTPUT_DIR, subject)
        os.makedirs(save_dir, exist_ok=True)

        try:
            fitz_doc = fitz.open(pdf_path)
            for xref in xrefs:
                dedup_key = f"{pdf_path}|{xref}"
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                try:
                    pix = fitz.Pixmap(fitz_doc, xref)
                    # Fix: correct CMYK detection (channels minus alpha > 3)
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_filename = f"{pdf_stem}_p{page_num}_xref{xref}.png"
                    img_path = os.path.join(save_dir, img_filename)
                    pix.save(img_path)
                    pix = None  # free memory

                    # Convert PNG → WebP in memory (smaller payload)
                    with Image.open(img_path) as im:
                        webp_buf = io.BytesIO()
                        im.save(webp_buf, format="WEBP", quality=80)
                        b64 = base64.b64encode(webp_buf.getvalue()).decode("utf-8")
                    saved_images.append(f"data:image/webp;base64,{b64}")
                    logger.info("[img] Extracted xref %s from %s", xref, file_name)
                except Exception as e:
                    logger.warning("[img] Failed xref %s in %s: %s", xref, pdf_path, e)
            fitz_doc.close()
        except Exception as e:
            logger.warning("[img] Could not open PDF %s: %s", pdf_path, e)

    return saved_images


# ── Request / Response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    subject: Optional[str] = None
    user_name: Optional[str] = None

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False

class OnboardRequest(BaseModel):
    name: str
    email: str
    college: str

class Source(BaseModel):
    source_type: str
    subject: str
    file_name: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    images: List[str]


# ── Helpers ───────────────────────────────────────────────────────────────────
USERS_FILE     = "./users.txt"
ANALYTICS_FILE = "./analytics.csv"

def lookup_user(name: Optional[str]) -> dict:
    """Look up user info from users.txt by matching the name (case-insensitive)."""
    fallback = {"name": name or "Unknown", "gmail": "", "college": ""}
    if not name:
        return fallback
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split("|")]
                if len(parts) >= 3 and parts[0].lower() == name.lower():
                    return {"name": parts[0], "gmail": parts[1], "college": parts[2]}
    except FileNotFoundError:
        pass
    return fallback


def log_analytics(req: QueryRequest, duration: float, num_sources: int, num_images: int):
    """Write a row to analytics.csv (fire-and-forget)."""
    try:
        user_info = lookup_user(req.user_name)
        file_exists = os.path.exists(ANALYTICS_FILE)
        with open(ANALYTICS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "name", "gmail", "college", "question", "subject", "time_taken_s", "num_sources", "num_images"])
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                user_info["name"],
                user_info["gmail"],
                user_info["college"],
                req.question,
                req.subject or "All",
                duration,
                num_sources,
                num_images,
            ])
    except Exception as e:
        logger.warning("[log] Failed to write analytics: %s", e)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    t0 = time.time()

    # ── Cache check ──────────────────────────────────────────────────────────
    cache_key = f"{req.question.strip().lower()}|{req.subject or 'all'}"
    if cache_key in _answer_cache:
        cached = _answer_cache[cache_key]
        duration = round(time.time() - t0, 1)
        log_analytics(req, duration, len(cached["sources"]), len(cached["images"]))
        logger.info("[query] CACHE HIT '%s' (%.1fs)", req.question[:60], duration)
        return QueryResponse(**cached)

    # ── ChromaDB retrieval + LLM ─────────────────────────────────────────────
    chain = get_chain(subject_filter=req.subject)

    t1 = time.time()
    result = await asyncio.to_thread(chain.invoke, {"query": req.question})
    t2 = time.time()
    logger.info("[timing] ChromaDB + LLM: %.2fs", t2 - t1)

    answer      = result["result"]
    source_docs = result.get("source_documents", [])

    # Deduplicated sources
    seen: set = set()
    sources: List[Source] = []
    for doc in source_docs:
        m   = doc.metadata
        key = f"{m.get('source_type')}|{m.get('subject')}|{m.get('file_name', m.get('rule', ''))}"
        if key not in seen:
            seen.add(key)
            sources.append(Source(
                source_type=m.get("source_type", ""),
                subject=m.get("subject", ""),
                file_name=m.get("file_name", m.get("rule", ""))
            ))

    # ── Image extraction (in background thread) ─────────────────────────────
    t3 = time.time()
    images = await asyncio.to_thread(extract_images_on_demand, source_docs)
    t4 = time.time()
    logger.info("[timing] Image extraction: %.2fs", t4 - t3)

    duration = round(time.time() - t0, 1)

    # ── Cache the result ─────────────────────────────────────────────────────
    response_data = {
        "answer": answer,
        "sources": [s.dict() for s in sources],
        "images": images,
    }
    if len(_answer_cache) >= MAX_CACHE_SIZE:
        # Evict oldest entry
        oldest_key = next(iter(_answer_cache))
        del _answer_cache[oldest_key]
    _answer_cache[cache_key] = response_data

    # ── Analytics ────────────────────────────────────────────────────────────
    log_analytics(req, duration, len(sources), len(images))

    logger.info("[timing] TOTAL: %.2fs (retrieval+LLM=%.2fs, images=%.2fs)",
                time.time() - t0, t2 - t1, t4 - t3)
    logger.info("[query] '%s' -> %d sources, %d images (%.1fs)", req.question[:60], len(sources), len(images), duration)
    return QueryResponse(answer=answer, sources=sources, images=images)


# ── Streaming endpoint ───────────────────────────────────────────────────────

@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """SSE streaming endpoint — sends answer tokens in real-time."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    t0 = time.time()

    # ── Cache check — return full answer instantly as a single SSE burst ─────
    cache_key = f"{req.question.strip().lower()}|{req.subject or 'all'}"
    if cache_key in _answer_cache:
        cached = _answer_cache[cache_key]

        async def cached_stream():
            yield f"data: {json.dumps({'type': 'token', 'content': cached['answer']})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'sources': cached['sources'], 'images': cached['images'], 'time': 0.1})}\n\n"

        duration = round(time.time() - t0, 1)
        log_analytics(req, duration, len(cached["sources"]), len(cached["images"]))
        logger.info("[stream] CACHE HIT '%s'", req.question[:60])
        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    # ── Retrieval phase (get context docs) ───────────────────────────────────
    t1 = time.time()

    # Build filter
    filters = [{"source_type": {"$in": ["module_notes", "ocr", "answer_guide"]}}]
    if req.subject:
        filters.append({"subject": {"$eq": req.subject}})
    where_filter = {"$and": filters} if len(filters) == 2 else filters[0]

    # Retrieve docs
    docs = await asyncio.to_thread(
        vectorstore.similarity_search,
        req.question, k=TOP_K, filter=where_filter
    )
    t2 = time.time()
    logger.info("[stream][timing] ChromaDB retrieval: %.2fs", t2 - t1)

    # Build context string
    context = "\n\n".join(doc.page_content for doc in docs)

    # Deduplicated sources
    seen: set = set()
    sources_list = []
    for doc in docs:
        m = doc.metadata
        key = f"{m.get('source_type')}|{m.get('subject')}|{m.get('file_name', m.get('rule', ''))}"
        if key not in seen:
            seen.add(key)
            sources_list.append({
                "source_type": m.get("source_type", ""),
                "subject": m.get("subject", ""),
                "file_name": m.get("file_name", m.get("rule", ""))
            })

    # Start image extraction in background
    image_task = asyncio.get_event_loop().run_in_executor(
        None, extract_images_on_demand, docs
    )

    # ── LLM streaming phase ──────────────────────────────────────────────────
    full_prompt = PROMPT_TEMPLATE.replace("{context}", context).replace("{question}", req.question)
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.2, num_predict=800)

    async def event_stream():
        full_answer = []
        t_llm_start = time.time()

        # Stream tokens from LLM
        for chunk in llm.stream(full_prompt):
            full_answer.append(chunk)
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

        t_llm_end = time.time()
        logger.info("[stream][timing] LLM generation: %.2fs", t_llm_end - t_llm_start)

        # Wait for images from background
        images = await image_task
        t_done = time.time()
        logger.info("[stream][timing] Image extraction: %.2fs", t_done - t_llm_end)

        answer_text = "".join(full_answer)
        duration = round(t_done - t0, 1)

        # Cache
        response_data = {"answer": answer_text, "sources": sources_list, "images": images}
        if len(_answer_cache) >= MAX_CACHE_SIZE:
            oldest_key = next(iter(_answer_cache))
            del _answer_cache[oldest_key]
        _answer_cache[cache_key] = response_data

        # Analytics
        log_analytics(req, duration, len(sources_list), len(images))

        # Send final metadata
        yield f"data: {json.dumps({'type': 'done', 'sources': sources_list, 'images': images, 'time': duration})}\n\n"

        logger.info("[stream] '%s' -> %d sources, %d images (%.1fs)",
                    req.question[:60], len(sources_list), len(images), duration)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok", "model": LLM_MODEL}


@app.post("/generate")
async def generate(req: GenerateRequest):
    llm = OllamaLLM(model=req.model, temperature=0.7)
    # Run blocking Ollama call in a thread
    res = await asyncio.to_thread(llm.invoke, req.prompt)
    return {"response": res}

@app.post("/onboard")
async def onboard(req: OnboardRequest):
    with open("users.txt", "a", encoding="utf-8") as f:
        f.write(f"{req.name} | {req.email} | {req.college}\n")
    return {"status": "ok"}
