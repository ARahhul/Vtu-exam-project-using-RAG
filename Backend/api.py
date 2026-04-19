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
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import os
import io
import base64
from PIL import Image

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ── Config (mirror query.py) ──────────────────────────────────────────────────
CHROMA_DB_PATH    = "./chroma_db"
IMAGES_OUTPUT_DIR = "./extracted_images"
EMBED_MODEL       = "nomic-embed-text"
LLM_MODEL         = "qwen3-vl:235b-cloud"
TOP_K             = 6

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

# ── App setup ────────────────────────────────────────────────────────────────
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

# ── Load vectorstore once at startup ─────────────────────────────────────────
print("[*] Loading vectorstore...")
embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)
print("[*] Vectorstore loaded.")


def build_chain(subject_filter: Optional[str] = None):
    search_kwargs = {"k": TOP_K}
    if subject_filter:
        search_kwargs["filter"] = {
            "$and": [
                {"source_type": {"$ne": "diagram"}},
                {"subject":     {"$eq": subject_filter}}
            ]
        }
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.2)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return chain


def extract_images_on_demand(source_docs) -> List[str]:
    """Same logic as query.py — extracts relevant images from PDFs."""
    import fitz
    saved_images = []
    seen_keys: set = set()

    for doc in source_docs:
        m = doc.metadata
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
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_filename = f"{pdf_stem}_p{page_num}_xref{xref}.png"
                    img_path = os.path.join(save_dir, img_filename)
                    pix.save(img_path)

                    # Convert PNG → WebP in memory (smaller payload, same quality)
                    with Image.open(img_path) as im:
                        webp_buf = io.BytesIO()
                        im.save(webp_buf, format="WEBP", quality=80)
                        b64 = base64.b64encode(webp_buf.getvalue()).decode("utf-8")
                    saved_images.append(f"data:image/webp;base64,{b64}")
                    pix = None
                except Exception:
                    pass
            fitz_doc.close()
        except Exception:
            pass

    return saved_images


# ── Request / Response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    subject: Optional[str] = None

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False


class Source(BaseModel):
    source_type: str
    subject: str
    file_name: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    images: List[str]


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chain = build_chain(subject_filter=req.subject)
    result = chain.invoke({"query": req.question})

    answer      = result["result"]
    source_docs = result.get("source_documents", [])

    # Deduplicated sources
    seen: set = set()
    sources: List[Source] = []
    for doc in source_docs:
        m = doc.metadata
        key = f"{m.get('source_type')}|{m.get('subject')}|{m.get('file_name', m.get('rule', ''))}"
        if key not in seen:
            seen.add(key)
            sources.append(Source(
                source_type=m.get("source_type", ""),
                subject=m.get("subject", ""),
                file_name=m.get("file_name", m.get("rule", ""))
            ))

    # extract_images_on_demand already returns data URIs — pass through directly
    images = extract_images_on_demand(source_docs)

    return QueryResponse(answer=answer, sources=sources, images=images)


@app.get("/health")
async def health():
    return {"status": "ok", "model": LLM_MODEL}

@app.post("/generate")
async def generate(req: GenerateRequest):
    llm = OllamaLLM(model=req.model, temperature=0.7)
    res = llm.invoke(req.prompt)
    return {"response": res}
