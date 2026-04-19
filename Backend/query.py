"""
VTU RAG Query Script — On-Demand Image Extraction
===================================================
Uses the ChromaDB built by ingest.py.

When a question is asked:
  1. Retrieves relevant text chunks from past papers, notes, answer guide
  2. Generates a VTU model answer using the LLM
  3. Checks if retrieved chunks have image references (xref IDs)
  4. If yes, extracts ONLY those images from the original PDF on-demand
     and saves them to ./extracted_images/ for viewing

Usage:
    python query.py
    python query.py --subject BAD401
"""

import os
import argparse
import fitz  # PyMuPDF — for on-demand image extraction
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CHROMA_DB_PATH    = "./chroma_db"
IMAGES_OUTPUT_DIR = "./extracted_images"
EMBED_MODEL       = "nomic-embed-text"
LLM_MODEL         = "kimi-k2.7b:cloud"
TOP_K             = 6

# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# HELPER: Extract images on-demand from PDF
# ─────────────────────────────────────────────
def extract_images_on_demand(source_docs):
    """
    Looks at retrieved chunks for image_xrefs metadata.
    If present, extracts those specific images from the original PDF
    and saves them to IMAGES_OUTPUT_DIR.
    Returns list of saved image paths.
    """
    saved_images = []
    seen_keys = set()  # avoid duplicate extraction

    for doc in source_docs:
        m = doc.metadata
        xrefs_str = m.get("image_xrefs", "")
        pdf_path  = m.get("pdf_path", "")

        if not xrefs_str or not pdf_path:
            continue

        if not os.path.exists(pdf_path):
            print(f"  [WARN] PDF not found: {pdf_path}")
            continue

        xrefs = [int(x.strip()) for x in xrefs_str.split(",") if x.strip()]
        subject   = m.get("subject", "unknown")
        file_name = m.get("file_name", "unknown")
        pdf_stem  = os.path.splitext(file_name)[0]
        page_num  = m.get("page", 0)
        if isinstance(page_num, str):
            page_num = int(page_num) if page_num.isdigit() else 0
        page_num += 1  # PyMuPDF is 0-indexed, display is 1-indexed

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

                    # Convert CMYK to RGB if needed
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    img_filename = f"{pdf_stem}_p{page_num}_xref{xref}.png"
                    img_path = os.path.join(save_dir, img_filename)
                    pix.save(img_path)
                    saved_images.append(img_path)
                    pix = None

                except Exception as e:
                    print(f"  [WARN] Could not extract xref {xref}: {e}")

            fitz_doc.close()

        except Exception as e:
            print(f"  [WARN] Could not open PDF {pdf_path}: {e}")

    return saved_images


# ─────────────────────────────────────────────
# HELPER: Show text sources
# ─────────────────────────────────────────────
def show_text_sources(source_docs):
    print("\n-- Text Sources Retrieved ----------------------------------------")
    seen = set()
    for doc in source_docs:
        m   = doc.metadata
        key = f"{m.get('source_type')} | {m.get('subject')} | {m.get('file_name', m.get('rule', ''))}"
        if key not in seen:
            print(f"  * {key}")
            seen.add(key)


# ─────────────────────────────────────────────
# HELPER: Show extracted images
# ─────────────────────────────────────────────
def show_extracted_images(image_paths):
    if not image_paths:
        print("\n-- No relevant images for this question --")
        print("----------------------------------------------------------\n")
        return

    print(f"\n-- Relevant Images ({len(image_paths)}) ------------------------------------")
    for i, path in enumerate(image_paths, 1):
        print(f"  [{i}] {path}")
    print("----------------------------------------------------------\n")


# ─────────────────────────────────────────────
# BUILD CHAIN
# ─────────────────────────────────────────────
def build_chain(vectorstore, subject_filter=None):
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


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="VTU RAG Query")
    parser.add_argument("--subject", type=str, default=None,
                        help="Filter by subject (e.g. BAD401, BCS401)")
    args = parser.parse_args()

    print("\n========== VTU RAG QUERY ==========")
    print(f"Model      : {LLM_MODEL}")
    print(f"Embeddings : {EMBED_MODEL}")
    print(f"DB Path    : {CHROMA_DB_PATH}")
    if args.subject:
        print(f"Subject    : {args.subject}")
    print("Type 'exit' to quit.\n")

    # Load vectorstore
    embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )

    chain = build_chain(vectorstore, subject_filter=args.subject)

    while True:
        try:
            question = input("? Your VTU Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if question.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break
        if not question:
            continue

        print("\n[*] Generating answer...\n")

        # ── Generate text answer
        result = chain.invoke({"query": question})
        print("=" * 60)
        print(result["result"])
        print("=" * 60)

        # ── Show text sources
        source_docs = result.get("source_documents", [])
        show_text_sources(source_docs)

        # ── Extract images ON-DEMAND from relevant chunks
        print("\n[*] Checking for relevant images...")
        image_paths = extract_images_on_demand(source_docs)
        show_extracted_images(image_paths)


if __name__ == "__main__":
    main()