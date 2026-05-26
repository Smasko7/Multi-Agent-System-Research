"""RAG tool — FAISS vector store over PDF corpus with Gemini embeddings."""

import os
import logging
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

_retriever_tool_cache = None


def _has_pdfs(corpus_dir: str) -> bool:
    return bool(list(Path(corpus_dir).glob("**/*.pdf")))


def _build_or_load_vectorstore():
    """Build FAISS vectorstore from PDFs, or load cached version."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vs_path = config.VECTORSTORE_DIR

    if Path(vs_path).exists() and list(Path(vs_path).iterdir()):
        logger.info("Loading existing vectorstore from %s", vs_path)
        return FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)

    logger.info("Building vectorstore from PDFs in %s", config.CORPUS_DIR)
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pdf_files = list(Path(config.CORPUS_DIR).glob("**/*.pdf"))
    all_docs = []
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf))
            all_docs.extend(loader.load())
        except Exception as exc:
            logger.warning("Failed to load %s: %s", pdf, exc)

    if not all_docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    Path(vs_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(vs_path)
    logger.info("Vectorstore saved to %s (%d chunks)", vs_path, len(chunks))
    return vectorstore


def get_retriever_tool():
    """Return a LangChain Tool wrapping FAISS retriever, or None if no PDFs."""
    global _retriever_tool_cache

    if _retriever_tool_cache is not None:
        return _retriever_tool_cache

    if not _has_pdfs(config.CORPUS_DIR):
        logger.info("No PDFs found in %s — RAG tool disabled.", config.CORPUS_DIR)
        return None

    try:
        from langchain_core.tools import Tool

        vectorstore = _build_or_load_vectorstore()
        if vectorstore is None:
            return None

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        def _retrieve(query: str) -> str:
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant documents found."
            return "\n\n---\n\n".join(
                f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
                for d in docs
            )

        pdf_names = sorted(p.stem for p in Path(config.CORPUS_DIR).glob("**/*.pdf"))
        corpus_list = ", ".join(pdf_names) if pdf_names else "(empty)"

        _retriever_tool_cache = Tool(
            name="document_retriever",
            func=_retrieve,
            description=(
                "Retrieve relevant passages from the local document corpus. "
                f"Corpus contents: {corpus_list}. "
                "STRONGLY PREFER this tool when the query references any of the "
                "above documents or their topics. Use web search only to supplement "
                "or for topics not covered by the corpus. "
                "Input should be a search query string."
            ),
        )
        return _retriever_tool_cache

    except Exception as exc:
        logger.warning("RAG setup failed: %s — continuing without RAG.", exc)
        return None
