"""Load notes from the Obsidian vault, chunk per-note, and build the Chroma index."""

import pickle
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

VAULT_DIR = Path(r"C:\Users\Maanas\OneDrive\Documents\Obsidian Vault")
INCLUDED_FOLDERS = ["Atomic Notes", "Content Plans", "Knowledge Base", "Research Papers", "Self"]

CHROMA_DIR = "chroma_db"
DOCS_PATH = "docs.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def strip_frontmatter(text: str) -> str:
    return FRONTMATTER_RE.sub("", text, count=1)


def load_notes() -> list[Document]:
    docs = []
    for folder in INCLUDED_FOLDERS:
        for path in (VAULT_DIR / folder).rglob("*.md"):
            text = strip_frontmatter(path.read_text(encoding="utf-8")).strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path.relative_to(VAULT_DIR)), "title": path.stem},
                )
            )
    return docs


def main():
    docs = load_notes()
    print(f"Loaded {len(docs)} notes.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
    print(f"Chroma index built at ./{CHROMA_DIR}")

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)
    print(f"Docs pickled to ./{DOCS_PATH} (for BM25 index)")


if __name__ == "__main__":
    main()
