import multiprocessing as mp
import shutil
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Annotated

import typer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

DB_DIR = "./chroma_db"
COLLECTION_NAME = "my_app_data"
MODEL_NAME = "all-MiniLM-L6-v2"

app = typer.Typer()


def get_vector_db() -> Chroma:
    logger.debug("Initializing Vector DB connection")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )


def walk_dirs(
    raw_base_path: str,
    document_queue: Queue[tuple[Document, int, int] | None],
    extensions: list[str] | None,
    recursive: bool,  # noqa: FBT001
) -> None:
    base_path = Path(raw_base_path).resolve()
    logger.info("Scanning directory: {base_path}", base_path=base_path)

    if extensions is not None:
        extensions = [e.lower() for e in extensions]

    pattern = "**/*" if recursive else "*"
    # Convert generator to list to get a total count for progress estimation
    all_files = [
        f for f in base_path.glob(pattern) if f.is_file() and (extensions is None or f.suffix.lower() in extensions)
    ]

    total_files = len(all_files)
    logger.info("Found {total} files matching criteria.", total=total_files)

    for index, file_path in enumerate(all_files, 1):
        try:
            stat = file_path.stat()
            mtime = stat.st_mtime
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            if not content.strip():
                logger.warning("Skipping empty file: {file_path}", file_path=file_path)
                continue

            doc = Document(
                page_content=content,
                metadata={"source": str(file_path), "filename": file_path.name, "mtime": mtime},
            )
            # Pass the doc along with current index and total
            document_queue.put((doc, index, total_files))

        except Exception as e:
            logger.exception("Failed to read {file_path}: {error}", file_path=file_path, error=e)
            continue

    document_queue.put(None)
    logger.debug("Producer finished walking directories.")


def consume_docs(document_queue: Queue[tuple[Document, int, int] | None]) -> None:
    db = get_vector_db()

    while True:
        item = document_queue.get()
        if item is None:
            break

        doc, current_idx, total = item
        remaining = total - current_idx

        logger.info(
            "Processing [{idx}/{total}] | Remaining: ~{rem} files | {name}",
            idx=current_idx,
            total=total,
            rem=remaining,
            name=doc.metadata["filename"],
        )

        existing = db.get(where={"source": doc.metadata["source"]})
        should_add = True

        if existing["ids"]:
            old_mtime = existing["metadatas"][0].get("mtime", 0)
            if doc.metadata["mtime"] > old_mtime:
                logger.info("Updating: {name} (Newer mtime detected)", name=doc.metadata["filename"])
                db.delete(ids=existing["ids"])
            else:
                logger.debug("Skipping: {name} (Already up to date)", name=doc.metadata["filename"])
                should_add = False

        if should_add:
            db.add_documents([doc])


@app.command()
def clear() -> None:
    """Completely deletes the vector database."""
    if Path(DB_DIR).exists():
        shutil.rmtree(DB_DIR)
        logger.success("Deleted database at {db_dir}", db_dir=DB_DIR)
    else:
        logger.warning("No database found to delete.")


@app.command()
def index(
    path: str = ".",
    ext: Annotated[list[str] | None, typer.Option(help="File extensions to include")] = None,
    *,
    recursive: Annotated[bool, typer.Option(help="Search subdirectories recursively")] = True,
) -> None:
    """Walks a directory and indexes files into the vector store with deduplication."""
    logger.info("Starting indexing process...")
    mp.set_start_method("spawn", force=True)
    document_queue: Queue[tuple[Document, int, int] | None] = mp.Queue(maxsize=64)

    producer = mp.Process(target=walk_dirs, args=(path, document_queue, ext, recursive))
    consumer = mp.Process(target=consume_docs, args=(document_queue,))

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()
    logger.success("Indexing complete!")


@app.command(name="query")
@app.command(name="search")
def query(
    text: Annotated[str, typer.Argument(help="The search query")],
    k: Annotated[int, typer.Option(help="Number of results")] = 3,
) -> None:
    """Search the index for the most relevant documents."""
    logger.info("Searching for: '{text}' (k={k})", text=text, k=k)
    db = get_vector_db()
    results = db.similarity_search(text, k=k)

    if not results:
        logger.warning("No matches found in the vector store.")
        return

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} | {doc.metadata.get('source')} ---")
        snippet = doc.page_content[:300].replace("\n", " ")
        print(f"{snippet}...")


if __name__ == "__main__":
    # Optional: Configure loguru to be a bit prettier
    logger.add("indexing.log", rotation="10 MB")
    app()
