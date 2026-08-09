from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.helpers import config


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""],
)


def split_text(text: str):
    chunks = _splitter.split_text(text)
    return [
        {"text": chunk, "metadata": {"chunk_index": i}}
        for i, chunk in enumerate(chunks)
    ]
