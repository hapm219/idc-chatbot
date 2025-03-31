import os
import json
import hashlib
from pathlib import Path
from PyPDF2 import PdfReader
import docx
from typing import List

def get_file_hash(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def extract_text_from_pdf(path: Path):
    with open(path, "rb") as file:
        reader = PdfReader(file)
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
                if len("".join(pages_text)) > 100000:
                    yield "".join(pages_text)
                    pages_text = []
        if pages_text:
            yield "".join(pages_text)

def extract_text_from_docx(path: Path):
    doc = docx.Document(path)
    paragraphs = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)
            if len("\n".join(paragraphs)) > 100000:
                yield "\n".join(paragraphs)
                paragraphs = []
    if paragraphs:
        yield "\n".join(paragraphs)

def extract_text_from_txt(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as file:
            return file.read()

def split_into_nodes(text: str, min_length: int = 500, max_length: int = 1500) -> List[str]:
    words = text.split()
    nodes = []
    current_node = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length and current_node:
            node_text = " ".join(current_node)
            if len(node_text) >= min_length:
                nodes.append(node_text)
                current_node = [word]
                current_length = len(word)
            else:
                current_node.append(word)
                current_length += len(word) + 1
        else:
            current_node.append(word)
            current_length += len(word) + 1
    
    if current_node:
        node_text = " ".join(current_node)
        if len(node_text) >= min_length:
            nodes.append(node_text)
    return [node.strip() for node in nodes if node.strip()]

def extract_text_with_metadata(path: Path, doc_type: str) -> List[dict]:
    documents = []
    if path.suffix.lower() == ".pdf":
        text_stream = extract_text_from_pdf(path)
    elif path.suffix.lower() == ".docx":
        text_stream = extract_text_from_docx(path)
    elif path.suffix.lower() == ".txt":
        text_stream = [extract_text_from_txt(path)]
    else:
        return []

    for text_chunk in text_stream:
        if text_chunk.strip():
            nodes = split_into_nodes(text_chunk, min_length=500, max_length=1500)
            documents.extend([{
                "text": node,
                "doc_type": doc_type,
                "metadata": {
                    "filename": path.name,
                    "doc_path": str(path),
                    "doc_size": os.path.getsize(path),
                    "doc_type": doc_type,
                }
            } for node in nodes])
    return documents

def save_documents(docs: List[dict], out_dir: Path, basename: str, batch_size: int = 100) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_file = out_dir / f"{basename}.txt"
    jsonl_file = out_dir / f"{basename}.jsonl"

    with open(txt_file, "w", encoding="utf-8") as txt_f, open(jsonl_file, "w", encoding="utf-8") as jsonl_f:
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            txt_f.write("\n\n".join(doc["text"] for doc in batch) + "\n\n")
            jsonl_f.writelines(json.dumps(doc, ensure_ascii=False) + "\n" for doc in batch)