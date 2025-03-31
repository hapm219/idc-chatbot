# data_indexing.py

import os
import time
import hashlib
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Cài đặt
REFINE_DIR = Path("data/refine_cleaner")  # ✅ Đã đổi sang refine_cleaner
STORAGE_DIR = Path("data/storage")
CACHE_PATH = STORAGE_DIR / "file_cache.json"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
FORCE_REINDEX = True

# Chuẩn bị storage
if FORCE_REINDEX and STORAGE_DIR.exists():
    print("🧹 FORCE_REINDEX = True → Xóa storage cũ.")
    shutil.rmtree(STORAGE_DIR)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Khởi tạo embedding model
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
index = VectorStoreIndex([], embed_model=embed_model)

# Hash cache
if CACHE_PATH.exists() and not FORCE_REINDEX:
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        old_cache = json.load(f)
else:
    old_cache = {}

new_cache = {}

# Đọc file refine_cleaner
all_files = list(REFINE_DIR.rglob("*.txt"))
print(f"📁 Tổng số file cần xử lý từ refine_cleaner: {len(all_files)}")

# Tiến hành indexing từng file với progress bar
for file_path in tqdm(all_files, desc="🔍 Indexing file", ncols=100):
    try:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        new_cache[str(file_path)] = file_hash

        if not FORCE_REINDEX and old_cache.get(str(file_path)) == file_hash:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Gán metadata dựa trên thư mục con
        if len(file_path.parts) >= 3:
            file_type = file_path.parts[-2]  # "manuals", "procedures", etc.
        else:
            file_type = "unknown"

        doc = Document(text=text, metadata={"file_path": str(file_path), "type": file_type})
        nodes = parser.get_nodes_from_documents([doc])
        index.insert_nodes(nodes)

    except Exception as e:
        print(f"❌ Lỗi khi xử lý {file_path.name}: {e}")

# Lưu index và cache
index.storage_context.persist(STORAGE_DIR)
with open(CACHE_PATH, "w", encoding="utf-8") as f:
    json.dump(new_cache, f, indent=2, ensure_ascii=False)

print("✅ Hoàn tất indexing.")
