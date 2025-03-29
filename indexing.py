import os
import time
import shutil
import hashlib
import json
import concurrent.futures
import gc
from pathlib import Path
from tqdm import tqdm
import numpy as np

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# === Cấu hình ===
RAW_DATA_DIR = "./data/rawdata"
STORAGE_DIR = "./data/storage"  # Có thể đổi sang /content/temp_storage nếu muốn
CACHE_PATH = os.path.join(STORAGE_DIR, "file_cache.json")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
FORCE_REINDEX = True

MAX_WORKERS = min(8, os.cpu_count() or 4)
print(f"⚙️ Sử dụng {MAX_WORKERS} luồng (CPU cores: {os.cpu_count()})")

# === Hàm tính hash file để kiểm tra thay đổi ===
def get_file_hash(file_path):
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

# === Load cache cũ nếu có ===
if os.path.exists(CACHE_PATH) and not FORCE_REINDEX:
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        old_cache = json.load(f)
else:
    old_cache = {}

# === Xử lý thư mục ===
if FORCE_REINDEX and os.path.exists(STORAGE_DIR):
    print("🧹 FORCE_REINDEX = True → Xoá toàn bộ thư mục storage để tạo lại index.")
    shutil.rmtree(STORAGE_DIR)

os.makedirs(STORAGE_DIR, exist_ok=True)

# === Đọc danh sách file hợp lệ ===
def get_all_files(directory):
    supported_exts = {".pdf", ".docx", ".txt", ".doc"}
    return [str(p) for p in Path(directory).rglob("*") if p.suffix.lower() in supported_exts]

all_files = get_all_files(RAW_DATA_DIR)
print(f"📁 Tổng số file phát hiện: {len(all_files)}")

# === Lọc file mới hoặc đã thay đổi ===
file_hashes = {}
files_to_process = []

for file_path in all_files:
    file_hash = get_file_hash(file_path)
    file_hashes[file_path] = file_hash
    if FORCE_REINDEX:
        files_to_process.append(file_path)
    elif file_hash and old_cache.get(file_path) != file_hash:
        files_to_process.append(file_path)

print(f"📌 Tổng số file cần index (mới hoặc thay đổi): {len(files_to_process)}")

# === Danh sách lỗi ===
error_files = []

# === Hàm xử lý từng file ===
def process_file(file_path):
    try:
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        return docs
    except Exception as e:
        error_files.append((file_path, str(e)))
        return []

# === Tách văn bản ===
parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# === Index song song ===
start_time = time.time()
all_nodes = []
print("🪄 Bắt đầu xử lý song song...")

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_file, path): path for path in files_to_process}
    progress_bar = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="🔍 Đang index")

    for future in progress_bar:
        result = future.result()
        for doc in result:
            nodes = parser.get_nodes_from_documents([doc])
            all_nodes.extend(nodes)

print(f"🧩 Tổng số đoạn văn (nodes) đã tạo: {len(all_nodes)}")

# === Khởi tạo mô hình embedding ===
print("📦 Đang khởi tạo mô hình embedding...")
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

# === Tạo embedding và index ===
print("🧠 Đang tạo index...")
index = VectorStoreIndex([], embed_model=embed_model)

for node in tqdm(all_nodes, desc="⚙️ Đang thêm vào index"):
    index.insert_nodes([node])

# === Lưu index ===
index.storage_context.persist(persist_dir=STORAGE_DIR)

# === Giải phóng RAM sau khi indexing
if 'embed_model' in locals():
    del embed_model
if 'index' in locals():
    del index
if 'all_nodes' in locals():
    del all_nodes
gc.collect()

# === Lưu cache mới ===
with open(CACHE_PATH, "w", encoding="utf-8") as f:
    json.dump(file_hashes, f, indent=2, ensure_ascii=False)

# === Tổng kết ===
end_time = time.time()
mins, secs = divmod(end_time - start_time, 60)
print(f"\n✅ Đã hoàn thành index {len(files_to_process) - len(error_files)} / {len(files_to_process)} file trong {int(mins)} phút {int(secs)} giây.")

# === Ghi log file lỗi (nếu có) ===
if error_files:
    error_path = os.path.join(STORAGE_DIR, "error_files.txt")
    with open(error_path, "w", encoding="utf-8") as f:
        for path, err in error_files:
            f.write(f"{path} -- {err}\n")
    print(f"⚠️ Có {len(error_files)} file lỗi. Chi tiết: {error_path}")
else:
    print("✅ Không có file nào lỗi.")