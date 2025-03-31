import os
import logging
import re
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, Manager
from collections import defaultdict
from functools import partial
from refine_utils import extract_text_with_metadata, save_documents, get_file_hash

RAW_DIR = Path("./data/rawdata")
REFINE_DIR = Path("./data/refine_data")
LOG_DIR = Path("./logs")
VALID_EXTS = {".pdf", ".docx", ".txt"}

REFINE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("refine.log"), logging.StreamHandler()])

def get_file_category(rel_path):
    category = rel_path.parts[0].lower()
    return "manual" if "manual" in category else "procedure"

def process_file(file, rel_path, processed_hashes, dup_files, success_files, failed_files):
    if not file.exists():
        logging.error(f"❌ File không tồn tại: {file}")
        failed_files.append(file)
        return
    try:
        file_hash = get_file_hash(file)
        if file_hash in processed_hashes:
            logging.info(f"🔁 Bỏ qua file trùng lặp nội dung: {file}")
            dup_files.append(file)
            return

        processed_hashes[file_hash] = True
        doc_type = get_file_category(rel_path)
        out_dir = REFINE_DIR / rel_path.parent
        basename = file.stem
        docs = extract_text_with_metadata(file, doc_type)

        if not docs:
            logging.warning(f"⚠️ Không trích xuất được nội dung từ file: {file}")
            failed_files.append(file)
            return

        save_documents(docs, out_dir, basename)
        success_files.append(file)
    except Exception as e:
        logging.error(f"❌ Lỗi xử lý file: {file} → {repr(e)}")
        failed_files.append(file)

def get_all_valid_files():
    all_entries = list(RAW_DIR.rglob("*"))
    all_files = [f for f in all_entries if f.is_file()]
    valid_files = [(f, f.relative_to(RAW_DIR)) for f in all_files if f.suffix.lower() in VALID_EXTS]
    invalid_files = [f for f in all_files if f.suffix and f.suffix.lower() not in VALID_EXTS]

    logging.info(f"📄 Tổng số FILE thực sự      : {len(all_files)}")
    logging.info(f"✅ Số file hợp lệ để xử lý   : {len(valid_files)}")
    logging.info(f"🚫 Số file KHÔNG hợp lệ      : {len(invalid_files)}")

    with open(LOG_DIR / "valid_files.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(str(f) for f, _ in valid_files))

    return valid_files

def process_file_wrapper(args, processed_hashes, dup_files, success_files, failed_files):
    file, rel_path = args
    return process_file(file, rel_path, processed_hashes, dup_files, success_files, failed_files)

if __name__ == "__main__":
    valid_files = get_all_valid_files()

    with Manager() as manager:
        processed_hashes = manager.dict()
        dup_files = manager.list()
        success_files = manager.list()
        failed_files = manager.list()

        with Pool(processes=min(os.cpu_count() * 2, 16)) as pool:
            with tqdm(total=len(valid_files), desc="🧹 Đang refine", ncols=100) as pbar:
                func = partial(process_file_wrapper,
                               processed_hashes=processed_hashes,
                               dup_files=dup_files,
                               success_files=success_files,
                               failed_files=failed_files)
                for _ in pool.imap_unordered(func, valid_files):
                    pbar.update(1)

        logging.info(f"🔁 Số file bị trùng nội dung : {len(dup_files)}")
        logging.info(f"✅ Số file xử lý thành công  : {len(success_files)}")
        logging.warning(f"❌ Số file không xử lý được  : {len(failed_files)}")

        # ✅ Sao chép danh sách sau khi Manager đóng
        success_files_list = list(success_files)

    # ✅ Thống kê thư mục đầu ra
    output_folders = defaultdict(int)
    for f in success_files_list:
        rel = f.relative_to(RAW_DIR)
        subfolder = rel.parent
        output_folders[subfolder] += 1

    print("\n📁 Các thư mục trong refine_data đã được xử lý:")
    with open(LOG_DIR / "refine_output_folders.txt", "w", encoding="utf-8") as logf:
        for folder, count in sorted(output_folders.items()):
            line = f"   - 📄 {count} file → 📂 {REFINE_DIR / folder}"
            print(line)
            logf.write(line + "\n")
