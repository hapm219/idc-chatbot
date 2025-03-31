import os
import re
from pathlib import Path
from tqdm import tqdm
from underthesea import sent_tokenize

REFINE_INPUT_DIR = Path("./data/refine_data")
REFINE_OUTPUT_DIR = Path("./data/refine_cleaner")
VALID_EXTS = {".txt"}

REFINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_lines(lines):
    clean = []
    for line in lines:
        line = line.strip()

        # Bỏ dòng trống
        if not line:
            continue

        # Bỏ dòng chỉ chứa số trang hoặc "Trang 1/4", "Page 2"
        if re.fullmatch(r"(trang|page)?\s*\d+(/\d+)?", line.strip(), flags=re.IGNORECASE):
            continue

        # Bỏ dòng ngắn <= 3 từ trôi nổi không đủ ngữ nghĩa
        if len(line.split()) <= 3:
            continue

        clean.append(line)
    return clean

def merge_short_lines(lines, min_len=40):
    merged = []
    buffer = ""

    for line in lines:
        if len(line) < min_len:
            buffer += (" " if buffer else "") + line
        else:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(line)

    if buffer:
        merged.append(buffer)
    return merged

def split_sentences_smart(text):
    return sent_tokenize(text)

def clean_file(in_path: Path, out_path: Path):
    lines = in_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = clean_lines(lines)
    lines = merge_short_lines(lines)
    text = " ".join(lines)
    sentences = split_sentences_smart(text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            sent = sent.strip()
            if sent:
                f.write(sent + "\n\n")

def get_all_txt_files(base_dir):
    return [f for f in base_dir.rglob("*") if f.suffix.lower() in VALID_EXTS and f.is_file()]

if __name__ == "__main__":
    input_files = get_all_txt_files(REFINE_INPUT_DIR)

    print(f"🧹 Đang làm sạch {len(input_files)} file refine với underthesea...")
    for in_file in tqdm(input_files, desc="🧠 Tách câu tiếng Việt", ncols=100):
        rel_path = in_file.relative_to(REFINE_INPUT_DIR)
        out_file = REFINE_OUTPUT_DIR / rel_path
        clean_file(in_file, out_file)

    print("✅ Hoàn tất làm sạch văn bản. Kết quả lưu vào:", REFINE_OUTPUT_DIR)
