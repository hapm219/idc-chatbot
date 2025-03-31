# chatbot.py

import os
import time
import textwrap
import torch
from langdetect import detect
from contextlib import contextmanager
from sentence_transformers import CrossEncoder
from llama_index.core import StorageContext, load_index_from_storage

from config import Config
from load_model import load_model_and_tokenizer, setup_llm, setup_embed_model
from clean_response import clean_response

HF_TOKEN = os.getenv("HF_TOKEN") or (lambda: (_ for _ in ()).throw(ValueError("❌ Chưa có HF_TOKEN!")))()

@contextmanager
def timer():
    start = time.time()
    yield
    print(f"⏱️ Thời gian phản hồi: {time.time() - start:.2f} giây")

def setup_index(storage_dir):
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    return load_index_from_storage(storage_context)

# Thiết lập
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Đang sử dụng thiết bị: {device.upper()} - {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

try:
    setup_embed_model()
    model, tokenizer = load_model_and_tokenizer(HF_TOKEN)
    llm = setup_llm(model, tokenizer)
    index = setup_index(Config.STORAGE_DIR)
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)
    reranker = CrossEncoder("BAAI/bge-reranker-base")
except Exception as e:
    print(f"❌ Lỗi khởi tạo: {e}")
    exit(1)

print("🤖 Chatbot IDC đã sẵn sàng. Gõ 'exit' để thoát.\n")

# Vòng lặp chính
while True:
    user_input = input("👤 Bạn: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Tạm biệt!")
        break

    lang = detect(user_input)
    print(f"🌐 Ngôn ngữ phát hiện: {lang}")
    print("🤔 Chatbot đang suy nghĩ...")

    with timer():
        results = query_engine.retrieve(user_input)
        pairs = [(user_input, r.node.text) for r in results]
        scores = reranker.predict(pairs)
        top_node = results[scores.argmax()].node.text

        system_prompt = f"""
        Bạn là trợ lý AI trả lời câu hỏi dựa trên tài liệu nội bộ. Trả lời chính xác theo dữ liệu.
        Nếu không chắc chắn, hãy nói rõ là chưa tìm thấy trong dữ liệu.
        Câu hỏi: {user_input}
        """
        response = llm.complete(top_node + "\n" + system_prompt)
        cleaned_text = clean_response(response)
        wrapped_text = textwrap.fill(cleaned_text, width=100)

        print("📄 Đoạn văn mô hình đang dùng để trả lời:")
        print("-", top_node[:300])
        prefix = "(Tiếng Việt)" if lang == "vi" else "(English)"
        print("💬", prefix, wrapped_text)