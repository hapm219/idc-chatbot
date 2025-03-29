import os
import time
import textwrap
import torch
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

# ==== Lấy token từ biến môi trường ====
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("❌ Chưa có HF_TOKEN! Thiết lập bằng: os.environ['HF_TOKEN'] = 'your_token_here'")

# ==== Cấu hình model ====
MODEL_REPO = "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"
MODEL_DIR = "/content/models/OpenHermes-AWQ"
STORAGE_DIR = "./data/storage"

# ==== Tải mô hình nếu chưa có ====
if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    print(f"⬇️ Đang tải mô hình từ Hugging Face: {MODEL_REPO}...")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_DIR,
        token=HF_TOKEN
    )
    print("✅ Đã tải xong mô hình.\n")

# ==== Cài embedding tiếng Việt chuẩn ====
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-base"
)
# ==== Kiểm tra thiết bị đang dùng: CPU hay GPU ====
device = "cuda" if torch.cuda.is_available() else "cpu"
device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
print(f"⚙️ Đang sử dụng thiết bị: {device.upper()} - {device_name}")

# ==== Load index ====
print(f"📦 Đang load index từ: {STORAGE_DIR}")
storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
index = load_index_from_storage(storage_context)

# ==== Load mô hình Transformers từ local ====
print(f"🚀 Đang khởi tạo LLM từ: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",  # Dùng GPU tự động
    local_files_only=True
)

llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.2,  # 👈 Ngăn lặp
    "pad_token_id": tokenizer.eos_token_id  # Ẩn warning
},
)

# ==== Tạo query engine ====
query_engine = index.as_query_engine(llm=llm)
print("🤖 Chatbot IDC đã sẵn sàng. Gõ 'exit' để thoát.\n")

# ==== Giao diện người dùng ====
while True:
    user_input = input("👤 Bạn: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Tạm biệt!")
        break

    print("🤔 Chatbot đang suy nghĩ...")
    start = time.time()
    prompt = f"""ưu tiên dựa vào nội dung đã được cung cấp trong tài liệu nội bộ nếu có.
    Nếu không chắc chắn, có thể nêu rõ là không tìm thấy trong dữ liệu.
    Câu hỏi: {user_input}
    """
    response = query_engine.query(prompt)
    print("📄 Đoạn văn mô hình đang dùng để trả lời:")
    for node in response.source_nodes:
        print("-", node.node.text[:300])

    wrapped_text = textwrap.fill(str(response), width=100)
    print("💬", wrapped_text)
    print(f"⏱️ Thời gian phản hồi: {time.time() - start:.2f} giây\n")
