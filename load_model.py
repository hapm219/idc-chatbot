# loadmodel.py

import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from config import Config

def load_model_and_tokenizer(hf_token: str):
    if not os.path.exists(os.path.join(Config.MODEL_DIR, "config.json")):
        print(f"⬇️ Đang tải mô hình từ Hugging Face: {Config.MODEL_REPO}...")
        snapshot_download(repo_id=Config.MODEL_REPO, local_dir=Config.MODEL_DIR, token=hf_token)
        print("✅ Đã tải xong mô hình.\n")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_DIR, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    return model, tokenizer

def setup_llm(model, tokenizer):
    return HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "pad_token_id": tokenizer.pad_token_id  # dùng đúng token đã gán
        }
    )

def setup_embed_model():
    Settings.embed_model = HuggingFaceEmbedding(model_name=Config.EMBED_MODEL)