import os
import subprocess

def run_chatbot_silently():
    print("🤖 Đang khởi động Chatbot IDC (ẩn cảnh báo hệ thống)...")
    subprocess.run(
        ["python3", "chatbot.py"],
        stderr=subprocess.DEVNULL  # Ẩn toàn bộ cảnh báo từ TensorFlow, XLA, cuDNN, v.v.
    )

if __name__ == "__main__":
    run_chatbot_silently()