import os
import subprocess

# ===== CẤU HÌNH =====
GITHUB_USER = "hapm219"
REPO_NAME = "idc-chatbot"
REPO_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git"

# ✅ Nhập token thủ công tại runtime
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    GITHUB_TOKEN = input("🔑 Nhập GitHub Token của bạn: ")

FULL_REPO_URL = f"https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{REPO_NAME}.git"

# ===== ĐƯỜNG DẪN LOCAL =====
PROJECT_DIR = "/content/drive/MyDrive/idc-chatbot"
os.chdir(PROJECT_DIR)

# ===== XÓA GIT CŨ (nếu có) và init lại =====
print("🧹 Đang làm sạch repo...")
subprocess.run(["rm", "-rf", ".git"])
subprocess.run(["git", "init"], check=True)
subprocess.run(["git", "config", "user.name", GITHUB_USER], check=True)
subprocess.run(["git", "config", "user.email", f"{GITHUB_USER}@example.com"], check=True)
subprocess.run(["git", "remote", "add", "origin", FULL_REPO_URL], check=True)

# ===== COMMIT & PUSH LẠI SẠCH =====
print("📦 Đang commit lại từ đầu...")
subprocess.run(["git", "add", "*.py"])
subprocess.run(["git", "add", "*.txt"])
subprocess.run(["git", "commit", "-m", "🚀 Version 1 sạch không chứa secrets"], check=True)
subprocess.run(["git", "branch", "-M", "main"], check=True)

print("⬆️ Đang push lên GitHub...")
subprocess.run(["git", "push", "--force", "-u", "origin", "main"], check=True)

print("✅ Đã push version 1 sạch lên GitHub!")
