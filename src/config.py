"""
Tải cấu hình từ file .env và thiết lập biến môi trường LangSmith.

⚠️  Import module này TRƯỚC KHI import bất kỳ thư viện LangChain nào.
    config.py tự động set LANGCHAIN_* vào os.environ khi được import.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Tải .env từ thư mục gốc của project (Lab/)
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")

# ── LangSmith — PHẢI set trước khi import LangChain ──────────────────────
_tracing = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "true"))
_langsmith_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "")
_langsmith_project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "day22-lab")
_langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)

# Support both current LANGSMITH_* names and legacy LANGCHAIN_* names.
os.environ["LANGSMITH_TRACING"] = _tracing
os.environ["LANGCHAIN_TRACING_V2"] = _tracing
os.environ["LANGSMITH_API_KEY"] = _langsmith_key
os.environ["LANGCHAIN_API_KEY"] = _langsmith_key
os.environ["LANGSMITH_PROJECT"] = _langsmith_project
os.environ["LANGCHAIN_PROJECT"] = _langsmith_project
os.environ["LANGSMITH_ENDPOINT"] = _langsmith_endpoint
os.environ["LANGCHAIN_ENDPOINT"] = _langsmith_endpoint

# ── Provider mặc định ─────────────────────────────────────────────────────
# Đổi giá trị PROVIDER trong .env: openai | gemini | anthropic | ollama | openrouter
PROVIDER = os.getenv("PROVIDER", "openai").lower()

# ── OpenAI ────────────────────────────────────────────────────────────────
OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL        = os.getenv("OPENAI_BASE_URL", "")   # để trống nếu dùng OpenAI chính thức
OPENAI_MODEL           = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ── Google Gemini ─────────────────────────────────────────────────────────
GOOGLE_API_KEY          = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL            = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_EMBEDDING_MODEL  = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))
RAGAS_EVALUATOR_MODELS = [
    item.strip()
    for item in os.getenv(
        "RAGAS_EVALUATOR_MODELS",
        "gemini-3.1-flash-lite-preview,gemini-3.1-flash-lite,gemini-3.1-flash-lite-preview,gemini-3.1-flash-lite",
    ).split(",")
    if item.strip()
]

# ── Anthropic ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# ── Ollama (local, không cần API key) ────────────────────────────────────
OLLAMA_BASE_URL         = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL            = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBEDDING_MODEL  = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ── OpenRouter ────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "cohere/north-mini-code:free")
OPENROUTER_EMBEDDING_MODEL = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL", "liquid/lfm-2.5-embedding-350m:free"
)
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "512"))
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── LangSmith ─────────────────────────────────────────────────────────────
LANGSMITH_API_KEY = _langsmith_key
LANGSMITH_PROJECT = _langsmith_project


def validate() -> bool:
    """
    Kiểm tra các biến môi trường bắt buộc đã được cấu hình.
    Trả về True nếu hợp lệ, False nếu thiếu.
    """
    missing = []

    if not LANGSMITH_API_KEY:
        missing.append("LANGCHAIN_API_KEY (LangSmith)")

    if PROVIDER == "openai" and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    elif PROVIDER == "gemini" and not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    elif PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    elif PROVIDER == "openrouter" and not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    # Ollama: không cần API key

    if missing:
        print("⚠️  Thiếu biến môi trường:")
        for m in missing:
            print(f"   - {m}")
        print("   Hãy kiểm tra file .env của bạn (xem .env.example để biết thêm).")
        return False

    print(f"✅ Config OK  |  Provider: {PROVIDER.upper()}  |  Project: {LANGSMITH_PROJECT}")
    return True


if __name__ == "__main__":
    validate()
