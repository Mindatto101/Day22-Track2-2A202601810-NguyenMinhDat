"""
Factory tạo LLM và Embeddings cho 5 providers: openai, gemini, anthropic, ollama, openrouter.

Cách dùng:
    from utils.llm_factory import get_llm, get_embeddings

    llm        = get_llm()            # dùng PROVIDER từ .env
    embeddings = get_embeddings()     # dùng PROVIDER từ .env

    llm_gemini = get_llm("gemini")    # chỉ định provider cụ thể
"""
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

_GEMINI_RATE_LIMITERS = {}
_GEMINI_RATE_LIMITER_LOCK = threading.Lock()
_GEMINI_EMBED_TIMES = deque()
_GEMINI_EMBED_LOCK = threading.Lock()


def _reserve_gemini_embedding_requests(count: int) -> None:
    """Reserve at most 90 embedding items in any rolling 60-second window."""
    with _GEMINI_EMBED_LOCK:
        while True:
            now = time.monotonic()
            while _GEMINI_EMBED_TIMES and now - _GEMINI_EMBED_TIMES[0] >= 60:
                _GEMINI_EMBED_TIMES.popleft()
            if len(_GEMINI_EMBED_TIMES) + count <= 90:
                _GEMINI_EMBED_TIMES.extend([now] * count)
                return
            wait_for = 60 - (now - _GEMINI_EMBED_TIMES[0]) + 0.5
            print(f"⏳ Chờ {wait_for:.1f}s để làm mới quota Gemini embeddings ...")
            time.sleep(wait_for)


def get_llm(
    provider: str = None,
    temperature: float = 0.0,
    model: str = None,
    requests_per_minute: int = None,
):
    """
    Trả về BaseChatModel tương ứng với provider được chọn.

    Args:
        provider    : "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                      Mặc định: đọc PROVIDER từ .env (config.PROVIDER)
        temperature : độ ngẫu nhiên (0.0 = tất định, 1.0 = sáng tạo)

    Returns:
        BaseChatModel instance sẵn sàng sử dụng

    Raises:
        ValueError nếu provider không hợp lệ
        ImportError nếu package tương ứng chưa được cài đặt
    """
    provider = (provider or config.PROVIDER).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.OPENAI_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": temperature,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.rate_limiters import InMemoryRateLimiter

        selected_model = model or config.GEMINI_MODEL
        safe_rpm = requests_per_minute or (12 if "flash-lite" in selected_model else 4)
        limiter_key = (selected_model, safe_rpm)
        with _GEMINI_RATE_LIMITER_LOCK:
            if limiter_key not in _GEMINI_RATE_LIMITERS:
                _GEMINI_RATE_LIMITERS[limiter_key] = InMemoryRateLimiter(
                    requests_per_second=safe_rpm / 60,
                    check_every_n_seconds=0.1,
                    max_bucket_size=1,
                )
        return ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
            max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
            max_retries=5,
            rate_limiter=_GEMINI_RATE_LIMITERS[limiter_key],
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    elif provider == "openrouter":
        # OpenRouter dùng OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=temperature,
            max_tokens=config.OPENROUTER_MAX_TOKENS,
            max_retries=5,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )


def get_embeddings(provider: str = None):
    """
    Trả về Embeddings instance tương ứng với provider được chọn.

    Lưu ý quan trọng:
        - Anthropic KHÔNG có Embeddings API → tự động fallback về OpenAI embeddings
        - OpenRouter cũng dùng OpenAI embeddings (không có API embeddings riêng)
        - Ollama cần model embedding riêng (mặc định: nomic-embed-text)
          Cài đặt: ollama pull nomic-embed-text

    Args:
        provider: "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                  Mặc định: đọc PROVIDER từ .env

    Returns:
        Embeddings instance sẵn sàng sử dụng
    """
    provider = (provider or config.PROVIDER).lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": config.OPENAI_EMBEDDING_MODEL,
            "api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return OpenAIEmbeddings(**kwargs)

    elif provider == "openrouter":
        # OpenRouter exposes an OpenAI-compatible embeddings endpoint.
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENROUTER_EMBEDDING_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            check_embedding_ctx_length=False,
            max_retries=5,
        )

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_core.embeddings import Embeddings
        delegate = GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )

        class RateLimitedGeminiEmbeddings(Embeddings):
            """Keep document embedding below Gemini's 100 requests/minute free-tier cap."""

            model = config.GEMINI_EMBEDDING_MODEL

            def embed_documents(self, texts):
                vectors = []
                batch_size = 90
                for start in range(0, len(texts), batch_size):
                    batch = texts[start:start + batch_size]
                    _reserve_gemini_embedding_requests(len(batch))
                    vectors.extend(delegate.embed_documents(batch, batch_size=batch_size))
                return vectors

            def embed_query(self, text):
                _reserve_gemini_embedding_requests(1)
                return delegate.embed_query(text)

        return RateLimitedGeminiEmbeddings()

    elif provider == "anthropic":
        # Anthropic không cung cấp Embeddings API → dùng OpenAI thay thế
        print("⚠️  Anthropic không có Embeddings API — đang dùng OpenAI embeddings thay thế.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )
