"""LLM wrapper that selects Anthropic (default) or Ollama.

The Anthropic backend defaults to the ``claude-opus-4-8`` model. The wrapper
exposes a single :func:`get_llm` factory plus an :func:`answer` helper that
applies the system prompt and renders the RAG answer prompt.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from generation.prompts import SYSTEM_PROMPT, build_answer_prompt

logger = logging.getLogger(__name__)

_llm: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    """Return a cached chat model for the configured provider.

    - ``anthropic`` -> :class:`langchain_anthropic.ChatAnthropic` with model
      ``claude-opus-4-8`` (overridable via ``ANTHROPIC_MODEL``).
    - ``ollama`` -> :class:`langchain_community.chat_models.ChatOllama`.
    """
    global _llm
    if _llm is not None:
        return _llm

    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        logger.info("Using Anthropic chat model: %s", settings.anthropic_model)
        _llm = ChatAnthropic(
            model=settings.anthropic_model,  # default: claude-opus-4-8
            api_key=settings.anthropic_api_key,
            max_tokens=4096,
            timeout=120,
        )
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        logger.info("Using Ollama chat model: %s", settings.ollama_model)
        _llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")

    return _llm


def answer(question: str, vector_context: str, graph_context: str) -> str:
    """Generate a grounded, cited answer from the assembled context."""
    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_answer_prompt(question, vector_context, graph_context)),
    ]
    response = llm.invoke(messages)
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)
