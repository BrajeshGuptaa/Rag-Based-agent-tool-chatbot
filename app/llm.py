from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from openai import AzureOpenAI, OpenAI

from .config import get_settings


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    settings = get_settings()
    if settings.use_github_models:
        if not settings.github_token:
            raise RuntimeError("GITHUB_TOKEN is missing")
        return OpenAI(api_key=settings.github_token, base_url=settings.github_models_base_url)
    if settings.use_azure_openai:
        if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
            raise RuntimeError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT is missing")
        return AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: List[str], model: Optional[str] = None) -> List[np.ndarray]:
    settings = get_settings()
    embed_model = model
    if not embed_model:
        if settings.use_github_models:
            embed_model = settings.github_embed_model
        elif settings.use_azure_openai and settings.azure_embed_deployment:
            embed_model = settings.azure_embed_deployment
        else:
            embed_model = settings.embed_model
    resp = _client().embeddings.create(model=embed_model, input=texts)
    return [np.array(item.embedding, dtype=np.float32) for item in resp.data]


def chat_completion(
    messages: Sequence[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
):
    settings = get_settings()
    payload: Dict[str, Any] = {
        "model": (
            model
            or (
                settings.github_chat_model
                if settings.use_github_models
                else (
                    settings.azure_chat_deployment
                    if settings.use_azure_openai and settings.azure_chat_deployment
                    else settings.chat_model
                )
            )
        ),
        "messages": list(messages),
        "temperature": temperature if temperature is not None else settings.temperature,
    }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    return _client().chat.completions.create(**payload)


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    settings = get_settings()
    pricing = settings.pricing.get(model)
    if not pricing:
        return 0.0
    prompt_cost = (prompt_tokens or 0) / 1_000_000 * pricing["prompt"]
    completion_cost = (completion_tokens or 0) / 1_000_000 * pricing["completion"]
    return round(prompt_cost + completion_cost, 6)
