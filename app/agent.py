import json
import time
import re
from typing import List

from .config import ABProfile, get_settings
from .llm import chat_completion, estimate_cost_usd
from .observability import log_event
from .retrieval import HybridRetriever
from .schemas import AgentRequest, AgentResponse, Citation, ToolExecution
from .storage import Storage
from .tools import execute_tool_call, get_tool_schemas


class AgentOrchestrator:
    def __init__(self, storage: Storage, retriever: HybridRetriever) -> None:
        self.storage = storage
        self.retriever = retriever

    def _build_context(self, chunks_with_scores) -> str:
        parts = []
        for chunk, score, bm25_score, dense_score in chunks_with_scores:
            parts.append(
                f"[{chunk.id}] {chunk.text}\nsource: {chunk.source} | scores -> bm25:{bm25_score:.2f} dense:{dense_score:.2f} blended:{score:.2f}"
            )
        return "\n\n".join(parts)

    def _as_citations(self, chunks_with_scores) -> List[Citation]:
        citations: List[Citation] = []
        for chunk, score, _, _ in chunks_with_scores:
            citations.append(
                Citation(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    score=score,
                    text=chunk.text[:4000],
                    source=chunk.source,
                    metadata=chunk.metadata,
                )
            )
        return citations

    def run(self, request: AgentRequest, profile_name: str) -> AgentResponse:
        settings = get_settings()
        profile: ABProfile = settings.ab_profiles.get(profile_name, settings.ab_profiles["control"])
        start = time.perf_counter()

        # Detect URLs in the query (used to trigger fetch_url when permitted).
        urls = re.findall(r"https?://[^\s)]+", request.query)
        first_url = urls[0].rstrip(".,") if urls else None
        allow_fetch_url = bool(first_url) and (
            not request.allowed_tools or "fetch_url" in request.allowed_tools
        )

        keywords = ("net worth", "valuation", "price", "funding", "market cap")
        use_web_first = bool(
            (request.allowed_tools and "web_search" in request.allowed_tools)
            or any(k in request.query.lower() for k in keywords)
        )

        if use_web_first:
            retrieved = []
            context = ""
            citations = []
        else:
            retrieved = self.retriever.search(
                request.query,
                top_k=profile.top_k,
                bm25_weight=profile.bm25_weight,
                embedding_weight=profile.embedding_weight,
                embed_model=profile.embed_model,
            )
            context = self._build_context(retrieved)
            citations = self._as_citations(retrieved)

        history_msgs = [{"role": h["role"], "content": h["content"]} for h in request.history]
        system_prompt = (
            "You are an agent that can plan and call tools. "
            "Use provided context and the available tools to answer. "
            "Prefer web_search when available for fresh, real-world facts. "
            "When you use information from context, cite with [chunk_id]."
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_msgs)
        messages.append(
            {
                "role": "user",
                "content": f"Question: {request.query}\n\nContext:\n{context}\n\nUse tools if they help.",
            }
        )

        tool_executions: List[ToolExecution] = []

        if use_web_first:
            try:
                # Execute web search directly, then summarize.
                # Preserve the user's query; the tool itself adds any special handling (e.g., net worth fallbacks).
                search_query = request.query

                # If a URL is present and fetch_url is allowed, fetch it first.
                if allow_fetch_url:
                    try:
                        t_start = time.perf_counter()
                        fetch_output = execute_tool_call(
                            "fetch_url",
                            {"url": first_url},
                        )
                        fetch_latency_ms = (time.perf_counter() - t_start) * 1000
                        tool_executions.append(
                            ToolExecution(
                                name="fetch_url",
                                args={"url": first_url},
                                output=fetch_output,
                                latency_ms=fetch_latency_ms,
                            )
                        )
                        fetched_text = (fetch_output.get("text") or "")[:4000]
                        messages.append({"role": "assistant", "content": f"Fetched page content from {first_url}."})
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Here is page text (truncated): {fetched_text}\n\nQuestion: {request.query}",
                            }
                        )
                    except Exception as fetch_exc:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f"Failed to fetch URL {first_url}: {fetch_exc}",
                            }
                        )

                t_start = time.perf_counter()
                search_output = execute_tool_call(
                    "web_search",
                    {"query": search_query, "max_results": 8},
                )
                latency_ms = (time.perf_counter() - t_start) * 1000
                tool_executions.append(
                    ToolExecution(
                        name="web_search",
                        args={"query": search_query, "max_results": 8},
                        output=search_output,
                        latency_ms=latency_ms,
                    )
                )
                messages.append({"role": "assistant", "content": "Web search results gathered."})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Here are web search snippets: {json.dumps(search_output)}\n\nQuestion: {request.query}\n\nAnswer concisely with a single up-to-date figure if possible. If unsure, say you could not confirm.",
                    }
                )
                first_completion = chat_completion(
                    messages=messages,
                    model=profile.chat_model,
                    tools=None,
                    tool_choice=None,
                )
                prompt_tokens = getattr(
                    first_completion.usage, "prompt_tokens", getattr(first_completion.usage, "input_tokens", 0)
                )
                completion_tokens = getattr(
                    first_completion.usage, "completion_tokens", getattr(first_completion.usage, "output_tokens", 0)
                )
                tool_calls = []
            except Exception as exc:
                answer = f"Web search failed: {exc}"
                cost_usd = 0.0
                latency_ms = (time.perf_counter() - start) * 1000
                log_event(
                    "agent_run",
                    {
                        "query": request.query,
                        "profile": profile_name,
                        "latency_ms": latency_ms,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cost_usd": cost_usd,
                        "tool_calls": [t.model_dump() for t in tool_executions],
                    },
                )
                return AgentResponse(
                    answer=answer,
                    tool_calls=tool_executions,
                    citations=[],
                    latency_ms=latency_ms,
                    cost_usd=cost_usd,
                    profile=profile_name,
                    hallucination_flag=True,
                )
        else:
            tool_schemas = get_tool_schemas(request.allowed_tools)
            first_completion = chat_completion(
                messages=messages,
                model=profile.chat_model,
                tools=tool_schemas,
                tool_choice="auto",
            )
            prompt_tokens = getattr(
                first_completion.usage, "prompt_tokens", getattr(first_completion.usage, "input_tokens", 0)
            )
            completion_tokens = getattr(
                first_completion.usage, "completion_tokens", getattr(first_completion.usage, "output_tokens", 0)
            )
            tool_calls = first_completion.choices[0].message.tool_calls or []
        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": first_completion.choices[0].message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": call.type,
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )
            for call in tool_calls:
                args = json.loads(call.function.arguments)
                t_start = time.perf_counter()
                output = execute_tool_call(call.function.name, args)
                latency_ms = (time.perf_counter() - t_start) * 1000
                tool_executions.append(
                    ToolExecution(
                        name=call.function.name,
                        args=args,
                        output=output,
                        latency_ms=latency_ms,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": json.dumps(output),
                    }
                )

        if tool_calls:
            final_completion = chat_completion(
                messages=messages,
                model=profile.chat_model,
                tool_choice="none",
            )
            prompt_tokens += getattr(
                final_completion.usage, "prompt_tokens", getattr(final_completion.usage, "input_tokens", 0)
            )
            completion_tokens += getattr(
                final_completion.usage, "completion_tokens", getattr(final_completion.usage, "output_tokens", 0)
            )
            answer = final_completion.choices[0].message.content or ""
        else:
            answer = first_completion.choices[0].message.content or ""
        cost_usd = estimate_cost_usd(prompt_tokens, completion_tokens, model=profile.chat_model)
        latency_ms = (time.perf_counter() - start) * 1000
        hallucination_flag = bool(not citations or (retrieved and retrieved[0][1] < 0.2))
        if use_web_first and tool_executions:
            hallucination_flag = False

        log_event(
            "agent_run",
            {
                "query": request.query,
                "profile": profile_name,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
                "tool_calls": [t.model_dump() for t in tool_executions],
            },
        )

        return AgentResponse(
            answer=answer,
            tool_calls=tool_executions,
            citations=citations,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            profile=profile_name,
            hallucination_flag=hallucination_flag,
        )
