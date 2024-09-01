import json
import re
import sqlite3
import time
import math
import csv
import io
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import httpx
import numexpr as ne
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from zoneinfo import ZoneInfo

from .config import get_settings
from urllib import robotparser
from .llm import chat_completion
from .retrieval import HybridRetriever
from .storage import Storage

try:  # Optional, used when cleaning_mode=readability
    import trafilatura
except Exception:  # pragma: no cover - best effort import
    trafilatura = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Any]


# Simple token bucket per domain to throttle fetch_url.
_RATE_LIMIT_BUCKETS: Dict[str, Dict[str, float]] = {}
_STORAGE = Storage()
_RETRIEVER = HybridRetriever(_STORAGE)


def _rate_limit(domain: str, tool: str) -> None:
    settings = get_settings()
    limits = settings.per_domain_rate_limits.get(
        domain, {"requests_per_minute": settings.rate_limit_requests_per_minute, "burst": settings.rate_limit_burst}
    )
    rate = limits["requests_per_minute"] / 60.0
    burst = limits["burst"]
    bucket = _RATE_LIMIT_BUCKETS.setdefault(domain, {"tokens": float(burst), "last": time.time()})
    now = time.time()
    bucket["tokens"] = min(burst, bucket["tokens"] + (now - bucket["last"]) * rate)
    if bucket["tokens"] < 1.0:
        raise ValueError(f"Rate limit exceeded for {domain} ({tool})")
    bucket["tokens"] -= 1.0
    bucket["last"] = now


def _is_allowed_url(url: str) -> str:
    settings = get_settings()
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise ValueError("Only http/https URLs are allowed")
    host = parsed.hostname or ""
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
        raise ValueError("IP literal hosts are not allowed")
    for pattern in settings.allowed_url_regexes:
        if re.match(pattern, url):
            return host
    raise ValueError("URL not allowed by whitelist")


def _robots_allowed(url: str) -> bool:
    settings = get_settings()
    if settings.robots_policy != "respect":
        return True
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        resp = httpx.get(robots_url, timeout=settings.max_fetch_time_seconds, follow_redirects=True)
    except Exception:
        return True  # best-effort: if robots cannot be fetched, allow
    if resp.status_code >= 400:
        return True
    rp = robotparser.RobotFileParser()
    rp.parse(resp.text.splitlines())
    return rp.can_fetch(settings.robots_user_agent, url)


def _clean_html(raw_html: str, mode: str, min_text_length: int = 200) -> str:
    if mode == "readability" and trafilatura:
        try:
            extracted = trafilatura.extract(raw_html, include_comments=False, include_tables=False)
            if extracted and len(extracted) >= min_text_length:
                return extracted.strip()
        except Exception:
            pass
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    return text.strip()


def _fetch_url(args: Dict[str, Any]) -> Any:
    url = args.get("url", "")
    headers = args.get("headers") or {}
    settings = get_settings()
    cleaning_mode = args.get("cleaning_mode", settings.scrape_cleaning_mode)
    domain = _is_allowed_url(url)
    if not _robots_allowed(url):
        raise ValueError("Blocked by robots.txt")
    _rate_limit(domain, "fetch_url")

    merged_headers = {
        "User-Agent": "GenericFetchBot/1.0 (+https://example.com/bot-info)",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    merged_headers.update(headers)

    timeout = httpx.Timeout(settings.max_fetch_time_seconds, read=settings.max_fetch_time_seconds)
    content = b""
    content_type = ""
    with httpx.stream("GET", url, headers=merged_headers, timeout=timeout, follow_redirects=True) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        for chunk in resp.iter_bytes():
            content += chunk
            if len(content) > settings.max_content_size_bytes:
                raise ValueError("Content too large")
    text = content.decode(resp.encoding or "utf-8", errors="ignore")
    cleaned = _clean_html(text, cleaning_mode)
    return {
        "url": url,
        "domain": domain,
        "content_type": content_type,
        "cleaning_mode": cleaning_mode,
        "text": cleaned,
    }


def _web_search(args: Dict[str, Any]) -> Any:
    query = args.get("query", "")
    max_results = int(args.get("max_results", 8))
    region = args.get("region", "us")
    settings = get_settings()

    # Prefer Serper (Google Search) when an API key is configured.
    if settings.serper_api_key:
        headers = {"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"}
        payload = {
            "q": query,
            "gl": region,
            "num": max_results,
        }
        resp = httpx.post("https://google.serper.dev/search", json=payload, headers=headers, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        serper_results = []
        for item in data.get("organic", []):
            serper_results.append(
                {"title": item.get("title"), "snippet": item.get("snippet"), "link": item.get("link")}
            )
        if not serper_results:
            for item in data.get("news", []):
                serper_results.append(
                    {"title": item.get("title"), "snippet": item.get("snippet"), "link": item.get("link")}
                )
        if serper_results:
            return serper_results[:max_results]

    # Fallback to DuckDuckGo if Serper is unavailable or returned nothing.
    results = []
    queries = [query]
    if "net worth" in query.lower():
        queries.append(f"{query} 2025 usd")
        queries.append(f"latest {query} Forbes")
    with DDGS() as search:
        for q in queries:
            for item in search.text(
                q,
                max_results=max_results,
                region="in-en",
                safesearch="off",
                backend="html",
            ):
                results.append(
                    {
                        "title": item.get("title"),
                        "snippet": item.get("body"),
                        "link": item.get("href"),
                    }
                )
            if results:
                break
        if not results:
            for q in queries:
                for item in search.news(q, max_results=max_results, region="in-en", safesearch="off"):
                    results.append(
                        {
                            "title": item.get("title"),
                            "snippet": item.get("body"),
                            "link": item.get("url"),
                        }
                    )
                if results:
                    break
    return results


def _news_search(args: Dict[str, Any]) -> Any:
    query = args.get("query", "")
    max_results = int(args.get("max_results", 8))
    region = args.get("region", "us")
    settings = get_settings()

    # Prefer Serper news when available.
    if settings.serper_api_key:
        headers = {"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"}
        payload = {"q": query, "gl": region, "num": max_results}
        resp = httpx.post("https://google.serper.dev/news", json=payload, headers=headers, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        serper_results = []
        for item in data.get("news", []):
            serper_results.append(
                {"title": item.get("title"), "snippet": item.get("snippet"), "link": item.get("link")}
            )
        if serper_results:
            return serper_results[:max_results]

    # Fallback: DuckDuckGo news.
    results = []
    with DDGS() as search:
        for item in search.news(query, max_results=max_results, region="in-en", safesearch="off"):
            results.append(
                {"title": item.get("title"), "snippet": item.get("body"), "link": item.get("url")}
            )
    return results


def _llm_router(args: Dict[str, Any]) -> Any:
    """Select a profile/model based on task_hint or priority."""
    task_hint = (args.get("task_hint") or "").lower()
    priority = args.get("priority", "balanced")
    settings = get_settings()

    # Simple heuristic: quality for reasoning/analysis, fast for speed, control as default.
    if "analysis" in task_hint or "long" in task_hint or priority == "quality":
        profile = "quality"
    elif "quick" in task_hint or "draft" in task_hint or priority == "fast":
        profile = "fast"
    else:
        profile = "control"
    chosen = settings.ab_profiles.get(profile, settings.ab_profiles["control"])
    return {
        "profile": profile,
        "chat_model": chosen.chat_model,
        "embed_model": chosen.embed_model,
    }


def _validate_tool_args(args: Dict[str, Any]) -> Any:
    """Validate tool arguments, mainly URLs/headers for fetch_url or web requests."""
    url = args.get("url")
    headers = args.get("headers") or {}
    result = {"url_allowed": None, "headers_allowed": True, "reasons": []}
    if url:
        try:
            host = _is_allowed_url(url)
            result["url_allowed"] = True
            result["host"] = host
        except Exception as exc:
            result["url_allowed"] = False
            result["reasons"].append(str(exc))
    # Simple header guard: block auth/cookie injection from user-supplied headers.
    forbidden = {"authorization", "cookie"}
    for k in headers.keys():
        if k.lower() in forbidden:
            result["headers_allowed"] = False
            result["reasons"].append(f"Header {k} not allowed")
    return result


def _search_docs(args: Dict[str, Any]) -> Any:
    """Expose hybrid retriever as a tool."""
    settings = get_settings()
    query = args.get("query", "")
    top_k = int(args.get("top_k", settings.top_k))
    bm25_weight = float(args.get("bm25_weight", settings.bm25_weight))
    embedding_weight = float(args.get("embedding_weight", settings.embedding_weight))
    embed_model = args.get("embed_model", settings.ab_profiles["control"].embed_model)
    _RETRIEVER.refresh()
    results = _RETRIEVER.search(
        query, top_k=top_k, bm25_weight=bm25_weight, embedding_weight=embedding_weight, embed_model=embed_model
    )
    out = []
    for chunk, score, bm25_score, dense_score in results:
        out.append(
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "score": score,
                "bm25_score": bm25_score,
                "dense_score": dense_score,
                "text": chunk.text[:1200],
                "source": chunk.source,
                "metadata": chunk.metadata,
            }
        )
    return out


def _summarize_page(args: Dict[str, Any]) -> Any:
    """Fetch (optional) and summarize text into key facts."""
    url = args.get("url")
    text = args.get("text") or ""
    cleaning_mode = args.get("cleaning_mode")
    fetched = None
    if url:
        fetched = _fetch_url({"url": url, "cleaning_mode": cleaning_mode})
        text = fetched.get("text") or text
    if not text:
        raise ValueError("No text to summarize")
    snippet = text[:4000]
    messages = [
        {"role": "system", "content": "Summarize into 5 concise bullets with key facts and numbers if present."},
        {"role": "user", "content": snippet},
    ]
    resp = chat_completion(messages=messages, model=None, temperature=0.2)
    summary = resp.choices[0].message.content if resp.choices else ""
    return {
        "summary": summary,
        "source_url": url,
        "used_text_len": len(snippet),
        "truncated": len(text) > len(snippet),
    }


def _plan(args: Dict[str, Any]) -> Any:
    """Lightweight planner suggesting steps/tools."""
    goal = args.get("goal", "")
    has_url = bool(re.search(r"https?://", goal))
    needs_fresh = any(k in goal.lower() for k in ["today", "latest", "recent", "news", "current"])
    steps = []
    if needs_fresh:
        steps.append({"step": "Fetch recent news", "tool": "news_search", "args": {"query": goal, "max_results": 5}})
    steps.append({"step": "General web search", "tool": "web_search", "args": {"query": goal, "max_results": 6}})
    if has_url:
        steps.append({"step": "Fetch page", "tool": "fetch_url", "args": {"url": None}})
        steps.append({"step": "Summarize page", "tool": "summarize_page", "args": {"url": None}})
    steps.append({"step": "Search internal docs", "tool": "search_docs", "args": {"query": goal, "top_k": 5}})
    steps.append({"step": "Compose answer", "tool": "none", "args": {}})
    return {"goal": goal, "steps": steps}


def _table_qa(args: Dict[str, Any]) -> Any:
    """Basic CSV/TSV/XLSX summary and simple stats."""
    content = args.get("content", "")
    delimiter = args.get("delimiter") or None
    query = args.get("query", "")
    fmt = args.get("format")
    if not content:
        raise ValueError("content is required (CSV/TSV or base64-encoded XLSX text not supported here)")
    buf = io.StringIO(content)
    dialect = "excel"
    if fmt == "tsv" or delimiter == "\t":
        dialect = "excel-tab"
    reader = csv.reader(buf, dialect=dialect, delimiter=delimiter or ("," if dialect == "excel" else "\t"))
    rows = list(reader)
    if not rows:
        return {"rows": 0, "columns": [], "summary": "No data"}
    header = rows[0]
    data_rows = rows[1:]
    col_count = len(header)
    numeric_stats = {}
    for idx, name in enumerate(header):
        vals = []
        for r in data_rows:
            if idx < len(r):
                try:
                    vals.append(float(r[idx]))
                except Exception:
                    continue
        if vals:
            numeric_stats[name] = {
                "count": len(vals),
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
            }
    summary = f"Rows: {len(data_rows)}, Columns: {col_count}. Numeric columns: {list(numeric_stats.keys())}"
    if query:
        summary += f" | Query not executed; only summary returned. (query='{query}')"
    return {"rows": len(data_rows), "columns": header, "numeric_stats": numeric_stats, "summary": summary}


def _python_sandbox(args: Dict[str, Any]) -> Any:
    """Run small Python snippets with restricted builtins."""
    code = args.get("code", "")
    if not code:
        raise ValueError("code is required")
    allowed_builtins = {"__builtins__": {"len": len, "range": range, "sum": sum, "min": min, "max": max, "abs": abs}}
    local_env: Dict[str, Any] = {}
    safe_globals = {"math": math}
    exec(code, {**allowed_builtins, **safe_globals}, local_env)
    return {k: v for k, v in local_env.items() if not k.startswith("__")}


def _geo_time(args: Dict[str, Any]) -> Any:
    """Compute distance or convert time zones."""
    mode = args.get("mode", "distance")
    if mode == "distance":
        lat1 = float(args["lat1"])
        lon1 = float(args["lon1"])
        lat2 = float(args["lat2"])
        lon2 = float(args["lon2"])
        # Haversine
        r = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        km = r * c
        return {"distance_km": km}
    elif mode == "timezone":
        ts = args.get("timestamp") or datetime.utcnow().isoformat()
        from_tz = args.get("from_tz", "UTC")
        to_tz = args.get("to_tz", "UTC")
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        converted = dt.replace(tzinfo=ZoneInfo(from_tz)).astimezone(ZoneInfo(to_tz))
        return {"converted": converted.isoformat()}
    else:
        raise ValueError("mode must be 'distance' or 'timezone'")


def _calculator(args: Dict[str, Any]) -> Any:
    expression = args.get("expression", "")
    return float(ne.evaluate(expression))


def _db_query(args: Dict[str, Any]) -> Any:
    query = args.get("query", "")
    db_path = args.get("db_path", "data/app.db")
    if not query.lower().strip().startswith("select"):
        raise ValueError("Only SELECT queries are allowed")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(query)
    cols = [c[0] for c in cur.description] if cur.description else []
    rows = cur.fetchall()
    conn.close()
    return {"columns": cols, "rows": rows}


def _call_internal_api(args: Dict[str, Any]) -> Any:
    method = args.get("method", "GET").upper()
    url = args["url"]
    payload = args.get("payload")
    headers = args.get("headers")
    resp = httpx.request(method, url, json=payload, headers=headers, timeout=10.0)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"status_code": resp.status_code, "text": resp.text}


AVAILABLE_TOOLS: Dict[str, Tool] = {
    "web_search": Tool(
        name="web_search",
        description="Search the web for fresh information via DuckDuckGo.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
        handler=_web_search,
    ),
    "calculator": Tool(
        name="calculator",
        description="Evaluate math expressions using a safe numeric evaluator.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        handler=_calculator,
    ),
    "database_query": Tool(
        name="database_query",
        description="Run read-only SQL queries against the local SQLite DB.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "db_path": {"type": "string", "default": "data/app.db"},
            },
            "required": ["query"],
        },
        handler=_db_query,
    ),
    "internal_api": Tool(
        name="internal_api",
        description="Call internal HTTP APIs for additional data.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "default": "GET"},
                "payload": {"type": "object"},
                "headers": {"type": "object"},
            },
            "required": ["url"],
        },
        handler=_call_internal_api,
    ),
    "fetch_url": Tool(
        name="fetch_url",
        description="Fetch and clean text content from a whitelisted URL with robots.txt respect and rate limits.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "headers": {"type": "object"},
                "cleaning_mode": {"type": "string", "enum": ["readability", "basic_bs4"]},
            },
            "required": ["url"],
        },
        handler=lambda args: _fetch_url(args),
    ),
    "news_search": Tool(
        name="news_search",
        description="Search recent news (Serper news when configured, otherwise DuckDuckGo news).",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
                "region": {"type": "string", "default": "us"},
            },
            "required": ["query"],
        },
        handler=_news_search,
    ),
    "llm_router": Tool(
        name="llm_router",
        description="Pick an A/B profile/model for the task.",
        parameters={
            "type": "object",
            "properties": {
                "task_hint": {"type": "string"},
                "priority": {"type": "string", "enum": ["balanced", "fast", "quality"], "default": "balanced"},
            },
        },
        handler=_llm_router,
    ),
    "validate_tool_args": Tool(
        name="validate_tool_args",
        description="Validate tool arguments (e.g., URLs/headers) before execution.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "headers": {"type": "object"},
            },
        },
        handler=_validate_tool_args,
    ),
    "search_docs": Tool(
        name="search_docs",
        description="Search internal documents (hybrid BM25 + embeddings).",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 6},
                "bm25_weight": {"type": "number"},
                "embedding_weight": {"type": "number"},
                "embed_model": {"type": "string"},
            },
            "required": ["query"],
        },
        handler=_search_docs,
    ),
    "summarize_page": Tool(
        name="summarize_page",
        description="Summarize provided text or fetched URL into key facts.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "text": {"type": "string"},
                "cleaning_mode": {"type": "string", "enum": ["readability", "basic_bs4"]},
            },
        },
        handler=_summarize_page,
    ),
    "planner": Tool(
        name="planner",
        description="Propose ordered steps and tools for a goal.",
        parameters={
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
            },
            "required": ["goal"],
        },
        handler=_plan,
    ),
    "table_qa": Tool(
        name="table_qa",
        description="Summarize CSV/TSV data and basic numeric stats.",
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "delimiter": {"type": "string"},
                "format": {"type": "string", "enum": ["csv", "tsv"]},
                "query": {"type": "string"},
            },
            "required": ["content"],
        },
        handler=_table_qa,
    ),
    "python_sandbox": Tool(
        name="python_sandbox",
        description="Run small Python snippets in a restricted sandbox.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string"},
            },
            "required": ["code"],
        },
        handler=_python_sandbox,
    ),
    "geo_time": Tool(
        name="geo_time",
        description="Distance (km) between coordinates or timezone conversion.",
        parameters={
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["distance", "timezone"], "default": "distance"},
                "lat1": {"type": "number"},
                "lon1": {"type": "number"},
                "lat2": {"type": "number"},
                "lon2": {"type": "number"},
                "timestamp": {"type": "string"},
                "from_tz": {"type": "string"},
                "to_tz": {"type": "string"},
            },
            "required": ["mode"],
        },
        handler=_geo_time,
    ),
}


def get_tool_schemas(selected: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    tools = AVAILABLE_TOOLS
    if selected:
        tools = {k: v for k, v in AVAILABLE_TOOLS.items() if k in selected}
    schemas = []
    for tool in tools.values():
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
        )
    return schemas


def execute_tool_call(name: str, args: Dict[str, Any]) -> Any:
    tool = AVAILABLE_TOOLS.get(name)
    if not tool:
        raise ValueError(f"Unknown tool {name}")
    return tool.handler(args)
