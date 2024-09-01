from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_type: Literal["text", "pdf", "url", "api"]
    content: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_id: Optional[str] = None


class ChunkMetadata(BaseModel):
    document_id: str
    chunk_index: int
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    text: str
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    query: str
    ab_profile: Optional[str] = None
    top_k: Optional[int] = None
    rerank_weight: Optional[float] = None
    history: List[Dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    latency_ms: float
    cost_usd: float
    hallucination_flag: bool = False
    profile: str


class AgentRequest(BaseModel):
    query: str
    ab_profile: Optional[str] = None
    allowed_tools: Optional[List[str]] = None
    history: List[Dict[str, str]] = Field(default_factory=list)


class ToolExecution(BaseModel):
    name: str
    args: Dict[str, Any]
    output: Any
    latency_ms: float


class AgentResponse(BaseModel):
    answer: str
    tool_calls: List[ToolExecution] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    latency_ms: float
    cost_usd: float
    profile: str
    hallucination_flag: bool = False
