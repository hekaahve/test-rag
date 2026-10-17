import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from services.ragservice import RAGService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_service = None
    try:
        rag_service = RAGService()
        rag_service.load_documents()
        rag_service.create_embeddings()
        app.state.rag_service = rag_service
    except Exception:
        logger.exception("RAG service initialization failed")
    yield
    app.state.rag_service = None


app = FastAPI(
    title="RAG API",
    description="Question-answering API backed by Azure Blob Storage and Azure OpenAI.",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=4000)]
    use_full_context: bool = True
    k: Annotated[int, Field(ge=1, le=20)] = 2

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank")
        return value


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health(request: Request) -> dict[str, str | int]:
    service = request.app.state.rag_service
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service is not ready")
    return {"status": "ok", "indexed_items": len(service.documents)}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    service = request.app.state.rag_service
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service is not ready")
    try:
        answer = service.query(
            payload.question,
            use_full_context=payload.use_full_context,
            k=payload.k,
        )
    except Exception as exc:
        logger.exception("Failed to answer chat request")
        raise HTTPException(status_code=502, detail="Failed to generate an answer") from exc
    return ChatResponse(answer=answer)