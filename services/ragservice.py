import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from services.blobservice import BlobService

SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.txt"
)


def load_system_prompt(path: Path = SYSTEM_PROMPT_PATH) -> str:
    """Read and validate the external system-prompt file."""
    try:
        system_prompt = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"System prompt file not found: {path}") from exc

    if not system_prompt:
        raise ValueError(f"System prompt file is empty: {path}")
    return system_prompt


class RAGService:
    """Load documents once and answer questions using vector retrieval."""

    def __init__(self) -> None:
        load_dotenv()

        required_settings = {
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv(
                "AZURE_OPENAI_DEPLOYMENT_NAME"
            ),
        }
        missing_settings = [name for name, value in required_settings.items() if not value]
        if missing_settings:
            raise ValueError(
                "Missing required environment settings: " + ", ".join(missing_settings)
            )

        self.client = OpenAI(
            base_url=required_settings["AZURE_OPENAI_ENDPOINT"],
            api_key=required_settings["AZURE_OPENAI_API_KEY"],
        )
        self.deployment = required_settings["AZURE_OPENAI_DEPLOYMENT_NAME"]
        self.system_prompt = load_system_prompt()
        self.model = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        self.documents = BlobService().list_blobs_flat()
        if not self.documents:
            raise ValueError("No non-empty document lines were found in the blob container")

        encoded_documents = self.model.encode(
            self.documents, convert_to_numpy=True
        )
        document_vectors = np.asarray(encoded_documents, dtype=np.float32)
        self.index = faiss.IndexFlatL2(document_vectors.shape[1])
        self.index.add(document_vectors)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_vectors = np.asarray(query_embedding, dtype=np.float32)
        count = min(k, len(self.documents))
        _, indices = self.index.search(query_vectors, count)
        return [self.documents[index] for index in indices[0]]

    def query(
        self, question: str, use_full_context: bool = True, k: int = 2
    ) -> str:
        context_documents = (
            self.documents if use_full_context else self.retrieve(question, k=k)
        )
        context = "\n".join(context_documents)
        query_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        response = self.client.responses.create(
            model=self.deployment,
            instructions=self.system_prompt,
            input=query_prompt,
        )
        return response.output_text or ""