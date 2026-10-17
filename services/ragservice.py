import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.blobservice import BlobService

SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.txt"
)


# Read and validate the external system-prompt file.
def load_system_prompt(path: Path = SYSTEM_PROMPT_PATH) -> str:
    try:
        system_prompt = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"System prompt file not found: {path}") from exc

    if not system_prompt:
        raise ValueError(f"System prompt file is empty: {path}")
    return system_prompt


# Load documents once and answer questions using vector retrieval.
class RAGService:

    # Load configuration and prepare the clients and empty index state.
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
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self.model: SentenceTransformer | None = None
        self.documents: list[str] = []
        self.index = None

    # Fetch text rows from Blob Storage and store them on the service.
    def load_documents(self) -> list[str]:
        self.documents = BlobService().list_blobs_flat()
        if not self.documents:
            raise ValueError("No non-empty document lines were found in the blob container")
        return self.documents
    
    # Step 1: Split documents into chunks
    def create_chunked_documents(self, documents: list[str]) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
        )
        chunks = [
            chunk
            for document in documents
            for chunk in text_splitter.split_text(document)
        ]
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks


    # Encode the loaded documents and build the FAISS search index.
    def create_embeddings(self) -> None:
        if not self.documents:
            raise RuntimeError("Load documents before creating embeddings")

        self.chunks = self.create_chunked_documents(self.documents)
        self.model = SentenceTransformer(self.embedding_model_name)
        encoded_documents = self.model.encode(
            self.chunks, convert_to_numpy=True
        )
        document_vectors = np.asarray(encoded_documents, dtype=np.float32)
        self.index = faiss.IndexFlatL2(document_vectors.shape[1])
        self.index.add(document_vectors)

    # Return the k document rows whose vectors are closest to the query.
    def retrieve(self, query: str, k: int = 5) -> list[str]:
        if self.model is None or self.index is None:
            raise RuntimeError("Create document embeddings before searching")
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_vectors = np.asarray(query_embedding, dtype=np.float32)
        count = min(k, len(self.chunks))
        _, indices = self.index.search(query_vectors, count)
        return [self.chunks[index] for index in indices[0]]

    # Generate an answer using either all documents or the retrieved matches.
    def query(
        self, question: str, use_full_context: bool = False, k: int = 5
    ) -> str:
        if not self.chunks:
            raise RuntimeError("Load documents before answering questions")
        context_documents = (
            self.chunks if use_full_context else self.retrieve(question, k=k)
        )
        context = "\n".join(context_documents)
        query_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        response = self.client.responses.create(
            model=self.deployment,
            instructions=self.system_prompt,
            input=query_prompt,
        )
        return response.output_text or ""