# requirements: sentence-transformers, faiss-cpu, openai

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
from typer import prompt
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from services.blobservice import BlobService

load_dotenv()
    
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

client = OpenAI(
    base_url=azure_endpoint,
    api_key=azure_api_key
)

# Step 2: Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings
blobservice = BlobService()
documents = blobservice.list_blobs_flat() 
embeddings = model.encode(documents)

# Step 3: Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Step 4: Retrieval function
def retrieve(query, k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# Step 5: RAG function
def rag_query(question, use_full_context=True, k=2):
    # Use all documents as context by default; retrieval remains optional.
    blobservice = BlobService()
    if use_full_context:
        context_docs = documents
    else:
        context_docs = retrieve(question, k=k)

    context = "\n".join(context_docs)

    # Create prompt
    prompt = f"""Answer the question based only on this context:
    Context:
    {context}
    Question: {question}
    Answer:"""

    # Generate response
    response = client.responses.create(
        # Azure OpenAI expects deployment name in model field.
        model=azure_deployment,
        input=prompt
    )

    return response.output_text or ""

# Test it
print(rag_query("Who coaches the Wednesday barbell session?"))