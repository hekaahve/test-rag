# requirements: sentence-transformers, faiss-cpu, openai

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # tämä rivi lataa .env-tiedoston

azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

client = OpenAI(
    base_url=azure_endpoint,
    api_key=azure_api_key
)

# Step 1: Prepare documents
documents = [
    "Our refund policy: 30 days, full refund with receipt.",
    "Shipping takes 3-5 business days for domestic orders.",
    "We accept Visa, Mastercard, and PayPal.",
    "Customer support: support@example.com or call 1-800-HELP"
]

# Step 2: Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings
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
def rag_query(question):
    # Retrieve relevant docs
    context = retrieve(question)

    # Create prompt
    prompt = f"""Answer the question based only on this context:

Context:
{chr(10).join(context)}

Question: {question}

Answer:"""

    # Generate response
    response = client.responses.create(
        # Azure OpenAI expects deployment name in model field.
        model=azure_deployment,
        input=question
    )

    return response.output_text or ""

# Test it
print(rag_query("How long does shipping take?"))
# Output: "Shipping takes 3-5 business days for domestic orders."