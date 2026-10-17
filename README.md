# RAG API

A FastAPI service that retrieves text from Azure Blob Storage, builds an in-memory
FAISS index, and answers questions with an Azure OpenAI deployment.

## Setup

1. Use Python 3.10 or newer and install dependencies:

   ```powershell
   pip install -r Requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in the Azure OpenAI and Blob Storage
   settings. The Blob client uses `DefaultAzureCredential`; for local development,
   sign in with Azure CLI using `az login`.

3. Start the API from the project directory:

   ```powershell
   uvicorn main:app --reload
   ```

The first startup downloads the embedding model and loads the documents from the
configured blob container. The API is then available at `http://127.0.0.1:8000`.

Edit [prompts/system_prompt.txt](prompts/system_prompt.txt) to change the model's
system instructions. Restart the API after changing the prompt so it is loaded again.

## Endpoints

- `GET /health` — reports readiness and the number of indexed text lines.
- `POST /chat` — accepts a question and returns an answer.
- `GET /docs` — interactive Swagger UI.

Example request body:

```json
{
  "question": "Who coaches the Wednesday barbell session?",
  "use_full_context": false,
  "k": 2
}
```

`use_full_context` defaults to `true`; set it to `false` to use FAISS retrieval.
The `k` value is limited to 1–20.

## Command-line use

The original one-question terminal flow remains available with `python chat.py`.

## Planned improvements

See [ROADMAP.md](ROADMAP.md).
