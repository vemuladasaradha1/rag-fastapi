# RAG Intelligence — Docker Project

This project converts the Colab FastAPI/RAG notebook into a normal Python
application that can be opened in Antigravity, run locally, containerized with
Docker, and deployed to a cloud container platform.

## Architecture

```text
Browser
   |
   v
FastAPI
   |
   +--> /chat --------> Groq LLM
   |
   +--> /llm-calling -> Groq LLM
   |
   +--> /rag ---------> Sentence Transformers
   |                         |
   |                         v
   |                       FAISS
   |                         |
   |                         v
   |                       Context
   |                         |
   |                         v
   |                       Groq
   |
   +--> /ui ----------> Browser UI
```

## Project structure

```text
rag_docker_project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── rag.py
├── data/
│   └── knowledge.txt
├── static/
├── .dockerignore
├── .gitignore
├── .env.example
├── Dockerfile
├── README.md
└── requirements.txt
```

## 1. Configure the Groq key

For local development, create `.env` from `.env.example`.

Then export the key in your shell, for example:

```bash
export GROQ_API_KEY="YOUR_KEY"
```

On Windows PowerShell:

```powershell
$env:GROQ_API_KEY="YOUR_KEY"
```

Do not put the real key inside Python source code or Dockerfile.

## 2. Run without Docker

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Open:

```text
http://localhost:8080/ui
```

Swagger:

```text
http://localhost:8080/docs
```

Health:

```text
http://localhost:8080/health
```

## 3. Build the Docker image

From the project root:

```bash
docker build -t rag-intelligence .
```

## 4. Run the container

PowerShell:

```powershell
docker run --rm -p 8080:8080 -e GROQ_API_KEY="$env:GROQ_API_KEY" rag-intelligence
```

Linux/macOS:

```bash
docker run --rm -p 8080:8080 -e GROQ_API_KEY="$GROQ_API_KEY" rag-intelligence
```

Then open:

```text
http://localhost:8080/ui
```

## 5. Important difference from Colab

The following Colab-only pieces were intentionally removed:

- `google.colab.userdata`
- `nest_asyncio`
- `pyngrok`
- `ngrok.connect(...)`
- `IPython.display.HTML`
- hard-coded ngrok URLs
- notebook `!pip install` commands

Docker/cloud servers do not need ngrok. The container itself listens on the
server's port.

## 6. Endpoints

### GET `/`

Basic API information.

### GET `/health`

Container health check.

### POST `/chat`

```json
{
  "message": "Explain RAG"
}
```

### POST `/llm-calling`

```json
{
  "message": "Explain LangGraph"
}
```

### POST `/rag`

```json
{
  "question": "What is RAG?"
}
```

### GET `/ui`

Browser interface.

### GET `/docs`

FastAPI Swagger UI.

## Next production steps

1. Move the documents from Python code into a real data ingestion pipeline.
2. Persist the FAISS index instead of rebuilding it on startup.
3. Add LangChain/LangGraph orchestration where required.
4. Add LangSmith tracing/evaluation.
5. Store `GROQ_API_KEY` in a cloud secret manager.
6. Add authentication before exposing private client data.
7. Add logging, rate limiting, and monitoring.
