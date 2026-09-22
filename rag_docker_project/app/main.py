import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import *
from app.rag import direct_llm_answer, rag_answer


app = FastAPI(
    title="RAG Intelligence API",
    description="FastAPI + Groq + Sentence Transformers + FAISS RAG application",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str


class RAGRequest(BaseModel):
    question: str


@app.get("/run")
def home():
    return {
        "message": "RAG Intelligence API is running",
        "docs": "/docs",
        "ui": "/ui",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat")
def chat(request: ChatRequest):
    answer = direct_llm_answer(request.message)
    return {
        "message": request.message,
        "answer": answer,
    }


@app.post("/llm-calling")
def llm_call(request: ChatRequest):
    answer = direct_llm_answer(request.message)
    return {
        "message": request.message,
        "response": answer,
    }


@app.post("/rag")
def rag(request: RAGRequest):
    answer = rag_answer(request.question)
    return {
        "question": request.question,
        "answer": answer,
    }


@app.get("/", response_class=HTMLResponse)
def ui():
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Intelligence</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, Arial, sans-serif;
            background: #0b1020;
            color: #eef2ff;
        }
        .shell {
            width: min(1000px, 94%);
            margin: 40px auto;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
            margin-bottom: 22px;
        }
        h1 { margin: 0; font-size: 30px; }
        .badge {
            padding: 7px 12px;
            border: 1px solid #26304d;
            border-radius: 999px;
            color: #a7f3d0;
            background: #11182b;
            font-size: 13px;
        }
        .card {
            background: #11182b;
            border: 1px solid #26304d;
            border-radius: 18px;
            padding: 22px;
            box-shadow: 0 15px 50px rgba(0,0,0,.25);
        }
        textarea {
            width: 100%;
            min-height: 150px;
            resize: vertical;
            padding: 16px;
            color: #eef2ff;
            background: #0b1020;
            border: 1px solid #303b5e;
            border-radius: 12px;
            outline: none;
            font-size: 16px;
        }
        textarea:focus { border-color: #7187ff; }
        .actions {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 14px;
            gap: 12px;
        }
        button {
            border: 0;
            border-radius: 10px;
            padding: 12px 22px;
            background: #6d7cff;
            color: white;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
        }
        button:disabled { opacity: .6; cursor: wait; }
        .answer {
            margin-top: 22px;
            padding: 20px;
            min-height: 130px;
            white-space: pre-wrap;
            line-height: 1.65;
            background: #0b1020;
            border: 1px solid #26304d;
            border-radius: 12px;
        }
        .muted { color: #94a3b8; font-size: 13px; }
        .suggestions {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 14px;
        }
        .suggestion {
            padding: 12px;
            border: 1px solid #26304d;
            border-radius: 10px;
            color: #cbd5e1;
            background: #0f1628;
            cursor: pointer;
        }
        .suggestion:hover { border-color: #7187ff; }
        @media (max-width: 700px) {
            .suggestions { grid-template-columns: 1fr; }
            .header { align-items: flex-start; flex-direction: column; }
        }
    </style>
</head>
<body>
<div class="shell">
    <div class="header">
        <div>
            <h1>🤖 RAG Intelligence</h1>
            <div class="muted">FastAPI · FAISS · Sentence Transformers · Groq</div>
        </div>
        <div class="badge">● API Online</div>
    </div>

    <div class="card">
        <textarea id="question"
            placeholder="Ask something about RAG, LangChain, LangGraph, FastAPI..."></textarea>

        <div class="suggestions">
            <div class="suggestion" onclick="setQuestion('What is RAG?')">What is RAG?</div>
            <div class="suggestion" onclick="setQuestion('What is LangChain?')">What is LangChain?</div>
            <div class="suggestion" onclick="setQuestion('What is LangGraph?')">What is LangGraph?</div>
        </div>

        <div class="actions">
            <span class="muted">Enter to send · Shift+Enter for a new line</span>
            <button id="askButton" onclick="askRAG()">Ask RAG</button>
        </div>

        <div id="answer" class="answer">Your answer will appear here...</div>
    </div>
</div>

<script>
function setQuestion(text) {
    document.getElementById("question").value = text;
    document.getElementById("question").focus();
}

async function askRAG() {
    const question = document.getElementById("question").value.trim();
    const answerBox = document.getElementById("answer");
    const button = document.getElementById("askButton");

    if (!question) {
        answerBox.textContent = "Please enter a question.";
        return;
    }

    button.disabled = true;
    answerBox.textContent = "Thinking...";

    try {
        const response = await fetch("/rag", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({question: question})
        });

        if (!response.ok) {
            throw new Error("HTTP " + response.status);
        }

        const data = await response.json();
        answerBox.textContent = data.answer;
    } catch (error) {
        answerBox.textContent = "Error: " + error.message;
    } finally {
        button.disabled = false;
    }
}

document.getElementById("question").addEventListener("keydown", function(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        askRAG();
    }
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
    )
