"""FastAPI wrapper around the LangGraph flow.

Run locally:  uvicorn main:app --reload --port 7860
Then:         POST /ask  {"query": "..."}  ->  {"answer", "sources", "confidence"}
"""

import os

from fastapi import FastAPI

from graph import ask
from schema import AnswerResponse, AskRequest, mock_mode

app = FastAPI(
    title="Zepto Support Assistant",
    description="Grounded policy Q&A over Zepto's document corpus.",
)


@app.get("/")
def health():
    """Health check, and a quick way to confirm which LLM mode is active."""
    return {
        "status": "ok",
        "mock_llm": mock_mode(),
        "mode": "mock (deterministic, no LLM calls)" if mock_mode() else "real LLM",
        "MOCK_LLM": os.getenv("MOCK_LLM"),
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_endpoint(request: AskRequest) -> AnswerResponse:
    """Route one query through the graph and return the validated response."""
    return ask(request.query)["response"]
