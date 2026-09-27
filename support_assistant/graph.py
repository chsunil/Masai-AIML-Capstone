"""LangGraph StateGraph: classify intent, then either retrieve-and-answer or
answer directly.

    classify_intent ---[policy_question]---> retrieve_and_answer ---> END
                     \\--[general_question]--> direct_answer      ---> END

Only the generation step inside each node branches on MOCK_LLM. The routing
itself never does, and retrieval always runs for real in both modes -- embedding
and ChromaDB need no API key and no network call.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from ingest import get_collection, get_model
from prompt import ANSWER_PROMPT, CLASSIFY_PROMPT
from schema import AnswerResponse, call_llm, mock_mode, parse_or_retry

TOP_K = 3
SNIPPET_CHARS = 200

# Keyword heuristic for mock-mode classification, exactly as the brief specifies.
POLICY_KEYWORDS = (
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
)

DIRECT_ANSWER_TEXT = "I can only answer questions about Zepto policies right now."


class State(TypedDict):
    query: str
    intent: str
    chunks: list[dict]
    response: AnswerResponse


def classify_intent(state: State) -> State:
    """Node 1 - route the query to retrieval or not."""
    query = state["query"]

    if mock_mode():
        # Graded baseline: pure keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = (
            "policy_question"
            if any(keyword in lowered for keyword in POLICY_KEYWORDS)
            else "general_question"
        )
    else:
        # Optional MOCK_LLM=0 extension.
        raw = call_llm(CLASSIFY_PROMPT.format(query=query)).strip().lower()
        intent = "policy_question" if "policy" in raw else "general_question"

    return {**state, "intent": intent}


def retrieve_and_answer(state: State) -> State:
    """Node 2 - retrieve the top-3 chunks, then generate an answer.

    Retrieval runs for real in BOTH modes. Only the generation step below
    branches on MOCK_LLM.
    """
    query = state["query"]

    results = get_collection().query(
        query_embeddings=get_model().encode([query]).tolist(), n_results=TOP_K
    )
    chunks = [
        {"id": chunk_id, "text": text, "distance": distance}
        for chunk_id, text, distance in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        )
    ]

    if mock_mode():
        # Graded baseline: canned template built from the single best chunk.
        snippet = chunks[0]["text"][:SNIPPET_CHARS]
        response = AnswerResponse(
            answer=f"Based on the retrieved context: {snippet}",
            sources=[chunk["id"] for chunk in chunks],
            confidence=1.0,
        )
    else:
        # Optional MOCK_LLM=0 extension: prompt the real LLM, grounded only in
        # the retrieved chunks, and validate its output against the schema.
        context = "\n\n".join(f"{c['id']}: {c['text']}" for c in chunks)
        prompt = ANSWER_PROMPT.format(context=context, query=query)
        response = parse_or_retry(prompt, call_llm(prompt))

    return {**state, "chunks": chunks, "response": response}


def direct_answer(state: State) -> State:
    """Node 3 - answer without retrieval."""
    if mock_mode():
        # Graded baseline: fixed canned string, no LLM call.
        response = AnswerResponse(
            answer=DIRECT_ANSWER_TEXT, sources=[], confidence=1.0
        )
    else:
        # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
        response = AnswerResponse(
            answer=call_llm(state["query"]).strip(), sources=[], confidence=0.5
        )

    return {**state, "chunks": [], "response": response}


def build_graph():
    """Wire the three nodes together with a conditional edge."""
    graph = StateGraph(State)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        lambda state: state["intent"],
        {
            "policy_question": "retrieve_and_answer",
            "general_question": "direct_answer",
        },
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


_compiled = None


def ask(query):
    """Run one query through the graph and return its validated AnswerResponse."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled.invoke({"query": query})


if __name__ == "__main__":
    import json

    # One query per branch, run with MOCK_LLM at its default.
    for query in [
        "How long does delivery take and when is it free?",
        "What is the capital of France?",
    ]:
        state = ask(query)
        print(f"\nQ: {query}")
        print(f"   intent: {state['intent']}")
        print(f"   {json.dumps(state['response'].model_dump(), indent=2)}")

    # Both branches must behave as the brief specifies.
    policy = ask("What is your refund policy?")
    assert policy["intent"] == "policy_question", policy["intent"]
    assert policy["response"].answer.startswith("Based on the retrieved context: ")
    assert len(policy["response"].sources) == TOP_K
    assert policy["chunks"][0]["id"] == "doc_02", policy["chunks"][0]["id"]

    general = ask("Tell me a joke")
    assert general["intent"] == "general_question", general["intent"]
    assert general["response"].answer == DIRECT_ANSWER_TEXT
    assert general["response"].sources == []

    print("\ngraph.py self-check passed")
