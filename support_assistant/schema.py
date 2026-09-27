"""Pydantic response schema, and the validated real-LLM call with retries.

In the graded default (mock) mode nothing here calls an LLM: graph.py builds an
AnswerResponse directly from its own code, so there is no model output that
could fail validation. The retry loop exists for the optional MOCK_LLM=0 path.
"""

import json
import os

from pydantic import BaseModel, Field, ValidationError

from prompt import CORRECTION_INSTRUCTION

MAX_RETRIES = 2  # 2 additional attempts after the first, as the brief specifies


class AskRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


def mock_mode():
    """True unless MOCK_LLM is explicitly set to 0.

    Unset or "1" -> mock. This is the default and the graded baseline.
    """
    return os.getenv("MOCK_LLM", "1") != "0"


def call_llm(prompt):
    """Optional MOCK_LLM=0 path: send the prompt to a real LLM.

    Only reached when MOCK_LLM=0 is explicitly set. Groq is used because it has
    a genuinely free tier; the key is read from the environment and is never
    committed.
    """
    from groq import Groq  # imported lazily so mock mode needs no such package

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return completion.choices[0].message.content


def parse_or_retry(prompt, raw):
    """Validate an LLM response against AnswerResponse, retrying on failure.

    Retries up to MAX_RETRIES additional times, each time appending a corrective
    instruction naming the validation error. Returns a clearly marked error
    response if every attempt fails.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            return AnswerResponse(**json.loads(raw))
        except (json.JSONDecodeError, ValidationError, TypeError) as error:
            if attempt == MAX_RETRIES:
                return AnswerResponse(
                    answer=(
                        "ERROR: the language model did not return a valid response "
                        f"after {MAX_RETRIES + 1} attempts. Last error: {error}"
                    ),
                    sources=[],
                    confidence=0.0,
                )
            raw = call_llm(prompt + CORRECTION_INSTRUCTION.format(error=error))

    raise AssertionError("unreachable")


if __name__ == "__main__":
    # Self-check for the retry path, without needing an LLM: a valid payload
    # parses, and an unparseable one degrades to a marked error response rather
    # than raising.
    good = parse_or_retry("", '{"answer": "hi", "sources": ["doc_01"], "confidence": 0.9}')
    assert good.answer == "hi" and good.sources == ["doc_01"], good

    import unittest.mock

    with unittest.mock.patch(f"{__name__}.call_llm", return_value="still not json"):
        bad = parse_or_retry("", "not json at all")
    assert bad.answer.startswith("ERROR:") and bad.confidence == 0.0, bad

    # Out-of-range confidence must be rejected by the schema.
    with unittest.mock.patch(f"{__name__}.call_llm", return_value='{"answer":"x","sources":[],"confidence":9}'):
        out_of_range = parse_or_retry("", '{"answer":"x","sources":[],"confidence":9}')
    assert out_of_range.answer.startswith("ERROR:"), out_of_range

    print("schema.py self-check passed")
    print(f"  mock_mode() with MOCK_LLM={os.getenv('MOCK_LLM')!r} -> {mock_mode()}")
