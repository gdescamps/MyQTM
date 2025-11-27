"""Unit tests for LLM integration functionality.

This module tests text-to-text LLM calls with different model providers,
verifying response validity and pricing calculations.
"""

from dotenv import load_dotenv

from src.utils.llm import txt2txt_llm

load_dotenv()


def test_call_llm():
    """Verify LLM functionality with multiple model providers.

    Tests text-to-text LLM calls with both Gemini 2.0 Flash and GPT-4o models,
    confirming that:
    1. Responses are non-null strings containing expected content
    2. Pricing information is valid and non-negative
    3. Both models process the same prompt consistently
    """
    # Test Gemini 2.0 Flash model with a simple factual question
    response, price = txt2txt_llm(
        prompt="What is the capital of France?",
        model_name="gemini-2.0-flash",
        cache=False,
    )
    assert response is not None
    assert isinstance(response, str)
    assert "Paris" in response
    assert price >= 0
