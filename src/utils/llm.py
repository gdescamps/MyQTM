import asyncio
import hashlib
import os
import pickle
import time
from threading import Lock

from google import genai

from src.utils.path import get_project_root

# Define the path to the cache file
CACHE_FILE = get_project_root() + "/llm_cache.pkl"

# Load the existing cache or initialize a new dictionary
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        _llm_cache = pickle.load(f)
else:
    _llm_cache = {}

prices = {
    "gemini-2.5-flash": {
        "input_price_per_million": 0.1,
        "output_price_per_million": 0.4,
    },
    "gemini-2.0-flash": {
        "input_price_per_million": 0.1,
        "output_price_per_million": 0.4,
    },
    "gemini-2.0-flash-lite": {
        "input_price_per_million": 0.075,
        "output_price_per_million": 0.3,
    },
}

_llm_cache_lock = Lock()
_llm_cache_update = False
_llm_cache_hits = 0
_llm_cache_misses = 0


def txt2txt_llm(prompt, model_name="gemini-2.0-flash", cache=True):
    global _llm_cache
    global _llm_cache_lock
    global _llm_cache_update
    global _llm_cache_hits
    global _llm_cache_misses

    if cache:
        # Generate a unique key based on the prompt and model
        cache_key = hashlib.sha256(f"{model_name}:{prompt}".encode("ascii")).hexdigest()
        # Check if the response is already in the cache
        cache_content = None
        with _llm_cache_lock:
            if cache_key in _llm_cache:
                cache_content = _llm_cache[cache_key]

        if cache_content is not None:
            _llm_cache_hits += 1
            return cache_content, 0.0
        else:
            _llm_cache_misses += 1

    if "gemini" in model_name:
        client = genai.Client(
            vertexai=True,
            project=os.getenv("PROJECT_ID"),
            location="us-east1",
        )

        while True:
            try:
                response = client.models.generate_content(
                    model=model_name, contents=prompt
                )
                break
            except Exception as e:
                time.sleep(10)

        input_token = response.usage_metadata.prompt_token_count
        output_token = response.usage_metadata.candidates_token_count
        price = input_token * prices[model_name]["input_price_per_million"]
        price += output_token * prices[model_name]["output_price_per_million"]
        price /= 1000000
        response = response.text

    # Save the response in the cache
    if cache:
        with _llm_cache_lock:
            _llm_cache[cache_key] = response
            _llm_cache_update = True

    return response, price


def save_llm_cache():
    global _llm_cache
    global _llm_cache_lock
    global _llm_cache_update

    if _llm_cache_update:
        with _llm_cache_lock:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(_llm_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            _llm_cache_update = False


def stats_llm_cache():
    global _llm_cache_hits
    global _llm_cache_misses

    total = _llm_cache_hits + _llm_cache_misses
    hit_rate = (_llm_cache_hits / total * 100) if total > 0 else 0
    print(
        f"LLM Cache Stats: Hits={_llm_cache_hits}, Misses={_llm_cache_misses}, Hit Rate={hit_rate:.2f}%"
    )
