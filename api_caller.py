"""
DevSoc AI/ML Crew Assignment - Module 1: LLM API Caller
-------------------------------------------------------
This script reads questions from an input text file, makes API calls to an LLM
(Google Gemini / Groq / OpenAI-compatible endpoint), and saves the structured
responses into a JSON file.

Author: AI/ML Applicant
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

# Optional dependency for loading local .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging for clear status updates
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper class to interface with LLM APIs securely."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.provider = self._detect_provider()

    def _detect_provider(self) -> str:
        if os.getenv("GEMINI_API_KEY") or (self.api_key and self.api_key.startswith("AIza")):
            return "gemini"
        elif os.getenv("GROQ_API_KEY") or (self.api_key and self.api_key.startswith("gsk_")):
            return "groq"
        elif os.getenv("OPENAI_API_KEY") or (self.api_key and self.api_key.startswith("sk-")):
            return "openai"
        return "gemini"  # Default

    def generate_response(self, prompt: str) -> str:
        """Dispatches prompt to the appropriate LLM provider."""
        if not self.api_key:
            raise ValueError(
                "API key not found! Please set GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY in your .env file."
            )

        if self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "groq":
            return self._call_groq(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        else:
            return self._call_gemini(prompt)

    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini REST API using requests."""
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1024
            }
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API (OpenAI-compatible) using requests."""
        import requests
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API using requests."""
        import requests
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


def load_questions_from_file(file_path: str) -> List[str]:
    """Reads questions line-by-line from a text file, skipping blank lines."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(questions)} questions from {file_path}")
    return questions


def save_responses_to_json(data: List[Dict[str, Any]], output_path: str) -> None:
    """Saves structured response records into a formatted JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"Successfully saved {len(data)} responses to {output_path}")


def main():
    input_file = os.path.join(os.path.dirname(__file__), "text.txt")
    output_file = os.path.join(os.path.dirname(__file__), "responses.json")

    logger.info("Initializing LLM client...")
    try:
        client = LLMClient()
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        sys.exit(1)

    questions = load_questions_from_file(input_file)
    results = []

    logger.info(f"Processing questions using {client.provider} ({client.model})...")

    for idx, question in enumerate(questions, start=1):
        logger.info(f"[{idx}/{len(questions)}] Processing: '{question}'")
        try:
            response_text = client.generate_response(question)
            results.append({
                "id": idx,
                "question": question,
                "response": response_text,
                "model": client.model,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            # Polite pause to avoid hitting strict rate limits
            time.sleep(1)
        except Exception as err:
            logger.error(f"Failed to fetch response for question {idx}: {err}")
            results.append({
                "id": idx,
                "question": question,
                "response": None,
                "error": str(err),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    save_responses_to_json(results, output_file)
    logger.info("Assignment Module 1 completed successfully.")


if __name__ == "__main__":
    main()
