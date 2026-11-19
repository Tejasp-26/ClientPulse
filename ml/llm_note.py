# ml/llm_note.py
# Calls Google Gemini to write a short note about a client.
# If the key is missing or any error occurs, falls back to simple_note().
# The LLM is only told facts that the code has already verified.

import os
from dotenv import load_dotenv
from ml.utils import simple_note

load_dotenv()


def build_prompt(client_facts: dict) -> str:
    """
    Build a prompt for Gemini using only verified facts.
    Tell the model exactly what it may and may not say.
    """
    reasons = client_facts.get("reasons", [])
    actions = client_facts.get("actions", [])
    reasons_str = "\n".join(f"  - {r}" for r in reasons) if reasons else "  - None identified"
    actions_str = "\n".join(f"  - {a}" for a in actions) if actions else "  - None suggested"

    prompt = f"""You are writing a brief account manager note about a B2B client.
Use ONLY the facts below. Do not add any other claims, statistics, or speculation.
Write exactly 3 sentences. Be direct and professional.

Client: {client_facts.get("client_id", "Unknown")}
Risk level: {client_facts.get("risk_level", "Unknown")} ({client_facts.get("risk", 0):.0%} risk score)
Plan: {client_facts.get("plan_type", "Unknown")}
Warning signs found:
{reasons_str}
Suggested actions:
{actions_str}

Write the 3-sentence note now:"""
    return prompt


def write_note(client_facts: dict) -> tuple[str, str]:
    """
    Try to call Gemini. On any failure, return the simple_note fallback.
    Returns: (note_text, source) where source is "gemini" or "fallback".
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        return simple_note(client_facts), "fallback"

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = build_prompt(client_facts)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        note = response.text.strip()
        return note, "gemini"
    except Exception as e:
        print(f"Gemini call failed ({e}); using fallback note.")
        return simple_note(client_facts), "fallback"
