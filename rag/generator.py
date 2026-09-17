import requests

from .config import (
    MODEL_NAME,
    QWEN_API_URL
)


def ask_qwen(prompt, system_prompt=None):

    if system_prompt is None:
        system_prompt = """
You are a document question answering assistant.

Use ONLY supplied context.

Never invent facts.

If answer is not found say:

I cannot find this information in the documents.
"""

    response = requests.post(
        QWEN_API_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt.strip()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 700
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()[
        "choices"
    ][0]["message"]["content"]
