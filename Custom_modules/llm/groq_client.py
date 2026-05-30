import os
from groq import Groq


def get_groq_client():
    # 1. Try to get the key from the environment
    api_key = os.environ.get("GROQ_API_KEY")

    # 2. Fallback to the hardcoded key if the environment variable is missing
    if not api_key:
        # ⚠️ SECURITY WARNING: Remove this fallback before sharing code or deploying to production
        my_key = "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"
        api_key = my_key

    # 3. Final safety check (in case both are missing)
    if not api_key:
        raise ValueError("GROQ_API_KEY not found and no fallback key provided.")

    # 4. Create and return the client
    client = Groq(api_key=api_key)
    return client


def ask_groq(prompt, model="llama-3.3-70b-versatile", temperature=0.2):
    client = get_groq_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a supply chain analytics assistant. "
                    "Answer only from the provided business metrics. "
                    "Be concise, business-oriented, and actionable."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content
