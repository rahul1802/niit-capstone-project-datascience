import os
from groq import Groq

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return Groq(api_key=api_key)

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
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content