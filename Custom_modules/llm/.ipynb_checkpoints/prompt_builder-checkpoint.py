def build_business_prompt(objective, question, metrics_text):
    return f"""
Business Objective:
{objective}

Business Question:
{question}

Analytics Summary:
{metrics_text}

Instructions:
- Answer only from the analytics summary.
- Mention top drivers, risk areas, and business action.
- Keep answer in 5-8 lines.
"""