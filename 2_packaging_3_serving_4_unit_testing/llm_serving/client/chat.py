"""Simple chat client for the vLLM OpenAI-compatible API."""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")


def chat(prompt: str, model: str | None = None) -> str:
    if model is None:
        models = client.models.list()
        model = models.data[0].id

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    prompts = [
        "Explain gradient descent in 2 sentences.",
        "What is the capital of France?",
        "Write a Python function that reverses a string.",
    ]

    for prompt in prompts:
        print(f"\n>>> {prompt}")
        print(chat(prompt))
