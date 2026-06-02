from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI


def first_key(value: str | None) -> str | None:
    if not value:
        return None
    for item in value.split(","):
        key = item.strip()
        if key:
            return key
    return None


def main() -> int:
    load_dotenv()

    api_key = first_key(os.getenv("MIMO_API_KEYS")) or os.getenv("MIMO_API_KEY")
    base_url = os.getenv("MIMO_BASE_URL", "https://opengateway.gitlawb.com/v1")
    model = os.getenv("MIMO_MODEL") or os.getenv("DEFAULT_MODEL", "mimo-v2.5-pro")

    if not api_key:
        print("Missing MIMO_API_KEYS or MIMO_API_KEY in .env")
        return 1

    http_client = httpx.Client(
        trust_env=False,
        headers={"Accept-Encoding": "identity"},
    )
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        default_headers={"Accept-Encoding": "identity"},
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Tra loi dung mot tu: OK"},
        ],
        temperature=0,
    )

    print(response.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

