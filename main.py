import os
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="lmstudio", base_url="http://localhost:1234/v1")


async def main() -> None:
    response = await client.responses.create(
        model="openai/gpt-oss-120b", input="こんにちわ"
    )
    print(response.output_text)


if __name__ == "__main__":
    asyncio.run(main())
