import asyncio
import json
import lib
from openai import AsyncOpenAI
from typing import Dict, Any

client = AsyncOpenAI(api_key="lmstudio", base_url="http://localhost:1234/v1")


async def main() -> None:
    mcp_config: Dict[str, Any] = {}
    try:
        with open("mcp.json", "r", encoding="UTF-8") as fp:
            mcp_config = json.load(fp)
    except Exception:
        pass

    mcp_connections: Dict[str, lib.Connection] = {}
    for name, server in mcp_config["mcpServers"].items():
        conn = lib.Connection(server)
        mcp_connections[name] = conn
        await conn.setup()
        await asyncio.sleep(1)
        if conn.is_exited():
            print(f"{name} is exited.")
        else:
            print(f"{name} is started.")

    response = await client.responses.create(
        model="openai/gpt-oss-120b", input="こんにちわ"
    )
    print(response.output_text)

    for name, conn in mcp_connections.items():
        await conn.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
