import asyncio
import json
import lib
from openai import AsyncOpenAI
from typing import Dict, Any

client = AsyncOpenAI(api_key="lmstudio", base_url="http://localhost:1234/v1")


def mcp_tools_to_openai(mcp_tools):
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
        }
        for tool in mcp_tools
    ]


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

    tools = []
    for name, conn in mcp_connections.items():
        # initialize
        await conn.send(
            {
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {
                        "roots": {"listChanged": True},
                        "sampling": {},
                        "elicitation": {},
                        "tools": {},
                    },
                    "clientInfo": {
                        "name": "ExampleClient",
                        "title": "Example Client Display Name",
                        "version": "1.0.0",
                    },
                },
            }
        )
        print(await conn.receive())
        # notifications/initialized
        await conn.send({"method": "notifications/initialized", "params": {}})
        await asyncio.sleep(1)
        # tools/list
        await conn.send({"method": "tools/list", "params": {}})
        tools_response = await conn.receive()
        print(tools_response)
        tools.extend(mcp_tools_to_openai(tools_response["result"]["tools"]))  # noqa

    print(tools)
    response = await client.responses.create(
        model="openai/gpt-oss-120b", input="こんにちわ"
    )
    print(response.output_text)

    for name, conn in mcp_connections.items():
        await conn.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
