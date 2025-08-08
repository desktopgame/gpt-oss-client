import asyncio
import json
import lib
import sys
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

    mcp_clients: Dict[str, lib.McpClient] = {}
    for name, server in mcp_config["mcpServers"].items():
        mcp_client = lib.McpClient(server)
        mcp_clients[name] = mcp_client
        await mcp_client.start()
        await asyncio.sleep(1)
        if mcp_client.is_exited():
            print(f"{name} is exited.")
        else:
            print(f"{name} is started.")

    tools = []
    tool2client: Dict[str, lib.McpClient] = {}
    for name, mcp_client in mcp_clients.items():
        await mcp_client.initialize()
        await mcp_client.notify_initialized()
        tool_list = await mcp_client.tools_list()
        tools.extend(tool_list)

        for tool_item in tool_list:
            tool2client[tool_item["function"]["name"]] = mcp_client

    input_list = [{"role": "user", "content": "現在の時刻を教えて"}]
    response = await client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=input_list,
        tools=tools,
        tool_choice="auto",
    )

    while True:
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls is not None and len(tool_calls) > 0:
            for tool_call in tool_calls:
                fn = tool_call.function
                args = json.loads(fn.arguments)
                target_client = tool2client[fn.name]
                response = await target_client.tools_call(fn.name, args)
                input_list.append({"role": "tool", "call_id": tool_call.id, "content": response["result"]["content"]})
                response = await client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=input_list,
                    tools=tools,
                    tool_choice="auto",
                )
                print(response.choices[0].message.content)
        else:
            print(response.choices[0].message.content)

        next_prompt = sys.stdin.readline().rstrip()
        input_list.append({"role": "user", "content": next_prompt})

        response = await client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=input_list,
            tools=tools,
            tool_choice="auto",
        )

    for name, mcp_client in mcp_clients.items():
        await mcp_client.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
