import asyncio
import json
import lib
import sys
from openai import AsyncOpenAI
from typing import Dict, Any
from halo import Halo

api_key = "lmstudio"
base_url = "http://localhost:1234/v1"
model = "openai/gpt-oss-120b"
system_prompt = "あなたは日本語で話す親切なアシスタントです。"

try:
    with open("gpt-oss-client.json", "r", encoding="UTF-8") as fp:
        config = json.load(fp)

        if "api_key" in config:
            api_key = config["api_key"]
        if "base_url" in config:
            base_url = config["base_url"]
        if "model" in config:
            model = config["model"]
except:
    pass

try:
    with open("system-prompt.txt", "r", encoding="UTF-8") as fp:
        system_prompt = fp.read()
except:
    pass

try:
    with open("system-prompt.md", "r", encoding="UTF-8") as fp:
        system_prompt = fp.read()
except:
    pass

client = AsyncOpenAI(api_key=api_key, base_url=base_url)
spinner_llm = Halo(text="Thinking", spinner="dots")
spinner_mcp = Halo(text="Running", spinner="dots")


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
        await mcp_client.notifications_initialized()
        tool_list = await mcp_client.tools_list()
        tools.extend(tool_list)

        for tool_item in tool_list:
            tool2client[tool_item["function"]["name"]] = mcp_client

    input_list = [{"role": "user", "content": system_prompt}]
    spinner_llm.start()
    response = await client.chat.completions.create(
        model=model,
        messages=input_list,
        tools=tools,
        tool_choice="auto",
    )
    spinner_llm.stop()

    print("=== チャットを開始します ===")

    async def turn(response):
        input_list.append(response.choices[0].message)
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls is not None and len(tool_calls) > 0:
            for tool_call in tool_calls:
                fn = tool_call.function
                args = json.loads(fn.arguments)
                target_client = tool2client[fn.name]

                spinner_mcp.start()
                response = await target_client.tools_call(fn.name, args)
                spinner_mcp.stop()
                print(response)
                if (
                    "method" in response
                    and response["method"] == "notifications/cancelled"
                ):
                    input_list.append(
                        {
                            "role": "tool",
                            "call_id": tool_call.id,
                            "content": f"failure a {fn.name} execute",
                        }
                    )
                    spinner_mcp.start()
                    response = await client.chat.completions.create(
                        model=model,
                        messages=input_list,
                        tools=tools,
                        tool_choice="auto",
                    )
                    spinner_mcp.stop()
                    await turn(response)
                else:
                    input_list.append(
                        {
                            "role": "tool",
                            "call_id": tool_call.id,
                            "content": response["result"]["content"],
                        }
                    )
                    spinner_llm.start()
                    response = await client.chat.completions.create(
                        model=model,
                        messages=input_list,
                        tools=tools,
                        tool_choice="auto",
                    )
                    spinner_llm.stop()
                    await turn(response)
        else:
            lines = response.choices[0].message.content.splitlines()
            lines = map(lambda line: f"> {line}", lines)
            lines = "\n".join(lines)
            print(lines)

    while True:
        next_prompt = sys.stdin.readline().rstrip()
        if next_prompt.startswith("/"):
            command = next_prompt[1:].rstrip()
            if command == "quit" or command == "exit":
                break
        input_list.append({"role": "user", "content": next_prompt})

        spinner_llm.start()
        response = await client.chat.completions.create(
            model=model,
            messages=input_list,
            tools=tools,
            tool_choice="auto",
        )
        spinner_llm.stop()
        await turn(response)

    for name, mcp_client in mcp_clients.items():
        await mcp_client.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
