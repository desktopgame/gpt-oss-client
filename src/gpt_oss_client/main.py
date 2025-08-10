import asyncio
import json
import sys
from openai import AsyncOpenAI
from typing import Dict, Any
from halo import Halo

try:
    from . import lib
except ImportError:
    import lib

config = lib.Config()
config.update()

api_key = config.api_key
base_url = config.base_url
model = config.model
auto_approve = config.auto_approve
context_length = config.context_length
system_prompt = config.system_prompt
mcp_config = config.mcp

client = AsyncOpenAI(api_key=api_key, base_url=base_url)
counter = lib.TokenCounter("gpt-oss-")
spinner_load = Halo(text="Loading", spinner="dots")
spinner_llm = Halo(text="Thinking", spinner="dots")
spinner_mcp = Halo(text="Running", spinner="dots")


async def main() -> None:
    mcp_clients: Dict[str, lib.McpClient] = {}
    for name, server in mcp_config["mcpServers"].items():
        mcp_client = lib.McpClient(server)
        mcp_clients[name] = mcp_client
        await mcp_client.start()

    spinner_load.start()
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
    response = await client.chat.completions.create(
        model=model,
        messages=input_list,
        tools=tools,
        tool_choice="auto",
    )
    input_list.append(response.choices[0].message.to_dict())
    spinner_load.stop()

    async def tool_use(response, tool_call):
        fn = tool_call.function
        if "method" in response and response["method"] == "notifications/cancelled":
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
            return response
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
            return response

    async def turn(response):
        input_list.append(response.choices[0].message.to_dict())
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls is not None and len(tool_calls) > 0:
            for tool_call in tool_calls:
                fn = tool_call.function
                args = json.loads(fn.arguments)
                if fn.name not in tool2client:
                    input_list.append(
                        {
                            "role": "system",
                            "call_id": tool_call.id,
                            "content": f"{fn.name} is not found",
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
                    return
                target_client = tool2client[fn.name]

                confirm_result = "y"
                if not auto_approve:
                    confirm_result = input(
                        f"$ want to use tool of `{fn.name}`, are you ok? [y/n]: "
                    ).lower()  # noqa
                if confirm_result == "y" or confirm_result == "yes":
                    spinner_mcp.start()
                    response = await target_client.tools_call(fn.name, args)
                    spinner_mcp.stop()
                    await turn(await tool_use(response, tool_call))
                else:
                    input_list.append(
                        {
                            "role": "user",
                            "call_id": tool_call.id,
                            "content": f"user rejected a {fn.name} execute",
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

            if context_length > 0:
                tokens = counter.count(input_list)
                parcent = tokens / context_length
                print(f"# token usage: {tokens}/{context_length} {parcent:.2%}")

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


def run():
    """Entry point for the command line script."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
