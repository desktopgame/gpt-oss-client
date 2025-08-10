import asyncio
import os
import json
import tiktoken
import pystache
from openai import AsyncOpenAI
from typing import Any, List, Dict
from pathlib import Path


class Config:
    def __init__(self):
        self.api_key = "lmstudio"
        self.base_url = "http://localhost:1234/v1"
        self.model = "openai/gpt-oss-120b"
        self.auto_approve = False
        self.context_length = 0
        self.system_prompt = "あなたは日本語で話す親切なアシスタントです。"
        self.mcp = {}

    def update(self):
        self.__load("gpt-oss-client.json", self.__hook_basic)
        self.__load("system-prompt.txt", self.__hook_system_prompt)
        self.__load("system-prompt.md", self.__hook_system_prompt)
        self.__load("mcp.json", self.__hook_mcp)

    def __load(self, name: str, on_load):
        try:
            load_at = Path.home().joinpath(name)
            with open(load_at, "r", encoding="UTF-8") as fp:
                on_load(fp)
        except:
            pass
        try:
            with open(name, "r", encoding="UTF-8") as fp:
                on_load(fp)
        except:
            pass

    def __hook_basic(self, fp):
        config = json.load(fp)

        if "api_key" in config:
            self.api_key = config["api_key"]
        if "base_url" in config:
            self.base_url = config["base_url"]
        if "model" in config:
            self.model = config["model"]
        if "auto_approve" in config:
            self.auto_approve = config["auto_approve"]
        if "context_length" in config:
            self.context_length = config["context_length"]

    def __hook_system_prompt(self, fp):
        self.system_prompt = fp.read()

    def __hook_mcp(self, fp):
        self.mcp = json.load(fp)

        def expand_macro(s: str) -> str:
            return pystache.render(s, {"cwd": os.getcwd()})

        if "mcpServers" in self.mcp:
            servers = self.mcp["mcpServers"]
            for name, server in servers.items():
                if "args" in server:
                    server["args"] = list(map(expand_macro, server["args"]))  # noqa


class StdioPipe:
    def __init__(self, config):
        self.config = config
        self.proc = None
        self.id = 0

    async def setup(self):
        env = os.environ
        if "env" in self.config:
            env = {**env, **self.config["env"]}

        cwd = None
        if "cwd" in self.config:
            cwd = self.config["cwd"]

        command = self.config["command"]
        args = self.config["args"]
        self.proc = await asyncio.subprocess.create_subprocess_shell(
            " ".join([command, *args]),
            shell=True,
            env=env,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def shutdown(self):
        try:
            await asyncio.wait_for(self.proc.communicate(), timeout=60) # noqa
        except Exception:
            self.proc.kill()

    async def _readline(self, timeout: float):
        stream = self.proc.stdout
        return await asyncio.wait_for(stream.readline(), timeout=timeout)

    async def send(self, data: Any):
        data["jsonrpc"] = "2.0"
        if "method" in data and not data["method"].startswith("notifications"):
            data["id"] = self.id
            self.id += 1

        self.proc.stdin.write(f"{json.dumps(data)}\n".encode())
        await self.proc.stdin.drain()
        await asyncio.sleep(1)

    async def receive(self) -> Any:
        line = await self._readline(10)
        return json.loads(line)

    def is_exited(self):
        status = self.proc.returncode
        return status is not None


class McpClient:
    def __init__(self, config):
        self.pipe = StdioPipe(config)

    def __mcp_tools_to_openai(self, mcp_tools):
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get(
                        "inputSchema", {"type": "object", "properties": {}}
                    ),
                },
            }
            for tool in mcp_tools
        ]

    async def start(self):
        await self.pipe.setup()

    async def initialize(self):
        await self.pipe.send(
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
                        "name": "gpt-oss-client",
                        "title": "gpt-oss-client",
                        "version": "0.4.0",
                    },
                },
            }
        )
        return await self.pipe.receive()

    async def notifications_initialized(self):
        await self.pipe.send({"method": "notifications/initialized", "params": {}})
        await asyncio.sleep(1)

    async def tools_list(self):
        await self.pipe.send({"method": "tools/list", "params": {}})
        response = await self.pipe.receive()
        if "method" in response and response["method"] == "roots/list":
            path_uri = Path(os.path.abspath(os.sep)).as_uri()
            await self.pipe.send({"roots": [{"name": "C", "uri": path_uri}]})
            response = await self.pipe.receive()
        return self.__mcp_tools_to_openai(response["result"]["tools"])

    async def tools_call(self, name: str, args: Any):
        params = {"name": name, "arguments": args}
        data = {"params": params, "method": "tools/call"}
        await self.pipe.send(data)
        return await self.pipe.receive()

    async def shutdown(self):
        await self.pipe.shutdown()

    def is_exited(self):
        return self.pipe.is_exited()


class TokenCounter:
    def __init__(self, model: str):
        self.enc = tiktoken.encoding_for_model(model)

    def count(self, messages):
        total = 0
        for m in messages:
            if "content" in m:
                text = m["content"]
                if text is not None:
                    if isinstance(text, list):
                        text = "".join(
                            part.get("text", "")
                            for part in text
                            if isinstance(part, dict)
                        )
                    total += len(self.enc.encode(text))
            if "tool_calls" in m:
                tool_calls = m["tool_calls"]
                if tool_calls is not None:
                    for tool_call in tool_calls:
                        total += len(self.enc.encode(tool_call.get("id", "")))
                        total += len(self.enc.encode(tool_call.get("type", "")))
                        function = tool_call.get("function", {})
                        total += len(self.enc.encode(function.get("name", "")))
                        total += len(self.enc.encode(function.get("arguments", "")))
        return total


class ChatManager:
    def __init__(self,
                 open_ai: AsyncOpenAI,
                 model: str,
                 system_prompt: str,
                 context_length: int,
                 mcp_clients: Dict[str, McpClient],
                 auto_approve: bool):
        self.open_ai = open_ai
        self.model = model
        self.system_prompt = system_prompt
        self.context_length = context_length
        self.mcp_clients = mcp_clients
        self.auto_approve = auto_approve
        self.input_list = []
        self.tools = []
        self.tool2client = {}
        self.counter = TokenCounter("gpt-oss-")

        def nop(evt):
            pass
        self.handle_llm_proc = nop
        self.handle_mcp_proc = nop
        self.handle_msg_proc = nop
        self.handle_use_proc = nop

    async def setup(self):
        self.tools = []
        self.tool2client = {}
        for _, mcp_client in self.mcp_clients.items():
            await mcp_client.initialize()
            await mcp_client.notifications_initialized()
            tool_list = await mcp_client.tools_list()
            self.tools.extend(tool_list)

            for tool_item in tool_list:
                self.tool2client[tool_item["function"]["name"]] = mcp_client

        self.input_list = [{"role": "user", "content": self.system_prompt}]
        response = await self.open_ai.chat.completions.create(
            model=self.model,
            messages=self.input_list,
            tools=self.tools,
            tool_choice="auto",
        )
        self.input_list.append(response.choices[0].message.to_dict())

    async def __tool_use(self, response, tool_call):
        fn = tool_call.function
        if "method" in response and response["method"] == "notifications/cancelled":
            self.input_list.append(
                {
                    "role": "tool",
                    "call_id": tool_call.id,
                    "content": f"failure a {fn.name} execute",
                }
            )
            self.handle_llm_proc("begin")
            response = await self.open_ai.chat.completions.create(
                model=self.model,
                messages=self.input_list,
                tools=self.tools,
                tool_choice="auto",
            )
            self.handle_llm_proc("end")
            return response
        else:
            self.input_list.append(
                {
                    "role": "tool",
                    "call_id": tool_call.id,
                    "content": response["result"]["content"],
                }
            )
            self.handle_llm_proc("begin")
            response = await self.open_ai.chat.completions.create(
                model=self.model,
                messages=self.input_list,
                tools=self.tools,
                tool_choice="auto",
            )
            self.handle_llm_proc("end")
            return response

    async def __turn(self, response):
        self.input_list.append(response.choices[0].message.to_dict())
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls is not None and len(tool_calls) > 0:
            for tool_call in tool_calls:
                fn = tool_call.function
                args = json.loads(fn.arguments)
                if fn.name not in self.tool2client:
                    self.input_list.append(
                        {
                            "role": "system",
                            "call_id": tool_call.id,
                            "content": f"{fn.name} is not found",
                        }
                    )
                    self.handle_llm_proc("begin")
                    response = await self.open_ai.chat.completions.create(
                        model=self.model,
                        messages=self.input_list,
                        tools=self.tools,
                        tool_choice="auto",
                    )
                    self.handle_llm_proc("end")
                    await self.__turn(response)
                    return
                target_client = self.tool2client[fn.name]

                confirm_result = "y"
                if not self.auto_approve:
                    confirm_result = self.handle_use_procc(fn.name).lower()  # noqa
                if confirm_result == "y" or confirm_result == "yes":
                    self.handle_mcp_proc("begin")
                    response = await target_client.tools_call(fn.name, args)
                    self.handle_mcp_proc("end")
                    await self.__turn(await self.__tool_use(response, tool_call))
                else:
                    self.input_list.append(
                        {
                            "role": "user",
                            "call_id": tool_call.id,
                            "content": f"user rejected a {fn.name} execute",
                        }
                    )
                    self.handle_llm_proc("begin")
                    response = await self.open_ai.chat.completions.create(
                        model=self.model,
                        messages=self.input_list,
                        tools=self.tools,
                        tool_choice="auto",
                    )
                    self.handle_llm_proc("end")
                    await self.__turn(response)
        else:
            self.handle_msg_proc(response)

    async def post(self, message: str):
        self.input_list.append({"role": "user", "content": message})

        self.handle_llm_proc("begin")
        response = await self.open_ai.chat.completions.create(
            model=self.model,
            messages=self.input_list,
            tools=self.tools,
            tool_choice="auto",
        )
        self.handle_llm_proc("end")
        await self.__turn(response)

    def token_count(self):
        return self.counter.count(self.input_list)
