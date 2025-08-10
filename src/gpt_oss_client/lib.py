import asyncio
import os
import json
import tiktoken
import pystache
from typing import Any
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
            await self.proc.communicate()
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
                        "version": "0.3.0",
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
