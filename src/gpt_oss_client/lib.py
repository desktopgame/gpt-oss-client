import asyncio
import os
import json
import tiktoken
from typing import Any
from pathlib import Path


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
        self.proc.stdin.close()
        try:
            while not self.proc.stdout.at_eof():
                await self._readline(1)
        except Exception:
            pass
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
                        "version": "0.2.0",
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
                        text = "".join(part.get("text","") for part in text if isinstance(part, dict))
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