import asyncio
import os
import json
from typing import Any


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
        while not self.proc.stdout.at_eof():
            await self._readline(1)
        self.proc.kill()

    async def _readline(self, timeout: float):
        stream = self.proc.stdout
        return await asyncio.wait_for(stream.readline(), timeout=timeout)

    async def send(self, data: Any):
        data["jsonrpc"] = "2.0"
        if not data["method"].startswith("notifications"):
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
                        "version": "0.1.0",
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
