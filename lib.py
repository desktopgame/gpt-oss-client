import asyncio
import os
import json
from typing import Any


class Connection:
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

        print(json.dumps(data))
        self.proc.stdin.write(f"{json.dumps(data)}\n".encode())
        await self.proc.stdin.drain()
        await asyncio.sleep(1)

    async def receive(self) -> Any:
        line = await self._readline(10)
        return json.loads(line)

    def is_exited(self):
        status = self.proc.returncode
        return status is not None
