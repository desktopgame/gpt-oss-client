import asyncio
import json
import sys
from openai import AsyncOpenAI
from typing import Dict, Any
from halo import Halo
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

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
    # init chat system
    mcp_clients: Dict[str, lib.McpClient] = {}
    for name, server in mcp_config["mcpServers"].items():
        mcp_client = lib.McpClient(server)
        mcp_clients[name] = mcp_client
        await mcp_client.start()

    def handle_llm_proc(method: str):
        if method == "begin":
            spinner_llm.start()
        elif method == "end":
            spinner_llm.stop()

    def handle_mcp_proc(method: str):
        if method == "begin":
            spinner_mcp.start()
        elif method == "end":
            spinner_mcp.stop()

    def handle_msg_proc(response):
        lines = response.choices[0].message.content.splitlines()
        lines = map(lambda line: f"> {line}", lines)
        lines = "\n".join(lines)
        print(lines)

        if context_length > 0:
            tokens = chat_manager.token_count()
            parcent = tokens / context_length
            print(f"# token usage: {tokens}/{context_length} {parcent:.2%}")

    def handle_use_proc(name):
        return input(f"$ want to use tool of `{name}`, are you ok? [y/n]: ")

    chat_manager = lib.ChatManager(
        client, model, system_prompt, context_length, mcp_clients, auto_approve
    )
    chat_manager.handle_llm_proc = handle_llm_proc
    chat_manager.handle_mcp_proc = handle_mcp_proc
    chat_manager.handle_msg_proc = handle_msg_proc
    chat_manager.handle_use_proc = handle_use_proc

    # init editor

    style = Style.from_dict(
        {
            "gutter": "bg:#222222 fg:#888888",
            "editor": "bg:#000000 fg:#e5e5e5",
            "minibuf": "bg:#1c1c1c fg:#d0d0d0",
            "modeline": "reverse bold",
        }
    )

    def line_prefix(line_no: int, wrap_count: int) -> list[tuple[str, str]]:
        num = f"{line_no + 1:>4} "
        return [("class:gutter", num)]

    def update_modeline(name="(unnamed)"):
        doc = editor.buffer.document
        row, col = doc.cursor_position_row + 1, doc.cursor_position_col + 1
        pct = int(row / max(1, doc.line_count) * 100)
        modeline.text = f"  {name}  {row}:{col}  {pct}%  UTF-8  "

    editor = TextArea(
        wrap_lines=True,
        scrollbar=True,
        style="class:editor",
        get_line_prefix=line_prefix,
    )
    minibuf = TextArea(height=1, prompt="M-x ", multiline=False, style="class:minibuf")
    modeline = TextArea(height=1, style="class:modeline", focusable=False)

    kb = KeyBindings()

    @kb.add("c-c")
    def _(e):
        e.app.exit()

    @kb.add("c-x", "c-e")
    def _(e):
        e.app.layout.focus(minibuf)

    @kb.add("c-x", "c-o")
    def _(e):
        e.app.layout.focus(editor)

    @kb.add("c-g")
    def _(e):
        minibuf.buffer.reset()
        e.app.invalidate()

    @kb.add("c-c")
    def _(e):
        e.app.exit()

    root = HSplit(
        [
            editor,
            modeline,
            minibuf,
        ]
    )
    app = Application(
        layout=Layout(root), key_bindings=kb, full_screen=True, style=style
    )
    # start

    spinner_load.start()
    await chat_manager.setup()
    spinner_load.stop()

    while True:
        next_prompt = sys.stdin.readline().rstrip()
        if next_prompt.startswith("/"):
            command = next_prompt[1:].rstrip()
            if command == "quit" or command == "exit":
                break
            if command == "edit":
                update_modeline()
                await app.run_async()
                continue
        await chat_manager.post(next_prompt)

    for name, mcp_client in mcp_clients.items():
        await mcp_client.shutdown()


def run():
    """Entry point for the command line script."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
