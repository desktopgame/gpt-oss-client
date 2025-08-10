import asyncio
import json
import sys
from openai import AsyncOpenAI
from typing import Dict, Any
from halo import Halo
from prompt_toolkit.application.current import get_app
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
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

edit_mode = False


async def main() -> None:
    global edit_mode

    # init common
    
    def line_prefix(line_no: int, wrap_count: int) -> list[tuple[str, str]]:
        num = f"{line_no + 1:>4} "
        return [("class:gutter", num)]

    style = Style.from_dict(
        {
            "gutter": "bg:#222222 fg:#888888",
            "editor": "bg:#000000 fg:#e5e5e5",
            "minibuf": "bg:#1c1c1c fg:#d0d0d0",
            "modeline": "reverse bold",
        }
    )

    editor = TextArea(
        wrap_lines=True,
        scrollbar=True,
        style="class:editor",
        get_line_prefix=line_prefix,
    )
    minibuf = TextArea(height=1, multiline=False, style="class:minibuf")
    modeline = TextArea(height=1, style="class:modeline", focusable=False)    
    modequeue = asyncio.Queue()

    def send_mode(mode: Any):
        asyncio.create_task(modequeue.put(mode))

    async def poll_mode():
        try:
            while True:
                mode = await modequeue.get()
                modeline.text = mode["text"]
                if "ticket" in mode:
                    await mode["ticket"]
                else:
                    await asyncio.sleep(mode["duration"])
        except Exception:
            pass

    asyncio.create_task(poll_mode())

    # init chat system

    mcp_clients: Dict[str, lib.McpClient] = {}
    for name, server in mcp_config["mcpServers"].items():
        mcp_client = lib.McpClient(server)
        mcp_clients[name] = mcp_client
        await mcp_client.start()

    chat_manager = lib.ChatManager(
        client, model, system_prompt, context_length, mcp_clients, auto_approve
    )

    chat_queue = asyncio.Queue()

    # init editor

    def chat_send(message: str):
        asyncio.create_task(chat_queue.put(message))

    async def chat_poll():
        try:
            while True:
                await chat_manager.post(await chat_queue.get())
        except Exception:
            pass

    asyncio.create_task(chat_poll())

    def accept_handler(buf: Buffer):
        prompt = buf.text
        if len(prompt.strip()) == 0:
            return

        minibuf.buffer.reset()
        get_app().invalidate()

        chat_send(prompt)

    minibuf.accept_handler = accept_handler

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

    # setup

    def handle_llm_proc(method: str):
        if edit_mode:
            if method == "begin":
                send_mode({"text": "Thinking...", "duration": 1})
            elif method == "end":
                pass
        else:
            if method == "begin":
                spinner_llm.start()
            elif method == "end":
                spinner_llm.stop()

    def handle_mcp_proc(method: str):
        if edit_mode:
            if method == "begin":
                send_mode({"text": "Running...", "duration": 1})
            elif method == "end":
                pass
        else:
            if method == "begin":
                spinner_mcp.start()
            elif method == "end":
                spinner_mcp.stop()

    def handle_msg_proc(response):
        lines = response.choices[0].message.content.splitlines()
        lines = map(lambda line: f"> {line}", lines)
        lines = "\n".join(lines)
        if edit_mode:
            editor.buffer.insert_text(f"{lines}\n")
            get_app().invalidate()

            if context_length > 0:
                tokens = chat_manager.token_count()
                parcent = tokens / context_length
                mode = {
                    "text": f"# token usage: {tokens}/{context_length} {parcent:.2%}",
                    "duration": 1
                }
                send_mode(mode)
        else:
            print(lines)

            if context_length > 0:
                tokens = chat_manager.token_count()
                parcent = tokens / context_length
                print(f"# token usage: {tokens}/{context_length} {parcent:.2%}")

    async def handle_use_proc(name):
        if edit_mode:
            fut = asyncio.get_event_loop().create_future()
            submitted = False

            def accept(_buf):
                nonlocal submitted
                if not submitted:
                    fut.set_result(minibuf.text)
                    submitted = True
                else:
                    send_mode("Please wait a moment...")

            tmp = minibuf.accept_handler
            minibuf.accept_handler = accept
            app.layout.focus(minibuf)

            try:
                ticket = asyncio.get_event_loop().create_future()
                mode = {
                    "text": f"want to use tool of `{name}`, are you ok? [y/n]",
                    "ticket": ticket
                }
                send_mode(mode)
                result = await fut
                ticket.set_result(None)
                return result
            finally:
                minibuf.accept_handler = tmp
        return input(f"$ want to use tool of `{name}`, are you ok? [y/n]: ")

    chat_manager.handle_llm_proc = handle_llm_proc
    chat_manager.handle_mcp_proc = handle_mcp_proc
    chat_manager.handle_msg_proc = handle_msg_proc
    chat_manager.handle_use_proc = handle_use_proc

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
                send_mode({"text": "Please feel free to ask us anything.", "duration": 1})
                edit_mode = True
                await app.run_async()
                edit_mode = False
                continue
        await chat_manager.post(next_prompt)

    modequeue.shutdown()

    for name, mcp_client in mcp_clients.items():
        await mcp_client.shutdown()


def run():
    """Entry point for the command line script."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
