import asyncio
import json
import sys
import io, re, functools
from rich.console import Console
from rich.markdown import Markdown
from sympy.parsing.latex import parse_latex
from sympy import pretty as sympy_pretty, sstr
from openai import AsyncOpenAI
from typing import Dict, Any
from halo import Halo
from prompt_toolkit.filters import Condition
from prompt_toolkit.filters.utils import to_filter
from prompt_toolkit.application.current import get_app
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl, FormattedTextControl
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.formatted_text import ANSI
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

console = Console(color_system="truecolor")

FENCE_RE    = re.compile(r"```.*?```", re.DOTALL)
INLINECODE  = re.compile(r"`[^`\n]*`")                   # インラインコード
BLOCK_RE    = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)    # $$ ... $$
INLINE_RE   = re.compile(r"(?<!\$)\$(.+?)\$(?!\$)", re.DOTALL)  # \$ は対象外

edit_mode = False
marker_id = 0
chat_lock = False
is_view_mode = [False]


@functools.lru_cache(maxsize=512)
def _latex_block_to_ascii(src: str) -> str:
    try:
        expr = parse_latex(src)
        # 罫線アートを壊さないためにラップ禁止 & 十分な幅を確保
        return sympy_pretty(expr, use_unicode=True, wrap_line=False, num_columns=9999)
    except Exception:
        return src  # 失敗は原文

@functools.lru_cache(maxsize=1024)
def _latex_inline_to_single(src: str) -> str:
    try:
        expr = parse_latex(src)
        # インラインは1行表現が崩れにくい（a^2, 1/x**2 など）
        return sstr(expr)
    except Exception:
        return src

def _stash_regions(text: str, patterns):
    holes = []
    def stash(m):
        holes.append(m.group(0))
        return f"⟪HOLE#{len(holes)-1}⟫"
    for pat in patterns:
        text = pat.sub(stash, text)
    return text, holes

def _unstash(text: str, holes):
    def putback(m): return holes[int(m.group(1))]
    return re.sub(r"⟪HOLE#(\d+)⟫", putback, text)

def render_markdown_and_latex(text: str) -> str:
    # 1) コード領域は退避（数式置換の対象外）
    stashed, holes = _stash_regions(text, [FENCE_RE, INLINECODE])

    # 2) LaTeX置換：ブロック→インラインの順
    def rep_block(m):
        body = m.group(1).strip()
        pretty = _latex_block_to_ascii(body)
        # コードフェンスで包んで Rich の整形・折返しを止める
        return "\n```text\n" + pretty + "\n```\n"

    def rep_inline(m):
        body = m.group(1).strip()
        return _latex_inline_to_single(body)

    stashed = BLOCK_RE.sub(rep_block, stashed)
    stashed = INLINE_RE.sub(rep_inline, stashed)

    # 3) 退避を戻す
    mixed = _unstash(stashed, holes)

    # 4) Markdown → ANSI
    buf = io.StringIO()
    console.file = buf
    console.print(Markdown(mixed), end="")  # code block内は等幅で崩れにくい
    console.file = None
    return buf.getvalue()


async def main() -> None:
    global edit_mode
    global marker_id
    global chat_lock

    # init common

    def line_prefix(line_no: int, wrap_count: int) -> list[tuple[str, str]]:
        num = f"{line_no + 1:>4} "
        return [("class:gutter", num)]

    style = Style.from_dict(
        {
            "gutter": "bg:#222222 fg:#888888",
            "editor": "bg:#000000 fg:#e5e5e5",
            "minibuf": "bg:#1c1c1c fg:#d0d0d0",
            "minibuf.disabled": "bg:#262626 fg:#777777 italic",
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

    view = FormattedTextControl()
    view_window = Window(view)

    workspace = HSplit(
        [
            editor,
            modeline,
            minibuf,
        ]
    )

    edit_container = ConditionalContainer(
        content=workspace,
        filter=Condition(lambda: not is_view_mode[0])
    )

    view_container = ConditionalContainer(
        content=view_window,
        filter=Condition(lambda: is_view_mode[0])
    )

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

    # init editor

    async def chat_submit(message: str):
        global chat_lock
        if chat_lock:
            return
        chat_lock = True
        global marker_id
        editor.buffer.insert_line_below()
        editor.buffer.insert_text(f"<chat_response_is_here:{marker_id}>")
        editor.buffer.insert_line_below()

        minibuf.buffer.read_only = to_filter(True)
        minibuf.window.style = "class:minibuf.disabled"
        await chat_manager.post(message)
        marker_id += 1
        minibuf.buffer.read_only = to_filter(False)
        minibuf.window.style = "class:minibuf"
        chat_lock = False

    async def chat_submit_selected():
        buf = editor.buffer
        sel = buf.selection_state
        if not sel:
            return
        a = buf.cursor_position
        b = sel.original_cursor_position
        start, end = (a, b) if a < b else (b, a)
        text = editor.buffer.text[start:end]
        next_line_pos = end
        while next_line_pos < len(editor.buffer.text):
            if editor.buffer.text[next_line_pos] == "\n":
                break
            next_line_pos += 1
        editor.buffer.cursor_position = next_line_pos
        await chat_submit(text)

    def accept_handler(buf: Buffer):
        message = buf.text
        if len(message.strip()) == 0:
            return

        minibuf.buffer.reset()
        get_app().invalidate()

        asyncio.create_task(chat_submit(message))

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

    @kb.add("c-x", "c-w")
    def _(e):
        asyncio.create_task(chat_submit_selected())  # noqa

    @kb.add("c-x", "c-x")
    def _(e):
        asyncio.create_task(chat_submit(editor.buffer.document.current_line))  # noqa

    @kb.add("c-g")
    def _(e):
        minibuf.buffer.reset()
        e.app.invalidate()

    @kb.add("f2")
    def _(event):
        is_view_mode[0] = not is_view_mode[0]
        if is_view_mode[0]:
            view.text = ANSI(render_markdown_and_latex(editor.buffer.text))
        event.app.invalidate()

    root = HSplit([
        edit_container,
        view_container,
    ])
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
            ph = f"<chat_response_is_here:{marker_id}>"
            buf = editor.buffer
            text = buf.text
            idx = text.find(ph)
            if idx >= 0:
                new_text = text[:idx] + f"{lines}\n" + text[idx + len(ph) :]
                buf.text = new_text
                buf.cursor_position = idx + len(lines) + 1
                get_app().invalidate()

            if context_length > 0:
                tokens = chat_manager.token_count()
                parcent = tokens / context_length
                mode = {
                    "text": f"# token usage: {tokens}/{context_length} {parcent:.2%}",
                    "duration": 1,
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

            tmp_accept_handler = minibuf.accept_handler
            tmp_read_only = minibuf.buffer.read_only
            tmp_style = minibuf.window.style
            minibuf.accept_handler = accept
            minibuf.buffer.read_only = to_filter(False)
            minibuf.window.style = "class:minibuf"
            app.layout.focus(minibuf)

            try:
                ticket = asyncio.get_event_loop().create_future()
                mode = {
                    "text": f"want to use tool of `{name}`, are you ok? [y/n]",
                    "ticket": ticket,
                }
                send_mode(mode)
                result = await fut
                ticket.set_result(None)
                return result
            finally:
                minibuf.accept_handler = tmp_accept_handler
                minibuf.buffer.read_only = tmp_read_only
                minibuf.window.style = tmp_style
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
            command_with_args = next_prompt[1:].rstrip().split(" ")
            command = command_with_args[0]
            args = command_with_args[1:]
            if command == "quit" or command == "exit":
                break
            elif command == "edit":
                file = ""
                if len(args) > 0:
                    file = args[0]
                if file != "":
                    try:
                        with open(file, "r", encoding="UTF-8") as fp:
                            editor.buffer.text = fp.read()
                            editor.buffer.cursor_position = 0
                    except Exception:
                        editor.buffer.text = ""
                        editor.buffer.cursor_position = 0
                else:
                    editor.buffer.text = ""
                    editor.buffer.cursor_position = 0

                send_mode(
                    {"text": "Please feel free to ask us anything.", "duration": 1}
                )
                edit_mode = True
                await app.run_async()
                edit_mode = False

                if file != "":
                    with open(file, "w", encoding="UTF-8") as fp:
                        fp.write(editor.buffer.text)
                continue
            elif command == "clear":
                chat_manager.clear()
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
