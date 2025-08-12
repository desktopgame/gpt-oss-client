import asyncio
import json
import sys
import io, re, functools
import types
from rich.console import Console
from rich.markdown import Markdown
from sympy.parsing.latex import parse_latex
from sympy import pretty as sympy_pretty, sstr
from openai import AsyncOpenAI
from typing import Dict, Any, Callable
from halo import Halo
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.filters.utils import to_filter
from prompt_toolkit.formatted_text import split_lines, StyleAndTextTuples
from prompt_toolkit.application.current import get_app
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.screen import Screen, _CHAR_CACHE, WritePosition
from prompt_toolkit.layout import Layout, HSplit, Window, UIContent, WindowAlign
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl, FormattedTextControl
from prompt_toolkit.layout.containers import ConditionalContainer, ScrollOffsets
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.key_binding.bindings.scroll import (
    scroll_one_line_down, scroll_one_line_up,
    scroll_page_down, scroll_page_up,
)
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
view_at = Point(0, 0)


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
    global view_at

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

    view_kb = KeyBindings()
    view: FormattedTextControl = None
    view_window: Window = None

    @view_kb.add('up')
    def _(e):
        # view_window._scroll_up()
        # if view_window.vertical_scroll > 0:
        #     view_window.vertical_scroll -= 1
        # scroll_one_line_up(e)
        global view_at
        if is_view_mode[0]:
            y = view_at.y - 1
            if y < 0:
                y = 0
            view_at = Point(view_at.x, y)
    @view_kb.add('down')
    def _(e):
        # if view_window.vertical_scroll < view_window.height:
        # view_window.vertical_scroll += 1
        # view_window._scroll_down()
        # scroll_one_line_down(e)
        global view_at
        if is_view_mode[0]:
            y = view_at.y + 1
            h = len(list(split_lines(view._get_formatted_text_cached())))
            if y >= h:
                y = h - 1
            view_at = Point(view_at.x, y)

    view = FormattedTextControl(focusable=True, show_cursor=True, key_bindings=view_kb,
    get_cursor_position=lambda: view_at)
    view_window = Window(
    content=view,
    wrap_lines=False,                    
    right_margins=[ScrollbarMargin()],   
    height=D(weight=1),
    cursorline=True)

    def override_copy_body(self,
        ui_content: UIContent,
        new_screen: Screen,
        write_position: WritePosition,
        move_x: int,
        width: int,
        vertical_scroll: int = 0,
        horizontal_scroll: int = 0,
        wrap_lines: bool = False,
        highlight_lines: bool = False,
        vertical_scroll_2: int = 0,
        always_hide_cursor: bool = False,
        has_focus: bool = False,
        align: WindowAlign = WindowAlign.LEFT,
        get_line_prefix: Callable[[int, int], Any] | None = None,):
        """
        Copy the UIContent into the output screen.
        Return (visible_line_to_row_col, rowcol_to_yx) tuple.

        :param get_line_prefix: None or a callable that takes a line number
            (int) and a wrap_count (int) and returns formatted text.
        """
        xpos = write_position.xpos + move_x
        ypos = write_position.ypos
        line_count = ui_content.line_count
        new_buffer = new_screen.data_buffer
        empty_char = _CHAR_CACHE["", ""]

        # Map visible line number to (row, col) of input.
        # 'col' will always be zero if line wrapping is off.
        visible_line_to_row_col: dict[int, tuple[int, int]] = {}

        # Maps (row, col) from the input to (y, x) screen coordinates.
        rowcol_to_yx: dict[tuple[int, int], tuple[int, int]] = {}

        def copy_line(
            line: StyleAndTextTuples,
            lineno: int,
            x: int,
            y: int,
            is_input: bool = False,
        ) -> tuple[int, int]:
            """
            Copy over a single line to the output screen. This can wrap over
            multiple lines in the output. It will call the prefix (prompt)
            function before every line.
            """
            if is_input:
                current_rowcol_to_yx = rowcol_to_yx
            else:
                current_rowcol_to_yx = {}  # Throwaway dictionary.

            # Draw line prefix.
            if is_input and get_line_prefix:
                prompt = self.to_formatted_text(get_line_prefix(lineno, 0))
                x, y = copy_line(prompt, lineno, x, y, is_input=False)

            # Scroll horizontally.
            skipped = 0  # Characters skipped because of horizontal scrolling.
            if horizontal_scroll and is_input:
                h_scroll = horizontal_scroll
                line = self.explode_text_fragments(line)
                while h_scroll > 0 and line:
                    h_scroll -= self.get_cwidth(line[0][1])
                    skipped += 1
                    del line[:1]  # Remove first character.

                x -= h_scroll  # When scrolling over double width character,
                # this can end up being negative.

            # Align this line. (Note that this doesn't work well when we use
            # get_line_prefix and that function returns variable width prefixes.)
            if align == WindowAlign.CENTER:
                line_width = self.fragment_list_width(line)
                if line_width < width:
                    x += (width - line_width) // 2
            elif align == WindowAlign.RIGHT:
                line_width = self.fragment_list_width(line)
                if line_width < width:
                    x += width - line_width

            col = 0
            wrap_count = 0
            for style, text, *_ in line:
                new_buffer_row = new_buffer[y + ypos]

                # Remember raw VT escape sequences. (E.g. FinalTerm's
                # escape sequences.)
                if "[ZeroWidthEscape]" in style:
                    new_screen.zero_width_escapes[y + ypos][x + xpos] += text
                    continue

                for c in text:
                    char = _CHAR_CACHE[c, style]
                    char_width = char.width

                    # Wrap when the line width is exceeded.
                    if wrap_lines and x + char_width > width:
                        visible_line_to_row_col[y + 1] = (
                            lineno,
                            visible_line_to_row_col[y][1] + x,
                        )
                        y += 1
                        wrap_count += 1
                        x = 0

                        # Insert line prefix (continuation prompt).
                        if is_input and get_line_prefix:
                            prompt = self.to_formatted_text(
                                get_line_prefix(lineno, wrap_count)
                            )
                            x, y = copy_line(prompt, lineno, x, y, is_input=False)

                        new_buffer_row = new_buffer[y + ypos]

                        if y >= write_position.height:
                            return x, y  # Break out of all for loops.

                    # Set character in screen and shift 'x'.
                    if x >= 0 and y >= 0 and x < width:
                        new_buffer_row[x + xpos] = char

                        # When we print a multi width character, make sure
                        # to erase the neighbors positions in the screen.
                        # (The empty string if different from everything,
                        # so next redraw this cell will repaint anyway.)
                        if char_width > 1:
                            for i in range(1, char_width):
                                new_buffer_row[x + xpos + i] = empty_char

                        # If this is a zero width characters, then it's
                        # probably part of a decomposed unicode character.
                        # See: https://en.wikipedia.org/wiki/Unicode_equivalence
                        # Merge it in the previous cell.
                        elif char_width == 0:
                            # Handle all character widths. If the previous
                            # character is a multiwidth character, then
                            # merge it two positions back.
                            for pw in [2, 1]:  # Previous character width.
                                if (
                                    x - pw >= 0
                                    and new_buffer_row[x + xpos - pw].width == pw
                                ):
                                    prev_char = new_buffer_row[x + xpos - pw]
                                    char2 = _CHAR_CACHE[
                                        prev_char.char + c, prev_char.style
                                    ]
                                    new_buffer_row[x + xpos - pw] = char2

                        # Keep track of write position for each character.
                        current_rowcol_to_yx[lineno, col + skipped] = (
                            y + ypos,
                            x + xpos,
                        )

                    col += 1
                    x += char_width
            return x, y

        # Copy content.
        def copy() -> int:
            y = -vertical_scroll_2
            lineno = vertical_scroll

            while y < write_position.height and lineno < line_count:
                # Take the next line and copy it in the real screen.
                line = ui_content.get_line(lineno)

                visible_line_to_row_col[y] = (lineno, horizontal_scroll)

                # Copy margin and actual line.
                x = 0
                x, y = copy_line(line, lineno, x, y, is_input=True)

                lineno += 1
                y += 1
            return y

        copy()

        def cursor_pos_to_screen_pos(row: int, col: int) -> Point:
            "Translate row/col from UIContent to real Screen coordinates."
            try:
                y, x = rowcol_to_yx[row, col]
            except KeyError:
                # When cursor is outside the visible area, calculate appropriate position
                # based on whether it's above or below the visible area
                if row < vertical_scroll:
                    # Cursor is above visible area - return top position
                    return Point(x=xpos, y=ypos)
                elif row >= vertical_scroll + write_position.height:
                    # Cursor is below visible area - return bottom position
                    return Point(x=xpos, y=ypos + write_position.height - 1)
                else:
                    # Cursor is on a visible line but at a non-visible column
                    # Try to find the closest visible position on the same line
                    for test_col in range(col, -1, -1):
                        if (row, test_col) in rowcol_to_yx:
                            y, x = rowcol_to_yx[row, test_col]
                            return Point(x=x, y=y)
                    # If still not found, return leftmost position on the line
                    return Point(x=xpos, y=ypos + row - vertical_scroll)
            else:
                return Point(x=x, y=y)

        # Set cursor and menu positions.
        if ui_content.cursor_position:
            screen_cursor_position = cursor_pos_to_screen_pos(
                ui_content.cursor_position.y, ui_content.cursor_position.x
            )

            if has_focus:
                new_screen.set_cursor_position(self, screen_cursor_position)

                if always_hide_cursor:
                    new_screen.show_cursor = False
                else:
                    new_screen.show_cursor = ui_content.show_cursor

                self._highlight_digraph(new_screen)

            if highlight_lines:
                self._highlight_cursorlines(
                    new_screen,
                    screen_cursor_position,
                    xpos,
                    ypos,
                    width,
                    write_position.height,
                )

        # Draw input characters from the input processor queue.
        if has_focus and ui_content.cursor_position:
            self._show_key_processor_key_buffer(new_screen)

        # Set menu position.
        if ui_content.menu_position:
            new_screen.set_menu_position(
                self,
                cursor_pos_to_screen_pos(
                    ui_content.menu_position.y, ui_content.menu_position.x
                ),
            )

        # Update output screen height.
        new_screen.height = max(new_screen.height, ypos + write_position.height)

        return visible_line_to_row_col, rowcol_to_yx

    view_window._copy_body = types.MethodType(override_copy_body, view_window)

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
    def _(e):
        global view_at
        is_view_mode[0] = not is_view_mode[0]
        if is_view_mode[0]:
            view.text = ANSI(render_markdown_and_latex(editor.buffer.text))
            view_at = Point(0, 0)
            e.app.layout.focus(view)
        else:
            e.app.layout.focus(editor)
        e.app.invalidate()
        if is_view_mode[0]:
            e.app.layout.focus(view)
        else:
            e.app.layout.focus(editor)

    root = HSplit([
        edit_container,
        view_container,
    ])
    app = Application(
        layout=Layout(root), key_bindings=kb, full_screen=True, style=style,
                  mouse_support=True
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
