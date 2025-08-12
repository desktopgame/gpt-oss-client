import asyncio
import os
import json
import tiktoken
import pystache
import io, re, functools
from sympy.parsing.latex import parse_latex
from sympy import pretty as sympy_pretty, sstr
from openai import AsyncOpenAI
from typing import Any, List, Dict, Callable, Optional
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import FilterOrBool
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.screen import Screen, _CHAR_CACHE, WritePosition
from prompt_toolkit.layout import Window, UIContent, WindowAlign, ColorColumn
from prompt_toolkit.layout.controls import UIControl
from prompt_toolkit.layout.containers import ScrollOffsets
from prompt_toolkit.layout.dimension import AnyDimension


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
            await asyncio.wait_for(self.proc.communicate(), timeout=60)  # noqa
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
                        "version": "0.5.0",
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
    def __init__(
        self,
        open_ai: AsyncOpenAI,
        model: str,
        system_prompt: str,
        context_length: int,
        mcp_clients: Dict[str, McpClient],
        auto_approve: bool,
    ):
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

        async def nop_async(evt):
            pass

        self.handle_llm_proc = nop
        self.handle_mcp_proc = nop
        self.handle_msg_proc = nop
        self.handle_use_proc = nop_async

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
                    confirm_result = (await self.handle_use_proc(fn.name)).lower()  # noqa
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

    def clear(self):
        if len(self.input_list) >= 2:
            self.input_list = self.input_list[:2]

    def token_count(self):
        return self.counter.count(self.input_list)


class PrettyPrinter:
    def __init__(self):
        self.console = Console(color_system="truecolor")

        self.FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
        self.INLINECODE = re.compile(r"`[^`\n]*`")
        self.BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
        self.INLINE_RE = re.compile(r"(?<!\$)\$(.+?)\$(?!\$)", re.DOTALL)

    @functools.lru_cache(maxsize=512)
    def _latex_block_to_ascii(self, src: str) -> str:
        try:
            expr = parse_latex(src)
            return sympy_pretty(
                expr, use_unicode=True, wrap_line=False, num_columns=9999
            )
        except Exception:
            return src

    @functools.lru_cache(maxsize=1024)
    def _latex_inline_to_single(self, src: str) -> str:
        try:
            expr = parse_latex(src)
            return sstr(expr)
        except Exception:
            return src

    def _stash_regions(self, text: str, patterns):
        holes = []

        def stash(m):
            holes.append(m.group(0))
            return f"⟪HOLE#{len(holes) - 1}⟫"

        for pat in patterns:
            text = pat.sub(stash, text)
        return text, holes

    def _unstash(self, text: str, holes):
        def putback(m):
            return holes[int(m.group(1))]

        return re.sub(r"⟪HOLE#(\d+)⟫", putback, text)

    def render_markdown_and_latex(self, text: str) -> str:
        stashed, holes = self._stash_regions(text, [self.FENCE_RE, self.INLINECODE])

        def rep_block(m):
            body = m.group(1).strip()
            pretty = self._latex_block_to_ascii(body)
            return "\n```text\n" + pretty + "\n```\n"

        def rep_inline(m):
            body = m.group(1).strip()
            return self._latex_inline_to_single(body)

        stashed = self.BLOCK_RE.sub(rep_block, stashed)
        stashed = self.INLINE_RE.sub(rep_inline, stashed)

        mixed = self._unstash(stashed, holes)

        buf = io.StringIO()
        self.console.file = buf
        self.console.print(Markdown(mixed), end="")
        self.console.file = None
        return buf.getvalue()


class CommandCompleter(Completer):
    def __init__(self):
        super().__init__()
        self.commands = [
            'quit',
            'exit',
            'edit',
            'clear'
        ]

    def get_completions(self, document, complete_event):
        if len(document.text) == 1 and document.text[0] == "/":
            for cmd in self.commands:
                yield Completion(cmd, start_position=0)
        elif document.text.startswith("/"):
            if " " not in document.text:
                progress = document.text[1:]
                matches = list(filter(lambda cmd: cmd == progress, self.commands))
                if len(matches) == 0:
                    for cmd in filter(lambda cmd: cmd.startswith(progress), self.commands):
                        yield Completion(cmd[len(progress):], display=cmd)
            else:
                if document.text.startswith("/edit"):
                    args = document.text[len("/edit"):]
                    parent: Optional[Path] = None
                    p: Optional[Path] = None
                    progress = ""
                    if len(args.strip()) == 0:
                        p = Path(os.getcwd())
                    else:
                        p = Path(args.strip())

                        if not p.exists():
                            progress = p.name
                            parent = p.parent
                            p = None
                    if p is not None:
                        for child in p.iterdir():
                            yield Completion(child.name, start_position=0)
                    else:
                        if parent is not None and parent.exists():
                            for child in parent.iterdir():
                                if child.name.startswith(progress):
                                    yield Completion(child.name[len(progress):], display=child.name)


class ScrollableWindow(Window):
    def __init__(
        self,
        content: UIControl | None = None,
        width: AnyDimension = None,
        height: AnyDimension = None,
        z_index: int | None = None,
        dont_extend_width: FilterOrBool = False,
        dont_extend_height: FilterOrBool = False,
        ignore_content_width: FilterOrBool = False,
        ignore_content_height: FilterOrBool = False,
        left_margins: Any | None = None,
        right_margins: Any | None = None,
        scroll_offsets: ScrollOffsets | None = None,
        allow_scroll_beyond_bottom: FilterOrBool = False,
        wrap_lines: FilterOrBool = False,
        get_vertical_scroll: Callable[[Window], int] | None = None,
        get_horizontal_scroll: Callable[[Window], int] | None = None,
        always_hide_cursor: FilterOrBool = False,
        cursorline: FilterOrBool = False,
        cursorcolumn: FilterOrBool = False,
        colorcolumns: (
            None | list[ColorColumn] | Callable[[], list[ColorColumn]]
        ) = None,
        align: WindowAlign | Callable[[], WindowAlign] = WindowAlign.LEFT,
        style: str | Callable[[], str] = "",
        char: None | str | Callable[[], str] = None,
        get_line_prefix: Any | None = None,
    ) -> None:
        super().__init__(
            content,
            width,
            height,
            z_index,
            dont_extend_width,
            dont_extend_height,
            ignore_content_width,
            ignore_content_height,
            left_margins,
            right_margins,
            scroll_offsets,
            allow_scroll_beyond_bottom,
            wrap_lines,
            get_vertical_scroll,
            get_horizontal_scroll,
            always_hide_cursor,
            cursorline,
            cursorcolumn,
            colorcolumns,
            align,
            style,
            char,
            get_line_prefix,
        )

    # original code is here: https://github.com/prompt-toolkit/python-prompt-toolkit
    # license: BSD

    def _copy_body(
        self,
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
        get_line_prefix: Callable[[int, int], Any] | None = None,
    ):
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
                #
                # MODIFIED: fix issue scrolls FormattedTextControl.
                #

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
