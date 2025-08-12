# gpt-oss-client

[日本語](README-ja_jp.md)

gpt-oss-client is a dedicated LLM client for gpt-oss that features two modes: Chat mode and Edit mode.  
It also supports function calling.

## How to install
You can install it using **uv**:
```
cd gpt-oss-client
uv tool install .
```

## How to use
There are no command‑line options. Running the executable starts the client in Chat mode:
```
gpt-oss-client
```

### Chat mode
In this mode, whatever text the user enters is sent directly to the LLM.
However, if a line starts with a slash (`/`), it is interpreted as a command. The currently supported commands are:

- `/quit` – Exit the client.
- `/exit` – Exit the client.
- `/edit` – Switch to Edit mode.
- `/edit filename.txt` – Open an arbitrary file and switch to Edit mode.
- `/clear` – Clear the conversation history.
- `/cd` – Change to an arbitrary directory.
- `/pwd` – Show the current working directory.

### Edit mode
In this mode you can edit text using an Emacs‑like UI, with shortcuts that mimic Emacs.  
The central text area is used for user input or for pasting LLM responses.  
Below it there is a one‑line label called the *mode line*, which displays the current operation and instructions for the user.  
Under that is another single‑line field known as the *mini‑buffer*; you type instructions for the LLM here. When you press **Enter**, the content of the mini‑buffer is sent to the LLM and its response is inserted into the text area.

The following shortcut keys are available in Edit mode:

- `Ctrl+C` – Exit back to Chat mode.
- `Ctrl+X Ctrl+E` – Focus the mini‑buffer.
- `Ctrl+X Ctrl+O` – Focus the text area.
- `Ctrl+X Ctrl+W` – Send the selected text in the text area to the LLM.
- `Ctrl+X Ctrl+X` – Send the current line (where the cursor is) to the LLM.
- `Ctrl+X Ctrl+A` – Send the entire contents of the text area to the LLM. The full response is shown in preview mode (press **F2** to return).
- `Ctrl+G` – Clear the mini‑buffer.
- `F2` – Toggle preview ON/OFF (Markdown/LaTeX rendering).

## Configuration
Place a configuration file in the directory where gpt‑oss‑client is started; the client will read it on launch.  
The basic settings are written in JSON.

**gpt-oss-client.json**
```json
{
    "api_key": "1234",
    "base_url": "http://localhost:1234/v1",
    "model": "openai/gpt-oss-20b",
    "allow_tools": [
        "read_file",
        "read_text_file",
        "list_directory",
        "list_directory_with_sizes",
        "directory_tree",
        "search_files",
        "get_file_info",
        "list_allowed_directories"
    ],
    "context_length": 15000
}
```

The system prompt can be written as plain text or Markdown.

**system-prompt.txt**
```text
You are a friendly assistant that communicates in Japanese.
```

Finally, the MCP (Model Context Protocol) configuration is stored in **mcp.json**.

**mcp.json**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\user\\example"
      ]
    }
  }
}
```

Variables can be used in the MCP configuration. Currently, only the `cwd` variable is available.
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "{{cwd}}"
      ]
    }
  }
}
```

If there is no configuration in the startup directory, the client looks for one in the user's home directory. If none is found, it runs with default settings.
