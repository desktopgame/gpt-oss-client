# gpt-oss-client
gpt-oss と対話的なインターフェイスでチャットできるCLIツールです。  
function calling に対応しています。

## How to install
uv を使ってインストールできます。
````
cd gpt-oss-client
uv tool install .
````

## How to use
オプションはありません。コマンドを実行すると対話的なインターフェイスが開始します。
````
gpt-oss-client
````

終了するには特殊な構文でコマンドを実行します。
```
/quit
```

## Configuration
gpt-oss-client が起動されるディレクトリに設定ファイルを置くことでそちらを読んで動作します。  
基本的な設定はJSONに記述します。

gpt-oss-client.json
```gpt-oss-client.json
{
    "api_key": "1234",
    "base_url": "http://localhost:1234/v1",
    "model": "openai/gpt-oss-20b",
    "auto_approve": false,
    "context_length": 15000
}
```

システムプロンプトはテキストまたはマークダウンで記述します。

system-prompt.txt
```system-prompt.txt
あなたは日本語で話す親切なアシスタントです。
```

最後に、MCPは `mcp.json` に記述します。

mcp.json
```mcp.json
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