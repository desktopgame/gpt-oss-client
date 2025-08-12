# gpt-oss-client
gpt-oss-client は、チャットモード／エディットモードの二つのモードを搭載した gpt-oss 専用のLLMクライアントです。  
function calling に対応しています。

## How to install
uv を使ってインストールできます。
````
cd gpt-oss-client
uv tool install .
````

## How to use
オプションはありません。コマンドを実行するとチャットモードで起動します。
````
gpt-oss-client
````

### Chat mode
このモードでは、ユーザーが入力した文字列がそのままLLMに送信されます。

ただし、スラッシュから始まる特定のワードを入力すると、コマンドとして解釈されます。  
現在サポートされているコマンドは以下の通りです。

/quit  
終了します。

/exit  
終了します。

/edit  
エディットモードに切り替えます。

/edit filename.txt  
任意のファイルを開いてエディットモードに切り替えます。

/clear  
会話履歴をクリアします。

/cd  
任意のディレクトリに移動します。

/pwd  
現在のディレクトリを表示します。

### Edit mode
このモードでは、emacs ライクなUIでテキストを編集できます。  
ショートカットも emacs を模倣しています。

画面中央のテキストエリアはユーザーの入力、またはLLMの返事を貼り付けるために使われます。  
その下の一行のラベルはモードラインと呼ばれ、実行中の操作やユーザーへの指示を表示します。  
その下の一行のテキストフィールドはミニバッファと呼ばれ、LLMへの指示を入力するために使われます。  
ミニバッファになにか入力してエンターすると、LLMに送信されその結果がテキストエリアに貼り付けられます。

また、エディットモードではいくつかショートカットキーを提供しています。

Ctrl+C  
終了

Ctrl+X Ctrl+E  
ミニバッファにフォーカス

Ctrl+X Ctrl+O  
テキストエリアにフォーカス

Ctrl+X Ctrl+W  
テキストエリアの選択されたテキストをLLMに送信する

Ctrl+X Ctrl+X  
テキストエリアの現在の一行をLLMに送信する

Ctrl+X Ctrl+A  
テキストエリアの全文をLLMに送信する。  
返事の全文をプレビューモードで表示する。（F2で戻れます。）

Ctrl+G  
ミニバッファをクリア

F2  
プレビューのON/OFFを切り替える（Markdown/Latexがレンダリングされます）

## Configuration
gpt-oss-client が起動されるディレクトリに設定ファイルを置くことでそちらを読んで動作します。  
基本的な設定はJSONに記述します。

gpt-oss-client.json
```gpt-oss-client.json
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

MCPの設定には変数を使うことができます。  
いまのところ、cwdという変数だけが存在します。
```mcp.json
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

起動時ディレクトリに設定がない場合、ユーザーディレクトリに読みに行きます。  
そこにもなければ、デフォルト設定で動作します。