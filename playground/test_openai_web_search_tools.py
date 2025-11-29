import os
import sys
from pathlib import Path
from openai import OpenAI

# Ensure the repository root is on sys.path so top-level imports like `settings`
# work even when this script is executed from the `playground/` directory.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from settings import OPENAI_API_KEY

# APIキーを環境変数に設定
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# クライアント初期化
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Web検索を使って質問に回答
response = client.responses.create(
    model="gpt-4o",  # モデル指定（例）
    tools=[{"type": "web_search_preview"}],  # Web検索ツールを有効化
    tool_choice={"type": "web_search_preview"},  # 強制的にWeb検索を使う
    input="最近のAI研究のトレンドについて教えて",
    store=True  # 結果を保存
)

# 結果の表示
if response.status == "completed":
    print("回答:", response.output_text)
else:
    print("検索に失敗しました")
