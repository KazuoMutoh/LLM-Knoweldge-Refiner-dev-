# Pythonの軽量イメージを利用
FROM python:3.11-slim

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# requirements.txt をコピーして依存関係をインストール
COPY requirements.txt .
RUN pip install -r requirements.txt

# ソースコードをコピー
COPY . .