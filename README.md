# PDF RAG Assistant

## 概要

PDFファイルを読み込み、質問に対して関連する部分を検索し、  
大規模言語モデル（OpenAI API）を使って回答を生成する簡単なRAG（Retrieval-Augmented Generation）のデモです。

## 技術スタック

- Python
- PyPDF2（PDFテキスト抽出）
- scikit-learn（TF-IDFによる簡易ベクトル検索）
- OpenAI API（LLMによる回答生成）

## セットアップ

```bash
pip install -r requirements.txt
