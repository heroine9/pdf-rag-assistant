import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# 実際の使用では環境変数を推奨：os.environ["OPENAI_API_KEY"] = "xxx"
client = OpenAI(api_key="YOUR_API_KEY_HERE")


def load_pdf_text(path):
    """PDFファイルからテキストを読み込む"""
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def build_index(chunks):
    """テキストチャンクからTF-IDFインデックスを構築する"""
    vectorizer = TfidfVectorizer().fit(chunks)
    vectors = vectorizer.transform(chunks)
    return vectorizer, vectors


def retrieve(query, vectorizer, vectors, chunks, top_k=3):
    """クエリに最も類似した上位top_kのチャンクを取得する"""
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, vectors)[0]
    # 類似度が高い順にtop_k件を取得
    top_indices = sims.argsort()[::-1][:top_k]
    selected = [chunks[i] for i in top_indices]
    return "\n\n".join(selected)


def ask_llm(question, context):
    """大規模言語モデルに質問し、コンテキストに基づいて回答を生成する"""
    prompt = f"""
You are a helpful assistant. Use the following document context to answer the question.
If you are not sure, say you are not sure.

Context:
{context}

Question:
{question}
"""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def split_text(text, chunk_size=500):
    """文字数ベースでテキストを分割する簡易チャンク化"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks


def main():
    pdf_path = input("PDFファイルのパスを入力してください：")
    question = input("質問を入力してください：")

    print("PDFを読み込み中…")
    text = load_pdf_text(pdf_path)
    chunks = split_text(text)

    print("インデックスを構築中…")
    vectorizer, vectors = build_index(chunks)

    print("関連内容を検索中…")
    context = retrieve(question, vectorizer, vectors, chunks)

    print("LLMに問い合わせ中…")
    answer = ask_llm(question, context)

    print("\n=== 回答 ===")
    print(answer)


if __name__ == "__main__":
    main()
