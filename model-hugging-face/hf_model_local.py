import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 感情分析用のモデルを指定
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# トークナイザーとモデルをローカルにダウンロード
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, local_files_only=False
)

# ダウンロードしたモデルとトークナイザーを保存
tokenizer.save_pretrained("./local_sentiment_model")
model.save_pretrained("./local_sentiment_model")

# ローカルから保存したモデルとトークナイザーをロード
local_tokenizer = AutoTokenizer.from_pretrained("./local_sentiment_model")
local_model = AutoModelForSequenceClassification.from_pretrained(
    "./local_sentiment_model"
)


def analyze_sentiment(text):
    # テキストをトークン化し、モデル入力形式に変換
    inputs = local_tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # モデルで推論を実行
    with torch.no_grad():
        outputs = local_model(**inputs)

    # 出力をソフトマックス関数で確率に変換
    probabilities = F.softmax(outputs.logits, dim=-1)

    # 最も確率の高いクラスを選択
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    # 感情ラベルを定義（モデルに応じて変更が必要な場合があります）
    sentiment_labels = ["Negative", "Positive"]

    # 予測された感情と確率を返す
    predicted_sentiment = sentiment_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()

    return predicted_sentiment, confidence


# テキストの処理例
text = "I love this movie! It's amazing!"
sentiment, confidence = analyze_sentiment(text)

print(f"Input text: {text}")
print(f"Predicted sentiment: {sentiment}")
print(f"Confidence: {confidence:.2f}")

# 別の例
text2 = "This book is terrible. I wouldn't recommend it to anyone."
sentiment2, confidence2 = analyze_sentiment(text2)

print(f"\nInput text: {text2}")
print(f"Predicted sentiment: {sentiment2}")
print(f"Confidence: {confidence2:.2f}")
