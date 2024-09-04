from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 事前にダウンロードしたモデルとトークナイザーをローカルキャッシュから読み込む
model_name_or_path = "./model_cache"  # ダウンロードしたモデルのパス
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

# テキストの入力（例: 感情分析）
input_text = "Hugging Face is making AI accessible!"

# テキストをトークン化
inputs = tokenizer(input_text, return_tensors="pt")

# モデルで予測
outputs = model(**inputs)

# 結果の表示
print("Logits:", outputs.logits)

# ロジットからクラスを選択（必要に応じて適用）
predicted_class = outputs.logits.argmax(dim=-1).item()
print(f"Predicted class: {predicted_class}")
