import torch
from transformers import AutoConfig, AutoTokenizer, LlamaForSequenceClassification

# モデルとトークナイザーのパスを指定
model_path = "./blue-lizard"

# トークナイザーをロード
tokenizer = AutoTokenizer.from_pretrained(model_path)

# モデルの設定をロード
config = AutoConfig.from_pretrained(model_path)

# モデルをロード
model = LlamaForSequenceClassification(config)
model_state_dict = {}

# 分割されたモデルファイルをロード
for i in range(1, 4):  # 3つの分割ファイルがあると仮定
    shard_file = f"{model_path}/model-0000{i}-of-00003.safetensors"
    shard_state_dict = torch.load(shard_file, map_location="cpu")
    model_state_dict.update(shard_state_dict)

# モデルの重みを設定
model.load_state_dict(model_state_dict)

# モデルを評価モードに設定
model.eval()


def predict(text):
    # 入力テキストをトークン化
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # 推論
    with torch.no_grad():
        outputs = model(**inputs)

    # 結果の解釈（例：分類タスクの場合）
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities[0].tolist()


# 使用例
text = "This is an example text for prediction."
predicted_class, probabilities = predict(text)

print(f"Predicted class: {predicted_class}")
print(f"Class probabilities: {probabilities}")
