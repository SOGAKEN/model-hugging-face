import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# 日本語の質問応答モデルを指定
model_name = "Deepreneur/blue-lizard"

# トークナイザーとモデルをローカルにダウンロード
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
model = AutoModelForQuestionAnswering.from_pretrained(
    model_name, local_files_only=False
)

# ダウンロードしたモデルとトークナイザーを保存
tokenizer.save_pretrained("./local_qa_model")
model.save_pretrained("./local_qa_model")

# ローカルから保存したモデルとトークナイザーをロード
local_tokenizer = AutoTokenizer.from_pretrained("./local_qa_model")
local_model = AutoModelForQuestionAnswering.from_pretrained("./local_qa_model")


def answer_question(question, context):
    # 質問とコンテキストをトークン化
    inputs = local_tokenizer(question, context, return_tensors="pt")

    # モデルで推論を実行
    with torch.no_grad():
        outputs = local_model(**inputs)

    # 回答の開始位置と終了位置を取得
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    # トークンを元の文字列に戻して回答を抽出
    answer = local_tokenizer.convert_tokens_to_string(
        local_tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0][answer_start:answer_end]
        )
    )

    return answer


# テスト用のコンテキストと質問
context = "日本は東アジアに位置する島国で、首都は東京です。東京は政治、経済、文化の中心地として知られています。"
question = "日本の首都は？"

# 質問に対する回答を取得
answer = answer_question(question, context)

print(f"質問: {question}")
print(f"回答: {answer}")

# 別の質問でテスト
question2 = "日本はどこに位置していますか？"
answer2 = answer_question(question2, context)

print(f"\n質問: {question2}")
print(f"回答: {answer2}")
