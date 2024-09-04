import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def setup_qa_system():
    # 日本語モデルを指定
    model_name = "tohoku-nlp/bert-base-japanese-v2"

    # トークナイザーとモデルをローカルにダウンロード
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # モデルの全てのパラメータを連続的（contiguous）にする
    for param in model.parameters():
        param.data = param.data.contiguous()

    # GPU使用可能な場合はGPUを使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # questionAnsweringパイプラインの作成
    qa_pipeline = pipeline(
        "question-answering", model=model, tokenizer=tokenizer, device=device
    )

    return qa_pipeline


def answer_question(qa_pipeline, question, context):
    result = qa_pipeline(question=question, context=context)
    return result["answer"]


def main():
    qa_pipeline = setup_qa_system()

    # テスト用のコンテキストと質問
    context = ""
    question = "日本の首都は？"

    # 質問に対する回答を取得
    answer = answer_question(qa_pipeline, question, context)

    print(f"質問: {question}")
    print(f"回答: {answer}")

    # 別の質問でテスト
    question2 = "日本はどこに位置していますか？"
    answer2 = answer_question(qa_pipeline, question2, context)

    print(f"\n質問: {question2}")
    print(f"回答: {answer2}")


if __name__ == "__main__":
    main()
