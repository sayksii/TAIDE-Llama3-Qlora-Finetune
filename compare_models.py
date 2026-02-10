import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# 預設測試問題（網通科技相關，對應 TCNNet 微調領域）
DEFAULT_QUESTIONS = [
    "什麼是 Wi-Fi 6E？它和 Wi-Fi 6 有什麼差別？",
    "請解釋什麼是 VPN，以及它的主要用途。",
    "請說明無線路由器的擺放位置對訊號的影響，以及該如何改善訊號覆蓋率？",
]


def generate_answer(model, tokenizer, question, max_new_tokens=256):
    """用模型生成一個回答"""
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except StopIteration:
        pass

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="比較微調前後的 TAIDE 模型")
    parser.add_argument("--model-path", type=str, default=".",
                        help="基礎模型路徑")
    parser.add_argument("--adapter-path", type=str, default="./lora-adapter",
                        help="LoRA adapter 路徑")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="每個回答最多產生的 token 數")
    parser.add_argument("--questions", nargs="+", default=None,
                        help="自訂測試問題（可多個）")
    args = parser.parse_args()

    questions = args.questions if args.questions else DEFAULT_QUESTIONS

    # ─── 1. 載入 Tokenizer ───
    print("載入 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # ─── 2. 載入 Base Model（4-bit） ───
    print("載入 base model（4-bit）...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # ─── 3. Base Model 回答 ───
    print("\n" + "=" * 60)
    print("【原始模型回答】")
    print("=" * 60)
    base_answers = []
    for i, q in enumerate(questions, 1):
        print(f"\n--- 問題 {i}: {q} ---")
        ans = generate_answer(model, tokenizer, q, args.max_new_tokens)
        base_answers.append(ans)
        print(ans)

    # ─── 4. 載入 LoRA Adapter ───
    print("\n\n載入 LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    # ─── 5. LoRA Model 回答 ───
    print("\n" + "=" * 60)
    print("【微調後模型回答】")
    print("=" * 60)
    lora_answers = []
    for i, q in enumerate(questions, 1):
        print(f"\n--- 問題 {i}: {q} ---")
        ans = generate_answer(model, tokenizer, q, args.max_new_tokens)
        lora_answers.append(ans)
        print(ans)

    # ─── 6. 並排比較 ───
    print("\n\n" + "=" * 60)
    print("【並排比較】")
    print("=" * 60)
    for i, q in enumerate(questions):
        print(f"\n{'─' * 50}")
        print(f"問題 {i + 1}: {q}")
        print(f"{'─' * 50}")
        print(f"  原始: {base_answers[i][:200]}{'...' if len(base_answers[i]) > 200 else ''}")
        print(f"  微調: {lora_answers[i][:200]}{'...' if len(lora_answers[i]) > 200 else ''}")


if __name__ == "__main__":
    main()
