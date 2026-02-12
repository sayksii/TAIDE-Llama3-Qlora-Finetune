"""
LLM-as-Judge 評估腳本：使用外部 LLM（Groq Llama 70B）評審微調前後模型的回答品質。

兩階段流程：
  generate: 載入本地 Base & LoRA 模型 → 生成回答 → 存成 answers.json
  judge:    讀取 answers.json → 送 Groq API 評分 → 存成 judge_results.json
  all:      一次完成 generate + judge
"""

import argparse
import json
import os
import random
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANSWERS_FILE = os.path.normpath(os.path.join(SCRIPT_DIR, "../results/answers.json"))
RESULTS_FILE = os.path.normpath(os.path.join(SCRIPT_DIR, "../results/judge_results.json"))

JUDGE_SYSTEM_PROMPT = """你是一位嚴格且公正的 AI 回答品質評審。你的任務是評估兩個模型對同一個問題的回答品質。
請根據提供的「標準答案」作為參考，從以下四個維度分別為每個模型的回答打分（1-10 分）。
你必須嚴格按照指定的 JSON 格式輸出結果，不要輸出任何其他內容。"""

JUDGE_USER_TEMPLATE = """## 問題
{question}

## 標準答案（參考）
{reference}

## 模型 A 的回答
{answer_a}

## 模型 B 的回答
{answer_b}

請從以下兩個維度分別為模型 A 和模型 B 打分（1-10 分）：

1. **正確性 (Correctness)**：回答的事實是否與標準答案一致？有無錯誤資訊？
2. **完整性 (Completeness)**：是否涵蓋了標準答案中的所有重點？有無遺漏關鍵內容？

請嚴格以以下 JSON 格式回覆，不要附加任何解釋：
{{
  "model_a": {{
    "correctness": <分數>,
    "completeness": <分數>,
    "comment": "<一句話簡評>"
  }},
  "model_b": {{
    "correctness": <分數>,
    "completeness": <分數>,
    "comment": "<一句話簡評>"
  }}
}}"""


# ═══════════════════════════════════════════════════
# 階段 1: generate — 本地模型生成回答
# ═══════════════════════════════════════════════════

def generate_answer(model, tokenizer, instruction, question, max_new_tokens=512):
    """用模型生成一個回答"""
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]
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


def run_generate(args):
    """階段 1：生成 Base 和 LoRA 模型的回答"""
    random.seed(args.seed)

    # 載入資料集
    print("載入資料集...")
    dataset = load_dataset("DataAgent/TCNNet-SFT-NetCom-zhTW-1.1M", split="train")
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    samples = dataset.select(indices)
    print(f"取樣 {len(samples)} 筆測試資料")

    # 載入 Tokenizer & Model
    print("載入 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print("載入 base model（4-bit）...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Base Model 生成
    print("\n" + "=" * 50)
    print("Base Model 生成回答")
    print("=" * 50)
    answers = []
    for i in range(len(samples)):
        inst = samples["instruction"][i]
        q = samples["input"][i]
        ref = samples["output"][i]
        print(f"  [{i+1}/{len(samples)}] {q[:40]}...")
        base_ans = generate_answer(model, tokenizer, inst, q, args.max_new_tokens)
        answers.append({
            "instruction": inst,
            "question": q,
            "reference": ref,
            "base_answer": base_ans,
            "lora_answer": None,
        })

    # 掛載 LoRA Adapter
    print("\n" + "=" * 50)
    print("掛載 LoRA Adapter，生成回答")
    print("=" * 50)
    print(f"載入 LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    for i in range(len(samples)):
        q = samples["input"][i]
        inst = samples["instruction"][i]
        print(f"  [{i+1}/{len(samples)}] {q[:40]}...")
        lora_ans = generate_answer(model, tokenizer, inst, q, args.max_new_tokens)
        answers[i]["lora_answer"] = lora_ans

    # 釋放 GPU
    del model
    torch.cuda.empty_cache()

    # 存檔
    os.makedirs(os.path.dirname(ANSWERS_FILE), exist_ok=True)
    with open(ANSWERS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "num_samples": len(samples),
                "adapter_path": args.adapter_path,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
            },
            "answers": answers,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n回答已保存到 {ANSWERS_FILE}")


# ═══════════════════════════════════════════════════
# 階段 2: judge — 用 Groq API 評分
# ═══════════════════════════════════════════════════

def call_groq(client, question, reference, answer_a, answer_b, model_name):
    """呼叫 Groq API 取得評分"""
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=question,
        reference=reference[:1500],  # 截斷過長的標準答案
        answer_a=answer_a[:1500],
        answer_b=answer_b[:1500],
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=500,
    )

    content = response.choices[0].message.content.strip()

    # 嘗試解析 JSON
    # 有時 LLM 會在 JSON 外面包 ```json ... ```
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    return json.loads(content)


def run_judge(args):
    """階段 2：用 Groq API 評分"""
    from groq import Groq

    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("錯誤：請提供 Groq API Key（--api-key 或設定 GROQ_API_KEY 環境變數）")
        return

    # 讀取回答
    if not os.path.exists(ANSWERS_FILE):
        print(f"錯誤：找不到 {ANSWERS_FILE}，請先執行 generate 階段")
        return

    with open(ANSWERS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    answers = data["answers"]
    print(f"讀取 {len(answers)} 筆回答")

    client = Groq(api_key=api_key)
    model_name = args.judge_model

    results = []
    base_scores = {"correctness": [], "completeness": []}
    lora_scores = {"correctness": [], "completeness": []}

    for i, item in enumerate(answers):
        print(f"\n[{i+1}/{len(answers)}] 評審中: {item['question'][:40]}...")

        # 隨機決定 A/B 順序（消除位置偏差）
        base_is_a = random.random() > 0.5

        if base_is_a:
            answer_a, answer_b = item["base_answer"], item["lora_answer"]
        else:
            answer_a, answer_b = item["lora_answer"], item["base_answer"]

        try:
            scores = call_groq(
                client, item["question"], item["reference"],
                answer_a, answer_b, model_name,
            )

            # 還原真實身份
            if base_is_a:
                base_s, lora_s = scores["model_a"], scores["model_b"]
            else:
                base_s, lora_s = scores["model_b"], scores["model_a"]

            for dim in base_scores:
                base_scores[dim].append(base_s[dim])
                lora_scores[dim].append(lora_s[dim])

            results.append({
                "question": item["question"],
                "base_scores": base_s,
                "lora_scores": lora_s,
            })

            print(f"  Base: {base_s}")
            print(f"  LoRA: {lora_s}")

        except Exception as e:
            print(f"  ⚠ 評審失敗: {e}")
            results.append({
                "question": item["question"],
                "error": str(e),
            })

        # Groq 免費版 rate limit，稍微等一下
        time.sleep(1)

    # 計算平均分
    print("\n" + "=" * 60)
    print("LLM-as-Judge 評估報告")
    print("=" * 60)

    dims = ["correctness", "completeness"]
    print(f"\n{'維度':<15} {'Base Model':>12} {'LoRA Model':>12} {'差異':>8}")
    print("-" * 50)

    summary = {}
    for dim in dims:
        b_avg = sum(base_scores[dim]) / len(base_scores[dim]) if base_scores[dim] else 0
        l_avg = sum(lora_scores[dim]) / len(lora_scores[dim]) if lora_scores[dim] else 0
        diff = l_avg - b_avg
        dim_zh = {"correctness": "正確性", "completeness": "完整性"}[dim]
        print(f"{dim_zh:<15} {b_avg:>12.2f} {l_avg:>12.2f} {diff:>+8.2f}")
        summary[dim] = {"base": b_avg, "lora": l_avg, "diff": diff}

    # 總分
    b_total = sum(sum(base_scores[d]) for d in dims) / (len(dims) * max(len(base_scores["correctness"]), 1))
    l_total = sum(sum(lora_scores[d]) for d in dims) / (len(dims) * max(len(lora_scores["correctness"]), 1))
    print("-" * 50)
    print(f"{'總平均':<15} {b_total:>12.2f} {l_total:>12.2f} {l_total - b_total:>+8.2f}")

    # 存檔
    output = {
        "config": {
            "judge_model": model_name,
            "num_evaluated": len([r for r in results if "error" not in r]),
            "seed": args.seed,
        },
        "summary": summary,
        "total": {"base": b_total, "lora": l_total},
        "details": results,
    }

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n詳細結果已保存到 {RESULTS_FILE}")


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge 評估：用外部 LLM 評審微調前後模型",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # generate
    gen_parser = subparsers.add_parser("generate", help="生成 Base & LoRA 模型回答")
    gen_parser.add_argument("--model-path", type=str, default="../model")
    gen_parser.add_argument("--adapter-path", type=str, default="../lora-adapter/checkpoint-250")
    gen_parser.add_argument("--num-samples", type=int, default=10)
    gen_parser.add_argument("--max-new-tokens", type=int, default=512)
    gen_parser.add_argument("--seed", type=int, default=42)

    # judge
    judge_parser = subparsers.add_parser("judge", help="用 Groq API 評分")
    judge_parser.add_argument("--api-key", type=str, default=None,
                              help="Groq API Key（或設定 GROQ_API_KEY 環境變數）")
    judge_parser.add_argument("--judge-model", type=str, default="llama-3.3-70b-versatile")
    judge_parser.add_argument("--seed", type=int, default=42)

    # all
    all_parser = subparsers.add_parser("all", help="一次完成 generate + judge")
    all_parser.add_argument("--model-path", type=str, default="../model")
    all_parser.add_argument("--adapter-path", type=str, default="../lora-adapter/checkpoint-250")
    all_parser.add_argument("--num-samples", type=int, default=10)
    all_parser.add_argument("--max-new-tokens", type=int, default=512)
    all_parser.add_argument("--api-key", type=str, default=None)
    all_parser.add_argument("--judge-model", type=str, default="llama-3.3-70b-versatile")
    all_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # resolve relative paths
    if hasattr(args, 'model_path'):
        args.model_path = os.path.normpath(os.path.join(SCRIPT_DIR, args.model_path))
    if hasattr(args, 'adapter_path'):
        args.adapter_path = os.path.normpath(os.path.join(SCRIPT_DIR, args.adapter_path))

    if args.command == "generate":
        run_generate(args)
    elif args.command == "judge":
        run_judge(args)
    elif args.command == "all":
        run_generate(args)
        run_judge(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
