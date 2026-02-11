"""
BERTScore 評估腳本：比較 Base Model 與 LoRA Model 的回答品質。

流程：
  1. 從 TCNNet 資料集隨機取樣 N 筆問題
  2. 載入 Base Model → 生成回答 → 掛載 LoRA Adapter → 生成回答
  3. 使用 BERTScore 計算「模型回答 vs 標準答案」的語意相似度
  4. 輸出比較報告
"""

import argparse
import json
import random
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from bert_score import score as bert_score


def generate_answer(model, tokenizer, instruction, question, max_new_tokens=256):
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


def main():
    parser = argparse.ArgumentParser(description="BERTScore 評估微調前後模型")
    parser.add_argument("--model-path", type=str, default="../model",
                        help="基礎模型路徑")
    parser.add_argument("--adapter-path", type=str, default="../lora-adapter/checkpoint-200",
                        help="LoRA adapter 路徑")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="從資料集取樣幾筆來測試")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="每個回答最多產生的 token 數")
    parser.add_argument("--seed", type=int, default=42,
                        help="隨機種子（確保每次取樣一致）")
    parser.add_argument("--bert-model", type=str, default="bert-base-chinese",
                        help="BERTScore 使用的 BERT 模型")
    args = parser.parse_args()

    random.seed(args.seed)

    # ─── 1. 載入並取樣資料集 ───
    print("載入資料集...")
    dataset = load_dataset("DataAgent/TCNNet-SFT-NetCom-zhTW-1.1M", split="train")
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    samples = dataset.select(indices)
    print(f"取樣 {len(samples)} 筆測試資料")

    instructions = samples["instruction"]
    questions = samples["input"]
    references = samples["output"]

    # ─── 2. 載入 Tokenizer & Model ───
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

    # ─── 3. Base Model 生成回答 ───
    print("\n" + "=" * 50)
    print("階段 1: Base Model 生成回答")
    print("=" * 50)

    base_preds = []
    for i, (inst, q) in enumerate(zip(instructions, questions)):
        print(f"  [{i+1}/{len(samples)}] {q[:40]}...")
        ans = generate_answer(model, tokenizer, inst, q, args.max_new_tokens)
        base_preds.append(ans)

    # ─── 4. 掛載 LoRA Adapter（不需重新載入模型） ───
    print("\n" + "=" * 50)
    print("階段 2: 掛載 LoRA Adapter，生成回答")
    print("=" * 50)
    print(f"載入 LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    lora_preds = []
    for i, (inst, q) in enumerate(zip(instructions, questions)):
        print(f"  [{i+1}/{len(samples)}] {q[:40]}...")
        ans = generate_answer(model, tokenizer, inst, q, args.max_new_tokens)
        lora_preds.append(ans)

    # 釋放模型記憶體，給 BERTScore 用
    del model
    torch.cuda.empty_cache()

    # ─── 5. 計算 BERTScore ───
    print("\n" + "=" * 50)
    print("計算 BERTScore...")
    print("=" * 50)

    P_base, R_base, F1_base = bert_score(
        base_preds, list(references),
        model_type=args.bert_model, lang="zh", verbose=True,
    )
    P_lora, R_lora, F1_lora = bert_score(
        lora_preds, list(references),
        model_type=args.bert_model, lang="zh", verbose=True,
    )

    # ─── 6. 輸出報告 ───
    print("\n" + "=" * 60)
    print("BERTScore 評估報告")
    print("=" * 60)

    avg_p_base, avg_r_base, avg_f1_base = P_base.mean().item(), R_base.mean().item(), F1_base.mean().item()
    avg_p_lora, avg_r_lora, avg_f1_lora = P_lora.mean().item(), R_lora.mean().item(), F1_lora.mean().item()

    print(f"\n{'模型':<15} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
    print("-" * 50)
    print(f"{'Base Model':<15} {avg_p_base:>10.4f} {avg_r_base:>10.4f} {avg_f1_base:>10.4f}")
    print(f"{'LoRA Model':<15} {avg_p_lora:>10.4f} {avg_r_lora:>10.4f} {avg_f1_lora:>10.4f}")
    print("-" * 50)
    print(f"{'提升幅度':<15} {avg_p_lora - avg_p_base:>+10.4f} {avg_r_lora - avg_r_base:>+10.4f} {avg_f1_lora - avg_f1_base:>+10.4f}")

    # 找出 LoRA 進步最多和退步最多的例子
    f1_diff = F1_lora - F1_base
    top_improve = f1_diff.argsort(descending=True)[:3]
    top_degrade = f1_diff.argsort()[:3]

    print("\n\n【LoRA 進步最多的 3 題】")
    for rank, idx in enumerate(top_improve, 1):
        idx = idx.item()
        print(f"\n  #{rank} (F1 提升: {f1_diff[idx].item():+.4f})")
        print(f"  問題: {questions[idx][:60]}...")
        print(f"  Base F1: {F1_base[idx].item():.4f} → LoRA F1: {F1_lora[idx].item():.4f}")

    print("\n\n【LoRA 退步最多的 3 題】")
    for rank, idx in enumerate(top_degrade, 1):
        idx = idx.item()
        print(f"\n  #{rank} (F1 下降: {f1_diff[idx].item():+.4f})")
        print(f"  問題: {questions[idx][:60]}...")
        print(f"  Base F1: {F1_base[idx].item():.4f} → LoRA F1: {F1_lora[idx].item():.4f}")

    # ─── 7. 保存詳細結果到 JSON ───
    results = {
        "config": {
            "num_samples": len(samples),
            "adapter_path": args.adapter_path,
            "bert_model": args.bert_model,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "summary": {
            "base": {"precision": avg_p_base, "recall": avg_r_base, "f1": avg_f1_base},
            "lora": {"precision": avg_p_lora, "recall": avg_r_lora, "f1": avg_f1_lora},
        },
        "details": [],
    }
    for i in range(len(samples)):
        results["details"].append({
            "question": questions[i],
            "reference": references[i][:200],
            "base_pred": base_preds[i][:200],
            "lora_pred": lora_preds[i][:200],
            "base_f1": F1_base[i].item(),
            "lora_f1": F1_lora[i].item(),
            "f1_diff": f1_diff[i].item(),
        })

    output_file = "../results/bertscore_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n詳細結果已保存到 {output_file}")


if __name__ == "__main__":
    main()
