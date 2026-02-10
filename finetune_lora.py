import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig


def format_dataset(examples, tokenizer):
    """將 instruction/input/output 格式轉為 chat template 格式"""
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": inp},
            {"role": "assistant", "content": output},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
    return {"text": texts}


def main():
    parser = argparse.ArgumentParser(description="QLoRA 微調 TAIDE Llama-3.1-8B-Chat")
    parser.add_argument("--model-path", type=str, default=".",
                        help="基礎模型路徑")
    parser.add_argument("--output-dir", type=str, default="./lora-adapter",
                        help="LoRA adapter 輸出路徑")
    parser.add_argument("--max-seq-length", type=int, default=1024,
                        help="最大序列長度")
    parser.add_argument("--epochs", type=int, default=1,
                        help="訓練輪數")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="learning rate")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="若 > 0，則只訓練這麼多 steps（用於測試）")
    args = parser.parse_args()

    # ─── 1. 載入 Tokenizer ───
    print("載入 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ─── 2. 載入並格式化資料集 ───
    print("載入資料集 DataAgent/TCNNet-SFT-NetCom-zhTW-1.1M...")
    dataset = load_dataset("DataAgent/TCNNet-SFT-NetCom-zhTW-1.1M", split="train")
    print(f"資料集共 {len(dataset)} 筆")

    print("格式化資料集（apply_chat_template）...")
    dataset = dataset.map(
        lambda examples: format_dataset(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    print(f"格式化完成。範例：\n{dataset[0]['text'][:300]}...")

    # ─── 3. 載入模型（4-bit QLoRA） ───
    print("載入模型（4-bit 量化）...")
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
    model = prepare_model_for_kbit_training(model)

    # ─── 4. 設定 LoRA ───
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ─── 5. 訓練 ───
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        max_length=args.max_seq_length,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    print("開始訓練...")
    trainer.train()

    # ─── 6. 保存 LoRA adapter ───
    print(f"保存 LoRA adapter 到 {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("完成！")


if __name__ == "__main__":
    main()
