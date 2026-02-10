import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import PeftModel


def sanitize_text(text: str) -> str:
    """清理可能的無效 surrogate 字元（例如 PowerShell 編碼問題）"""
    try:
        return text.encode('utf-8', 'replace').decode('utf-8')
    except Exception:
        return text


def count_tokens(tokenizer, text: str) -> int:
    """計算文字的 token 數量"""
    s = sanitize_text(str(text))
    try:
        return len(tokenizer(s).input_ids)
    except Exception:
        # 若 tokenizer 無法處理（例如編碼問題），退回簡單的字元/空白估算
        return len(s.split())


def clip_history_to_max_tokens(tokenizer, conv: list, max_tokens: int):
    """裁剪歷史對話，確保 token 數不超過上限（O(n) 做法）"""
    total = sum(count_tokens(tokenizer, m["content"]) for m in conv)
    while conv and total > max_tokens:
        total -= count_tokens(tokenizer, conv[0]["content"])
        conv.pop(0)


def load_model(model_path: str, quant_mode: str):
    """載入模型，支援 none / 8bit / 4bit 量化模式"""
    use_bnb = False
    try:
        import bitsandbytes  # noqa: F401
        use_bnb = True
    except Exception:
        use_bnb = False

    if quant_mode == "8bit" and use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
            )
        except ValueError as e:
            msg = str(e)
            if "Some modules are dispatched on the CPU or the disk" in msg:
                print("⚠ GPU VRAM 不足，啟用 fp32 CPU offload（速度較慢）")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            else:
                raise
    elif quant_mode == "4bit" and use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
            )
        except ValueError as e:
            msg = str(e)
            if "Some modules are dispatched on the CPU or the disk" in msg:
                print("⚠ GPU VRAM 不足，4-bit 載入失敗，嘗試不量化載入...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            else:
                raise
    else:
        # fallback: no quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    return model


def main():
    # 1. 解析命令列參數
    parser = argparse.ArgumentParser(description="TAIDE 互動式聊天程式")
    parser.add_argument(
        "--quant", choices=["none", "8bit", "4bit"], default="4bit",
        help="quantization: none, 8bit or 4bit",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32768,
        help="歷史對話的最大 token 數",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="每次回覆最多產生的 token 數",
    )
    parser.add_argument(
        "--adapter-path", type=str, default=None,
        help="LoRA adapter 路徑（例如 ./lora-adapter），不指定則用原始模型",
    )
    args = parser.parse_args()

    max_tokens = args.max_tokens
    max_new_tokens = args.max_new_tokens

    # 2. 指定模型資料夾路徑
    model_path = "."

    # 3. 載入 Tokenizer（移除 use_fast=False，使用預設的 fast tokenizer）
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. 載入模型
    model = load_model(model_path, args.quant)

    # 5. 若指定了 adapter，載入 LoRA
    if args.adapter_path:
        print(f"載入 LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        print("LoRA adapter 已載入。")

    print("模型與 tokenizer 已載入。輸入 'exit' 或按 Ctrl-D 結束。")

    # 5. 互動式對話迴圈
    conversation = []  # 區域變數，格式: [{"role": "user"/"assistant", "content": ...}, ...]

    try:
        while True:
            try:
                user = input("You: ")
            except EOFError:
                break
            if not user:
                continue
            if user.strip().lower() in ("exit", "quit"):
                break

            user_str = sanitize_text(str(user).strip())
            if not user_str:
                continue

            # 加入使用者訊息到歷史
            conversation.append({"role": "user", "content": user_str})

            # 裁剪過長的歷史
            clip_history_to_max_tokens(tokenizer, conversation, max_tokens)

            # 使用模型內建的 chat template 組 prompt
            prompt = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
            )

            inputs = tokenizer(prompt, return_tensors="pt")

            # 將 inputs 移到模型所在裝置
            try:
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except StopIteration:
                pass

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

            # 只解碼新產生的 token
            input_len = inputs["input_ids"].shape[1]
            gen_ids = outputs[0]
            if gen_ids.shape[0] > input_len:
                new_tokens = gen_ids[input_len:]
            else:
                new_tokens = gen_ids[0:0]
            resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # 儲存助理回應到歷史
            conversation.append({"role": "assistant", "content": resp})

            # 再次裁剪以保證不超過 token 上限
            clip_history_to_max_tokens(tokenizer, conversation, max_tokens)

            print("Assistant:", resp)

    except KeyboardInterrupt:
        print('\n終止互動')


if __name__ == "__main__":
    main()