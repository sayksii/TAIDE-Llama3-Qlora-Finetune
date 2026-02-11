# TAIDE Llama-3.1-8B-Chat — Quick Start

## 環境準備

```bash
# 啟動 conda 環境
conda activate taide-lab

# 安裝依賴（首次）
pip install peft trl accelerate bitsandbytes bert-score
```

## 檢查 GPU

```bash
python scripts/check_gpu.py
```

## 聊天（原始模型）

```bash
cd scripts
python run_taide.py
```

可選參數：
```bash
python run_taide.py --quant 4bit          # 量化模式：none / 8bit / 4bit（預設 4bit）
python run_taide.py --max-new-tokens 256  # 回覆長度上限（預設 512）
python run_taide.py --max-tokens 4096     # 歷史上限 tokens（預設 32768）
```

## LoRA 微調

```bash
cd scripts

# 完整訓練（2,226 筆 × 1 epoch）
python finetune_lora.py

# 快速測試（只跑 2 步，確認 pipeline 正常）
python finetune_lora.py --max-steps 2
```

可選參數：
```bash
python finetune_lora.py --epochs 3              # 訓練輪數（預設 1）
python finetune_lora.py --lr 1e-4               # learning rate（預設 2e-4）
python finetune_lora.py --max-seq-length 2048    # 最大序列長度（預設 1024）
python finetune_lora.py --output-dir ../my-lora  # 輸出路徑（預設 ../lora-adapter）
```

## 聊天（微調後模型）

```bash
cd scripts

# 使用完整訓練完的 adapter
python run_taide.py --adapter-path ../lora-adapter

# 使用中途的 checkpoint（例如 checkpoint-200）
python run_taide.py --adapter-path ../lora-adapter/checkpoint-200
```

## 比較微調前後

```bash
cd scripts
python compare_models.py

# 自訂問題
python compare_models.py --questions "什麼是防火牆？" "如何設定 VPN？"

# 比較特定 checkpoint
python compare_models.py --adapter-path ../lora-adapter/checkpoint-200
```

## LLM-as-Judge 評估 (推薦)

使用 Groq Llama-3 70B 作為評審，比對微調前後的回答品質。

```bash
cd scripts

# 設定 Groq API Key (Windows PowerShell)
$env:GROQ_API_KEY="gsk_xxxx"

# 一次完成生成+評分 (10 筆資料)
python eval_llm_judge.py all --num-samples 10

# 分階段執行：
# 1. 生成回答 (需 GPU)
python eval_llm_judge.py generate --num-samples 10

# 2. 評分 (不需 GPU，需連網)
python eval_llm_judge.py judge
```

## BERTScore 評估

```bash
cd scripts

# 正式評估（50 筆）
python eval_bertscore.py

# 快速測試（2 筆）
python eval_bertscore.py --num-samples 2 --max-new-tokens 50
```

