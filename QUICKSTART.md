# TAIDE Llama-3.1-8B-Chat — Quick Start

## 環境準備

```bash
# 啟動 conda 環境
conda activate taide-lab

# 安裝依賴（首次）
pip install peft trl accelerate bitsandbytes
```

## 檢查 GPU

```bash
python check_gpu.py
```

## 聊天（原始模型）

```bash
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
python finetune_lora.py --output-dir ./my-lora   # 輸出路徑（預設 ./lora-adapter）
```

## 聊天（微調後模型）

```bash
# 使用完整訓練完的 adapter
python run_taide.py --adapter-path ./lora-adapter

# 使用中途的 checkpoint（例如 checkpoint-200）
python run_taide.py --adapter-path ./lora-adapter/checkpoint-200
```

## 比較微調前後

```bash
python compare_models.py

# 自訂問題
python compare_models.py --questions "什麼是防火牆？" "如何設定 VPN？"

# 比較特定 checkpoint
python compare_models.py --adapter-path ./lora-adapter/checkpoint-200
```
