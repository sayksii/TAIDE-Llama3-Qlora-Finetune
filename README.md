# TAIDE Llama-3.1-8B-Chat LoRA Fine-Tuning

本專案旨在對 TAIDE Llama-3.1-LX-8B-Chat 模型進行 QLoRA 微調，使其更擅長網通科技領域（如 Wi-Fi 6E, VPN, 路由器等）的問答。

## 專案目標

- **Base Model**: `TAIDE/Llama-3.1-TAIDE-LX-8B-Chat`
- **Dataset**: `DataAgent/TCNNet-SFT-NetCom-zhTW-1.1M` (網通科技領域)
- **Method**: QLoRA (4-bit Quantization + LoRA adapters)
- **Goal**: 提升模型在特定科技領域的回答專業度，同時保留原本的對話能力。

## 主要功能

- **`finetune_lora.py`**:
    - 完整的 QLoRA 訓練流程。
    - 支援 `apply_chat_template`，確保訓練與推論格式一致。
    - 針對 12GB VRAM 優化（4-bit loading, gradient accumulation）。
    - 訓練過程中監控 Loss 與 Token Accuracy。

- **`run_taide.py`**:
    - 互動式聊天介面。
    - 支援載入原始模型或掛載 LoRA adapter (`--adapter-path`)。
    - 支援 4-bit/8-bit 量化載入。

- **`compare_models.py`**:
    - 並排比較「原始模型」與「微調後模型」的回答。
    - 內建網通測試問題集。

## 快速開始

詳細的操作指令請參考 **[QUICKSTART.md](./QUICKSTART.md)**。

```bash
# 1. 安裝依賴
pip install peft trl accelerate bitsandbytes

# 2. 開始微調 (預設 1 epoch)
python finetune_lora.py --epochs 3

# 3. 測試並比較結果
python compare_models.py
```

## 專案結構

- `finetune_lora.py`: 微調主程式
- `run_taide.py`: 聊天主程式
- `compare_models.py`: 模型比較工具
- `QUICKSTART.md`: 詳細指令說明
- `REPORT.md`: LoRA 指標說明 (Loss & Accuracy)
- `lora-adapter/`: 微調後的權重 (Adapter) 輸出目錄

## 硬體需求

- **GPU**: 建議 NVIDIA RTX 3090/4090 或至少 12GB VRAM (如 RTX 4070)
- **RAM**: 建議 16GB 以上
- **System**: Windows (WSL2 推薦) 或 Linux

## License

TAIDE 模型遵循其官方授權條款。本微調專案程式碼為 MIT License。
