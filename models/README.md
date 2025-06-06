# 語音處理模型資料夾

本資料夾用於存放系統使用的預訓練模型，主要由 SpeechBrain 提供，自動下載並快取在此目錄。

## 主要模型說明

### speechbrain_recognition/

這是系統的核心模型目錄，包含 ECAPA-TDNN 說話者識別模型的檔案：

- **模型名稱**：ECAPA-TDNN
- **來源**：speechbrain/spkrec-ecapa-voxceleb
- **功能**：從音訊中提取說話者特徵向量，用於說話者識別與驗證
- **輸入格式**：16kHz 單聲道音訊
- **輸出維度**：192 維特徵向量

這些檔案由 SpeechBrain 庫管理，首次使用時會自動下載。請勿手動修改此目錄中的檔案，以免影響系統功能。

## 模型快取機制

SpeechBrain 採用智能快取機制：

1. 首次使用某模型時，會自動從 HuggingFace 下載並存放於對應資料夾
2. 後續使用時，會直接從本地快取載入，無需重新下載
3. 模型結構和權重檔案以 PyTorch 格式存儲

## 相關模型置於其他資料夾

- **語者分離模型**：位於 `pretrained_models/sepformer-whamr16k/` 目錄
- **特徵向量資料**：嵌入向量不再存放於檔案系統，已遷移至 Weaviate 資料庫
