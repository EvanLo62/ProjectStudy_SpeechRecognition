# 語音識別系統 (Speech Recognition System)

## 專案概述

本專案實現了基於 SpeechBrain 的說話者識別系統，能夠從語音檔案中提取說話者特徵向量，並通過餘弦相似度比較判斷說話者身份。系統能夠自動更新現有說話者的特徵資料或註冊新的說話者。

> **重要提示**: 目前 v3 版本 (`main_identify_v3.py`) 為穩定版本，可正常使用。v4 版本 (`main_identify_v4_weaviate.py`) 仍在開發階段，存在問題，尚不建議使用。

## 音檔取樣率與比對閾值注意事項

系統使用 16kHz 取樣率處理音訊，關於不同取樣率的處理效果說明：

- **16kHz 音檔**: 提供最佳的識別準確度，建議使用此取樣率
- **高於 16kHz 音檔**: 會被降頻至 16kHz 處理，效果良好
- **8kHz 音檔**: 會被升頻至 16kHz，但比對結果變化較大，準確度較低
- **其他取樣率**: 直接重新採樣至 16kHz

> **重要**: 目前系統的閾值 (`THRESHOLD_LOW=0.2`, `THRESHOLD_UPDATE=0.34`, `THRESHOLD_NEW=0.36`) 是基於 8kHz 音檔設計的。若主要處理 16kHz 或更高品質的音檔，建議調小閾值以提高識別準確度。較高取樣率的音檔特徵相似度通常更高（距離值更小）。

## 功能特點

- **說話者識別**：使用 SpeechBrain 的 ECAPA-TDNN 模型提取語音特徵向量
- **動態更新**：根據相似度自動更新現有說話者的特徵資料
- **新說話者識別**：當語音與現有特徵差異較大時，自動註冊為新說話者
- **批次處理**：支援處理單個音檔或整個目錄的音檔
- **日誌記錄**：所有操作都會記錄到 output_log.txt 檔案中

## 系統需求

- Python 3.9+
- Windows 作業系統
- 相關 Python 套件 (詳見 requirements.txt)

## 安裝步驟

1. 確保您已安裝 Python 3.9 或更高版本
2. 複製專案到本地：
   ```
   git clone <專案儲存庫網址>
   cd ProjectStudy_SpeechRecognition
   ```
3. 建立虛擬環境並啟用：
   ```
   python -m venv venv_speech
   venv_speech\Scripts\activate
   ```
4. 安裝相依套件：
   ```
   pip install -r requirements.txt
   ```

## 使用方法

### 初始化

首次使用系統時會自動下載 SpeechBrain 的預訓練模型並初始化環境。

### 處理單個音檔

```python
from main_identify_v3 import process_audio_file

process_audio_file("path_to_audio.wav")
```

### 處理整個目錄

```python
from main_identify_v3 import process_audio_directory

process_audio_directory("path_to_directory")
```

## 閾值設定說明

系統使用三個閾值來判斷如何處理輸入的語音：

- **THRESHOLD_LOW** (0.2)：若相似度極高（距離<0.2），視為重複語音，不進行更新
- **THRESHOLD_UPDATE** (0.34)：若相似度較高（距離<0.34），更新現有說話者特徵
- **THRESHOLD_NEW** (0.36)：若距離在0.34-0.36之間，匹配到說話者但不更新；若距離>0.36，則註冊為新說話者

## 嵌入特徵檔案格式

系統儲存的嵌入特徵文件格式為：
```
embeddingFiles/
  ├── <說話者資料夾>/
  │    └── <說話者ID>_<檔案編號>_<更新次數>.npy
```

例如：`embeddingFiles/n1/n1_1_3.npy` 代表說話者n1的第1個特徵檔案，已經更新過3次。

## 工作原理

1. **特徵提取**：
   - 讀取音訊檔案
   - 將音訊重新採樣到16kHz
   - 使用SpeechBrain模型提取嵌入向量

2. **特徵比較**：
   - 計算新向量與資料庫中所有向量的餘弦距離
   - 找出距離最小的向量（最相似）

3. **決策邏輯**：
   - 根據相似度閾值決定：跳過/更新/匹配/新建

4. **資料管理**：
   - 使用加權移動平均更新現有特徵
   - 為新說話者創建新資料夾和特徵檔案

## 進階使用

系統支援同步更新嵌入特徵檔案的說話者ID，透過呼叫：
```python
import sync_npy_username
sync_npy_username.sync_all_folders()
```

## 錯誤排除

- 若遇到模型載入錯誤，請確認 SpeechBrain 已正確安裝：`pip install speechbrain`
- 若處理音檔時出現錯誤，檢查音檔格式是否為 wav，若不是可使用 m4a-wav.py 進行轉換

## 文件目錄說明

- `main_identify_v3.py`：主程式檔案
- `sync_npy_username.py`：用於同步嵌入檔案的使用者名稱
- `m4a-wav.py`：音檔格式轉換工具
- `models/`：存放下載的SpeechBrain模型
- `embeddingFiles/`：存放生成的嵌入向量檔案
- `output_log.txt`：操作日誌
