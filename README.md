# 語音識別系統 (Speech Recognition System)

## 專案概述

本專案實現了基於 SpeechBrain 的說話者識別系統，能夠從語音檔案中提取說話者特徵向量，並透過餘弦相似度比較判斷說話者身份。系統能夠自動更新現有說話者的特徵資料或註冊新的說話者。

> **重要提示**: 目前 v4 版本 (`main_identify_v4_weaviate.py`) 為最新穩定版本，使用 Weaviate 資料庫進行特徵向量存儲和檢索。舊版 v3 (`main_identify_v3.py`) 使用檔案系統儲存特徵向量，仍可使用但不再積極維護。

## 系統需求

- Python 3.9+
- Windows 作業系統
- Docker Desktop (用於運行 Weaviate 資料庫)
- 相關 Python 套件 (詳見 requirements.txt)

## 安裝步驟

1. 確保您已安裝 Python 3.9 或更高版本以及 Docker Desktop
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

## Docker 和 Weaviate 設定 (v4 版本必要步驟)

1. 確保 Docker Desktop 已安裝並運行
2. 啟動 Weaviate 容器：
   ```
   cd weaviate_study
   docker-compose up -d
   ```
3. 等待容器啟動完成（通常需要 30-60 秒）
4. 初始化 Weaviate 結構：
   ```
   python create_collections.py
   ```
   
> **注意**: 首次啟動後，必須執行 `create_collections.py` 來建立必要的資料集合。此步驟只需執行一次，除非您刪除了 Weaviate 容器或資料卷。

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
- **向量資料庫**：使用 Weaviate 進行高效的向量檢索和管理（v4版本）
- **日誌記錄**：所有操作都會記錄到 output_log.txt 檔案中

## 使用方法 (v4 版本)

### 初始化

首次使用系統時會自動下載 SpeechBrain 的預訓練模型並初始化環境。請確保 Docker 和 Weaviate 已正確設定。

### 處理單個音檔

```python
from main_identify_v4_weaviate import SpeakerIdentifier

identifier = SpeakerIdentifier()
identifier.process_audio_file("path_to_audio.wav")
```

### 處理整個目錄

```python
from main_identify_v4_weaviate import SpeakerIdentifier

identifier = SpeakerIdentifier()
identifier.process_audio_directory("path_to_directory")
```

### 添加嵌入向量到現有說話者

```python
from main_identify_v4_weaviate import SpeakerIdentifier

identifier = SpeakerIdentifier()
identifier.add_embedding_to_existing_speaker("path_to_audio.wav", "speaker_uuid")
```

## 閾值設定說明

系統使用三個閾值來判斷如何處理輸入的語音：

- **THRESHOLD_LOW** (0.2)：若相似度極高（距離<0.2），視為重複語音，不進行更新
- **THRESHOLD_UPDATE** (0.34)：若相似度較高（距離<0.34），更新現有說話者特徵
- **THRESHOLD_NEW** (0.36)：若距離在0.34-0.36之間，匹配到說話者但不更新；若距離>0.36，則註冊為新說話者

## 資料結構 (v4 版本)

在 v4 版本中，所有特徵向量和說話者資訊都存儲在 Weaviate 資料庫中，分為兩個主要集合：

1. **Speaker**：存儲說話者基本信息
   - `name`: 說話者名稱
   - `create_time`: 建立時間
   - `last_active_time`: 最後活動時間
   - `voiceprint_ids`: 關聯的聲紋向量ID列表

2. **VoicePrint**：存儲聲紋向量資料
   - `create_time`: 建立時間
   - `updated_time`: 更新時間
   - `update_count`: 更新次數
   - `speaker_name`: 說話者名稱
   - `vector`: 聲紋嵌入向量（用於相似度搜尋）
   - 參考關係 `speaker`: 指向 Speaker 集合中的對應記錄

## 將舊版本資料遷移到 Weaviate (v3 到 v4)

如果您之前使用 v3 版本並有現有的嵌入向量文件 (.npy)，可以使用以下命令將其遷移至 Weaviate 資料庫：

```
python weaviate_study/npy_to_weaviate.py
```

此工具會從 `embeddingFiles/` 目錄中讀取所有 .npy 文件並將它們匯入至 Weaviate 資料庫。

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

4. **資料管理** (v4 版本)：
   - 使用加權移動平均更新現有特徵向量
   - 在 Weaviate 中建立新的說話者和聲紋向量記錄
   - 維護說話者與聲紋向量之間的關聯關係

## 錯誤排除

### Docker/Weaviate 相關問題
- 如果無法連接到 Weaviate，確認 Docker 是否正在運行
- 執行 `docker ps` 檢查 Weaviate 容器是否正常啟動
- 確保已執行 `create_collections.py` 建立必要的資料集合
- 若需重置 Weaviate 資料，可使用 `weaviate_study/tool_delete_all.py`

### 其他常見問題
- 若遇到模型載入錯誤，請確認 SpeechBrain 已正確安裝：`pip install speechbrain`
- 若處理音檔時出現錯誤，檢查音檔格式是否為 wav，若不是可使用 m4a-wav.py 進行轉換
- 若遇到日期格式錯誤，確保系統時間設定正確（Weaviate 要求嚴格的 RFC3339 日期格式）

## 文件目錄說明

- `main_identify_v4_weaviate.py`：v4 版本主程式檔案（使用 Weaviate 資料庫）
- `weaviate_study/`：Weaviate 相關工具和設定
  - `docker-compose.yml`：Weaviate Docker 容器設定檔
  - `create_collections.py`：建立 Weaviate 集合結構
  - `npy_to_weaviate.py`：將 .npy 檔案匯入 Weaviate
  - `tool_search.py`：Weaviate 搜尋工具
  - `tool_delete_all.py`：清除 Weaviate 資料
- `models/`：存放下載的 SpeechBrain 模型
- `output_log.txt`：操作日誌

## 授權資訊

本專案採用 [授權名稱] 授權，詳見 LICENSE 檔案。
