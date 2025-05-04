# 語音識別與分離系統 (Speech Recognition and Separation System)

## 專案概述

本專案實現了基於深度學習的說話者識別與語音分離系統，能夠從語音檔案或即時音訊中提取說話者特徵向量，進行身份識別，並可將多人混合語音分離為獨立的音訊流。系統能夠自動更新現有說話者的特徵資料或註冊新的說話者。

> **重要提示**: 目前 v5 版本 (`main_identify_v5.py`) 為最新穩定版本，使用 Weaviate 資料庫進行特徵向量存儲和檢索。v2.1.0 版整合系統 (`speaker_system_v2.py`) 提供完整的語者分離與即時識別功能。舊版 v3 (`main_identify_v3.py`) 使用檔案系統儲存特徵向量，仍可使用但不再積極維護。

## 系統需求

- Python 3.9+
- Windows 作業系統
- Docker Desktop (用於運行 Weaviate 資料庫)
- 麥克風設備 (用於即時錄音功能)
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

## Docker 和 Weaviate 設定 (v5 和整合系統必要步驟)

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

## 主要功能模組

### 1. 語者識別模組 (main_identify_v5.py)

v5 版本提供獨立的語者識別功能，可處理單個音檔或整批音檔：

- **說話者識別**：使用 SpeechBrain 的 ECAPA-TDNN 模型提取語音特徵向量
- **動態更新**：根據相似度自動更新現有說話者的特徵資料
- **新說話者識別**：當語音與現有特徵差異較大時，自動註冊為新說話者
- **批次處理**：支援處理單個音檔或整個目錄的音檔
- **向量資料庫**：使用 Weaviate 進行高效的向量檢索和管理

### 2. 整合式語者分離與識別系統 (speaker_system_v2.py)

v2.1.0 整合系統提供一站式語音處理解決方案，具備以下功能：

- **即時錄音**：從麥克風直接捕獲音訊流
- **語者分離**：使用 SpeechBrain Sepformer 模型將混合語音分離為獨立聲道
- **即時識別**：分離後直接進行語者識別，無需等待完整錄音
- **音訊增強**：應用降噪技術提高分離品質
- **結果視覺化**：實時顯示識別結果，包括明確標示新增的說話者

## 音檔取樣率與比對閾值注意事項

系統使用 16kHz 取樣率處理音訊，關於不同取樣率的處理效果說明：

- **16kHz 音檔**: 提供最佳的識別準確度，建議使用此取樣率
- **高於 16kHz 音檔**: 會被降頻至 16kHz 處理，效果良好
- **8kHz 音檔**: 會被升頻至 16kHz，但比對結果變化較大，準確度較低
- **其他取樣率**: 直接重新採樣至 16kHz

> **重要**: 目前系統的閾值 (`THRESHOLD_LOW=0.27`, `THRESHOLD_UPDATE=0.34`, `THRESHOLD_NEW=0.36`) 是基於實際測試設計的。若要調整識別敏感度，可根據實際需求調整這些閾值。

## 使用方法

### 1. 語者識別模組 (v5 版本)

```python
from main_identify_v5 import SpeakerIdentifier, set_output_enabled

# 控制是否顯示詳細輸出
set_output_enabled(False)

# 初始化識別器
identifier = SpeakerIdentifier()

# 處理單個音檔
identifier.process_audio_file("path_to_audio.wav")

# 或處理整個目錄
identifier.process_audio_directory("path_to_directory")

# 添加嵌入向量到現有說話者
identifier.add_embedding_to_existing_speaker("path_to_audio.wav", "speaker_uuid")
```

### 2. 整合式語者分離與識別系統

直接執行整合系統進行即時錄音、分離和識別：

```
python speaker_system_v2.py
```

運行後系統將：
1. 啟動麥克風錄音
2. 每 6 秒處理一次音訊 (有 50% 重疊)
3. 分離語音並即時識別
4. 在終端顯示結果
5. 將分離後的音檔儲存至 `16K-model/Audios-16K-IDTF/` 目錄

按下 `Ctrl+C` 可停止錄音和識別過程。

## 閾值設定說明

系統使用三個閾值來判斷如何處理輸入的語音：

- **THRESHOLD_LOW** (0.27)：若相似度極高（距離<0.27），視為重複語音，不進行更新
- **THRESHOLD_UPDATE** (0.34)：若相似度較高（距離<0.34），更新現有說話者特徵
- **THRESHOLD_NEW** (0.36)：若距離在0.34-0.36之間，匹配到說話者但不更新；若距離>0.36，則註冊為新說話者

## 資料結構 (v5 版本)

在 v5 版本中，所有特徵向量和說話者資訊都存儲在 Weaviate 資料庫中，分為兩個主要集合：

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

## 將舊版本資料遷移到 Weaviate (v3 到 v5)

如果您之前使用 v3 版本並有現有的嵌入向量文件 (.npy)，可以使用以下命令將其遷移至 Weaviate 資料庫：

```
python weaviate_study/npy_to_weaviate.py
```

此工具會從 `embeddingFiles/` 目錄中讀取所有 .npy 文件並將它們匯入至 Weaviate 資料庫。即使目前使用 v5，仍然可以透過此工具保留舊有的說話者資料。

## 語者識別模組工作原理 (v5 版本 - main_identify_v5.py)

v5 語者識別系統是專為處理單一說話者音檔設計的獨立模組，提供高度精確的說話者身份識別功能。其工作流程如下：

1. **音訊處理與預處理**:
   - 智能處理不同取樣率的音訊檔案 (8kHz, 16kHz, 44.1kHz 等)
   - 自動轉換立體聲為單聲道，並正規化音訊資料
   - 對音訊進行長度限制，確保處理效率 (最多 10 秒)

2. **特徵向量提取**:
   - 使用 SpeechBrain 的 ECAPA-TDNN 模型提取 192 維的聲紋嵌入向量
   - 模型經過 VoxCeleb 數據集訓練，可捕捉說話者獨特的聲音特徵

3. **向量檢索與比對**:
   - 利用 Weaviate 向量資料庫進行高效相似度搜尋
   - 計算新提取向量與資料庫中向量的餘弦距離
   - 根據距離排序，找出最相似的說話者

4. **決策邏輯處理**:
   - 根據三級閾值系統 (THRESHOLD_LOW, THRESHOLD_UPDATE, THRESHOLD_NEW) 進行判斷
   - 動態更新機制：使用加權移動平均法更新現有向量，保持聲紋鮮活度
   - 自動判斷新說話者，無需手動操作

5. **資料管理與持久化**:
   - 維護說話者與聲紋向量之間的關聯關係
   - 記錄每次更新資訊，包括時間戳和更新次數
   - 支援單例模式，避免重複初始化資源

v5 版本還實現了資源高效利用、詳細日誌記錄以及靈活的輸出控制機制，使其特別適合在需要高精確度說話者識別的場景使用，例如音訊檔案分析、安全驗證系統等。

## 語者分離與識別整合系統工作原理 (v2 版本 - speaker_system_v2.py)

v2.1.0 整合系統結合了語者分離和識別功能，專為處理多人對話場景設計：

1. **錄音與分段**:
   - 從麥克風捕獲音訊流
   - 以滑動視窗方式分段音訊 (默認 6 秒，50% 重疊)
   - 檢測能量是否超過閾值以過濾靜音

2. **語者分離**:
   - 使用 SpeechBrain Sepformer 模型將混合語音分離
   - 將音訊重新取樣至 16kHz (模型要求)
   - 應用頻譜閘控降噪增強分離效果

3. **特徵提取與比對**:
   - 對分離後的每個語音流提取嵌入向量
   - 與 Weaviate 資料庫中的現有向量比對
   - 計算餘弦距離判斷相似度

4. **識別決策**:
   - 根據相似度閾值判斷是否為新說話者
   - 適時更新現有說話者的聲紋特徵
   - 即時返回識別結果並顯示

5. **結果輸出**:
   - 將分離後的音檔儲存為 WAV 格式
   - 即時顯示識別結果與狀態
   - 記錄處理日誌

## v3 版本資訊 (main_identify_v3.py - 歷史參考用)

v3 版本使用檔案系統儲存特徵向量，具有以下特點：

- 使用 `.npy` 檔案儲存向量資料
- 資料結構為按說話者 ID 組織的目錄
- 每個說話者目錄下有多個 `.npy` 檔案表示不同時間的聲紋樣本
- 相容性更好，不依賴外部資料庫

雖然 v3 不再積極維護，但其檔案系統儲存方式仍可作為 Weaviate 資料庫無法使用時的備用選項。

## 錯誤排除

### Docker/Weaviate 相關問題
- 如果無法連接到 Weaviate，確認 Docker 是否正在運行
- 執行 `docker ps` 檢查 Weaviate 容器是否正常啟動
- 確保已執行 `create_collections.py` 建立必要的資料集合
- 若需重置 Weaviate 資料，可使用 `weaviate_study/tool_delete_all.py`

### 語者分離與識別系統問題
- 如果錄音品質不佳，檢查麥克風設定和環境噪音
- 若分離效果不理想，可能是原始音訊中說話者聲音重疊過多
- 若識別結果不準確，可嘗試調整閾值或增加該說話者的訓練樣本

### 其他常見問題
- 若遇到模型載入錯誤，請確認 SpeechBrain 已正確安裝：`pip install speechbrain`
- 若處理音檔時出現錯誤，檢查音檔格式是否為 wav，若不是可使用 m4a-wav.py 進行轉換
- 若遇到日期格式錯誤，確保系統時間設定正確（Weaviate 要求嚴格的 RFC3339 日期格式）

## 文件目錄說明

- `main_identify_v5.py`：v5 版本語者識別主程式（使用 Weaviate 資料庫）
- `speaker_system_v2.py`：整合式語者分離與識別系統
- `weaviate_study/`：Weaviate 相關工具和設定
  - `docker-compose.yml`：Weaviate Docker 容器設定檔
  - `create_collections.py`：建立 Weaviate 集合結構
  - `npy_to_weaviate.py`：將 v3 版本的 .npy 檔案匯入 Weaviate
  - `tool_search.py`：Weaviate 搜尋工具
  - `tool_delete_all.py`：清除 Weaviate 資料
- `models/`：存放下載的 SpeechBrain 模型
- `output_log.txt`：操作日誌
- `16K-model/Audios-16K-IDTF/`：存放分離後的音檔與原始混合音檔

## 授權資訊

本專案採用 [授權名稱] 授權，詳見 LICENSE 檔案。
