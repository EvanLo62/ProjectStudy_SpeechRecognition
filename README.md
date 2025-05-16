# Voice_ID: 語者分離與識別系統

## 專案概述

### 語音識別與分離系統 (Speech Recognition and Separation System)
本專案實現了基於深度學習的語者識別與語音分離系統，能夠從語音檔案或即時音訊中提取語者特徵向量，進行身份識別，並將多人混合語音分離為獨立的音訊流。系統能夠自動更新現有語者的聲紋資料或註冊新的語者。

> **重要提示**: 目前 v5 版本 (`VID_identify_v5.py`) 為最新穩定版本，使用 Weaviate 資料庫進行特徵向量存儲和檢索。v2.1.5 版整合系統 (`VID_system_v2.py`) 提供完整的語者分離與即時識別功能，並整合語者管理模組 (`VID_manager.py`)。

> **語者分離模型限制**：目前系統使用的是 SpeechBrain 的 16kHz 雙說話者（2人）預訓練模型，因此在使用時只能分離兩個說話者的混合語音。若有三人或更多人同時說話的情況，系統會將其合併為兩個主要聲源或可能造成分離效果不佳。

## 專案架構

```
ProjectStudy_SpeechRecognition/
│
├── VID_main.py           # 主程式入口點
├── VID_identify_v5.py    # 語者識別引擎 (v5.1.2)
├── VID_system_v2.py      # 整合式語音分離與識別系統 (v2.1.5)
├── VID_manager.py        # 語者與聲紋管理模組
├── VID_database.py       # 資料庫服務抽象層
├── VID_logger.py         # 統一日誌系統
│
├── 16K-model/            # 音訊檔案目錄
│   └── Audios-16K-IDTF/  # 分離後的音檔和混合音檔
│
├── embeddingFiles/       # 語者嵌入向量儲存 (舊版v3使用)
│   └── [說話者ID]/       # 各語者的嵌入向量
│
├── models/               # 存放下載的模型檔案
│   ├── sepformer_whamr_enhancement/
│   └── speechbrain_recognition/
│
├── pretrained_models/    # 預訓練模型
│   └── sepformer-whamr16k/
│
├── weaviate_study/       # Weaviate資料庫工具和設定
│   ├── create_collections.py   # 建立集合結構
│   ├── docker-compose.yml      # Docker配置檔
│   ├── npy_to_weaviate.py      # 將.npy轉換至Weaviate
│   ├── tool_delete_all.py      # 清空資料庫
│   ├── tool_delete.py          # 刪除特定資料
│   └── tool_search.py          # 搜尋工具
│
└── archive/              # 歷史和測試用檔案
    └── deprecated/       # 已棄用的舊版模組
```

## 系統需求

- Python 3.9+
- Windows 作業系統 (Windows 11)
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

## Docker 和 Weaviate 設定 (必要步驟)

1. 確保 Docker Desktop 已安裝並運行
2. 啟動 Weaviate 容器：
   ```
   cd weaviate_study
   docker-compose up -d
   ```
3. 等待容器啟動完成（通常需要 30-60 秒）
4. 初始化 Weaviate 結構：
   ```
   python weaviate_study\create_collections.py
   ```
   
> **注意**: 首次啟動後，必須執行 `create_collections.py` 來建立必要的資料集合。此步驟只需執行一次，除非您刪除了 Weaviate 容器或資料卷。

## 主要功能模組

### 1. 主程式入口點 (VID_main.py)

提供系統的主要入口點及使用者介面：

- **整合式功能選單**：可選擇啟動即時語者分離與識別，或管理說話者和聲紋
- **系統狀態檢查**：在啟動時檢查 Weaviate 連線狀態，確保系統正常運作
- **錯誤處理**：提供完整的例外處理，確保程式運行穩定

### 2. 語者識別引擎 (VID_identify_v5.py)

v5.1.2 版本提供獨立的語者識別功能：

- **說話者識別**：使用 SpeechBrain 的 ECAPA-TDNN 模型提取語音特徵向量
- **動態更新**：根據相似度自動更新現有語者的特徵資料
- **新語者識別**：當語音與現有特徵差異較大時，自動註冊為新語者
- **批次處理**：支援處理單個音檔或整個目錄的音檔
- **向量資料庫**：使用 Weaviate 進行高效的向量檢索和管理
- **時間戳記支援**：可從外部傳入時間戳功能
- **多聲紋映射**：支援將多個不同條件下的聲紋映射到單個語者

### 3. 整合式語者分離與識別系統 (VID_system_v2.py)

v2.1.5 整合系統提供一站式語音處理解決方案：

- **即時錄音**：從麥克風直接捕獲音訊流
- **語者分離**：使用 SpeechBrain Sepformer 模型將混合語音分離為獨立聲道
- **即時識別**：分離後直接進行語者識別，無需等待完整錄音
- **音訊增強**：應用頻譜閘控降噪技術提高分離品質
- **結果視覺化**：實時顯示識別結果，包括明確標示新增的語者

### 4. 語者管理模組 (VID_manager.py)

提供完整的語者和聲紋管理功能：

- **語者查詢與列表**：列出所有已註冊的語者及其基本資訊
- **語者信息編輯**：修改語者名稱和其他相關屬性
- **聲紋管理**：查看語者關聯的所有聲紋向量及其屬性
- **聲紋遷移**：將聲紋從一個語者轉移到另一個語者
- **刪除功能**：移除語者或特定聲紋向量
- **資料庫檢查與修復**：提供資料庫健康檢查和修復功能

### 5. 資料庫服務 (VID_database.py)

封裝所有與 Weaviate 資料庫的互動：

- **單例模式**：確保全局只有一個資料庫連接實例
- **統一接口**：提供清晰的資料庫操作介面
- **語者操作**：建立、讀取、更新、刪除語者資訊
- **聲紋操作**：建立、讀取、更新、刪除聲紋向量
- **向量搜索**：基於相似度的語者匹配與搜索
- **資料一致性**：確保語者和聲紋資料之間的引用完整性

### 6. 統一日誌系統 (VID_logger.py)

提供集中式的日誌系統，可被專案中所有模組使用：

- **統一格式**：統一的日誌格式和顏色設置
- **模組識別**：在日誌中包含模組名稱
- **彈性文件處理**：支援輸出到不同的日誌檔案
- **單例模式**：確保每個命名的日誌器只創建一次

## 音檔取樣率與比對閾值注意事項

系統使用 16kHz 取樣率處理音訊，關於不同取樣率的處理效果說明：

- **16kHz 音檔**: 提供最佳的識別準確度，建議使用此取樣率
- **高於 16kHz 音檔**: 會被降頻至 16kHz 處理，效果良好
- **8kHz 音檔**: 會被升頻至 16kHz，但比對結果變化較大，準確度較低
- **其他取樣率**: 直接重新採樣至 16kHz

> **重要**: 目前系統的閾值已根據實際測試進行優化 (`THRESHOLD_LOW=0.26`, `THRESHOLD_UPDATE=0.34`, `THRESHOLD_NEW=0.385`)。若要調整識別敏感度，可根據實際需求調整這些閾值。

## 使用方法

### 1. 啟動整合式系統

直接執行主程式：

```
python VID_main.py
```

運行後系統將顯示功能選單：
1. 啟動即時語者分離與識別：開始錄音並識別
2. 管理說話者和聲紋：啟動語者管理選單
0. 退出

選擇選項1後：
- 啟動麥克風錄音
- 每 6 秒處理一次音訊 (有 50% 重疊)
- 分離語音並即時識別
- 在終端顯示結果
- 將分離後的音檔儲存至 `16K-model/Audios-16K-IDTF/` 目錄

按下 `Ctrl+C` 可停止錄音和識別過程。

### 2. 語者管理

通過整合系統選單或直接執行：

```
python VID_manager.py
```

語者管理系統介面提供以下選項：
1. 列出所有說話者
2. 檢視說話者詳細資訊
3. 更改說話者名稱
4. 轉移聲紋到其他說話者
5. 刪除說話者
6. 資料庫清理與修復
0. 離開

### 3. 使用語者識別引擎的程式碼範例

```python
from VID_identify_v5 import SpeakerIdentifier, set_output_enabled

# 控制是否顯示詳細輸出
set_output_enabled(False)

# 初始化識別器
identifier = SpeakerIdentifier()

# 處理單個音檔
result = identifier.process_audio_file("path_to_audio.wav")

# 或處理整個目錄
results = identifier.process_audio_directory("path_to_directory")

# 添加嵌入向量到現有說話者
identifier.add_voiceprint_to_speaker("path_to_audio.wav", "speaker_uuid")
```

## 關鍵參數與閾值設定

系統使用三個閾值來判斷如何處理輸入的語音：

- **THRESHOLD_LOW** (0.26)：若相似度極高（距離<0.26），視為重複語音，不進行更新
- **THRESHOLD_UPDATE** (0.34)：若相似度較高（距離<0.34），更新現有說話者特徵
- **THRESHOLD_NEW** (0.385)：若距離在0.34-0.385之間，為現有說話者添加新聲紋；若距離>0.39，則註冊為新說話者

其他重要參數：
- **WINDOW_SIZE** (6)：處理窗口大小（秒）
- **OVERLAP** (0.5)：窗口重疊率
- **MIN_ENERGY_THRESHOLD** (0.005)：最小能量閾值，用於過濾靜音
- **DEVICE_INDEX** (None)：麥克風設備索引，預設使用系統默認設備

## 資料庫結構 (Weaviate)

系統使用 Weaviate 向量資料庫儲存語者資訊和聲紋向量，分為兩個主要集合：

1. **Speaker**：存儲語者基本資訊
   - `uuid`: 唯一識別碼
   - `name`: 語者名稱
   - `create_time`: 建立時間
   - `last_active_time`: 最後活動時間
   - `voiceprint_ids`: 關聯的聲紋向量ID列表

2. **VoicePrint**：存儲聲紋向量資料
   - `uuid`: 唯一識別碼
   - `create_time`: 建立時間
   - `updated_time`: 更新時間
   - `update_count`: 更新次數
   - `speaker_name`: 語者名稱
   - `audio_source`: 來源音檔資訊
   - `vector`: 聲紋嵌入向量（用於相似度搜尋）
   - 參考關係 `speaker`: 指向 Speaker 集合中的對應記錄

## 將舊版本資料遷移到 Weaviate

如果您之前使用 v3 版本並有現有的嵌入向量文件 (.npy)，可以使用以下命令將其遷移至 Weaviate 資料庫：

```
python weaviate_study\npy_to_weaviate.py
```

此工具會從 `embeddingFiles/` 目錄中讀取所有 .npy 文件並將它們匯入至 Weaviate 資料庫。

## 錯誤排除

### Docker/Weaviate 相關問題
- 如果無法連接到 Weaviate，確認 Docker 是否正在運行
- 執行 `docker ps` 檢查 Weaviate 容器是否正常啟動
- 確保已執行 `create_collections.py` 建立必要的資料集合
- 若需重置 Weaviate 資料，可使用 `weaviate_study\create_collections.py`

### 語者分離與識別系統問題
- 如果錄音品質不佳，檢查麥克風設定和環境噪音
- 若分離效果不理想，可能是原始音訊中說話者聲音重疊過多
- 若識別結果不準確，可嘗試調整閾值或增加該說話者的訓練樣本

### 其他常見問題
- 若遇到模型載入錯誤，請確認 SpeechBrain 已正確安裝：`pip install speechbrain`
- 若處理音檔時出現錯誤，檢查音檔格式是否為 wav 格式
- 若遇到日期格式錯誤，確保系統時間設定正確（Weaviate 要求嚴格的 RFC3339 日期格式）

## 最新更新

- **2025-05-16**: 更新集中式日誌系統 (VID_logger.py)，提供統一的日誌格式和管理
- **2025-05-15**: 更新語者管理模組 (`VID_manager.py`)與資料庫服務抽象層(VID_database.py)，新增功能與改進資料庫操作介面
- **2025-05-07**: 更新 README.md 以反映最新系統功能與版本
- **2025-05-06**: 更新 VID_identify_v5.py 到 v5.1.2，支援外部時間戳和多聲紋映射
- **2025-05-06**: 更新 VID_system_v2.py 到 v2.1.5，整合語者管理功能

## 未來規劃

1. 增強語者分離功能，支援三人以上混合語音
2. 開發網頁介面，提供更友善的使用體驗
3. 增加語音轉文字功能，實現完整的語音內容分析
4. 加入情感分析模組，提供語者情緒狀態追蹤
5. 優化資源使用，減少計算資源需求

## 授權資訊

本專案採用 MIT 授權，詳見 LICENSE 檔案。

## 文件目錄說明

- `VID_identify_v5.py`：v5.1.2 版本語者識別主程式（使用 Weaviate 資料庫）
- `VID_system_v2.py`：v2.1.5 整合式語者分離與識別系統
- `VID_manager.py`：語者與聲紋管理模組
- `weaviate_study/`：Weaviate 相關工具和設定
  - `docker-compose.yml`：Weaviate Docker 容器設定檔
  - `create_collections.py`：建立 Weaviate 集合結構
  - `npy_to_weaviate.py`：將 v3 版本的 .npy 檔案匯入 Weaviate
  - `tool_search.py`：Weaviate 搜尋工具
  - `tool_delete_all.py`：清除 Weaviate 資料
  - `tool_delete_speaker.py`：刪除特定語者資料
- `models/`：存放下載的 SpeechBrain 模型
- `output_log.txt`：操作日誌
- `16K-model/Audios-16K-IDTF/`：存放分離後的音檔與原始混合音檔