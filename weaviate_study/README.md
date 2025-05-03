# Weaviate 向量資料庫應用工具集

本資料夾包含與 Weaviate 向量資料庫相關的工具、設定及範例程式。Weaviate 是一個高性能的向量資料庫，在本專案中用於儲存和檢索語音特徵向量，支援語者識別系統的核心功能。

## 重要檔案說明

### 1. docker-compose.yml

Weaviate 容器設定檔，用於快速部署 Weaviate 服務。

```bash
# 啟動 Weaviate 服務
docker-compose up -d

# 停止服務
docker-compose down
```

預設配置在本機啟動 Weaviate，無需驗證，可立即使用。

### 2. create_collections.py

初始化 Weaviate 資料結構的必要腳本，建立兩個核心集合：

- **Speaker**: 儲存說話者基本資訊
- **VoicePrint**: 儲存聲紋向量和相關元數據

⚠️ **重要**: 首次使用系統前必須執行此腳本建立資料結構！

### 3. 資料操作工具

- **npy_to_weaviate.py**: 將 v3 版本的 .npy 檔案遷移至 Weaviate 資料庫
- **tool_search.py**: 在 Weaviate 中搜尋說話者或聲紋資訊
- **tool_delete_all.py**: 清空 Weaviate 資料庫中的所有資料
- **tool_delete_speaker.py**: 刪除特定說話者及其聲紋資料

### 4. 開發測試檔案

- **test_create_collection_1.py** 與 **test_create_collection_2.py**: 不同版本的集合建立測試腳本，僅供開發參考

## 使用指南

### Weaviate 服務管理

1. 確認 Docker Desktop 已啟動
2. 在此資料夾下執行 `docker-compose up -d` 啟動服務
3. 執行 `python create_collections.py` 初始化資料結構 (僅首次需要)
4. Weaviate 會在背景持續運行，無需每次使用都重新啟動

### 資料遷移 (從 v3 版本遷移資料)

若要從舊版系統遷移資料：

```bash
python npy_to_weaviate.py
```

此操作會掃描 `embeddingFiles/` 資料夾中的所有 .npy 檔案，為每個檔案創建相應的說話者和聲紋記錄。

### 資料庫維護

若需要重設資料庫或進行特定操作：

```bash
# 搜尋特定說話者資訊
python tool_search.py

# 清空所有資料
python tool_delete_all.py
```

## 進階使用

Weaviate 提供完整的 REST API 和 GraphQL 介面，可以從瀏覽器訪問 `http://localhost:8080/v1/graphql` 進行查詢操作。

更多關於 Weaviate 的資訊，請參考[官方文檔](https://weaviate.io/developers/weaviate/current/)。