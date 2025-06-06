# 專案存檔資料夾 (Archive)

本資料夾包含專案開發過程中的歷史檔案、過時版本和參考資料。這些檔案雖不再用於主要開發流程，但仍作為開發歷史和技術參考保留。

## 資料夾結構

### deprecated/

此子資料夾包含已棄用的舊版程式碼和腳本，主要包括：

- **16kHz.py**: 早期的 16kHz 音訊處理腳本，已被現有的音訊處理功能取代
- **main_identify_v1.py**: 第一版語者識別系統，使用簡單的距離比對
- **main_identify_v2.py**: 第二版語者識別，引入閾值判斷機制
- 其他早期開發的測試程式、功能原型和不再使用的工具

## 使用說明

本資料夾中的檔案主要供研究和歷史參考之用，不建議在當前專案中直接使用。若要了解專案的演進過程或查看早期實現方式，這些檔案可能會有所幫助。

### 開發參考

如果您想了解特定功能的早期實現方式，可以參考這些歷史檔案：

```python
# 例如，查看早期的音訊處理方式
python archive/deprecated/16kHz.py

# 查看 v1 版本的語者識別邏輯
python archive/deprecated/main_identify_v1.py
```

## 技術演進歷程

本專案歷經多次重要技術升級：

1. **檔案管理階段 (v1-v3)**
   - 基於檔案系統的特徵向量儲存 
   - 使用 NumPy (.npy) 檔案格式儲存向量
   - 簡單距離計算進行匹配
   - 依賴人工命名和管理說話者身份

2. **向量資料庫階段 (v4-v5)**
   - 採用 Weaviate 向量資料庫
   - 高效向量索引和相似度搜尋
   - 完整的說話者-聲紋關聯結構
   - 引入動態更新機制和閾值策略

3. **語者分離整合階段 (v2.x)**
   - 整合 Sepformer 進行多人語音分離
   - 開發即時錄音與識別流程
   - 多線程處理提高系統響應速度
   - 增強音訊預處理與降噪功能

4. **語者管理階段 (v5.x + 管理模組)**
   - 完整的語者與聲紋管理系統
   - 支援多聲紋映射到單一語者
   - 聲紋轉移與合併功能
   - 用戶友好的管理界面

archive 資料夾保留了這一演進過程的痕跡，對於理解系統架構的發展有重要價值。

## 注意事項

- 此資料夾中的程式碼不保證與當前系統相容
- 部分功能可能依賴於已不再使用的函式庫或 API
- 請勿在生產環境中使用這些檔案
- 若需存取舊版模型權重，請參考 `models/legacy/` 資料夾
- 閱讀歷史程式碼時，建議參考同時期的文件說明以理解上下文

## 未來參考價值

這些歷史資料對以下情況具有參考價值：

1. 需要回滾到舊版本時的參考依據
2. 理解系統設計決策的歷史背景
3. 新功能開發時的想法啟發
4. 作為技術演示和學術研究的素材

最後更新：2025-05-07