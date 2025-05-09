# 專案存檔資料夾 (Archive)

本資料夾包含專案開發過程中的歷史檔案、過時版本和參考資料。這些檔案雖不再用於主要開發流程，但仍作為開發歷史和技術參考保留。

## 資料夾結構

### deprecated/

此子資料夾包含已棄用的舊版程式碼和腳本，主要包括：

- **16kHz.py**: 早期的 16kHz 音訊處理腳本，已被現有的音訊處理功能取代
- 其他早期開發的測試程式、功能原型和不再使用的工具

## 使用說明

本資料夾中的檔案主要供研究和歷史參考之用，不建議在當前專案中直接使用。若要了解專案的演進過程或查看早期實現方式，這些檔案可能會有所幫助。

### 開發參考

如果您想了解特定功能的早期實現方式，可以參考這些歷史檔案：

```python
# 例如，查看早期的音訊處理方式
python archive/deprecated/16kHz.py --help
```

## 技術演進

本專案從初始的檔案系統儲存方式 (v3)，到使用 Weaviate 向量資料庫 (v4、v5)，再到整合語者分離功能 (speaker_system_v2.py)，經歷了多次技術升級。archive 資料夾保留了這一演進過程的痕跡，對於理解系統架構的發展有重要價值。

## 注意事項

- 此資料夾中的程式碼不保證與當前系統相容
- 部分功能可能依賴於已不再使用的函式庫或 API
- 請勿在生產環境中使用這些檔案