#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主程式：語音識別系統 (Weaviate 版本)
使用 SpeechBrain 提取聲紋向量並儲存到 Weaviate 資料庫中

支援從檔案或直接從記憶體中的音訊數據處理聲紋特徵
"""

import os
import re
import sys
import uuid
import numpy as np
import torch
import soundfile as sf
from scipy.spatial.distance import cosine
from scipy.signal import resample_poly
import warnings
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, BinaryIO

# 明確定義退出碼
class ExitCode:
    """系統退出碼定義"""
    SUCCESS = 0                    # 正常結束
    PACKAGE_ERROR = 1              # 套件載入失敗
    DB_CONNECTION_ERROR = 2        # 資料庫連線失敗
    COLLECTION_NOT_EXIST = 3       # Weaviate 集合不存在
    FILE_NOT_FOUND = 4             # 檔案或目錄不存在
    AUDIO_PROCESSING_ERROR = 5     # 音訊處理錯誤
    PARAMETER_ERROR = 6            # 參數錯誤
    UNKNOWN_ERROR = 99             # 未知錯誤

# 隱藏多餘的警告與日誌
warnings.filterwarnings("ignore")
logging.getLogger("speechbrain").setLevel(logging.ERROR)

# 自定義 Tee 類別：同時輸出到螢幕和 log 檔案
class Tee:
    def __init__(self, file_name: str, mode: str = "w") -> None:
        """
        建立同時將輸出寫入檔案和標準輸出的類別
        
        Args:
            file_name: 輸出日誌檔案名稱 
            mode: 開啟檔案的模式，預設為覆寫 ('w')
        """
        self.file = open(file_name, mode, encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message: str) -> None:
        """寫入訊息到檔案和標準輸出"""
        self.file.write(message)
        self.stdout.write(message)

    def flush(self) -> None:
        """刷新檔案和標準輸出緩衝區"""
        self.file.flush()
        self.stdout.flush()

# 將標準輸出重定向到 log 檔
sys.stdout = Tee("output_log.txt")

# 載入必要的套件
try:
    from speechbrain.inference import SpeakerRecognition
    import weaviate # type: ignore
    import io
    print("成功載入所有必要套件")
except ImportError as e:
    print(f"套件載入失敗: {e}")
    print("請確認已安裝必要套件：pip install speechbrain weaviate-client")
    sys.exit(ExitCode.PACKAGE_ERROR)

# 系統配置參數
class Config:
    """系統配置參數集中管理"""
    # 時區設定
    TAIPEI_TIMEZONE = timezone(timedelta(hours=8))  # 台北時區 (UTC+8)
    
    # 音訊處理參數
    TARGET_SAMPLE_RATE = 16000  # 模型期望的採樣率
    MAX_AUDIO_SECONDS = 10      # 最大音訊長度（秒）
    
    # 相似度閾值
    THRESHOLD_LOW = 0.2      # 過於相似，不更新
    THRESHOLD_UPDATE = 0.34  # 觸發更新嵌入向量
    THRESHOLD_NEW = 0.36     # 判定為新說話者
    
    # Weaviate 集合名稱
    COLLECTION_SPEAKER = "Speaker"
    COLLECTION_VOICEPRINT = "VoicePrint"

# 載入 SpeechBrain 模型
try:
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/speechbrain_recognition"
    )
    print("SpeechBrain 模型載入成功！")
except Exception as e:
    print(f"SpeechBrain 模型載入失敗: {e}")
    print("請確認已安裝 SpeechBrain 或檢查網路連接")
    sys.exit(ExitCode.PACKAGE_ERROR)

def connect_to_weaviate() -> Any:
    """
    連接到本地 Weaviate 資料庫
    
    Returns:
        weaviate.WeaviateClient: Weaviate 客戶端實例或 None (如果連接失敗)
    
    Raises:
        ConnectionError: 當無法連接到 Weaviate 時
    """
    try:
        client = weaviate.connect_to_local()
        print("成功連接到 Weaviate 本地資料庫")
        return client
    except Exception as e:
        error_msg = f"無法連接到 Weaviate 資料庫: {str(e)}"
        print(error_msg)
        sys.exit(ExitCode.DB_CONNECTION_ERROR)

def check_collections_exist(client: Any) -> bool:
    """
    檢查 Weaviate 中是否存在所需的集合
    
    Args:
        client: Weaviate 客戶端實例
        
    Returns:
        bool: 如果所需集合都存在則返回 True，否則返回 False
    """
    speaker_exists = client.collections.exists(Config.COLLECTION_SPEAKER)
    voiceprint_exists = client.collections.exists(Config.COLLECTION_VOICEPRINT)
    
    all_exists = speaker_exists and voiceprint_exists
    
    if not speaker_exists:
        print(f"警告: {Config.COLLECTION_SPEAKER} 集合不存在")
    if not voiceprint_exists:
        print(f"警告: {Config.COLLECTION_VOICEPRINT} 集合不存在")
    
    return all_exists

def format_date_rfc3339(dt: Optional[datetime] = None) -> str:
    """
    將日期格式化為符合 RFC3339 標準的字串，包含時區資訊
    
    Args:
        dt: 日期時間對象，若未提供則使用目前時間
        
    Returns:
        str: RFC3339 格式的日期字串
    """
    # 若未提供時間，使用當前時間
    if dt is None:
        dt = datetime.now(Config.TAIPEI_TIMEZONE)
    
    # 確保日期時間有時區資訊
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=Config.TAIPEI_TIMEZONE)
    
    # 格式化為 RFC3339
    return dt.isoformat()

def resample_audio(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    使用 scipy 進行高品質音訊重新採樣
    
    Args:
        signal: 原始音訊訊號
        orig_sr: 原始採樣率
        target_sr: 目標採樣率
        
    Returns:
        np.ndarray: 重新採樣後的音訊訊號
    """
    return resample_poly(signal, target_sr, orig_sr)

def process_audio_data(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    處理音訊數據並提取嵌入向量
    
    Args:
        signal: 音訊訊號 (numpy 陣列)
        sample_rate: 音訊的採樣率 (Hz)
        
    Returns:
        np.ndarray: 聲紋嵌入向量
        
    Raises:
        ValueError: 當音訊處理失敗時
    """
    try:
        # 確保是單聲道
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        
        # 重新取樣到 16kHz (適用於 SpeechBrain 模型)
        if sample_rate != Config.TARGET_SAMPLE_RATE:
            # 先降頻到 8kHz 再升頻到 16kHz  測試模式
            signal_8k = resample_audio(signal, sample_rate, 8000)
            signal_16k = resample_audio(signal_8k, 8000, Config.TARGET_SAMPLE_RATE)
        else:
            signal = signal[:Config.TARGET_SAMPLE_RATE * Config.MAX_AUDIO_SECONDS]
            signal_16k = resample_audio(signal, sample_rate, Config.TARGET_SAMPLE_RATE)
        
        # 轉換為 PyTorch 張量並限制長度
        signal_16k = torch.tensor(signal_16k, dtype=torch.float32).unsqueeze(0)
        max_length = Config.TARGET_SAMPLE_RATE * Config.MAX_AUDIO_SECONDS  # 限制音訊長度
        signal_16k = signal_16k[:, :max_length] if signal_16k.shape[1] > max_length else signal_16k
        
        # 使用模型提取嵌入向量
        embedding = model.encode_batch(signal_16k).squeeze().numpy()
        return embedding
    
    except Exception as e:
        raise ValueError(f"處理音訊數據時發生錯誤: {str(e)}")

def extract_embedding_from_file(audio_path: str) -> np.ndarray:
    """
    從音檔提取嵌入向量
    
    Args:
        audio_path: 音檔的路徑
        
    Returns:
        np.ndarray: 聲紋嵌入向量
        
    Raises:
        FileNotFoundError: 當音檔不存在時
        ValueError: 當音檔無法處理時
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音檔 {audio_path} 不存在")
    
    try:
        # 讀取音檔
        signal, sr = sf.read(audio_path)
        return process_audio_data(signal, sr)
    except Exception as e:
        raise ValueError(f"處理音檔 {audio_path} 時發生錯誤: {str(e)}")

def extract_embedding_from_bytes(audio_bytes: Union[bytes, BinaryIO], sample_rate: int, 
                                format_hint: Optional[str] = None) -> np.ndarray:
    """
    從字節數據中提取音訊嵌入向量
    
    Args:
        audio_bytes: 音訊二進制數據或二進制流
        sample_rate: 音訊的採樣率 (Hz)
        format_hint: 音訊格式提示 (如 'WAV', 'MP3'，可選)
        
    Returns:
        np.ndarray: 聲紋嵌入向量
        
    Raises:
        ValueError: 當音訊數據無法處理時
    """
    try:
        # 如果是 bytes 對象，轉換為 IO 流
        if isinstance(audio_bytes, bytes):
            audio_bytes = io.BytesIO(audio_bytes)
        
        # 使用 soundfile 讀取音訊數據
        signal, file_sr = sf.read(audio_bytes, format=format_hint)
        
        # 如果提供的採樣率與文件中讀取的不同，使用提供的採樣率
        actual_sr = sample_rate if sample_rate != file_sr else file_sr
        
        return process_audio_data(signal, actual_sr)
    
    except Exception as e:
        raise ValueError(f"處理音訊二進制數據時發生錯誤: {str(e)}")

def find_closest_speaker(client: Any, new_embedding: np.ndarray) -> Tuple[Optional[str], Optional[str], float, List[Dict]]:
    """
    在 Weaviate 資料庫中尋找與新嵌入向量最相似的聲紋
    
    Args:
        client: Weaviate 客戶端實例
        new_embedding: 新的聲紋嵌入向量
        
    Returns:
        Tuple: (說話者ID, 聲紋ID, 餘弦距離, 所有比對結果的列表)
    """
    try:
        # 獲取 VoicePrint 集合
        voiceprint_collection = client.collections.get(Config.COLLECTION_VOICEPRINT)
        
        # !!! TEST_MODE !!! 為了測試，使用向量搜尋找到最相似的多個聲紋
        print("\n!!! TEST_MODE !!! 尋找最相似的 5 個聲紋")
        results = voiceprint_collection.query.near_vector(
            near_vector=new_embedding,
            limit=5,  # 取前 5 個最相似的結果，便於測試
            distance=True,  # 附帶餘弦距離
            include_vector=True  # 測試模式：包含向量資訊以進行更詳細的比較
        )
        
        # 檢查是否有結果
        if not results.objects:
            print("資料庫中沒有任何聲紋記錄")
            return None, None, float("inf"), []
        
        # 收集所有比對結果
        all_matches = []
        for obj in results.objects:
            speaker_id = None
            # 嘗試獲取關聯的說話者
            if hasattr(obj, 'references') and obj.references.get('speaker'):
                speaker_refs = obj.references.get('speaker')
                if speaker_refs and len(speaker_refs) > 0:
                    speaker_id = speaker_refs[0]  # 取第一個關聯的說話者
            
            # 將結果加入列表，直接使用 VoicePrint 中的 speaker_name 屬性，避免額外查詢
            match_info = {
                'voiceprint_id': obj.uuid,
                'speaker_id': speaker_id,
                'speaker_name': obj.properties.get('speaker_name', 'unknown'),
                'distance': obj.distance,
                'update_count': obj.properties.get('update_count', 0),
                'vector': obj.vector  # 測試模式：保存向量資訊
            }
            all_matches.append(match_info)
            
            # !!! TEST_MODE !!! 輸出詳細比對資訊
            print(f"比對結果 #{idx} - 說話者: {match_info['speaker_name']}, "
                  f"聲紋ID: {match_info['voiceprint_id']}, "
                  f"餘弦距離: {match_info['distance']:.4f}, "
                  f"更新次數: {match_info['update_count']}")
                    
        # 找出最接近的結果 (餘弦距離最小)
        best_match = min(all_matches, key=lambda x: x['distance'])
        
        # !!! TEST_MODE !!! 輸出最佳匹配結果
        print(f"\n最佳匹配: 說話者={best_match['speaker_name']}, "
              f"距離={best_match['distance']:.6f}")
        
        return best_match['speaker_id'], best_match['voiceprint_id'], best_match['distance'], all_matches
    
    except Exception as e:
        print(f"尋找最相似聲紋時發生錯誤: {str(e)}")
        import traceback
        print(f"詳細錯誤追蹤: {traceback.format_exc()}")
        return None, None, float("inf"), []

def update_voiceprint(client: Any, voiceprint_id: str, new_embedding: np.ndarray, speaker_name: str) -> str:
    """
    更新現有聲紋向量 (使用加權平均)
    
    Args:
        client: Weaviate 客戶端實例
        voiceprint_id: 要更新的聲紋ID
        new_embedding: 新的嵌入向量
        speaker_name: 說話者名稱
        
    Returns:
        str: 更新後的聲紋ID (如果成功)
    """
    try:
        # 獲取集合
        voiceprint_collection = client.collections.get(Config.COLLECTION_VOICEPRINT)
        
        # 獲取現有的聲紋記錄
        result = voiceprint_collection.query.fetch_object_by_id(
            voiceprint_id,
            include_vector=True
        )
        
        if not result:
            print(f"無法找到聲紋ID: {voiceprint_id}")
            return ""
        
        # 獲取現有的嵌入向量和更新次數
        current_vector = result.vector
        current_update_count = result.properties.get('update_count', 1)
        
        # 計算加權平均更新向量 (保留舊向量的權重)
        updated_embedding = (current_vector * current_update_count + new_embedding) / (current_update_count + 1)
        new_update_count = current_update_count + 1
        
        # 更新資料庫中的向量
        voiceprint_collection.data.update(
            uuid=voiceprint_id,
            properties={
                "updated_time": format_date_rfc3339(),
                "update_count": new_update_count
            },
            vector=updated_embedding
        )
        
        print(f"(更新) 說話者 '{speaker_name}' 的聲紋 {voiceprint_id} 已更新 (更新次數: {new_update_count})")
        return voiceprint_id
        
    except Exception as e:
        print(f"更新聲紋時發生錯誤: {str(e)}")
        return ""

def create_new_speaker(client: Any, new_embedding: np.ndarray) -> Tuple[str, str]:
    """
    建立新的說話者和聲紋記錄
    
    Args:
        client: Weaviate 客戶端實例
        new_embedding: 新的嵌入向量
        
    Returns:
        Tuple[str, str]: (說話者ID, 聲紋ID)
    """
    try:
        # 獲取集合
        speaker_collection = client.collections.get(Config.COLLECTION_SPEAKER)
        voiceprint_collection = client.collections.get(Config.COLLECTION_VOICEPRINT)
        
        # 生成新的說話者名稱 (如 "n1", "n2" 等)
        new_speaker_name = generate_unique_speaker_name(client)
        
        # 為說話者和聲紋生成 UUID
        speaker_id = str(uuid.uuid4())
        voiceprint_id = str(uuid.uuid4())
        
        # 當前時間
        current_time = format_date_rfc3339()
        
        # 先建立說話者
        speaker_collection.data.insert(
            uuid=speaker_id,
            properties={
                "name": new_speaker_name,
                "create_time": current_time,
                "last_active_time": current_time,
                "voiceprint_ids": [voiceprint_id]
            }
        )
        
        # 再建立聲紋，並關聯到說話者
        voiceprint_collection.data.insert(
            uuid=voiceprint_id,
            properties={
                "create_time": current_time,
                "updated_time": current_time,
                "update_count": 1,
                "speaker_name": new_speaker_name
            },
            references={
                "speaker": [speaker_id]
            },
            vector=new_embedding
        )
        
        print(f"(新說話者) 已建立新說話者 '{new_speaker_name}' (ID: {speaker_id}) 和對應聲紋 (ID: {voiceprint_id})")
        return speaker_id, voiceprint_id
        
    except Exception as e:
        print(f"建立新說話者時發生錯誤: {str(e)}")
        return "", ""

def update_speaker_last_active(client: Any, speaker_id: str) -> None:
    """
    更新說話者的最後活動時間
    
    Args:
        client: Weaviate 客戶端實例
        speaker_id: 說話者ID
    """
    try:
        speaker_collection = client.collections.get(Config.COLLECTION_SPEAKER)
        speaker_collection.data.update(
            uuid=speaker_id,
            properties={
                "last_active_time": format_date_rfc3339()
            }
        )
    except Exception as e:
        print(f"更新說話者活動時間時發生錯誤: {str(e)}")

def generate_unique_speaker_name(client: Any) -> str:
    """
    根據現有資料生成新的唯一說話者名稱 (例如 n1, n2, ...)
    
    Args:
        client: Weaviate 客戶端實例
        
    Returns:
        str: 新的唯一說話者名稱
    """
    try:
        # 獲取說話者集合
        speaker_collection = client.collections.get(Config.COLLECTION_SPEAKER)
        
        # 查詢現有的說話者名稱
        results = speaker_collection.query.fetch_objects(
            limit=100,  # 假設不會超過100個說話者
            include_vector=False
        )
        
        # 從現有名稱中提取數字部分
        numbers = []
        pattern = re.compile(r'n(\d+)')
        
        for obj in results.objects:
            name = obj.properties.get('name', '')
            match = pattern.match(name)
            if match:
                numbers.append(int(match.group(1)))
        
        # 產生新的編號
        next_number = max(numbers) + 1 if numbers else 1
        return f"n{next_number}"
        
    except Exception as e:
        print(f"生成唯一說話者名稱時發生錯誤: {str(e)}")
        # 出現錯誤時使用時間戳作為名稱的一部分，確保唯一性
        timestamp = int(datetime.now().timestamp())
        return f"n{timestamp}"

def get_speaker_name(client: Any, speaker_id: str) -> str:
    """
    根據說話者ID獲取說話者名稱
    
    Args:
        client: Weaviate 客戶端實例
        speaker_id: 說話者ID
        
    Returns:
        str: 說話者名稱
    """
    try:
        speaker_collection = client.collections.get(Config.COLLECTION_SPEAKER)
        result = speaker_collection.query.fetch_object_by_id(
            speaker_id,
            include_vector=False
        )
        
        if result:
            return result.properties.get('name', 'unknown')
        return 'unknown'
        
    except Exception as e:
        print(f"獲取說話者名稱時發生錯誤: {str(e)}")
        return 'unknown'

def process_audio_embedding(client: Any, new_embedding: np.ndarray, source_info: str = "直接處理") -> Optional[Tuple[str, str, float]]:
    """
    處理聲紋嵌入向量：比對資料庫，決定更新或建立新說話者
    
    Args:
        client: Weaviate 客戶端實例
        new_embedding: 聲紋嵌入向量
        source_info: 資料來源描述，用於日誌顯示
        
    Returns:
        Optional[Tuple[str, str, float]]: (說話者ID, 說話者名稱, 餘弦距離) 或 None (如果處理失敗)
    """
    try:
        print(f"\n處理來源: {source_info}")
        
        # 在資料庫中尋找最相似的聲紋
        speaker_id, voiceprint_id, best_distance, all_matches = find_closest_speaker(client, new_embedding)
        
        # 最佳匹配的聲紋
        best_match = next((m for m in all_matches if m['voiceprint_id'] == voiceprint_id), None)
        speaker_name = best_match['speaker_name'] if best_match else 'unknown'
        
        # 根據距離決定操作
        if best_distance < Config.THRESHOLD_LOW:
            # 過於相似，不進行更新
            if speaker_id:
                print(f"(跳過) 嵌入向量過於相似 (距離 = {best_distance:.4f})，不進行更新。")
                print(f"該音訊與說話者 '{speaker_name}' 的聲紋相同。")
                # 在非關鍵路徑上非同步更新最後活動時間
                # 這裡只做輕量級的記錄，不影響返回結果
                try:
                    update_speaker_last_active(client, speaker_id)
                except Exception as e:
                    print(f"注意: 更新活動時間時發生錯誤: {str(e)}")
                return speaker_id, speaker_name, best_distance
            else:
                print(f"(跳過) 嵌入向量過於相似 (距離 = {best_distance:.4f})，但找不到對應的說話者資訊。")
                return None
        
        elif best_distance < Config.THRESHOLD_UPDATE:
            # 相似度適中，更新現有聲紋
            if voiceprint_id and speaker_id:
                # 非同步更新聲紋，但不等待結果
                try:
                    update_voiceprint(client, voiceprint_id, new_embedding, speaker_name)
                    # 後台更新最後活動時間
                    update_speaker_last_active(client, speaker_id)
                except Exception as e:
                    print(f"注意: 更新聲紋時發生錯誤: {str(e)}")
                
                print(f"該音訊與說話者 '{speaker_name}' 相符。")
                return speaker_id, speaker_name, best_distance
            else:
                print(f"無法更新：找不到有效的聲紋ID或說話者ID")
                return None
        
        elif best_distance < Config.THRESHOLD_NEW:
            # 相似但不足以更新，只顯示匹配信息
            if speaker_id:
                print(f"(匹配) 該音訊與說話者 '{speaker_name}' 的聲紋相似 (距離 = {best_distance:.4f})，但未進行更新。")
                # 後台更新最後活動時間
                try:
                    update_speaker_last_active(client, speaker_id)
                except Exception as e:
                    print(f"注意: 更新活動時間時發生錯誤: {str(e)}")
                return speaker_id, speaker_name, best_distance
            else:
                print(f"(匹配) 找到相似聲紋 (距離 = {best_distance:.4f})，但找不到對應的說話者資訊。")
                return None
        
        else:
            # 建立新的說話者和聲紋
            new_speaker_id, _ = create_new_speaker(client, new_embedding)
            if new_speaker_id:
                # 直接使用上面獲取的結果
                new_speaker_name = f"n{int(datetime.now().timestamp())}" if not speaker_name else speaker_name
                print(f"識別為新說話者: '{new_speaker_name}'")
                return new_speaker_id, new_speaker_name, float('inf')
            return None
    
    except Exception as e:
        print(f"處理嵌入向量時發生錯誤: {str(e)}")
        return None

def process_audio_file(client: Any, audio_file: str) -> Optional[Tuple[str, str, float]]:
    """
    處理單個音檔：提取嵌入向量，比對資料庫，決定更新或建立新說話者
    
    Args:
        client: Weaviate 客戶端實例
        audio_file: 音檔路徑
        
    Returns:
        Optional[Tuple[str, str, float]]: (說話者ID, 說話者名稱, 餘弦距離) 或 None (如果處理失敗)
    """
    try:
        if not os.path.exists(audio_file):
            print(f"音檔 {audio_file} 不存在，取消處理。")
            sys.exit(ExitCode.FILE_NOT_FOUND)
        
        # 提取嵌入向量
        new_embedding = extract_embedding_from_file(audio_file)
        
        # 處理嵌入向量
        return process_audio_embedding(client, new_embedding, f"音檔 {audio_file}")
    
    except Exception as e:
        print(f"處理音檔 {audio_file} 時發生錯誤: {str(e)}")
        sys.exit(ExitCode.AUDIO_PROCESSING_ERROR)

def process_audio_bytes(client: Any, audio_bytes: Union[bytes, BinaryIO], sample_rate: int, 
                        format_hint: Optional[str] = None, source_info: str = "音訊數據") -> Optional[Tuple[str, str, float]]:
    """
    處理音訊二進制數據：提取嵌入向量，比對資料庫，決定更新或建立新說話者
    
    Args:
        client: Weaviate 客戶端實例
        audio_bytes: 音訊二進制數據或二進制流
        sample_rate: 音訊的採樣率 (Hz)
        format_hint: 音訊格式提示 (如 'WAV', 'MP3'，可選)
        source_info: 資料來源描述，用於日誌顯示
        
    Returns:
        Optional[Tuple[str, str, float]]: (說話者ID, 說話者名稱, 餘弦距離) 或 None (如果處理失敗)
    """
    try:
        # 從二進制數據提取嵌入向量
        new_embedding = extract_embedding_from_bytes(audio_bytes, sample_rate, format_hint)
        
        # 處理嵌入向量
        return process_audio_embedding(client, new_embedding, source_info)
    
    except Exception as e:
        print(f"處理音訊二進制數據時發生錯誤: {str(e)}")
        sys.exit(ExitCode.AUDIO_PROCESSING_ERROR)

def process_audio_numpy(client: Any, signal: np.ndarray, sample_rate: int, 
                       source_info: str = "numpy 音訊陣列") -> Optional[Tuple[str, str, float]]:
    """
    直接處理 numpy 陣列格式的音訊數據
    
    Args:
        client: Weaviate 客戶端實例
        signal: 音訊信號 numpy 陣列
        sample_rate: 音訊的採樣率 (Hz)
        source_info: 資料來源描述，用於日誌顯示
        
    Returns:
        Optional[Tuple[str, str, float]]: (說話者ID, 說話者名稱, 餘弦距離) 或 None (如果處理失敗)
    """
    try:
        # 處理音訊數據並提取嵌入向量
        new_embedding = process_audio_data(signal, sample_rate)
        
        # 處理嵌入向量
        return process_audio_embedding(client, new_embedding, source_info)
    
    except Exception as e:
        print(f"處理 numpy 音訊數據時發生錯誤: {str(e)}")
        sys.exit(ExitCode.AUDIO_PROCESSING_ERROR)

def process_audio_directory(client: Any, directory: str) -> List[Tuple[str, str, str, float]]:
    """
    處理指定資料夾內所有支援的音檔
    
    Args:
        client: Weaviate 客戶端實例
        directory: 音檔資料夾路徑
        
    Returns:
        List[Tuple[str, str, str, float]]: 成功處理的音檔列表，每個項目為 (檔案路徑, 說話者ID, 說話者名稱, 餘弦距離)
    """
    results = []
    
    if not os.path.exists(directory):
        print(f"資料夾 {directory} 不存在，取消處理。")
        sys.exit(ExitCode.FILE_NOT_FOUND)
    
    # 支援更多音訊格式
    audio_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg"))]
    
    if not audio_files:
        print(f"資料夾 {directory} 中沒有支援的音檔。")
        return results
    
    print(f"發現 {len(audio_files)} 個音檔於 {directory}，開始處理...")
    
    for audio_file in audio_files:
        try:
            result = process_audio_file(client, audio_file)
            if result:
                speaker_id, speaker_name, distance = result
                results.append((audio_file, speaker_id, speaker_name, distance))
        except Exception as e:
            print(f"處理 {audio_file} 時發生錯誤: {str(e)}")
    
    print(f"\n完成處理資料夾 {directory} 中所有音檔。")
    return results

def main() -> None:
    """
    主程式入口點：連接 Weaviate、檢查集合、處理命令列參數
    """
    print("======= 語音識別系統 (Weaviate 版本) =======")
    print(f"執行時間: {format_date_rfc3339()}")
    
    try:
        # 連接到 Weaviate
        client = connect_to_weaviate()
        
        try:
            # 檢查必要的集合是否存在
            if not check_collections_exist(client):
                print("請先建立必要的 Weaviate 集合，可以執行 weaviate_study/create_collections.py")
                sys.exit(ExitCode.COLLECTION_NOT_EXIST)
            
            # 處理命令列參數
            if len(sys.argv) > 1:
                path = sys.argv[1]
                if os.path.isfile(path):
                    print("處理單一音檔模式")
                    result = process_audio_file(client, path)
                    if result:
                        speaker_id, speaker_name, distance = result
                        print(f"\n識別結果:")
                        print(f"說話者: {speaker_name}")
                        print(f"ID: {speaker_id}")
                        print(f"相似度距離: {distance:.4f}")
                elif os.path.isdir(path):
                    print("處理資料夾模式")
                    results = process_audio_directory(client, path)
                    if results:
                        print(f"\n識別結果摘要:")
                        for file_path, speaker_id, speaker_name, distance in results:
                            print(f"檔案: {os.path.basename(file_path)}, 說話者: {speaker_name}, 距離: {distance:.4f}")
                else:
                    print(f"無效的路徑: {path}")
                    sys.exit(ExitCode.PARAMETER_ERROR)
            else:
                print("使用方式: python main_identify_v4_weaviate.py <音檔路徑或資料夾路徑>")
                print("範例:")
                print("  - 處理單一檔案: python main_identify_v4_weaviate.py testFiles/audioFile/1-0.wav")
                print("  - 處理整個資料夾: python main_identify_v4_weaviate.py testFiles/audioFile")
                print("未提供參數，退出程式")
                sys.exit(ExitCode.PARAMETER_ERROR)
        
        finally:
            # 關閉 Weaviate 連接
            client.close()
            print("Weaviate 連接已關閉")
    
    except Exception as e:
        print(f"執行過程中發生錯誤: {str(e)}")
        sys.exit(ExitCode.UNKNOWN_ERROR)
    
    print("======= 程式執行結束 =======")
    sys.exit(ExitCode.SUCCESS)

if __name__ == "__main__":
    main() 