"""
2025.3.10 此檔案更模組化、增加可閱讀性，為未來增改功能做準備

(用 Scipy 做重新採樣，從 8kHz 升到 16kHz。)
語音辨識 檔案介紹

    使用方式: 
        呼叫 process_audio_file(audio_file) 函式，audio_file 須為 wav 檔，程式會抓前 10 秒聲音進行判斷

    產生結果:
        1. 將新音檔的嵌入向量與所有已保存的說話者嵌入（分散在各自的子資料夾內）進行餘弦距離比對。
        2. 若找出距離足夠近的檔案，視為同一個說話者，並依照距離決定是否進行更新或僅作匹配。
        3. 若新音檔與所有嵌入距離都太遠，則視為新說話者，自動建立一個新的說話者資料夾並存檔。

    檔名與資料夾命名規則 (新版本):
        - 每個說話者有獨立子資料夾，資料夾名稱代表說話者識別碼（預設以 n 開頭，可手動更名）。
        - 檔案命名格式： <用戶名>_<檔案獨立編號>_<更新次數>.npy  
          例如：新說話者預設資料夾命名為 n1，新建立的第一個檔案命名為 n1_1_1.npy，
          更新後會直接覆蓋，檔案獨立編號保持不變，更新次數增加（如 n1_1_2.npy）。
          
    更新與匹配機制:
        - 當新嵌入與現有嵌入距離 < THRESHOLD_LOW，視為過於相似，不作處理。
        - 當 THRESHOLD_LOW <= 距離 < THRESHOLD_UPDATE，利用加權移動平均更新，並直接覆蓋原有 npy 檔（更新次數增加）。
        - 當 THRESHOLD_UPDATE <= 距離 < THRESHOLD_NEW，僅提示匹配，但不更新。
        - 當距離 >= THRESHOLD_NEW，判定為新說話者，建立新資料夾並儲存新檔案。
"""

import os
import random
import re
import string
import sys
import numpy as np
import torch
import torchaudio
import soundfile as sf  # torchaudio 套件需要
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from scipy.signal import resample_poly
import warnings
import logging
import subprocess

import sync_npy_username  # (保留) 用來呼叫 ffmpeg

# 隱藏多餘的警告和日誌
warnings.filterwarnings("ignore")
logging.getLogger("speechbrain").setLevel(logging.ERROR)

# 自定義 Tee 類別，用來同時輸出到螢幕和 log 檔案
class Tee:
    def __init__(self, file_name, mode="w"):
        self.file = open(file_name, mode, encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

# 重定向標準輸出到 log 檔案
sys.stdout = Tee("output_log.txt")

# 載入 SpeechBrain 語音辨識模型
from speechbrain.inference import SpeakerRecognition

# 全域參數設定
EMBEDDING_DIR = "embeddingFiles"  # (全域) 所有說話者嵌入資料的根目錄（內含子資料夾）
THRESHOLD_LOW = 0.2     # (全域) 若距離過近，視為相同且不更新
THRESHOLD_UPDATE = 0.34 # (全域) 若距離介於低閾值與此值之間，則更新嵌入向量（直接覆蓋檔案）
THRESHOLD_NEW = 0.36    # (全域) 超過此距離，判定為新說話者

# 嘗試載入 SpeechBrain 語音辨識模型
try:
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",  # (註解) SpeechBrain 預訓練模型來源
        savedir="models/speechbrain_recognition"       # (註解) 模型存放路徑
    )
    print("SpeechBrain 模型加載成功！")
except ImportError:
    print("SpeechBrain 未正確安裝，請運行: pip install speechbrain")
    exit()

def resample_audio(signal, orig_sr, target_sr):
    """
    使用 scipy.signal.resample_poly 進行高品質重新採樣。
    """
    up = target_sr  # (註解) 設定升頻因子
    down = orig_sr  # (註解) 設定降頻因子
    return resample_poly(signal, up, down)  # (註解) 執行重新採樣

def extract_embedding(audio_path):
    """
    從音檔中提取語音嵌入向量 (embedding)。
    - 限制音檔長度為 10 秒。
    - 將音訊降頻至 8kHz，再升頻至 16kHz，以符合模型要求。
    """
    signal, sr = sf.read(audio_path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)  # (註解) 多聲道轉為單聲道

    # 降頻至 8kHz，再升頻至 16kHz
    signal_8k = resample_audio(signal, sr, 8000)  # (註解) 先降頻
    signal_16k = resample_audio(signal_8k, 8000, 16000)  # (註解) 再升頻
    
    # 轉為 PyTorch 張量，並限制長度為 10 秒
    signal_16k = torch.tensor(signal_16k, dtype=torch.float32).unsqueeze(0)  # (註解) 新增 batch 維度
    max_length = 16000 * 10  # (註解) 10 秒內的取樣數
    signal_16k = signal_16k[:, :max_length] if signal_16k.shape[1] > max_length else signal_16k

    # 使用 SpeechBrain 模型提取嵌入向量
    embedding = model.encode_batch(signal_16k).squeeze().numpy()  # (註解) 移除不必要的維度並轉為 numpy array
    return embedding

def list_all_embedding_files():
    """
    遍歷 EMBEDDING_DIR 內所有子資料夾，返回所有嵌入檔案資訊。
    
    回傳:
        List of tuples: (speaker_folder, file_name, full_path)
    """
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR)  # (註解) 若資料夾不存在則創建
    embedding_files = []
    for folder in os.listdir(EMBEDDING_DIR):
        folder_path = os.path.join(EMBEDDING_DIR, folder)
        if os.path.isdir(folder_path):  # (註解) 僅處理子資料夾
            for file in os.listdir(folder_path):
                if file.endswith(".npy"):
                    full_path = os.path.join(folder_path, file)
                    embedding_files.append((folder, file, full_path))
    return embedding_files

def compare_all_embeddings(new_embedding):
    """
    將新嵌入向量與所有現有說話者資料夾中的嵌入檔進行比較，計算餘弦距離。
    
    回傳:
        best_match: (speaker_folder, file_name, distance) 若無檔案則為 None
        best_distance: 最小的距離值（若無檔案則為 inf）
        distances: 所有比對結果的列表，元素為 (speaker_folder, file_name, distance)
    """
    embedding_files = list_all_embedding_files()
    if not embedding_files:
        print("尚無任何已存的嵌入檔案。")
        return None, float('inf'), []
    
    distances = []
    for folder, file, full_path in embedding_files:
        saved_embedding = np.load(full_path)  # (註解) 載入每個 npy 檔案的嵌入向量
        distance = cosine(new_embedding, saved_embedding)  # (註解) 計算餘弦距離
        distances.append((folder, file, distance))
        print(f"比對 - 說話者資料夾: {folder}, 檔案: {file}, 餘弦距離: {distance:.4f}")
    
    best_match = min(distances, key=lambda x: x[2])
    return best_match, best_match[2], distances

def update_embedding(old_embedding, new_embedding, update_count):
    """
    使用加權移動平均更新嵌入向量。
    公式： updated = (old * update_count + new) / (update_count + 1)
    
    參數:
        old_embedding: 原有嵌入向量
        new_embedding: 新提取的嵌入向量
        update_count: 原有更新次數
    
    回傳:
        updated_embedding: 更新後的嵌入向量
        new_update_count: 更新後的次數 (update_count + 1)
    """
    updated_embedding = (old_embedding * update_count + new_embedding) / (update_count + 1)
    return updated_embedding, update_count + 1

def parse_file_info(file_name):
    """
    (新增) 從檔案名稱中解析出用戶名、檔案獨立編號、更新次數。
    檔案命名格式預期為: <用戶名>_<檔案獨立編號>_<更新次數>.npy
    """
    m = re.match(r"^(.*?)_(\d+)_(\d+)\.npy$", file_name)
    if m:
        speaker = m.group(1)             # 用戶名
        file_id = int(m.group(2))          # 檔案獨立編號
        update_count = int(m.group(3))     # 更新次數
        return speaker, file_id, update_count
    else:
        # (新增註解) 若解析失敗，採預設值
        speaker = file_name.split("_")[0]
        file_id = 1
        update_count = 1
        return speaker, file_id, update_count

def build_file_name(speaker, file_id, update_count):
    """
    (新增) 根據用戶名、檔案獨立編號、更新次數組合成檔案名稱。
    回傳格式: <用戶名>_<檔案獨立編號>_<更新次數>.npy
    """
    return f"{speaker}_{file_id}_{update_count}.npy"

def update_speaker_embedding(speaker_folder, file_name, new_embedding):
    """
    (修改) 更新指定說話者資料夾中某個 npy 檔案的嵌入向量，
    直接覆蓋舊檔版本，僅更新 update_count 參數（檔案獨立編號保持不變）。
    """
    full_path = os.path.join(EMBEDDING_DIR, speaker_folder, file_name)
    old_embedding = np.load(full_path)  # (註解) 載入原有嵌入向量
    # 解析舊檔案資訊：用戶名、檔案獨立編號、更新次數
    speaker, file_id, update_count = parse_file_info(file_name)
    updated_embedding, new_update_count = update_embedding(old_embedding, new_embedding, update_count)
    # 組合新的檔名，檔案獨立編號不變，但更新次數 +1
    new_file_name = build_file_name(speaker, file_id, new_update_count)
    new_full_path = os.path.join(EMBEDDING_DIR, speaker_folder, new_file_name)
    np.save(new_full_path, updated_embedding)  # (註解) 儲存更新後的嵌入向量至新檔案
    os.remove(full_path)  # (註解) 刪除舊檔以達到覆蓋效果
    print(f"(更新) 在 {speaker_folder} 資料夾中, 原檔案 {file_name} 更新為 {new_file_name} (更新次數: {new_update_count})")
    return new_file_name

def handle_new_speaker(new_embedding):
    """
    (修改) 處理新說話者的情形，建立新的說話者資料夾，並將嵌入向量儲存為第一個檔案。
    檔名格式為 <用戶名>_<檔案獨立編號>_<更新次數>.npy，其中檔案獨立編號與更新次數初始皆為 1。
    """
    new_speaker_folder = generate_unique_speaker_folder()
    new_file_name = build_file_name(new_speaker_folder, 1, 1)  # (註解) 新檔案獨立編號與更新次數均設為 1
    new_full_path = os.path.join(EMBEDDING_DIR, new_speaker_folder, new_file_name)
    np.save(new_full_path, new_embedding)
    print(f"(新說話者) 建立新資料夾 {new_speaker_folder} 並儲存檔案: {new_file_name}")
    return new_speaker_folder, new_file_name

def match_speaker(speaker_folder, file_name, best_distance):
    """
    (新增) 處理比對到的說話者，但未進行更新的情形。
    """
    print(f"(匹配) 該音檔與 {speaker_folder} 資料夾中的 {file_name} 相似 (距離 = {best_distance:.4f})，但未進行更新。")

def generate_unique_speaker_folder():
    """
    根據現有資料夾生成新的唯一說話者資料夾名稱，
    採用 "n" 加上自動編號的方式，例如 n1, n2, ...
    """
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR)
    existing_folders = [d for d in os.listdir(EMBEDDING_DIR) if os.path.isdir(os.path.join(EMBEDDING_DIR, d))]
    unknown_numbers = []
    for folder in existing_folders:
        m = re.match(r'n(\d+)', folder)
        if m:
            unknown_numbers.append(int(m.group(1)))
    next_number = max(unknown_numbers) + 1 if unknown_numbers else 1
    new_folder_name = f"n{next_number}"
    os.makedirs(os.path.join(EMBEDDING_DIR, new_folder_name))
    return new_folder_name

def process_audio_file(audio_file):
    """
    主流程函式：
      1. 提取新音檔的嵌入向量
      2. 與現有所有說話者的嵌入檔進行比對
      3. 根據餘弦距離決定：
         - 距離 < THRESHOLD_LOW: 過於相似，不作任何更新
         - THRESHOLD_LOW <= 距離 < THRESHOLD_UPDATE: 利用加權移動平均更新，直接覆蓋原有檔案（更新次數增加）
         - THRESHOLD_UPDATE <= 距離 < THRESHOLD_NEW: 視為匹配，但不更新
         - 距離 >= THRESHOLD_NEW: 判定為新說話者，建立新資料夾並儲存新檔案
    """
    print(f"\n處理音檔: {audio_file}")
    if not os.path.exists(audio_file):
        print(f"音檔 {audio_file} 不存在，取消處理。")
        return

    new_embedding = extract_embedding(audio_file)
    best_match, best_distance, all_distances = compare_all_embeddings(new_embedding)

    if best_distance < THRESHOLD_LOW:
        # (修改) 非常相似，跳過更新
        speaker_folder = best_match[0]
        print(f"(跳過) 嵌入向量過於相似 (距離 = {best_distance:.4f})，不進行更新。")
        print(f"該音檔與說話者 {speaker_folder} 的檔案相同。")
    elif best_distance < THRESHOLD_UPDATE:
        # (修改) 更新情形：呼叫 update_speaker_embedding 以直接覆蓋更新原有檔案
        speaker_folder, file_name, _ = best_match
        update_speaker_embedding(speaker_folder, file_name, new_embedding)
        print(f"該音檔與說話者 {speaker_folder} 相符，且已更新嵌入檔案。")
    elif best_distance < THRESHOLD_NEW:
        # (修改) 匹配但不更新：呼叫 match_speaker 處理
        speaker_folder, file_name, _ = best_match
        match_speaker(speaker_folder, file_name, best_distance)
    else:
        # (修改) 新說話者：呼叫 handle_new_speaker 建立新資料夾並儲存新檔案
        handle_new_speaker(new_embedding)

def process_audio_directory(directory):
    """
    處理指定資料夾內所有 .wav 檔案，逐一進行語音處理。
    """
    if not os.path.exists(directory):
        print(f"資料夾 {directory} 不存在，取消處理。")
        return

    audio_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".wav")]
    if not audio_files:
        print(f"資料夾 {directory} 中沒有 .wav 檔案。")
        return

    print(f"發現 {len(audio_files)} 個音檔於 {directory}，開始處理...")
    for audio_file in audio_files:
        try:
            process_audio_file(audio_file)
        except Exception as e:
            print(f"處理 {audio_file} 時發生錯誤: {e}")
    print(f"\n完成處理資料夾 {directory} 中所有音檔。")

if __name__ == "__main__":
    sync_npy_username.sync_all_folders()   #檢查更新npy檔用戶名

    # 測試單一音檔的處理
    # test_audio_file = "test_audioFile/0363/363-1.wav"  # (測試) 音檔路徑
    # process_audio_file(test_audio_file)
    # print()
    
    # 測試處理整個資料夾（如有需要，解除下列註解）
    # test_directory = "test_audioFile/0009"  # (測試) 資料夾路徑
    # process_audio_directory(test_directory)
    # print()
