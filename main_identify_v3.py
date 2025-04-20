import os
import re
import sys
import numpy as np
import torch
import soundfile as sf  # torchaudio 套件需要
from scipy.spatial.distance import cosine
from scipy.signal import resample_poly
import warnings
import logging
import sync_npy_username  # 用來呼叫 ffmpeg 或檢查更新

# 隱藏多餘的警告與日誌
warnings.filterwarnings("ignore")
logging.getLogger("speechbrain").setLevel(logging.ERROR)

# 自定義 Tee 類別：同時輸出到螢幕和 log 檔案
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

# 將標準輸出重定向到 log 檔
sys.stdout = Tee("output_log.txt")

# 載入 SpeechBrain 語音辨識模型
from speechbrain.inference import SpeakerRecognition

# 全域參數設定
EMBEDDING_DIR = "embeddingFiles"  # 所有說話者嵌入資料的根目錄
THRESHOLD_LOW = 0.2     # 過於相似，不更新
THRESHOLD_UPDATE = 0.34 # 更新嵌入向量
THRESHOLD_NEW = 0.36    # 判定為新說話者

# 在模組載入時只初始化一次 embedding 模型
try:
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/speechbrain_recognition"
    )
    print("SpeechBrain 模型加載成功！")
except ImportError:
    print("SpeechBrain 未正確安裝，請運行: pip install speechbrain")
    exit()

def resample_audio(signal, orig_sr, target_sr):
    """使用 scipy 進行高品質重新採樣"""
    return resample_poly(signal, target_sr, orig_sr)

def extract_embedding(audio_path: str) -> np.ndarray:
    """
    提取音檔的嵌入向量，根據音檔取樣率智能處理
    
    Args:
        audio_path: 音檔路徑
        
    Returns:
        np.ndarray: 音檔的嵌入向量
        
    處理流程:
        1. 若音檔為 16kHz，則直接使用
        2. 若音檔為 8kHz，則直接升頻到 16kHz
        3. 若音檔取樣率高於 16kHz，則降頻到 16kHz
        4. 其他取樣率，則重新採樣到 16kHz
    """
    try:
        signal, sr = sf.read(audio_path)
        
        # 處理立體聲轉單聲道
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
            
        # 根據取樣率處理
        if sr == 16000:
            # 已是 16kHz，直接使用
            signal_16k = signal
        elif sr == 8000:
            # 若為 8kHz，直接升頻到 16kHz
            signal_16k = resample_audio(signal, 8000, 16000)
        elif sr > 16000:
            # 若高於 16kHz，直接降頻到 16kHz
            signal_16k = resample_audio(signal, sr, 16000)
        else:
            # 其他取樣率，重新採樣到 16kHz
            signal_16k = resample_audio(signal, sr, 16000)
        
        # 轉換為 PyTorch 張量
        signal_16k = torch.tensor(signal_16k, dtype=torch.float32).unsqueeze(0)
        
        # 限制音檔長度（最多 10 秒）
        max_length = 16000 * 10
        if signal_16k.shape[1] > max_length:
            signal_16k = signal_16k[:, :max_length]
            
        # 提取嵌入向量
        embedding = model.encode_batch(signal_16k).squeeze().numpy()
        return embedding
        
    except Exception as e:
        print(f"提取嵌入向量時發生錯誤: {e}")
        raise

def list_all_embedding_files():
    """遍歷 EMBEDDING_DIR 下所有子資料夾，返回所有嵌入檔案資訊"""
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR)
    embedding_files = []
    for folder in os.listdir(EMBEDDING_DIR):
        folder_path = os.path.join(EMBEDDING_DIR, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".npy"):
                    full_path = os.path.join(folder_path, file)
                    embedding_files.append((folder, file, full_path))
    return embedding_files

def compare_all_embeddings(new_embedding):
    """計算新嵌入向量與所有現有嵌入的餘弦距離"""
    embedding_files = list_all_embedding_files()
    if not embedding_files:
        print("尚無任何已存的嵌入檔案。")
        return None, float('inf'), []
    distances = []
    for folder, file, full_path in embedding_files:
        saved_embedding = np.load(full_path)
        distance = cosine(new_embedding, saved_embedding)
        distances.append((folder, file, distance))
        print(f"比對 - 說話者資料夾: {folder}, 檔案: {file}, 餘弦距離: {distance:.4f}")
    best_match = min(distances, key=lambda x: x[2])
    return best_match, best_match[2], distances

def update_embedding(old_embedding, new_embedding, update_count):
    """使用加權移動平均更新嵌入向量"""
    updated_embedding = (old_embedding * update_count + new_embedding) / (update_count + 1)
    return updated_embedding, update_count + 1

def parse_file_info(file_name):
    """從檔案名稱中解析出用戶名、檔案編號與更新次數"""
    m = re.match(r"^(.*?)_(\d+)_(\d+)\.npy$", file_name)
    if m:
        speaker = m.group(1)
        file_id = int(m.group(2))
        update_count = int(m.group(3))
        return speaker, file_id, update_count
    else:
        speaker = file_name.split("_")[0]
        return speaker, 1, 1

def build_file_name(speaker, file_id, update_count):
    """組合檔案名稱：<用戶名>_<檔案編號>_<更新次數>.npy"""
    return f"{speaker}_{file_id}_{update_count}.npy"

def update_speaker_embedding(speaker_folder, file_name, new_embedding):
    """更新指定說話者資料夾中某個 npy 檔案的嵌入向量"""
    full_path = os.path.join(EMBEDDING_DIR, speaker_folder, file_name)
    old_embedding = np.load(full_path)
    speaker, file_id, update_count = parse_file_info(file_name)
    updated_embedding, new_update_count = update_embedding(old_embedding, new_embedding, update_count)
    new_file_name = build_file_name(speaker, file_id, new_update_count)
    new_full_path = os.path.join(EMBEDDING_DIR, speaker_folder, new_file_name)
    np.save(new_full_path, updated_embedding)
    os.remove(full_path)
    print(f"(更新) 在 {speaker_folder} 資料夾中, 原檔案 {file_name} 更新為 {new_file_name} (更新次數: {new_update_count})")
    return new_file_name

def handle_new_speaker(new_embedding):
    """處理新說話者：建立新的資料夾並儲存嵌入向量"""
    new_speaker_folder = generate_unique_speaker_folder()
    new_file_name = build_file_name(new_speaker_folder, 1, 1)
    new_full_path = os.path.join(EMBEDDING_DIR, new_speaker_folder, new_file_name)
    np.save(new_full_path, new_embedding)
    print(f"(新說話者) 建立新資料夾 {new_speaker_folder} 並儲存檔案: {new_file_name}")
    return new_speaker_folder, new_file_name

def match_speaker(speaker_folder, file_name, best_distance):
    """處理比對到的說話者，但未進行更新"""
    print(f"(匹配) 該音檔與 {speaker_folder} 資料夾中的 {file_name} 相似 (距離 = {best_distance:.4f})，但未進行更新。")

def generate_unique_speaker_folder():
    """根據現有資料夾生成新的唯一說話者資料夾名稱 (例如 n1, n2, ...)"""
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR)
    existing_folders = [d for d in os.listdir(EMBEDDING_DIR) if os.path.isdir(os.path.join(EMBEDDING_DIR, d))]
    numbers = []
    for folder in existing_folders:
        m = re.match(r'n(\d+)', folder)
        if m:
            numbers.append(int(m.group(1)))
    next_number = max(numbers) + 1 if numbers else 1
    new_folder = f"n{next_number}"
    os.makedirs(os.path.join(EMBEDDING_DIR, new_folder))
    return new_folder

def process_audio_file(audio_file):
    """
    主流程：提取音檔嵌入，與現有嵌入比對，
    根據餘弦距離決定：更新、匹配或建立新說話者
    """
    print(f"\n處理音檔: {audio_file}")
    if not os.path.exists(audio_file):
        print(f"音檔 {audio_file} 不存在，取消處理。")
        return
    new_embedding = extract_embedding(audio_file)
    best_match, best_distance, _ = compare_all_embeddings(new_embedding)
    if best_distance < THRESHOLD_LOW:
        speaker_folder = best_match[0]
        print(f"(跳過) 嵌入向量過於相似 (距離 = {best_distance:.4f})，不進行更新。")
        print(f"該音檔與說話者 {speaker_folder} 的檔案相同。")
    elif best_distance < THRESHOLD_UPDATE:
        speaker_folder, file_name, _ = best_match
        update_speaker_embedding(speaker_folder, file_name, new_embedding)
        print(f"該音檔與說話者 {speaker_folder} 相符，且已更新嵌入檔案。")
    elif best_distance < THRESHOLD_NEW:
        speaker_folder, file_name, _ = best_match
        match_speaker(speaker_folder, file_name, best_distance)
    else:
        handle_new_speaker(new_embedding)

def process_audio_directory(directory):
    """處理指定資料夾內所有 .wav 檔案"""
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
    # 首次執行時先檢查更新 npy 檔用戶名
    sync_npy_username.sync_all_folders()
    
    # 主程式執行: 若要處理單一檔案或資料夾，可解除下列註解
    process_audio_file("path_to_audio.wav")
    # process_audio_directory("path_to_directory")
