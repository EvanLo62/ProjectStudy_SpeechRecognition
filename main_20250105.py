import os
import random
import string
import numpy as np
import torch
import torchaudio
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from speechbrain.pretrained import SpeakerRecognition
import warnings

# 忽略不必要的警告訊息，保持輸出整潔
warnings.filterwarnings("ignore")

# 嘗試載入 SpeechBrain 語音辨識模型
try:
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",  # SpeechBrain 預訓練模型的來源
        savedir="models/speechbrain_recognition"     # 模型儲存的路徑
    )
    print("SpeechBrain 模型加載成功！")
except ImportError:
    print("SpeechBrain 未正確安裝，請運行: pip install speechbrain")
    exit()

# 嵌入向量存放的目錄
EMBEDDING_DIR = "vectorFile"
# 設定距離閾值
THRESHOLD_LOW = 0.2  # 過於相似的閾值，小於此值的嵌入向量不更新
THRESHOLD_UPDATE = 0.4  # 進行更新的最大距離閾值
THRESHOLD_NEW = 0.55  # 超過此距離，視為新說話者


def extract_embedding(audio_path, normalize=False):
    """
    從音檔中提取語音嵌入向量 (embedding)。
    - 限制音檔的最大長度為 10 秒。
    - 將音訊轉換為單聲道，並以 16kHz 取樣。

    參數:
    audio_path (str): 音檔的路徑
    normalize (bool): 是否對向量進行歸一化

    回傳:
    np.ndarray: 提取的語音嵌入向量
    """
    # 加載音訊檔案，返回信號和取樣率
    signal, fs = torchaudio.load(audio_path)

    # 限制音訊檔案的最大長度（10 秒）
    max_length = fs * 10
    signal = signal[:, :max_length] if signal.shape[1] > max_length else signal

    # 如果是多聲道音訊，將其轉換為單聲道
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    # 如果取樣率不是 16kHz，進行重取樣
    if fs != 16000:
        signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(signal)

    # 使用 SpeechBrain 模型提取嵌入向量
    embedding = model.encode_batch(signal).squeeze().numpy()

    # 如果需要，對嵌入向量進行歸一化
    if normalize:
        embedding = embedding / norm(embedding)

    return embedding


def compare_all_npy(new_embedding):
    """
    將新嵌入向量與目錄中的所有現有 `.npy` 檔案進行比較。
    - 計算每個檔案的餘弦距離，並返回距離結果。
    - 測試用功能：顯示每個檔案與新嵌入的距離。

    參數:
    new_embedding (np.ndarray): 新提取的語音嵌入向量

    回傳:
    str: 與新嵌入最相近的檔案名稱
    float: 最小的餘弦距離
    list: 所有檔案的距離列表
    """
    # 確保嵌入向量目錄存在
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR)
    
    # 搜索目錄中的 .npy 檔案
    npy_files = [f for f in os.listdir(EMBEDDING_DIR) if f.endswith(".npy")]
    if not npy_files:
        print("No existing .npy files found in embedding directory.")
        return None, float('inf'), []

    # 記錄所有距離的列表
    distances = []
    print("=== Comparing with existing embeddings ===")  # 測試用，顯示所有比較結果
    for npy_file in npy_files:
        file_path = os.path.join(EMBEDDING_DIR, npy_file)
        saved_embedding = np.load(file_path)  # 加載保存的嵌入向量

        # 計算新嵌入與現有嵌入的餘弦距離
        distance = cosine(new_embedding, saved_embedding)
        distances.append((npy_file, distance))
        print(f"File: {npy_file}, Distance: {distance:.4f}")  # 測試用，逐一顯示距離
    
    # 找到距離最小的檔案
    best_match = min(distances, key=lambda x: x[1])
    return best_match[0], best_match[1], distances


def update_embedding(old_embedding, new_embedding, n_samples):
    """
    使用加權移動平均更新嵌入向量。
    - 更新過程中考慮已存在樣本的數量，對新向量進行加權。

    參數:
    old_embedding (np.ndarray): 現有的嵌入向量
    new_embedding (np.ndarray): 新的嵌入向量
    n_samples (int): 已經累積的樣本數

    回傳:
    np.ndarray: 更新後的嵌入向量
    int: 更新後的樣本數量
    """
    updated_embedding = (old_embedding * n_samples + new_embedding) / (n_samples + 1)
    return updated_embedding, n_samples + 1

def generate_unique_filename():
    """
    生成新人的檔名
    隨機生成三位字母組合並確保唯一性
    """
    while True:
        # 隨機生成兩個大寫字母
        random_name = ''.join(random.choices(string.ascii_uppercase, k=3))
        # 檢查是否已經有這個名稱的檔案
        existing_files = [f for f in os.listdir(EMBEDDING_DIR) if f.startswith(f"n{random_name}")]
        if not existing_files:
            return f"n{random_name}"


def process_audio_file(audio_file):
    """
    主邏輯函式，處理導入音檔並進行判斷。
    - 提取嵌入向量並與現有的 `.npy` 比較。
    - 判斷是否更新嵌入向量或創建新的說話者資料。

    參數:
    audio_file (str): 音檔的路徑
    """
    print(f"\nProcessing file: {audio_file}")  # 測試用，顯示導入的音檔名稱
    new_embedding = extract_embedding(audio_file)
    
    # 比較新嵌入與現有嵌入向量
    match_file, best_distance, all_distances = compare_all_npy(new_embedding)
    
    if best_distance < THRESHOLD_LOW:
        # 如果距離小於低閾值，視為過於相似，跳過更新
        print(f"Embedding too similar (distance = {best_distance:.4f}), skipping update.")
    elif best_distance < THRESHOLD_UPDATE:
        # 距離在合理範圍內，進行嵌入更新
        old_embedding = np.load(os.path.join(EMBEDDING_DIR, match_file))
        n_samples = int(match_file.split("_")[-1].split(".")[0])  # 提取樣本數量
        updated_embedding, updated_samples = update_embedding(old_embedding, new_embedding, n_samples)
        
        # 保存更新後的嵌入向量，並刪除舊檔案
        new_filename = f"{match_file.split('_')[0]}_{updated_samples}.npy"
        np.save(os.path.join(EMBEDDING_DIR, new_filename), updated_embedding)
        os.remove(os.path.join(EMBEDDING_DIR, match_file))
        print(f"Updated: {match_file} -> {new_filename}")
    elif best_distance < THRESHOLD_NEW:
        # 距離接近匹配但不更新
        print(f"Matched with {match_file}, but no update performed (distance = {best_distance:.4f}).")
    else:
        # 如果距離超過新說話者閾值，新增新檔案
        new_filename = f"{generate_unique_filename()}_1.npy"
        np.save(os.path.join(EMBEDDING_DIR, new_filename), new_embedding)
        print(f"New speaker detected, saved as: {new_filename}")


if __name__ == "__main__":
    test_audio_file = "audioFile/4-0.wav"  # 測試用音檔路徑
    process_audio_file(test_audio_file)
