# 一開始測試的檔案，算歐幾里得距離的

import os
import numpy as np
import torchaudio
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from speechbrain.inference import SpeakerRecognition
import warnings

# 隱藏執行過程中的警告
warnings.filterwarnings("ignore")

# 初始化 ECAPA-TDNN 模型
def load_model():
    """加載 SpeechBrain 的 ECAPA-TDNN 模型"""
    try:
        model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/speechbrain_recognition"
        )
        print("SpeechBrain 模型加載成功！")
        return model
    except ImportError:
        print("SpeechBrain 未正確安裝，請運行 pip install speechbrain")
        exit()

# 提取音訊的前 10 秒
def preprocess_audio(audio_path):
    """加載並處理音訊檔案，返回前 10 秒的信號"""
    signal, fs = torchaudio.load(audio_path)
    if fs != 16000:
        signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(signal)
    # 截取前 10 秒的信號
    signal = signal[:, :16000 * 10]  # 10 秒對應 16000 Hz * 10
    return signal

# 提取嵌入向量並正則化
def extract_embedding(model, audio_path):
    """提取音訊嵌入向量並正則化"""
    signal = preprocess_audio(audio_path)
    embedding = model.encode_batch(signal).squeeze().numpy()
    normalized_embedding = embedding / norm(embedding)  # 正則化處理
    return normalized_embedding

# 讀取已儲存的嵌入向量
def load_saved_embeddings(embedding_dir):
    """從指定目錄讀取所有已儲存的嵌入向量"""
    embeddings = []
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
    for filename in os.listdir(embedding_dir):
        if filename.endswith(".npy"):
            saved_vector = np.load(os.path.join(embedding_dir, filename))
            embeddings.append((filename, saved_vector))
    return embeddings

# 比對嵌入向量
def compare_embedding(new_embedding, embeddings, threshold):
    """比對新嵌入向量與已儲存的嵌入向量"""
    for filename, saved_embedding in embeddings:
        distance = norm(new_embedding - saved_embedding)  # 計算歐幾里得距離
        print(f"函式比對-Match found: {filename}, Distance: {distance:.4f}")
        if distance <= threshold:
            return filename, distance
    return None, None

# 儲存新的嵌入向量
def save_new_embedding(new_embedding, embedding_dir, embeddings):
    """為新語者生成嵌入檔案並儲存"""
    new_id = len(embeddings) + 1
    save_path = os.path.join(embedding_dir, f"{new_id}.npy")
    np.save(save_path, new_embedding)
    print(f"New embedding saved as {save_path}")

# 主邏輯
def main(new_audio_file, embedding_dir, threshold):
    """處理新的音訊檔案，並進行語者比對"""
    print(f"Processing new audio file: {new_audio_file}")

    # 加載模型
    model = load_model()

    # 提取新音檔的嵌入向量
    new_embedding = extract_embedding(model, new_audio_file)

    # 讀取已儲存的嵌入向量
    embeddings = load_saved_embeddings(embedding_dir)

    # 比對新嵌入向量
    match_file, distance = compare_embedding(new_embedding, embeddings, threshold)
    if match_file:
        print(f"Match found: {match_file}, Distance: {distance:.4f}")
    else:
        print("No match found. Adding new speaker to the database...")
        save_new_embedding(new_embedding, embedding_dir, embeddings)

# 範例執行
if __name__ == "__main__":
    new_audio_file = "audioFile/1-4.wav"  # 新的音檔路徑
    embedding_dir = "vectorFile"  # 嵌入向量存放資料夾
    threshold = 0.8  # 歐幾里得距離閾值
    main(new_audio_file, embedding_dir, threshold)
