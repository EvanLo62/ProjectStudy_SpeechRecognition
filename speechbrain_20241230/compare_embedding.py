# 2. 比對音檔與現有npy檔的餘弦距離

import os
import numpy as np
import torch
import torchaudio
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from speechbrain.pretrained import SpeakerRecognition
import warnings

warnings.filterwarnings("ignore")

try:
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="models/speechbrain_recognition"
    )
    print("SpeechBrain 模型加載成功！")
except ImportError:
    print("SpeechBrain 未正確安裝，請運行: pip install speechbrain")
    exit()

EMBEDDING_DIR = "vectorFile"
THRESHOLD = 0.4  # 餘弦距離門檻 (可自行調整)

def extract_embedding(audio_path, normalize=False):
    """
    與前面相同的函式
    """
    signal, fs = torchaudio.load(audio_path)
    
    max_length = fs * 10
    signal = signal[:, :max_length] if signal.shape[1] > max_length else signal
    
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    
    if fs != 16000:
        signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(signal)
    
    embedding = model.encode_batch(signal).squeeze().numpy()
    
    if normalize:
        embedding = embedding / norm(embedding)
    
    return embedding

def compare_all_npy(new_embedding, threshold=THRESHOLD):
    """
    逐一讀取 EMBEDDING_DIR 內的 .npy 檔案，
    計算與 new_embedding 的餘弦距離 (cosine distance)，
    並列印每一個的結果。
    最後回傳「最低距離」及對應的檔名 (若低於門檻)。
    """
    if not os.path.exists(EMBEDDING_DIR):
        print(f"Directory '{EMBEDDING_DIR}' does not exist. No embeddings to compare.")
        return None, None
    
    npy_files = [f for f in os.listdir(EMBEDDING_DIR) if f.endswith(".npy")]
    if not npy_files:
        print("No .npy files found in embedding directory.")
        return None, None

    best_distance = float("inf")
    best_match_file = None

    print("=== Compare with all existing .npy files ===")
    for npy_file in npy_files:
        file_path = os.path.join(EMBEDDING_DIR, npy_file)
        saved_embedding = np.load(file_path)
        
        distance = cosine(new_embedding, saved_embedding)
        print(f"{npy_file}: Cosine distance = {distance:.4f}")
        
        # 更新最小距離
        if distance < best_distance:
            best_distance = distance
            best_match_file = npy_file

    # 判斷最小距離是否小於門檻
    if best_distance <= threshold:
        return best_match_file, best_distance
    else:
        return None, best_distance

def main(audio_file):
    """
    1. 提取新音檔embedding
    2. 與所有 .npy 比對，逐一印出距離
    3. 最後給出最相似檔案(若有)或顯示No match
    """
    print(f"Now comparing new audio: {audio_file}")
    new_embedding = extract_embedding(audio_file)

    match_file, best_distance = compare_all_npy(new_embedding, THRESHOLD)
    if match_file is not None:
        print(f"\n[Result] Best match => {match_file}, distance = {best_distance:.4f} (<= threshold={THRESHOLD})")
    else:
        # 代表全部都沒低於門檻
        print(f"\n[Result] No match found! (best distance={best_distance:.4f} > threshold={THRESHOLD})")

if __name__ == "__main__":
    test_audio_file = "audioFile/1-3.wav"
    main(test_audio_file)
