# 1. 用音檔產生Embeding的npy檔

import os
import numpy as np
import torch
import torchaudio
from numpy.linalg import norm
from speechbrain.pretrained import SpeakerRecognition
import warnings

# 隱藏執行過程中的警告
warnings.filterwarnings("ignore")

# SpeechBrain ECAPA-TDNN 模型 (用於提取聲紋嵌入)
try:
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="models/speechbrain_recognition"
    )
    print("SpeechBrain 模型加載成功！")
except ImportError:
    print("SpeechBrain 未正確安裝，請運行: pip install speechbrain")
    exit()

# 全域參數
EMBEDDING_DIR = "vectorFile"

def extract_embedding(audio_path, normalize=False):
    """
    提取音檔的嵌入向量，並選擇是否進行正則化 (L2-normalization)。
    
    :param audio_path: 音檔的路徑
    :param normalize: 是否進行向量正則化
    :return: 嵌入向量 (numpy array)
    """
    signal, fs = torchaudio.load(audio_path)
    
    # 取前 10 秒
    max_length = fs * 10
    signal = signal[:, :max_length] if signal.shape[1] > max_length else signal
    
    # 單聲道
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    
    # 若非 16kHz，重采樣
    if fs != 16000:
        signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(signal)
    
    # 提取嵌入
    embedding = model.encode_batch(signal).squeeze().numpy()
    
    # (選擇性) L2-normalize
    if normalize:
        embedding = embedding / norm(embedding)
    
    return embedding

def save_embedding(embedding):
    """
    將輸入的 embedding 存檔；檔名根據目前資料夾內的檔案數量來決定。
    """
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR)

    existing_files = [f for f in os.listdir(EMBEDDING_DIR) if f.endswith(".npy")]
    new_id = len(existing_files) + 1
    save_path = os.path.join(EMBEDDING_DIR, f"{new_id}.npy")
    
    np.save(save_path, embedding)
    print(f"New embedding saved as {save_path}")

def main(audio_file):
    """
    1. 從指定的音檔讀取並提取 embedding
    2. 存檔到 vectorFile 資料夾
    """
    print(f"Extracting embedding from: {audio_file}")
    embedding = extract_embedding(audio_file)
    save_embedding(embedding)

if __name__ == "__main__":
    # 你要處理的音檔 (可自行修改)
    new_audio_file = "audioFile/2-2.wav"
    main(new_audio_file)
