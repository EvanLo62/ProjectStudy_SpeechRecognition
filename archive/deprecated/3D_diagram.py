# 顯示3D立體圖，看每個音檔的距離

import os
import numpy as np
import torchaudio
import torch
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # 用於 3D 繪圖
import matplotlib.pyplot as plt
from speechbrain.inference import SpeakerRecognition

# # 載入 .env 檔案
# load_dotenv()
# HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
# if not HUGGINGFACE_TOKEN:
#     raise ValueError("HUGGINGFACE_TOKEN is not found. Make sure .env file exists and contains the token.")

# Step 1: 載入模型
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


# Step 2: 讀取所有嵌入向量檔案
vector_dir = "embeddingFiles"
vector_files = [f for f in os.listdir(vector_dir) if f.endswith(".npy")]

vectors = []
for vector_file in vector_files:
    vector_path = os.path.join(vector_dir, vector_file)
    vectors.append((vector_file, np.load(vector_path)))

# Step 3: 提取新的音訊嵌入
def extract_embedding(audio_path):
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

    return embedding

# test_audio_file = "audioFile/4-0.wav"  # 測試用音檔路徑
# print(f"\nProcessing file: {test_audio_file}")  # 測試用，顯示導入的音檔名稱
# new_embedding = extract_embedding(test_audio_file)

# 收集所有嵌入向量
all_vectors = [saved_vector for _, saved_vector in vectors]
# all_vectors.append(new_embedding)
# labels = [vector_file for vector_file, _ in vectors] + ["New"]  # 使用檔案名稱作為標籤

labels = [vector_file for vector_file, _ in vectors] # 使用檔案名稱作為標籤

# # PCA 降維到 2D
# pca = PCA(n_components=2)
# reduced_vectors = pca.fit_transform(all_vectors)

# # 繪製散點圖
# plt.figure(figsize=(8, 6))
# plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c='blue', alpha=0.7)
# for i, label in enumerate(labels):
#     plt.annotate(label, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
# plt.title("Embedding Distribution")
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.grid(True)
# plt.show()

# PCA 降維到 3D
pca = PCA(n_components=3)
reduced_vectors = pca.fit_transform(all_vectors)

# 繪製 3D 散點圖
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], c='blue', alpha=0.7)

# 添加標籤
for i, label in enumerate(labels):
    ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], label)

ax.set_title("3D Embedding Distribution")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.show()