# 顯示3D立體圖，看每個音檔的距離

import os
import numpy as np
from pyannote.audio import Model
import torchaudio
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # 用於 3D 繪圖
import matplotlib.pyplot as plt

# 載入 .env 檔案
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN is not found. Make sure .env file exists and contains the token.")

# Step 1: 載入模型
model = Model.from_pretrained("pyannote/embedding", use_auth_token=HUGGINGFACE_TOKEN)

# Step 2: 讀取所有嵌入向量檔案
vector_dir = "vectorFile"
vector_files = [f for f in os.listdir(vector_dir) if f.endswith(".npy")]

vectors = []
for vector_file in vector_files:
    vector_path = os.path.join(vector_dir, vector_file)
    vectors.append((vector_file, np.load(vector_path)))

# Step 3: 提取新的音訊嵌入
new_audio_file = "audioFile/1-3.wav"  # 新的音訊檔案
print(f"new_audio_file is {new_audio_file}")
waveform, sample_rate = torchaudio.load(new_audio_file)

# 重採樣到 16000 Hz
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

# 提取片段
start_time = 0
end_time = 10
start_sample = int(start_time * sample_rate)
end_sample = int(end_time * sample_rate)

waveform_segment = waveform[:, start_sample:end_sample]
if waveform_segment.shape[0] > 1:
    waveform_segment = waveform_segment.mean(dim=0)

waveform_segment = waveform_segment.unsqueeze(0)

# 提取嵌入
new_embedding = model(waveform_segment).detach().numpy().flatten()

# 收集所有嵌入向量
all_vectors = [saved_vector for _, saved_vector in vectors]
all_vectors.append(new_embedding)
labels = [vector_file for vector_file, _ in vectors] + ["New"]  # 使用檔案名稱作為標籤

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