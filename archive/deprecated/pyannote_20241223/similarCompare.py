# 歐基里德距離

import os
import numpy as np
from pyannote.audio import Model
from numpy.linalg import norm  # 使用 NumPy 的 norm
import torchaudio
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN is not found. Make sure .env file exists and contains the token.")

# Step 1: 載入模型
model = Model.from_pretrained("pyannote/embedding", use_auth_token=HUGGINGFACE_TOKEN)

# Step 2: 讀取所有嵌入向量檔案
vector_dir = "ResearchArchives/vectorFile_test"
vector_files = [f for f in os.listdir(vector_dir) if f.endswith(".npy")]

vectors = []
for vector_file in vector_files:
    vector_path = os.path.join(vector_dir, vector_file)
    vectors.append((vector_file, np.load(vector_path)))

# Step 3: 提取新的音訊嵌入
new_audio_file = "audioFile/2-1.wav"  # 新的音訊檔案
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

# Step 4: 比較嵌入向量
print("Comparing new embedding with saved vectors:")

# 設定閾值
euclidean_threshold = 1.0  # 根據數據分布調整閾值
min_distance = float('inf')
matched_file = None

# 歐幾里德距離計算無需正規化
# 因為嵌入向量本身的距離直接表示特徵差異

for vector_file, saved_vector in vectors:
    # 計算歐幾里德距離
    euclidean_distance = norm(new_embedding - saved_vector)
    print(f"Euclidean distance with {vector_file}: {euclidean_distance:.4f}")
    
    # 找到最小距離
    if euclidean_distance < min_distance:
        min_distance = euclidean_distance
        matched_file = vector_file

# Step 5: 判斷是否是已知聲紋
if min_distance < euclidean_threshold:
    print(f"Matched with {matched_file} (Euclidean distance: {min_distance:.4f})")
else:
    print("No match found. Adding new speaker to the database...")
    # 新聲紋保存
    new_id = len(vectors) + 1
    save_path = os.path.join(vector_dir, f"{new_id}.npy")
    np.save(save_path, new_embedding)
    print(f"New embedding saved as {save_path}")
