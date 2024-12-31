# 測歐基里德距離

import os
import numpy as np
from pyannote.audio import Model
import torchaudio
from numpy.linalg import norm
from dotenv import load_dotenv

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

if not vector_files:
    raise RuntimeError("No embedding vectors found in the directory.")

vectors = []
for vector_file in vector_files:
    vector_path = os.path.join(vector_dir, vector_file)
    vectors.append((vector_file, np.load(vector_path)))

# Step 3: 提取新的音訊嵌入
new_audio_file = "audioFile/1-1.wav"
if not os.path.exists(new_audio_file):
    raise FileNotFoundError(f"File not found: {new_audio_file}")

waveform, sample_rate = torchaudio.load(new_audio_file)
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

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

# 比較嵌入向量
print("Comparing new embedding with saved vectors:")
for vector_file, saved_vector in vectors:
    saved_vector = saved_vector.flatten()
    
    # 計算歐幾里得距離
    euclidean_distance = norm(new_embedding - saved_vector)
    print(f"Euclidean distance with {vector_file}: {euclidean_distance:.4f}")
