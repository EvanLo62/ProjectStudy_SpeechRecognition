# 用官方推薦的方式去跑模型，測每個音檔之間的餘弦距離

import os
import numpy as np
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cosine
from numpy.linalg import norm
from dotenv import load_dotenv
from pyannote.core import Segment

# 載入 .env 檔案
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN is not found. Make sure .env file exists and contains the token.")

# Step 1: 載入模型
model = Model.from_pretrained("pyannote/embedding", use_auth_token=HUGGINGFACE_TOKEN)
inference = Inference(model, window="whole")

# Step 2: 讀取所有音訊檔案
comparison_audio_files = [
    "audioFile/1-1.wav",
    "audioFile/1-2.wav",
    "audioFile/1-4.wav",
    "audioFile/2-1.wav",
    "audioFile/2-2.wav"
]

# Step 3: 提取基準音訊嵌入
base_audio_file = "audioFile/1-3.wav"  # 基準音檔
print(f"Base audio file is {base_audio_file}")
base_embedding = inference(base_audio_file).flatten()

# Step 4: 比較嵌入向量
print("Comparing base embedding with other audio files:")

for audio_file in comparison_audio_files:
    comparison_embedding = inference(audio_file).flatten()
    cosine_distance = cosine(base_embedding, comparison_embedding)
    print(f"Cosine distance with {audio_file}: {cosine_distance:.4f}")
    # euclidean_distance = norm(base_embedding - comparison_embedding)
    # print(f"Euclidean distance with {audio_file}: {euclidean_distance:.4f}")
    
