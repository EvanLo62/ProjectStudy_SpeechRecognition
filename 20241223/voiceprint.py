import os
import numpy as np
from pyannote.audio import Model
import torchaudio
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN is not found. Make sure .env file exists and contains the token.")

# Step 1: 載入預訓練模型
model = Model.from_pretrained("pyannote/embedding", use_auth_token=HUGGINGFACE_TOKEN)

# Step 2: 載入音訊檔案
audio_file = "audioFile/1-1.wav"  # 音訊檔案
waveform, sample_rate = torchaudio.load(audio_file)

# Step 3: 定義要處理的音訊片段
start_time = 0
end_time = 10
start_sample = int(start_time * sample_rate)
end_sample = int(end_time * sample_rate)

# 提取該片段的波形
waveform_segment = waveform[:, start_sample:end_sample]
if waveform_segment.shape[0] > 1:
    waveform_segment = waveform_segment.mean(dim=0)

waveform_segment = waveform_segment.unsqueeze(0)  # 添加批次維度

# Step 4: 提取該片段的聲紋嵌入向量
embedding = model(waveform_segment)
embedding_vector = embedding.detach().numpy().flatten()  # 壓平為 1-D

# Step 5: 自動檢查目標資料夾並生成新檔名
output_dir = "vectorFile"
os.makedirs(output_dir, exist_ok=True)  # 確保資料夾存在

# 找到目前的檔案數量
existing_files = [f for f in os.listdir(output_dir) if f.endswith(".npy")]
next_index = len(existing_files) + 1  # 下個檔案的編號
output_file = os.path.join(output_dir, f"{next_index}.npy")

# 儲存嵌入向量
np.save(output_file, embedding_vector)
print(f"Embedding vector has been saved to {output_file}.")
