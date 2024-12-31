import os
import torchaudio

def resample_audio(input_path, output_path, target_sr=16000):
    # 讀取音檔
    waveform, sr = torchaudio.load(input_path)

    # 若原始取樣率不等於目標取樣率，就重采樣
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # 存成新的音檔
    torchaudio.save(output_path, waveform, sample_rate=target_sr)
    print(f"Converted '{input_path}' from {sr} Hz to {target_sr} Hz, saved as '{output_path}'.")

# 測試用
if __name__ == "__main__":
    name = "2-2"
    input_file = f"audioFile/{name}.wav"      # 你原本的音檔
    output_file = f"audioFile/{name}.wav" # 轉檔後的音檔
    resample_audio(input_file, output_file, target_sr=16000)
