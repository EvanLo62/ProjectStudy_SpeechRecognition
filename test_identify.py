import os
import numpy as np
import torch
import torchaudio
import pyaudio
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from speechbrain.inference import SepformerSeparation as separator
from speechbrain.inference import SpeakerRecognition
import noisereduce as nr

# 基本錄音參數
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
TARGET_RATE = 16000
WINDOW_SIZE = 6
OVERLAP = 0.5
DEVICE_INDEX = None

# 音訊處理參數
MIN_ENERGY_THRESHOLD = 0.005
NOISE_REDUCE_STRENGTH = 0.1

# 全域參數設定
EMBEDDING_DIR = "embeddingFiles"  # 所有說話者嵌入資料的根目錄
THRESHOLD_LOW = 0.2     # 過於相似，不更新
THRESHOLD_UPDATE = 0.34 # 更新嵌入向量
THRESHOLD_NEW = 0.36    # 判定為新說話者

# 輸出目錄
OUTPUT_DIR = "16K-model/Audios-16K-IDTF"
IDENTIFIED_DIR = "16K-model/Identified-Speakers"

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== 語者分離部分 ======================

class AudioSeparator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用設備: {self.device}")
        # 使用16KHz分離模型，分離兩語者
        self.model = separator.from_hparams(
            source="speechbrain/sepformer-whamr16k",
            savedir='pretrained_models/sepformer-whamr16k',
            run_opts={"device": self.device}
        )
        
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=RATE,
            new_freq=TARGET_RATE
        ).to(self.device)
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures = []
        self.is_recording = False
        self.output_files = []  # 儲存分離後的音檔路徑
        logger.info("AudioSeparator 初始化完成")

    def spectral_gating(self, audio):
        """應用頻譜閘控降噪"""
        noise_sample = audio[:max(int(TARGET_RATE * 0.1), 1)]
        return nr.reduce_noise(
            y=audio,
            y_noise=noise_sample,
            sr=TARGET_RATE,
            prop_decrease=NOISE_REDUCE_STRENGTH,
            n_jobs=-1
        )

    def enhance_separation(self, separated_signals):
        """增強分離效果，僅應用一次降噪以避免過度處理"""
        enhanced_signals = torch.zeros_like(separated_signals)
        
        for i in range(separated_signals.shape[2]):
            current_signal = separated_signals[0, :, i].cpu().numpy()
            denoised_signal = self.spectral_gating(current_signal)
            length = min(len(denoised_signal), separated_signals.shape[1])
            enhanced_signals[0, :length, i] = torch.from_numpy(denoised_signal).to(self.device)
        
        return enhanced_signals

    def process_audio(self, audio_data):
        """處理音訊格式"""
        try:
            # 轉換為 float32
            if FORMAT == pyaudio.paInt16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # 能量檢測：過低則略過
            energy = np.mean(np.abs(audio_float))
            if energy < MIN_ENERGY_THRESHOLD:
                logger.debug(f"音訊能量 ({energy}) 低於閾值 ({MIN_ENERGY_THRESHOLD})")
                return None
            
            # 重塑為正確形狀
            if len(audio_float.shape) == 1:
                audio_float = audio_float.reshape(-1, CHANNELS)

            # 調整形狀以符合模型輸入：[channels, time]
            audio_tensor = torch.from_numpy(audio_float).T.float()

            # 如果是雙聲道而模型只支援單聲道則取平均
            if audio_tensor.shape[0] == 2:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            # 移至 GPU 並重新取樣至 8kHz
            audio_tensor = audio_tensor.to(self.device)
            resampled = self.resampler(audio_tensor)
            
            # 確保形狀正確
            if len(resampled.shape) == 1:
                resampled = resampled.unsqueeze(0)
            
            return resampled
            
        except Exception as e:
            logger.error(f"音訊處理錯誤：{e}")
            return None

    def record_and_process(self, output_dir):
        """錄音並處理"""
        os.makedirs(output_dir, exist_ok=True)
        mixed_audio_buffer = []
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=DEVICE_INDEX
            )
            
            logger.info("開始錄音...")
            
            # 計算緩衝區大小與重疊數據
            samples_per_window = int(WINDOW_SIZE * RATE)
            window_frames = int(samples_per_window / CHUNK)
            overlap_frames = int((OVERLAP * RATE) / CHUNK)
            slide_frames = window_frames - overlap_frames
            
            buffer = []
            segment_index = 0
            self.is_recording = True
            
            while self.is_recording:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frame = np.frombuffer(data, dtype=np.float32 if FORMAT == pyaudio.paFloat32 else np.int16)
                    buffer.append(frame)
                    mixed_audio_buffer.append(frame.copy())
                except IOError as e:
                    logger.warning(f"錄音時發生IO錯誤：{e}")
                    continue
                
                if len(buffer) >= window_frames:
                    segment_index += 1
                    audio_data = np.concatenate(buffer[:window_frames])
                    audio_tensor = self.process_audio(audio_data)
                    
                    if audio_tensor is not None:
                        logger.info(f"處理片段 {segment_index}")
                        future = self.executor.submit(
                            self.separate_and_save,
                            audio_tensor,
                            output_dir,
                            segment_index
                        )
                        self.futures.append(future)
                    
                    # 保留重疊部分
                    buffer = buffer[slide_frames:]
                    self.futures = [f for f in self.futures if not f.done()]
                    
        except Exception as e:
            logger.error(f"錄音過程中發生錯誤：{e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            for future in self.futures:
                try:
                    future.result(timeout=10.0)
                except Exception as e:
                    logger.error(f"處理任務發生錯誤：{e}")
            
            self.executor.shutdown(wait=True)
            
            # 儲存原始混合音訊為單獨檔案
            timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
            mixed_output_file = ""
            
            if mixed_audio_buffer:
                try:
                    mixed_audio = np.concatenate(mixed_audio_buffer)
                    mixed_audio = mixed_audio.reshape(-1, CHANNELS)
                    
                    mixed_output_file = os.path.join(
                        output_dir,
                        f"mixed_audio_{timestamp}.wav"
                    )
                    
                    mixed_tensor = torch.from_numpy(mixed_audio).T.float()
                    torchaudio.save(
                        mixed_output_file,
                        mixed_tensor,
                        RATE  # 使用原始採樣率 44100Hz 儲存
                    )
                    logger.info(f"已儲存原始混合音訊：{mixed_output_file}")
                except Exception as e:
                    logger.error(f"儲存混合音訊時發生錯誤：{e}")
            
            logger.info("錄音結束，資源已清理")
            return mixed_output_file

    def separate_and_save(self, audio_tensor, output_dir, segment_index):
        """分離並儲存音訊"""
        try:
            with torch.no_grad():
                separated = self.model.separate_batch(audio_tensor)
                enhanced_separated = self.enhance_separation(separated)
                
                timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
                for i in range(enhanced_separated.shape[2]):
                    speaker_audio = enhanced_separated[:, :, i].cpu()
                    
                    # 正規化音量
                    max_val = torch.max(torch.abs(speaker_audio))
                    if max_val > 0:
                        speaker_audio = speaker_audio / max_val * 0.9
                    
                    final_audio = speaker_audio[0].numpy()
                    final_tensor = torch.from_numpy(final_audio).unsqueeze(0)
                    
                    output_file = os.path.join(
                        output_dir,
                        f"speaker{i+1}_{timestamp}_{segment_index}.wav"
                    )
                    
                    torchaudio.save(
                        output_file,
                        final_tensor,
                        TARGET_RATE
                    )
                    
                    # 記錄輸出檔案路徑，稍後用於識別
                    self.output_files.append(output_file)
                
            logger.info(f"片段 {segment_index} 處理完成")
            
        except Exception as e:
            logger.error(f"處理片段 {segment_index} 時發生錯誤：{e}")

    def stop_recording(self):
        """停止錄音"""
        self.is_recording = False
        logger.info("準備停止錄音...")

    def get_output_files(self):
        """獲取所有分離後的音檔路徑"""
        return self.output_files


# ================== 語者識別部分 ======================

class SpeakerIdentifier:
    def __init__(self):
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/speechbrain_recognition"
        )
        logger.info("SpeakerIdentifier 初始化完成")
        
        # 確保嵌入目錄存在
        if not os.path.exists(EMBEDDING_DIR):
            os.makedirs(EMBEDDING_DIR)
        
        # 確保識別結果目錄存在
        if not os.path.exists(IDENTIFIED_DIR):
            os.makedirs(IDENTIFIED_DIR)

    def extract_embedding(self, audio_path):
        """提取音檔的嵌入向量"""
        try:
            signal, fs = torchaudio.load(audio_path)
            
            # 處理立體聲轉單聲道
            if signal.size(0) > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)
            
            # 根據取樣率處理
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                signal = resampler(signal)
            
            # 限制音檔長度（最多 10 秒）
            max_length = 16000 * 10
            if signal.size(1) > max_length:
                signal = signal[:, :max_length]
                
            # 提取嵌入向量
            embedding = self.model.encode_batch(signal).cpu().numpy().squeeze()
            return embedding
            
        except Exception as e:
            logger.error(f"提取嵌入向量時發生錯誤: {e}")
            raise

    def list_all_embedding_files(self):
        """遍歷 EMBEDDING_DIR 下所有子資料夾，返回所有嵌入檔案資訊"""
        embedding_files = []
        for folder in os.listdir(EMBEDDING_DIR):
            folder_path = os.path.join(EMBEDDING_DIR, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(".npy"):
                        full_path = os.path.join(folder_path, file)
                        embedding_files.append((folder, file, full_path))
        return embedding_files

    def compare_all_embeddings(self, new_embedding):
        """計算新嵌入向量與所有現有嵌入的餘弦距離"""
        from scipy.spatial.distance import cosine
        
        embedding_files = self.list_all_embedding_files()
        if not embedding_files:
            logger.info("尚無任何已存的嵌入檔案。")
            return None, float('inf'), []
            
        distances = []
        for folder, file, full_path in embedding_files:
            saved_embedding = np.load(full_path)
            distance = cosine(new_embedding, saved_embedding)
            distances.append((folder, file, distance))
            
        best_match = min(distances, key=lambda x: x[2])
        return best_match, best_match[2], distances

    def identify_speaker(self, audio_file):
        """識別音檔對應的說話者"""
        logger.info(f"識別音檔: {audio_file}")
        
        if not os.path.exists(audio_file):
            logger.error(f"音檔 {audio_file} 不存在")
            return None, float('inf')
            
        # 提取嵌入向量
        new_embedding = self.extract_embedding(audio_file)
        
        # 比對最相似的說話者
        best_match, best_distance, _ = self.compare_all_embeddings(new_embedding)
        
        if best_match is None:
            logger.info("無法找到匹配的說話者，可能是首次使用系統")
            return None, float('inf')
            
        # 顯示比對結果
        speaker_folder = best_match[0]
        logger.info(f"最相似的說話者: {speaker_folder}, 距離: {best_distance:.4f}")
        
        return speaker_folder, best_distance

    def process_audio_files(self, audio_files):
        """處理一批音檔進行識別"""
        results = {}
        
        for audio_file in audio_files:
            try:
                speaker, distance = self.identify_speaker(audio_file)
                
                # 根據閾值決定是否確認為該說話者
                if distance < THRESHOLD_NEW:
                    result = f"{speaker} (距離: {distance:.4f})"
                else:
                    result = f"未知說話者 (距離: {distance:.4f})"
                    
                results[audio_file] = (speaker, distance, result)
                logger.info(f"識別結果: {result}")
                
                # 將識別結果加入到檔名並複製到識別結果目錄
                self.save_identified_audio(audio_file, speaker, distance)
                
            except Exception as e:
                logger.error(f"處理 {audio_file} 時發生錯誤: {e}")
                results[audio_file] = (None, None, f"處理錯誤: {str(e)}")
                
        return results
        
    def save_identified_audio(self, audio_file, speaker, distance):
        """將識別後的音檔保存到專用目錄，並在檔名添加識別結果"""
        try:
            filename = os.path.basename(audio_file)
            basename, ext = os.path.splitext(filename)
            
            # 如果是未知說話者
            if distance >= THRESHOLD_NEW:
                new_filename = f"{basename}_unknown{ext}"
            else:
                new_filename = f"{basename}_{speaker}_{distance:.4f}{ext}"
                
            new_path = os.path.join(IDENTIFIED_DIR, new_filename)
            
            # 讀取並保存音檔
            audio, sr = torchaudio.load(audio_file)
            torchaudio.save(new_path, audio, sr)
            
            logger.info(f"已保存識別後的音檔: {new_path}")
            
        except Exception as e:
            logger.error(f"保存識別後的音檔時發生錯誤: {e}")


# ================== 主流程 ======================

def separate_and_identify():
    """先分離再識別的主流程"""
    try:
        # 步驟1: 初始化語者分離器
        separator = AudioSeparator()
        
        # 步驟2: 錄音並進行語者分離
        logger.info("開始錄音並分離語者...")
        mixed_audio_file = separator.record_and_process(OUTPUT_DIR)
        
        # 等待所有分離任務完成
        logger.info("錄音完成，正在進行最終處理...")
        
        # 步驟3: 獲取分離後的所有音檔
        separated_files = separator.get_output_files()
        logger.info(f"共產生 {len(separated_files)} 個分離後的音檔")
        
        # 步驟4: 初始化語者識別器
        identifier = SpeakerIdentifier()
        
        # 步驟5: 對分離後的音檔進行識別
        logger.info("開始對分離後的音檔進行語者識別...")
        results = identifier.process_audio_files(separated_files)
        
        # 步驟6: 顯示最終識別結果
        print("\n最終識別結果:")
        for audio_file, (speaker, distance, result) in results.items():
            print(f"音檔: {os.path.basename(audio_file)} -> {result}")
            
        print(f"\n原始混合音檔: {mixed_audio_file}")
        print(f"分離後的音檔已保存至: {OUTPUT_DIR}")
        print(f"識別後的音檔已保存至: {IDENTIFIED_DIR}")
        
    except KeyboardInterrupt:
        logger.info("\n接收到停止信號")
        if 'separator' in locals():
            separator.stop_recording()
    except Exception as e:
        logger.error(f"程式執行時發生錯誤：{e}")

# 主程式進入點
if __name__ == "__main__":
    separate_and_identify()