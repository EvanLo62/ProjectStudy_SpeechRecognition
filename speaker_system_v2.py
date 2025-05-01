"""
===============================================================================
即時語者分離與識別系統 (Real-time Speech Separation and Speaker Identification System)
===============================================================================

版本：v2.0.0
作者：CYouuu, EvanLo62
最後更新：2025-05-02

功能摘要：
-----------
本系統實現了先進的即時語音處理功能，能夠在混合語音環境中實時將不同語者的聲音分離，
並利用深度學習模型對每位語者進行辨識與追蹤。主要特點包括：

 1. 即時處理：邊錄音邊處理，不需等待完整錄音
 2. 語者分離：能夠將多位語者的混合語音分離成獨立的音檔
 3. 即時識別：分離後立即進行語者識別，顯示實時識別結果
 4. 聲紋更新：自動更新語者聲紋向量，提高識別準確率
 5. 單例模式：避免模型和資料庫重複初始化，提高效能

技術架構：
-----------
 - 語者分離模型: SpeechBrain Sepformer (16kHz)
 - 語者識別模型: SpeechBrain ECAPA-TDNN 模型
 - 向量資料庫: Weaviate，用於儲存和檢索說話者嵌入向量
 - 即時處理: 多執行緒並行處理，邊錄音邊識別
 - 音訊增強: 頻譜閘控降噪，提高分離品質

Weaviate 資料庫設定：
-----------
 - 安裝並啟動 Weaviate 向量資料庫，使用docker-compose.yml配置：
   ```
   docker-compose up -d
   ```
 - 執行 `create_collections.py` 建立必要的2個集合：
   ```
   python create_collections.py
   ```
 - 若要匯入現有語者嵌入向量，可執行：
   ```
   python weaviate_study/npy_to_weaviate.py
   ```

處理流程：
-----------
 1. 錄音：連續從麥克風接收音訊流
 2. 分塊處理：每6秒音訊(可自訂)為一個處理單元，重疊率50%
 3. 分離處理：將每段混合音訊分離為獨立的聲音流
 4. 即時識別：對每位分離後的語者立即進行識別
 5. 顯示結果：即時顯示每段識別結果及識別型態

使用方式：
-----------
 1. 直接運行主程式:
    ```
    python speaker_system_v2.py
    ```

 2. 按下 Ctrl+C 停止錄音和識別

前置需求：
-----------
 - Python 3.9+
 - PyTorch with torchaudio
 - SpeechBrain
 - PyAudio (錄音功能)
 - Weaviate 向量資料庫 (需通過 Docker 啟動)
 - 其他依賴套件 (見 requirements.txt)

系統參數：
-----------
 - THRESHOLD_LOW = 0.27: 過於相似，不更新向量
 - THRESHOLD_UPDATE = 0.34: 相似度足夠，更新向量
 - THRESHOLD_NEW = 0.36: 超過此值視為新語者
 - WINDOW_SIZE = 6: 處理窗口大小（秒）
 - OVERLAP = 0.5: 窗口重疊率

輸出結果：
-----------
 - 分離後的音檔: 16K-model/Audios-16K-IDTF/ 目錄下
 - 混合音檔: 同目錄下，前綴為 mixed_audio_
 - 日誌檔案: system_output.log

詳細資訊：
-----------
請參考專案文件: https://github.com/LCY000/ProjectStudy_SpeechRecognition

===============================================================================
"""

import os
import sys
import numpy as np
import torch
import torchaudio
import pyaudio
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Any, Union
from speechbrain.inference import SepformerSeparation as separator
from speechbrain.inference import SpeakerRecognition
import noisereduce as nr
# 導入 scipy.signal.resample_poly 進行重採樣
from scipy.signal import resample_poly

# 導入 main_identify_v5 模組
import main_identify_v5 as speaker_id

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

# 全域參數設定，使用 v5 版本的閾值
EMBEDDING_DIR = "embeddingFiles"  # 所有說話者嵌入資料的根目錄
THRESHOLD_LOW = speaker_id.THRESHOLD_LOW     # 過於相似，不更新
THRESHOLD_UPDATE = speaker_id.THRESHOLD_UPDATE # 更新嵌入向量
THRESHOLD_NEW = speaker_id.THRESHOLD_NEW    # 判定為新說話者

# 輸出目錄
OUTPUT_DIR = "16K-model/Audios-16K-IDTF"
IDENTIFIED_DIR = "16K-model/Identified-Speakers"

# 自訂日誌格式，移除多餘前綴
class CustomFormatter(logging.Formatter):
    """自訂日誌格式，根據日誌級別使用不同顏色"""
    
    # 不同日誌級別的顏色代碼
    COLORS = {
        logging.DEBUG: '\033[94m',     # 藍色
        logging.INFO: '\033[92m',      # 綠色
        logging.WARNING: '\033[93m',   # 黃色
        logging.ERROR: '\033[91m',     # 紅色
        logging.CRITICAL: '\033[41m',  # 紅底
    }
    RESET = '\033[0m'  # 結束顏色
    
    def __init__(self, use_color: bool = True) -> None:
        """
        初始化格式器
        
        Args:
            use_color: 是否使用顏色輸出
        """
        super().__init__(fmt='%(message)s', datefmt='%H:%M:%S')
        self.use_color = use_color
    
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日誌記錄
        
        Args:
            record: 日誌記錄物件
            
        Returns:
            str: 格式化後的日誌字串
        """
        # 根據時間和級別生成前綴
        if hasattr(record, 'simple') and record.simple:
            # 簡單輸出，無前綴
            log_fmt = '%(message)s'
        else:
            # 標準輸出，添加時間和級別
            log_fmt = '[%(asctime)s] %(levelname)s: %(message)s'
        
        # 設置格式字串
        self._style._fmt = log_fmt
        
        # 如果使用顏色，則為不同級別設置顏色
        if self.use_color and not (hasattr(record, 'simple') and record.simple):
            color = self.COLORS.get(record.levelno, '')
            reset = self.RESET
            record.levelname = f"{color}{record.levelname}{reset}"
        
        result = super().format(record)
        return result

# 設定日誌系統
def setup_logging(log_file: str = "system_output.log", 
                 console_level: int = logging.INFO, 
                 file_level: int = logging.DEBUG) -> logging.Logger:
    """
    設定日誌系統，同時輸出到控制台和文件
    
    Args:
        log_file: 日誌文件路徑
        console_level: 控制台輸出的日誌級別
        file_level: 文件輸出的日誌級別
        
    Returns:
        logging.Logger: 配置好的日誌物件
    """
    # 重要：關閉 root logger 的傳播，避免重複輸出
    logging.getLogger().handlers = []
    
    # 創建日誌物件
    logger = logging.getLogger("speaker_system")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 重要：關閉傳播以避免重複輸出
    
    # 清除現有的處理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(CustomFormatter(use_color=True))
    
    # 文件處理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levellevel)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # 添加處理器到日誌物件
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # 設置 SpeechBrain 和其他庫的日誌級別
    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    
    return logger

# 初始化日誌系統
logger = setup_logging()

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

    def process_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """
        處理音訊格式：將原始錄音資料轉換為模型可用的格式
        
        流程：
        1. 轉換資料型別為 float32 (若非已是 float32)
        2. 能量檢測以過濾靜音部分 
        3. 重塑資料以符合模型輸入格式
        4. 轉換為 PyTorch 張量並移至正確設備 (CPU/GPU)
        5. 重新取樣至 16kHz (模型需要)
        
        Args:
            audio_data: 原始音訊資料 (NumPy 陣列)
            
        Returns:
            torch.Tensor: 處理後的音訊張量，若能量過低則回傳 None
        """
        try:
            # 步驟1: 轉換資料型別
            # 若原始格式為 16 位整數 (paInt16)，則歸一化至 [-1, 1] 範圍
            if FORMAT == pyaudio.paInt16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # 步驟2: 能量檢測 - 計算平均絕對振幅
            energy = np.mean(np.abs(audio_float))
            # 若能量低於閾值 (靜音或噪音)，則略過此段音訊
            if energy < MIN_ENERGY_THRESHOLD:
                logger.debug(f"音訊能量 ({energy}) 低於閾值 ({MIN_ENERGY_THRESHOLD})")
                return None
            
            # 步驟3a: 重塑資料 - 若為一維陣列，轉換為 [time, channels] 形狀
            if len(audio_float.shape) == 1:
                audio_float = audio_float.reshape(-1, CHANNELS)

            # 步驟3b: 調整形狀 - 轉換為模型輸入格式 [channels, time]
            audio_tensor = torch.from_numpy(audio_float).T.float()

            # 步驟3c: 聲道調整 - 若為雙聲道而模型只支援單聲道，則對聲道取平均
            if audio_tensor.shape[0] == 2:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            # 步驟4: 移至處理設備 (CPU/GPU)
            audio_tensor = audio_tensor.to(self.device)
            
            # 步驟5: 使用 torchaudio 重採樣至 16kHz (模型要求)
            # 原始取樣率 44.1kHz 重採樣至 16kHz
            resampled = self.resampler(audio_tensor)
            
            # 步驟6: 確保形狀正確 - 若降為一維，恢復為二維張量 [1, time]
            if len(resampled.shape) == 1:
                resampled = resampled.unsqueeze(0)
            
            return resampled
            
        except Exception as e:
            # 記錄所有處理過程中的錯誤並返回 None
            logger.error(f"音訊處理錯誤：{e}")
            return None

    def record_and_process(self, output_dir):
        """
        錄音並處理音訊的主要方法
        
        流程：
        1. 初始化錄音設備並開始串流
        2. 逐塊收集音訊並加入緩衝區
        3. 當緩衝區達到一定長度時，觸發處理
        4. 使用多執行緒處理多個音訊片段
        5. 完成後儲存原始混合音訊
        
        Args:
            output_dir: 輸出目錄路徑
            
        Returns:
            str: 儲存的混合音訊檔案路徑
        """
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        # 儲存原始混合音訊的緩衝區
        mixed_audio_buffer = []
        
        try:
            # 步驟1: 初始化錄音裝置
            p = pyaudio.PyAudio()
            stream = p.open(
                format=FORMAT,  # 音訊格式 (float32)
                channels=CHANNELS,  # 聲道數 (單聲道)
                rate=RATE,  # 取樣率 (44100Hz)
                input=True,  # 啟用輸入
                frames_per_buffer=CHUNK,  # 每次讀取的取樣數
                input_device_index=DEVICE_INDEX  # 輸入裝置索引
            )
            
            logger.info("開始錄音...")
            
            # 步驟2: 計算緩衝區參數
            # 根據設定的視窗大小計算每個處理片段需要的幀數
            samples_per_window = int(WINDOW_SIZE * RATE)  # 總取樣數 (6秒 * 44100Hz)
            window_frames = int(samples_per_window / CHUNK)  # 每個視窗的幀數
            # 計算重疊部分的幀數 (50% 重疊)
            overlap_frames = int((OVERLAP * RATE) / CHUNK)
            # 計算每次移動的幀數 (視窗大小 - 重疊部分)
            slide_frames = window_frames - overlap_frames
            
            # 初始化處理變數
            buffer = []  # 處理緩衝區
            segment_index = 0  # 片段計數器
            self.is_recording = True  # 錄音控制旗標
            
            # 步驟3: 錄音主循環
            while self.is_recording:
                try:
                    # 讀取一個音訊區塊
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    # 將二進位資料轉換為數值陣列
                    frame = np.frombuffer(data, dtype=np.float32 if FORMAT == pyaudio.paFloat32 else np.int16)
                    # 加入處理緩衝區
                    buffer.append(frame)
                    # 同時保存一份到混合音訊緩衝區 (最終儲存用)
                    mixed_audio_buffer.append(frame.copy())
                except IOError as e:
                    logger.warning(f"錄音時發生IO錯誤：{e}")
                    continue
                
                # 步驟4: 當累積足夠資料時進行處理
                if len(buffer) >= window_frames:
                    segment_index += 1
                    # 合併緩衝區中的幀以建立當前視窗的完整音訊資料
                    audio_data = np.concatenate(buffer[:window_frames])
                    # 處理音訊 (轉換格式、能量檢測、重採樣)
                    audio_tensor = self.process_audio(audio_data)
                    
                    # 步驟5: 如果音訊有效 (能量足夠)，啟動語者分離處理
                    if (audio_tensor is not None):
                        logger.info(f"處理片段 {segment_index}")
                        # 使用多執行緒非同步處理音訊分離
                        future = self.executor.submit(
                            self.separate_and_identify,  # 分離和儲存方法
                            audio_tensor,            # 處理後的音訊張量
                            output_dir,              # 輸出目錄
                            segment_index            # 片段索引
                        )
                        self.futures.append(future)
                    
                    # 步驟6: 移動視窗 (保留重疊部分)
                    buffer = buffer[slide_frames:]
                    # 清理已完成的任務
                    self.futures = [f for f in self.futures if not f.done()]
                    
        except Exception as e:
            logger.error(f"錄音過程中發生錯誤：{e}")
        finally:
            # 步驟7: 清理資源
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # 等待所有分離任務完成
            for future in self.futures:
                try:
                    future.result(timeout=10.0)
                except Exception as e:
                    logger.error(f"處理任務發生錯誤：{e}")
            
            self.executor.shutdown(wait=True)
            
            # 步驟8: 儲存原始混合音訊
            timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
            mixed_output_file = ""
            
            if mixed_audio_buffer:
                try:
                    # 合併所有錄製的音訊
                    mixed_audio = np.concatenate(mixed_audio_buffer)
                    mixed_audio = mixed_audio.reshape(-1, CHANNELS)
                    
                    # 建立輸出檔案路徑
                    mixed_output_file = os.path.join(
                        output_dir,
                        f"mixed_audio_{timestamp}.wav"
                    )
                    
                    # 轉換為 PyTorch 張量並儲存
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

    def separate_and_identify(self, audio_tensor: torch.Tensor, output_dir: str, segment_index: int) -> None:
        """
        分離並儲存音訊，然後立即進行語音識別
        
        Args:
            audio_tensor: 處理後的音訊張量
            output_dir: 輸出目錄
            segment_index: 片段索引
        """
        try:
            audio_files = []  # 儲存當前處理的音檔路徑
            
            with torch.no_grad():
                # 步驟1: 語者分離
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
                    
                    # 建立輸出檔案路徑
                    output_file = os.path.join(
                        output_dir,
                        f"speaker{i+1}_{timestamp}_{segment_index}.wav"
                    )
                    
                    # 儲存音檔
                    torchaudio.save(
                        output_file,
                        final_tensor,
                        TARGET_RATE
                    )
                    
                    # 記錄輸出檔案路徑
                    audio_files.append(output_file)
                    self.output_files.append(output_file)
            
            # 步驟2: 即時進行語者識別
            if audio_files:
                # 使用簡單日誌方式輸出片段分離訊息
                log_record = logging.LogRecord(
                    "speaker_system", logging.INFO, "", 0, 
                    f"片段 {segment_index} 分離完成，開始進行即時語者識別...", (), None
                )
                log_record.simple = True  # 使用簡單格式
                logger.handle(log_record)
                
                try:
                    # 初始化或獲取語者識別器實例 (單例模式)
                    identifier = SpeakerIdentifier()
                    
                    # 識別當前分離的音檔
                    results = identifier.process_audio_files(audio_files)
                    
                    # 使用簡化格式輸出識別結果
                    result_message = []
                    for audio_file, (speaker, distance, result) in results.items():
                        # 使用簡單格式輸出識別結果
                        basename = os.path.basename(audio_file)
                        result_message.append(f"【{basename} → {result}】")
                    
                    if result_message:
                        log_record = logging.LogRecord(
                            "speaker_system", logging.INFO, "", 0, 
                            f"識別結果: {' '.join(result_message)}", (), None
                        )
                        log_record.simple = True
                        logger.handle(log_record)
                        
                except Exception as e:
                    logger.error(f"識別片段 {segment_index} 時發生錯誤：{e}")
            
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
    """說話者識別類，負責呼叫 v5 版本的語者識別功能，使用單例模式"""
    
    _instance = None
    
    def __new__(cls) -> 'SpeakerIdentifier':
        """實現單例模式，確保全局只有一個實例"""
        if cls._instance is None:
            cls._instance = super(SpeakerIdentifier, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """初始化說話者識別器，使用 v5 版本的 SpeakerIdentifier"""
        # 若已初始化，則跳過
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        try:
            # 初始化 v5 模組 (它也會使用單例模式，避免重複加載模型)
            self.identifier = speaker_id.SpeakerIdentifier()
            
            # 設置詳細度，減少非必要輸出
            self.identifier.set_verbose(False)
            
            logger.info("語者識別器初始化完成")
            self._initialized = True
        except Exception as e:
            logger.error(f"初始化語者識別器時發生錯誤：{e}")
            raise
    
    def process_audio_files(self, audio_files: list) -> dict:
        """
        處理多個音檔並進行說話者識別
        
        Args:
            audio_files: 音檔路徑列表
            
        Returns:
            dict: 音檔路徑 -> (說話者名稱, 相似度, 識別結果描述)
        """
        results = {}
        
        try:
            for audio_file in audio_files:
                if not os.path.exists(audio_file):
                    logger.warning(f"音檔不存在: {audio_file}")
                    results[audio_file] = (None, -1, "檔案不存在")
                    continue
                
                logger.info(f"識別音檔: {os.path.basename(audio_file)}")
                
                # 呼叫 v5 版本的語者識別功能
                result = self.identifier.process_audio_file(audio_file)
                
                if result:
                    speaker_id_, speaker_name, distance = result
                    
                    # 根據距離判斷識別結果
                    if distance < THRESHOLD_LOW:
                        result_desc = f"說話者 {speaker_name} (非常相似)"
                    elif distance < THRESHOLD_UPDATE:
                        result_desc = f"說話者 {speaker_name} (已更新聲紋)"
                    elif distance < THRESHOLD_NEW:
                        result_desc = f"說話者 {speaker_name} (相似但未更新)"
                    else:
                        result_desc = f"新說話者 {speaker_name}"
                    
                    results[audio_file] = (speaker_name, distance, result_desc)
                    logger.info(f"結果: {result_desc}")
                else:
                    results[audio_file] = (None, -1, "識別失敗")
                    logger.warning("識別失敗")
        except Exception as e:
            logger.error(f"處理音檔時發生錯誤：{e}")
        
        return results


# ================== 主流程 ======================

def main():
    """
    先分離再識別的主流程 - 即時版本
    
    這個版本下，語音分離和識別是並行處理的：
    1. 語音分離後立即進行識別
    2. 不需要等待所有音檔全部錄製完畢才開始識別
    3. 可以實時顯示識別結果
    """
    try:
        # 步驟1: 設定 v5 的日誌輸出
        speaker_id.setup_logging(log_file="output_log.txt", mode="w")
        logger.info("語者分離與識別系統啟動 (即時識別模式)")
        
        # 步驟2: 初始化語者分離器
        separator = AudioSeparator()
        
        # 步驟3: 錄音並進行語者分離 (分離的同時進行識別)
        # 注意：識別功能已整合到 separate_and_save 方法中
        logger.info("開始錄音並執行即時分離與識別...")
        mixed_audio_file = separator.record_and_process(OUTPUT_DIR)
        
        # 步驟4: 錄音結束後的摘要
        separated_files = separator.get_output_files()
        logger.info(f"處理完成，共產生 {len(separated_files)} 個分離後的音檔")
        
        # 步驟5: 顯示錄音結果
        print(f"\n原始混合音檔: {mixed_audio_file}")
        print(f"分離後的音檔已保存至: {OUTPUT_DIR}")
        
        # 步驟6: 恢復原始標準輸出
        speaker_id.restore_stdout()
        
    except KeyboardInterrupt:
        logger.info("\n接收到停止信號")
        if 'separator' in locals():
            separator.stop_recording()
        speaker_id.restore_stdout()
    except Exception as e:
        logger.error(f"程式執行時發生錯誤：{e}")
        speaker_id.restore_stdout()

# 主程式進入點
if __name__ == "__main__":
    main()