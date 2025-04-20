#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音檔監控系統 (Audio Monitoring System)

本模組監控指定資料夾內新增的音檔，並自動傳送至語音辨識處理系統進行處理。
作為語音辨識系統的自動化入口點，連接外部音檔來源與 main_identify_v3.py 的核心處理功能。

功能:
1. 持續監控 'incoming_audio' 資料夾
2. 偵測新建立的 .wav 音檔
3. 確保檔案完整寫入後再進行處理
4. 調用 main_identify_v3 模組進行語音辨識與身份匹配

與 main_identify_v3.py 的關係:
- 本模組作為自動化前端，負責檔案監控與觸發處理
- main_identify_v3.py 作為核心引擎，負責語音嵌入提取與身份比對
- 本模組調用 main_identify_v3 中的 process_audio_file 函數處理每個新檔案
"""

import time
import os
from typing import Optional, Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from main_identify_v3 import process_audio_file

def wait_for_file_complete(file_path: str, wait_time: float = 0.1, retries: int = 5) -> bool:
    """
    檢查檔案大小是否穩定，確保檔案已完全寫入。
    如果檔案在連續幾次檢查中大小都不變，則認為檔案已完成寫入。
    
    Args:
        file_path: 要檢查的檔案路徑
        wait_time: 每次檢查間隔的秒數
        retries: 重複檢查的次數
        
    Returns:
        bool: 如果檔案大小穩定且大於0，則返回True；否則返回False
    """
    last_size: int = -1
    for _ in range(retries):
        try:
            current_size = os.path.getsize(file_path)
            if current_size == last_size and current_size > 0:
                return True
            last_size = current_size
        except Exception as e:
            print(f"檢查檔案大小時發生錯誤: {e}")
        time.sleep(wait_time)
    return False


class AudioFileHandler(FileSystemEventHandler):
    """
    音檔監控處理器
    
    監控檔案系統事件，當有新的.wav檔案建立時，等待檔案完全寫入後，
    將其傳遞給語音辨識系統進行處理。
    """
    
    def on_created(self, event) -> None:
        """
        當新檔案被建立時觸發此方法
        
        Args:
            event: 檔案系統事件物件，包含檔案路徑等資訊
        """
        if not event.is_directory and event.src_path.endswith('.wav'):
            print(f"檢測到新檔案：{event.src_path}")
            # 等待檔案完全寫入後再處理
            if wait_for_file_complete(event.src_path):
                process_audio_file(event.src_path)
            else:
                print(f"檔案 {event.src_path} 寫入不完整，跳過處理。")


def start_monitoring(watch_directory: str = 'incoming_audio') -> None:
    """
    啟動監控系統，監控指定資料夾中新增的音檔
    
    Args:
        watch_directory: 要監控的資料夾路徑
    """
    if not os.path.exists(watch_directory):
        os.makedirs(watch_directory)
        print(f"已建立監控資料夾：{watch_directory}")
        
    event_handler = AudioFileHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_directory, recursive=False)
    observer.start()
    print(f"開始監控資料夾：{watch_directory}")
    print("系統已啟動，等待新的音檔...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    print("監控已停止。")


if __name__ == '__main__':
    start_monitoring()
