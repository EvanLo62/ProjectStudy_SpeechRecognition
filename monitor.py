import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from main_identify_v3 import process_audio_file

def wait_for_file_complete(file_path, wait_time=0.1, retries=5):
    """
    檢查檔案大小是否穩定，確保檔案已完全寫入。
    如果檔案在連續幾次檢查中大小都不變，則認為檔案已完成寫入。
    """
    last_size = -1
    for _ in range(retries):
        try:
            current_size = os.path.getsize(file_path)
            if current_size == last_size and current_size > 0:
                return True
            last_size = current_size
        except Exception:
            pass
        time.sleep(wait_time)
    return False

class AudioFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.wav'):
            print(f"檢測到新檔案：{event.src_path}")
            # 等待檔案完全寫入後再處理
            if wait_for_file_complete(event.src_path):
                process_audio_file(event.src_path)
            else:
                print(f"檔案 {event.src_path} 寫入不完整，跳過處理。")

if __name__ == '__main__':
    watch_directory = 'incoming_audio'
    if not os.path.exists(watch_directory):
        os.makedirs(watch_directory)
    event_handler = AudioFileHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_directory, recursive=False)
    observer.start()
    print(f"開始監控資料夾：{watch_directory} \n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    print("監控已停止。")
