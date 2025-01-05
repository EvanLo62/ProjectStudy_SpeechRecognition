import speech_recognition as sr

# 初始化辨識器
recognizer = sr.Recognizer()

# 載入音頻檔案
audio_file = "test.wav"  # 替換為你的音頻文件路徑
with sr.AudioFile(audio_file) as source:
    audio_data = recognizer.record(source)  # 讀取整個音頻檔

# 語音轉文字
try:
    text = recognizer.recognize_google(audio_data, language="zh-TW")
    print("辨識結果：", text)
except sr.UnknownValueError:
    print("無法理解音頻內容")
except sr.RequestError as e:
    print(f"API 請求出錯：{e}")