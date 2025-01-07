from pydub import AudioSegment

# 設定檔案路徑
name = "1-11n"
input_file = f"{name}.m4a"  # 輸入的 m4a 檔案
output_file = f"{name}.wav"  # 輸出的 wav 檔案

# 使用 pydub 加載音訊並轉換格式
audio = AudioSegment.from_file(input_file, format="m4a")

# 取前 10 秒 (pydub 使用毫秒為單位，所以 10 秒是 10 * 1000 = 10000)
first_10_seconds = audio[:10000]

# 輸出成 wav 檔
first_10_seconds.export(output_file, format="wav")

print("轉換完成！只輸出前十秒，檔案儲存為：", output_file)
