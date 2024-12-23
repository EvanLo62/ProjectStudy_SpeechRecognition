from pydub import AudioSegment

# 設定檔案路徑
input_file = "2-1.m4a"  # 輸入的 m4a 檔案
output_file = "1-2.wav"  # 輸出的 wav 檔案

# 使用 pydub 加載音訊並轉換格式
audio = AudioSegment.from_file(input_file, format="m4a")
audio.export(output_file, format="wav")

print("轉換完成！檔案儲存為：", output_file)
