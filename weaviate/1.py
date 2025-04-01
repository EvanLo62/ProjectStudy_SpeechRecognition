import weaviate
import json

# 連接到本地的 Weaviate 服務 (預設在 http://localhost:8080)
client = weaviate.connect_to_local()

# 測試連線
if client.is_ready():
    print("Weaviate 連線成功!")

    

    metainfo = client.get_meta()
    print(json.dumps(metainfo, indent=2))  # Print the meta information in a readable format

else:
    print("Weaviate 無法連線，請檢查服務是否正在運行。")

assert client.is_live()  # This will raise an exception if the client is not live

client.close()