#!/usr/bin/env python3
"""
Weaviate 資料庫刪除工具

此工具用於簡單快速地刪除 Weaviate 資料庫中的說話者和聲紋資料。
可以選擇刪除:
1. 特定說話者及其所有相關聲紋資料
2. 特定說話者的單個聲紋資料
3. 按名稱搜尋說話者
"""

import weaviate  # type: ignore
import os
import sys
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from datetime import datetime, timezone
from weaviate.collections.classes.filters import Filter


def format_rfc3339(dt: Optional[datetime] = None) -> str:
    """
    將日期時間格式化為符合 RFC3339 標準的字串，包含時區信息
    
    Args:
        dt: 要格式化的 datetime 對象，若為 None 則使用當前時間
        
    Returns:
        str: RFC3339 格式的日期時間字串
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        # 若沒有時區信息，則假設為 UTC
        dt = dt.replace(tzinfo=timezone.utc)
    
    # 格式化為 RFC3339 格式: YYYY-MM-DDThh:mm:ss.sssZ
    return dt.isoformat().replace('+00:00', 'Z')


def connect_to_weaviate() -> Optional[weaviate.WeaviateClient]:
    """
    連接到本地 Weaviate 資料庫
    
    Returns:
        Optional[weaviate.WeaviateClient]: Weaviate 客戶端連接，如果連接失敗則返回 None
    """
    try:
        client = weaviate.connect_to_local()
        print("成功連接到 Weaviate 資料庫！")
        
        # 檢查必要的集合是否存在
        if not client.collections.exists("VoicePrint") or not client.collections.exists("Speaker"):
            print("警告：Weaviate 中缺少必要的集合 (VoicePrint / Speaker)!")
            print("請先運行 weaviate_study/create_collections.py 建立所需的集合")
            return None
            
        return client
        
    except Exception as e:
        print(f"無法連接到 Weaviate 資料庫：{e}")
        print("請確認：")
        print("1. Docker 服務是否正在運行")
        print("2. Weaviate 容器是否已經啟動")
        print("3. weaviate_study/docker-compose.yml 中的配置是否正確")
        print("使用命令 'docker-compose -f weaviate_study/docker-compose.yml up -d' 啟動 Weaviate")
        return None


def get_all_speakers(client: weaviate.WeaviateClient) -> List[Dict[str, Any]]:
    """
    獲取資料庫中的所有說話者資訊
    
    Args:
        client: Weaviate 客戶端連接
        
    Returns:
        List[Dict[str, Any]]: 說話者資訊列表，每個說話者包含 UUID 和名稱等資訊
    """
    try:
        speaker_collection = client.collections.get("Speaker")
        results = speaker_collection.query.fetch_objects(
            limit=100,  # 限制獲取的數量，可以根據需求調整
            return_properties=["name", "create_time", "last_active_time", "voiceprint_ids"]
        )
        
        speakers = []
        for obj in results.objects:
            voiceprint_count = len(obj.properties.get("voiceprint_ids", [])) if obj.properties.get("voiceprint_ids") else 0
            speaker_info = {
                "uuid": obj.uuid,
                "name": obj.properties.get("name", "未命名"),
                "create_time": obj.properties.get("create_time", "未知"),
                "last_active_time": obj.properties.get("last_active_time", "未知"),
                "voiceprint_count": voiceprint_count
            }
            speakers.append(speaker_info)
        
        return speakers
        
    except Exception as e:
        print(f"獲取說話者列表時發生錯誤: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")
        return []


def search_speakers_by_name(client: weaviate.WeaviateClient, name_query: str) -> List[Dict[str, Any]]:
    """
    根據名稱搜尋說話者
    
    Args:
        client: Weaviate 客戶端連接
        name_query: 名稱搜尋關鍵字
        
    Returns:
        List[Dict[str, Any]]: 符合條件的說話者資訊列表
    """
    try:
        speaker_collection = client.collections.get("Speaker")
        
        # 使用 where 過濾器根據名稱搜尋
        results = speaker_collection.query.fetch_objects(
            limit=100,
            return_properties=["name", "create_time", "last_active_time", "voiceprint_ids"],
            filters = Filter.by_property("name").like(f"*{name_query}*")
        )
        
        speakers = []
        for obj in results.objects:
            voiceprint_count = len(obj.properties.get("voiceprint_ids", [])) if obj.properties.get("voiceprint_ids") else 0
            speaker_info = {
                "uuid": obj.uuid,
                "name": obj.properties.get("name", "未命名"),
                "create_time": obj.properties.get("create_time", "未知"),
                "last_active_time": obj.properties.get("last_active_time", "未知"),
                "voiceprint_count": voiceprint_count
            }
            speakers.append(speaker_info)
        
        return speakers
        
    except Exception as e:
        print(f"搜尋說話者時發生錯誤: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")
        return []


def get_speaker_voiceprints(client: weaviate.WeaviateClient, speaker_id: str) -> List[Dict[str, Any]]:
    """
    獲取指定說話者的所有聲紋資料
    
    Args:
        client: Weaviate 客戶端連接
        speaker_id: 說話者的 UUID
        
    Returns:
        List[Dict[str, Any]]: 說話者的聲紋資料列表
    """
    try:
        voice_print_collection = client.collections.get("VoicePrint")
        
        # 修正：使用正確的 Weaviate v4 參考過濾器語法
        filters = Filter.by_ref("speaker").by_id().equal(speaker_id)
        
        results = voice_print_collection.query.fetch_objects(
            limit=100,
            return_properties=["speaker_name", "update_count", "create_time", "updated_time"],
            filters=filters
        )
        
        voiceprints = []
        for obj in results.objects:
            voiceprint_info = {
                "uuid": obj.uuid,
                "speaker_name": obj.properties.get("speaker_name", "未知"),
                "update_count": obj.properties.get("update_count", 0),
                "create_time": obj.properties.get("create_time", "未知"),
                "updated_time": obj.properties.get("updated_time", "未知")
            }
            voiceprints.append(voiceprint_info)
        
        return voiceprints
        
    except Exception as e:
        print(f"獲取說話者聲紋資料時發生錯誤: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")
        return []


def delete_voiceprint(client: weaviate.WeaviateClient, voiceprint_id: str) -> bool:
    """
    刪除單個聲紋資料
    
    Args:
        client: Weaviate 客戶端連接
        voiceprint_id: 要刪除的聲紋資料 UUID
        
    Returns:
        bool: 操作是否成功
    """
    try:
        voice_print_collection = client.collections.get("VoicePrint")
        
        # 先獲取聲紋資訊，以便輸出訊息
        voiceprint_obj = voice_print_collection.query.fetch_object_by_id(
            uuid=voiceprint_id,
            return_properties=["speaker_name", "update_count"]
        )
        
        if not voiceprint_obj:
            print(f"找不到 ID 為 {voiceprint_id} 的聲紋資料")
            return False
        
        speaker_name = voiceprint_obj.properties.get("speaker_name", "未知")
        update_count = voiceprint_obj.properties.get("update_count", 0)
        
        # 執行刪除操作
        voice_print_collection.data.delete_by_id(voiceprint_id)
        print(f"已成功刪除說話者 '{speaker_name}' 的聲紋資料（更新次數: {update_count}，ID: {voiceprint_id}）")
        return True
        
    except Exception as e:
        print(f"刪除聲紋資料時發生錯誤: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")
        return False


def update_speaker_voiceprint_ids(client: weaviate.WeaviateClient, speaker_id: str, voiceprint_ids: List[str]) -> bool:
    """
    更新說話者的聲紋 ID 列表
    
    Args:
        client: Weaviate 客戶端連接
        speaker_id: 說話者 ID
        voiceprint_ids: 更新後的聲紋 ID 列表
        
    Returns:
        bool: 操作是否成功
    """
    try:
        speaker_collection = client.collections.get("Speaker")
        
        # 更新說話者資訊
        speaker_collection.data.update(
            uuid=speaker_id,
            properties={
                "voiceprint_ids": voiceprint_ids,
                "last_active_time": format_rfc3339()
            }
        )
        
        return True
        
    except Exception as e:
        print(f"更新說話者聲紋 ID 列表時發生錯誤: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")
        return False


def delete_speaker(client: weaviate.WeaviateClient, speaker_id: str) -> bool:
    """
    刪除說話者（不包括其聲紋資料）
    
    Args:
        client: Weaviate 客戶端連接
        speaker_id: 要刪除的說話者 ID
        
    Returns:
        bool: 操作是否成功
    """
    try:
        speaker_collection = client.collections.get("Speaker")
        
        # 先獲取說話者資訊，以便輸出訊息
        speaker_obj = speaker_collection.query.fetch_object_by_id(
            uuid=speaker_id,
            return_properties=["name"]
        )
        
        if not speaker_obj:
            print(f"找不到 ID 為 {speaker_id} 的說話者")
            return False
        
        speaker_name = speaker_obj.properties.get("name", "未知")
        
        # 執行刪除操作
        speaker_collection.data.delete_by_id(speaker_id)
        print(f"已成功刪除說話者 '{speaker_name}' (ID: {speaker_id})")
        return True
        
    except Exception as e:
        print(f"刪除說話者時發生錯誤: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")
        return False


def delete_speaker_and_voiceprints(client: weaviate.WeaviateClient, speaker_id: str) -> bool:
    """
    刪除說話者及其所有聲紋資料
    
    Args:
        client: Weaviate 客戶端連接
        speaker_id: 要刪除的說話者 ID
        
    Returns:
        bool: 操作是否成功
    """
    try:
        # 先獲取說話者資訊
        speaker_collection = client.collections.get("Speaker")
        speaker_obj = speaker_collection.query.fetch_object_by_id(
            uuid=speaker_id,
            return_properties=["name", "voiceprint_ids"]
        )
        
        if not speaker_obj:
            print(f"找不到 ID 為 {speaker_id} 的說話者")
            return False
        
        speaker_name = speaker_obj.properties.get("name", "未知")
        voiceprint_ids = speaker_obj.properties.get("voiceprint_ids", [])
        
        # 獲取相關聲紋資料
        voiceprints = get_speaker_voiceprints(client, speaker_id)
        
        # 刪除所有聲紋資料
        voice_print_collection = client.collections.get("VoicePrint")
        deleted_count = 0
        
        for voiceprint in voiceprints:
            try:
                voice_print_collection.data.delete_by_id(voiceprint["uuid"])
                deleted_count += 1
            except Exception as e:
                print(f"刪除聲紋資料 {voiceprint['uuid']} 時發生錯誤: {str(e)}")
                # 繼續刪除其他聲紋資料
        
        # 刪除說話者
        try:
            speaker_collection.data.delete_by_id(speaker_id)
            print(f"已成功刪除說話者 '{speaker_name}' (ID: {speaker_id}) 及其 {deleted_count}/{len(voiceprints)} 個聲紋資料")
            return True
        except Exception as e:
            print(f"刪除說話者 {speaker_id} 時發生錯誤: {str(e)}")
            print(f"已刪除 {deleted_count}/{len(voiceprints)} 個聲紋資料，但說話者刪除失敗")
            return False
        
    except Exception as e:
        print(f"刪除說話者及其聲紋資料時發生錯誤: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")
        return False


def remove_voiceprint_from_speaker(client: weaviate.WeaviateClient, speaker_id: str, voiceprint_id: str) -> bool:
    """
    將聲紋 ID 從說話者的聲紋列表中移除（不刪除聲紋資料本身）
    
    Args:
        client: Weaviate 客戶端連接
        speaker_id: 說話者 ID
        voiceprint_id: 要移除的聲紋 ID
        
    Returns:
        bool: 操作是否成功
    """
    try:
        speaker_collection = client.collections.get("Speaker")
        
        # 獲取說話者資訊
        speaker_obj = speaker_collection.query.fetch_object_by_id(
            uuid=speaker_id,
            return_properties=["name", "voiceprint_ids"]
        )
        
        if not speaker_obj:
            print(f"找不到 ID 為 {speaker_id} 的說話者")
            return False
        
        speaker_name = speaker_obj.properties.get("name", "未知")
        voiceprint_ids = speaker_obj.properties.get("voiceprint_ids", [])
        
        # 移除指定的聲紋 ID
        if voiceprint_id in voiceprint_ids:
            voiceprint_ids.remove(voiceprint_id)
            
            # 更新說話者資訊
            speaker_collection.data.update(
                uuid=speaker_id,
                properties={
                    "voiceprint_ids": voiceprint_ids,
                    "last_active_time": format_rfc3339()
                }
            )
            
            print(f"已從說話者 '{speaker_name}' 的聲紋列表中移除聲紋 ID {voiceprint_id}")
            return True
        else:
            print(f"聲紋 ID {voiceprint_id} 不在說話者 '{speaker_name}' 的聲紋列表中")
            return False
        
    except Exception as e:
        print(f"從說話者聲紋列表移除聲紋 ID 時發生錯誤: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")
        return False


def display_speakers(speakers: List[Dict[str, Any]]) -> None:
    """
    顯示說話者列表
    
    Args:
        speakers: 說話者資訊列表
    """
    if not speakers:
        print("沒有找到任何說話者。")
        return
    
    print(f"\n找到 {len(speakers)} 個說話者:")
    print("=" * 70)
    print(f"{'編號':<4} {'名稱':<15} {'聲紋數量':<8} {'UUID':<36} {'建立時間'}")
    print("-" * 70)
    
    for idx, speaker in enumerate(speakers, 1):
        # 處理 create_time 可能是 datetime 物件的情況
        create_time_str = ""
        if isinstance(speaker['create_time'], datetime):
            create_time_str = speaker['create_time'].strftime("%Y-%m-%d")
        else:
            # 如果是字串，嘗試取前 10 個字元
            create_time_str = str(speaker['create_time'])[:10]
            
        print(f"{idx:<4} {speaker['name']:<15} {speaker['voiceprint_count']:<8} {speaker['uuid']} {create_time_str}")


def format_datetime_safe(dt_value: Any, max_len: int = 10) -> str:
    """
    安全地格式化日期時間值，處理不同的日期時間格式
    
    Args:
        dt_value: 日期時間值，可能是日期時間物件或字串
        max_len: 輸出字串的最大長度
        
    Returns:
        str: 格式化後的日期時間字串
    """
    if isinstance(dt_value, datetime):
        # 如果是 datetime 物件，格式化為字串
        return dt_value.strftime("%Y-%m-%d %H:%M:%S")
    else:
        # 其他情況，轉換為字串並取適當長度
        return str(dt_value)[:max_len]


def display_voiceprints(voiceprints: List[Dict[str, Any]]) -> None:
    """
    顯示聲紋資料列表
    
    Args:
        voiceprints: 聲紋資料列表
    """
    if not voiceprints:
        print("沒有找到任何聲紋資料。")
        return
    
    print(f"\n找到 {len(voiceprints)} 個聲紋資料:")
    print("=" * 90)
    print(f"{'編號':<4} {'說話者':<15} {'更新次數':<8} {'UUID':<36} {'建立時間':<20} {'更新時間'}")
    print("-" * 90)
    
    for idx, voiceprint in enumerate(voiceprints, 1):
        # 處理 create_time 和 updated_time 可能是 datetime 物件的情況
        create_time_str = format_datetime_safe(voiceprint['create_time'], max_len=19)
        updated_time_str = format_datetime_safe(voiceprint['updated_time'], max_len=19)
            
        print(f"{idx:<4} {voiceprint['speaker_name']:<15} {voiceprint['update_count']:<8} "
              f"{voiceprint['uuid']} {create_time_str:<20} {updated_time_str}")


def handle_delete_single_voiceprint(client: weaviate.WeaviateClient) -> None:
    """
    處理刪除單個聲紋資料的操作
    
    Args:
        client: Weaviate 客戶端連接
    """
    # 獲取所有說話者
    speakers = get_all_speakers(client)
    
    if not speakers:
        print("資料庫中沒有任何說話者，無法執行刪除操作。")
        return
    
    # 顯示所有說話者
    display_speakers(speakers)
    
    try:
        # 選擇說話者
        speaker_idx = input("\n請選擇要查看聲紋資料的說話者編號 (1-N 或輸入 0 返回): ").strip()
        if speaker_idx == "0":
            return
        
        idx = int(speaker_idx) - 1
        if 0 <= idx < len(speakers):
            speaker = speakers[idx]
            
            # 獲取說話者的聲紋資料
            voiceprints = get_speaker_voiceprints(client, speaker["uuid"])
            
            if not voiceprints:
                print(f"說話者 '{speaker['name']}' 沒有任何聲紋資料。")
                return
            
            # 顯示聲紋資料
            display_voiceprints(voiceprints)
            
            # 選擇聲紋資料
            voiceprint_idx = input("\n請選擇要刪除的聲紋資料編號 (1-N 或輸入 0 返回): ").strip()
            if voiceprint_idx == "0":
                return
            
            vp_idx = int(voiceprint_idx) - 1
            if 0 <= vp_idx < len(voiceprints):
                voiceprint = voiceprints[vp_idx]
                
                # 確認刪除
                confirm = input(f"\n確定要刪除說話者 '{voiceprint['speaker_name']}' 的聲紋資料 "
                                f"(更新次數: {voiceprint['update_count']})？(y/N): ").strip().lower()
                
                if confirm == 'y':
                    # 執行刪除
                    success = delete_voiceprint(client, voiceprint["uuid"])
                    
                    if success:
                        # 從說話者的聲紋列表中移除此 ID
                        remove_voiceprint_from_speaker(client, speaker["uuid"], voiceprint["uuid"])
                        print("刪除操作完成！")
                    else:
                        print("刪除操作失敗，請查看上方錯誤訊息。")
                else:
                    print("已取消刪除操作。")
            else:
                print(f"無效的選擇！可用選項範圍是 1 到 {len(voiceprints)}")
        else:
            print(f"無效的選擇！可用選項範圍是 1 到 {len(speakers)}")
            
    except ValueError:
        print("請輸入有效的數字！")
    except Exception as e:
        print(f"處理刪除聲紋資料操作時發生錯誤: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")


def handle_delete_speaker(client: weaviate.WeaviateClient) -> None:
    """
    處理刪除說話者及其所有聲紋資料的操作
    
    Args:
        client: Weaviate 客戶端連接
    """
    # 獲取所有說話者
    speakers = get_all_speakers(client)
    
    if not speakers:
        print("資料庫中沒有任何說話者，無法執行刪除操作。")
        return
    
    # 顯示所有說話者
    display_speakers(speakers)
    
    try:
        # 選擇說話者
        speaker_idx = input("\n請選擇要刪除的說話者編號 (1-N 或輸入 0 返回): ").strip()
        if speaker_idx == "0":
            return
        
        idx = int(speaker_idx) - 1
        if 0 <= idx < len(speakers):
            speaker = speakers[idx]
            
            # 顯示說話者詳細資訊
            print(f"\n說話者詳細資訊:")
            print(f"名稱: {speaker['name']}")
            print(f"UUID: {speaker['uuid']}")
            print(f"建立時間: {speaker['create_time']}")
            print(f"最後活動時間: {speaker['last_active_time']}")
            print(f"聲紋數量: {speaker['voiceprint_count']}")
            
            # 獲取聲紋資料以顯示更多細節
            if (speaker['voiceprint_count'] > 0):
                voiceprints = get_speaker_voiceprints(client, speaker["uuid"])
                display_voiceprints(voiceprints)
            
            # 確認刪除
            confirm = input(f"\n確定要刪除說話者 '{speaker['name']}' 及其所有聲紋資料？(y/N): ").strip().lower()
            
            if confirm == 'y':
                # 執行刪除
                success = delete_speaker_and_voiceprints(client, speaker["uuid"])
                
                if success:
                    print("刪除操作完成！")
                else:
                    print("刪除操作失敗或部分失敗，請查看上方錯誤訊息。")
            else:
                print("已取消刪除操作。")
        else:
            print(f"無效的選擇！可用選項範圍是 1 到 {len(speakers)}")
            
    except ValueError:
        print("請輸入有效的數字！")
    except Exception as e:
        print(f"處理刪除說話者操作時發生錯誤: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")


def handle_search_speaker(client: weaviate.WeaviateClient) -> None:
    """
    處理搜尋說話者的操作
    
    Args:
        client: Weaviate 客戶端連接
    """
    search_query = input("\n請輸入要搜尋的說話者名稱 (可部分符合): ").strip()
    
    if not search_query:
        print("搜尋關鍵字不能為空！")
        return
    
    # 執行搜尋
    speakers = search_speakers_by_name(client, search_query)
    
    # 顯示搜尋結果
    display_speakers(speakers)
    
    if speakers:
        # 提供後續操作選項
        action = input("\n請選擇操作: (1) 刪除說話者及其聲紋 (2) 查看聲紋詳情 (0) 返回: ").strip()
        
        if action == "1":
            # 選擇要刪除的說話者
            speaker_idx = input("\n請選擇要刪除的說話者編號 (1-N 或輸入 0 返回): ").strip()
            if speaker_idx == "0":
                return
            
            try:
                idx = int(speaker_idx) - 1
                if 0 <= idx < len(speakers):
                    speaker = speakers[idx]
                    
                    # 確認刪除
                    confirm = input(f"\n確定要刪除說話者 '{speaker['name']}' 及其所有聲紋資料？(y/N): ").strip().lower()
                    
                    if confirm == 'y':
                        # 執行刪除
                        success = delete_speaker_and_voiceprints(client, speaker["uuid"])
                        
                        if success:
                            print("刪除操作完成！")
                        else:
                            print("刪除操作失敗或部分失敗，請查看上方錯誤訊息。")
                    else:
                        print("已取消刪除操作。")
                else:
                    print(f"無效的選擇！可用選項範圍是 1 到 {len(speakers)}")
            except ValueError:
                print("請輸入有效的數字！")
                
        elif action == "2":
            # 選擇要查看的說話者
            speaker_idx = input("\n請選擇要查看聲紋詳情的說話者編號 (1-N 或輸入 0 返回): ").strip()
            if speaker_idx == "0":
                return
            
            try:
                idx = int(speaker_idx) - 1
                if 0 <= idx < len(speakers):
                    speaker = speakers[idx]
                    
                    # 獲取聲紋資料
                    voiceprints = get_speaker_voiceprints(client, speaker["uuid"])
                    display_voiceprints(voiceprints)
                    
                    # 提供刪除單個聲紋的選項
                    if voiceprints:
                        delete_option = input("\n是否要刪除特定聲紋？(y/N): ").strip().lower()
                        if delete_option == 'y':
                            handle_delete_single_voiceprint(client)
                else:
                    print(f"無效的選擇！可用選項範圍是 1 到 {len(speakers)}")
            except ValueError:
                print("請輸入有效的數字！")


def show_menu() -> str:
    """
    顯示主菜單並獲取用戶選項
    
    Returns:
        str: 用戶選擇的選項
    """
    print("\n" + "=" * 50)
    print(" Weaviate 語音資料庫刪除工具 ".center(50, "="))
    print("=" * 50)
    print("1. 列出所有說話者")
    print("2. 搜尋說話者")
    print("3. 刪除說話者及其所有聲紋資料")
    print("4. 刪除特定聲紋資料")
    print("5. 依聲紋 UUID 刪除聲紋")  # 新增選項
    print("0. 退出程式")
    print("-" * 50)
    
    return input("請選擇操作 (0-5): ").strip()


def main() -> None:
    """
    主程序函數
    """
    print("Weaviate 語音資料庫刪除工具")
    print("===========================")
    
    # 連接 Weaviate
    client = connect_to_weaviate()
    if client is None:
        print("無法連接到 Weaviate 資料庫，程式退出。")
        return
    
    try:
        while True:
            choice = show_menu()
            
            if choice == "0":
                print("\n謝謝使用，再見！")
                break
                
            elif choice == "1":
                # 列出所有說話者
                speakers = get_all_speakers(client)
                display_speakers(speakers)
                
            elif choice == "2":
                # 搜尋說話者
                handle_search_speaker(client)
                
            elif choice == "3":
                # 刪除說話者及其所有聲紋資料
                handle_delete_speaker(client)
                
            elif choice == "4":
                # 刪除特定聲紋資料
                handle_delete_single_voiceprint(client)
            elif choice == "5":
                # 依聲紋 UUID 刪除聲紋（簡化版，不查詢說話者）
                voiceprint_id = input("請輸入要刪除的聲紋 UUID: ").strip()
                if not voiceprint_id:
                    print("聲紋 UUID 不能為空！")
                else:
                    success = delete_voiceprint_simple(client, voiceprint_id)
                    if success:
                        print("刪除操作完成！")
                    else:
                        print("刪除操作失敗，請檢查 UUID 是否正確或查看錯誤訊息。")
            else:
                print("無效的選項，請重新選擇！")
            
            # 等待用戶確認繼續
            if choice != "0":
                input("\n按 Enter 鍵繼續...")
            
    except KeyboardInterrupt:
        print("\n\n程式被中斷，正在退出...")
    except Exception as e:
        print(f"\n執行過程中發生錯誤: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")
    finally:
        # 關閉連接
        if client:
            client.close()
            print("已關閉 Weaviate 連接")


def delete_voiceprint_by_id(client: weaviate.WeaviateClient, voiceprint_id: str) -> bool:
    """
    直接刪除指定聲紋，並自動從對應說話者的聲紋列表移除該聲紋 ID。

    Args:
        client: Weaviate 客戶端連接
        voiceprint_id: 要刪除的聲紋 UUID
    
    Returns:
        bool: 操作是否成功
    """
    try:
        voice_print_collection = client.collections.get("VoicePrint")
        # 只查詢 speaker_name
        voiceprint_obj = voice_print_collection.query.fetch_object_by_id(
            uuid=voiceprint_id,
            return_properties=["speaker_name"]
        )
        if not voiceprint_obj:
            print(f"找不到 ID 為 {voiceprint_id} 的聲紋資料")
            return False
        speaker_name = voiceprint_obj.properties.get("speaker_name")
        if not speaker_name:
            print(f"聲紋資料未包含說話者名稱，無法自動移除聲紋 ID")
            return False
        # 反查說話者 UUID
        speakers = search_speakers_by_name(client, speaker_name)
        if not speakers:
            print(f"找不到名稱為 '{speaker_name}' 的說話者，無法自動移除聲紋 ID")
            return False
        # 若有多個同名，只移除第一個
        speaker_id = speakers[0]["uuid"]
        # 執行聲紋刪除
        success = delete_voiceprint(client, voiceprint_id)
        if not success:
            return False
        # 從說話者的聲紋列表移除該聲紋 ID
        remove_voiceprint_from_speaker(client, speaker_id, voiceprint_id)
        print(f"已完成聲紋 {voiceprint_id} 的刪除與說話者列表更新。")
        return True
    except Exception as e:
        print(f"刪除指定聲紋時發生錯誤: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")
        return False


def delete_voiceprint_simple(client: weaviate.WeaviateClient, voiceprint_id: str) -> bool:
    """
    只根據 UUID 直接刪除 VoicePrint，不查詢任何屬性，也不處理說話者關聯。

    Args:
        client: Weaviate 客戶端連接
        voiceprint_id: 要刪除的聲紋 UUID
    Returns:
        bool: 是否成功刪除
    """
    try:
        voice_print_collection = client.collections.get("VoicePrint")
        voice_print_collection.data.delete_by_id(voiceprint_id)
        print(f"已成功刪除 VoicePrint (ID: {voiceprint_id})")
        return True
    except Exception as e:
        print(f"刪除 VoicePrint 時發生錯誤: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    main()