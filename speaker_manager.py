"""
語者與聲紋管理系統 (Weaviate 版)
==================================

本模組提供語者（Speaker）與聲紋（VoicePrint）在 Weaviate 資料庫的管理功能。
CLI 互動介面與資料操作分離，結構現代、易於維護。

依賴：
    - weaviate-client
    - Python 3.9+
"""

import os
import re
import sys
import uuid
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone, timedelta
from weaviate.classes.query import Filter # type: ignore

try:
    import weaviate  # type: ignore
except ImportError:
    print("請先安裝 weaviate-client：pip install weaviate-client")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging 設定
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("speaker_manager")

# ---------------------------------------------------------------------------
# UUID 工具
# ---------------------------------------------------------------------------
UUID_PATTERN = re.compile(r"^[0-9a-fA-F-]{36}$")

def valid_uuid(value: str) -> bool:
    """檢查字串是否為有效 UUID 格式。"""
    return bool(UUID_PATTERN.match(value))

# ---------------------------------------------------------------------------
# SpeakerManager (Weaviate 操作)
# ---------------------------------------------------------------------------

class SpeakerManager:
    """
    封裝所有與 Weaviate 資料庫互動的語者管理操作。
    """
    CLASS_NAME = "Speaker"
    VP_CLASS_NAME = "VoicePrint"

    def __init__(self, client: weaviate.WeaviateClient):
        self._client = client

    def list_all_speakers(self) -> List[Dict[str, Any]]:
        """列出所有說話者。"""
        try:
            results = (
                self._client.collections.get(self.CLASS_NAME)
                .query.fetch_objects()
            )
            speakers: List[Dict[str, Any]] = []
            for obj in results.objects:
                speakers.append({
                    "uuid": str(obj.uuid),
                    "name": obj.properties.get("name", "未命名"),
                    "create_time": obj.properties.get("create_time", "未知"),
                    "last_active_time": obj.properties.get("last_active_time", "未知"),
                    "voiceprint_count": len(obj.properties.get("voiceprint_ids", [])),
                })
            speakers.sort(key=lambda s: s["last_active_time"], reverse=True)
            return speakers
        except Exception as exc:
            logger.error(f"列出說話者時發生錯誤: {exc}")
            return []

    def get_speaker(self, speaker_uuid: str) -> Optional[Any]:
        """取得單一說話者物件。"""
        try:
            return (
                self._client.collections.get(self.CLASS_NAME)
                .query.fetch_object_by_id(uuid=speaker_uuid)
            )
        except Exception as exc:
            logger.error(f"獲取說話者詳細資訊時發生錯誤: {exc}")
            return None

    def update_speaker_name(self, speaker_uuid: str, new_name: str) -> bool:
        """
        更改說話者名稱，並同步更新所有該語者底下聲紋的 speaker_name。
        """
        try:
            # 1.先更新 Speaker 本身
            sp_col = self._client.collections.get(self.CLASS_NAME)
            sp_col.data.update(uuid=speaker_uuid, properties={"name": new_name})

            # 2.拿回這個 Speaker 物件，讀出 voiceprint_ids
            sp_obj = sp_col.query.fetch_object_by_id(uuid=speaker_uuid)
            vp_ids = sp_obj.properties.get("voiceprint_ids", [])

            # 3.逐一更新每支 VoicePrint
            vp_col = self._client.collections.get(self.VP_CLASS_NAME)
            for vp_id in vp_ids:
                vp_col.data.update(
                    uuid=vp_id,
                    properties={"speaker_name": new_name}
                )

            return True
        except Exception as exc:
            logger.error(f"更改說話者名稱時發生錯誤: {exc}")
            return False

    def delete_speaker(self, speaker_uuid: str) -> bool:
        """刪除說話者。"""
        try:
            collection = self._client.collections.get(self.CLASS_NAME)
            collection.data.delete_by_id(uuid=speaker_uuid)
            return True
        except Exception as exc:
            logger.error(f"刪除說話者時發生錯誤: {exc}")
            return False

    def transfer_voiceprints(
        self, source_uuid: str, dest_uuid: str, voiceprint_ids: Optional[List[str]] = None
    ) -> bool:
        """
        將來源說話者的聲紋轉移到目標說話者，並同步更新聲紋的 speaker_id 與 speaker_name。
        若來源說話者已無聲紋，則自動刪除該說話者。
        """
        try:
            collection = self._client.collections.get(self.CLASS_NAME)
            src_obj = collection.query.fetch_object_by_id(uuid=source_uuid)
            dest_obj = collection.query.fetch_object_by_id(uuid=dest_uuid)
            if not src_obj or not dest_obj:
                logger.warning("來源或目標說話者不存在。")
                return False
            src_vps = set(src_obj.properties.get("voiceprint_ids", []))
            dest_vps = set(dest_obj.properties.get("voiceprint_ids", []))
            move_set = src_vps if voiceprint_ids is None else set(voiceprint_ids)
            dest_vps.update(move_set)
            src_vps.difference_update(move_set)
            # 更新來源與目標說話者的聲紋
            collection.data.update(uuid=source_uuid, properties={"voiceprint_ids": list(src_vps)})
            collection.data.update(uuid=dest_uuid, properties={"voiceprint_ids": list(dest_vps)})
            # 取得目標語者名稱
            dest_name = dest_obj.properties.get("name", "未命名")
            # 批次更新被轉移聲紋的 speaker_id 與 speaker_name
            vp_collection = self._client.collections.get(self.VP_CLASS_NAME)
            for vp_id in move_set:
                try:
                    vp_collection.data.update(uuid=vp_id, properties={
                        "speaker_name": dest_name
                    }, references={"speaker": [dest_uuid]})
                except Exception as e:
                    logger.error(f"轉移聲紋 {vp_id} 時發生錯誤: {e}")
            # 若來源說話者已無聲紋，自動刪除
            if not src_vps:
                try:
                    collection.data.delete_by_id(uuid=source_uuid)
                    logger.info(f"來源說話者 {source_uuid} 已無聲紋，自動刪除。")
                except Exception as del_exc:
                    logger.error(f"自動刪除來源說話者時發生錯誤: {del_exc}")
            return True
        except Exception as exc:
            logger.error(f"轉移聲紋時發生錯誤: {exc}")
            return False

    def cleanup(self) -> None:
        """資料庫清理（示意）。"""
        logger.info("呼叫成功 (功能尚未開發)。")

# ---------------------------------------------------------------------------
# CLI (Command‑line Interface)
# ---------------------------------------------------------------------------

class SpeakerManagerCLI:
    """命令列互動介面 (CLI)。"""
    MENU = (
        """
============================================================
                       語者與聲紋管理系統
============================================================
1. 列出所有說話者
2. 檢視說話者詳細資訊
3. 更改說話者名稱
4. 轉移聲紋到其他說話者
5. 刪除說話者
6. 資料庫清理與修復
0. 離開
------------------------------------------------------------
"""
    )

    def __init__(self, manager: SpeakerManager) -> None:
        self.manager = manager
        self.index2uuid: Dict[str, str] = {}

    @staticmethod
    def _print_speakers_table(speakers: List[Dict[str, Any]]) -> None:
        header = f"{'No.':<4} | {'ID':<36} | {'名稱':<20} | {'聲紋數量':<10} | {'最後活動時間'}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for idx, sp in enumerate(speakers, start=1):
            print(
                f"{idx:<4} | {sp['uuid']:<36} | {sp['name']:<20} | "
                f"{sp['voiceprint_count']:<10} | {sp['last_active_time']}"
            )
        print("-" * len(header))

    def _resolve_id(self, raw: str) -> Optional[str]:
        raw = raw.strip()
        return self.index2uuid.get(raw) if raw.isdigit() else raw

    def _action_list(self) -> None:
        speakers = self.manager.list_all_speakers()
        if not speakers:
            print("目前沒有說話者紀錄。")
            return
        self.index2uuid = {str(i): sp["uuid"] for i, sp in enumerate(speakers, start=1)}
        self._print_speakers_table(speakers)

    def _action_view(self) -> None:
        raw = input("請輸入說話者序號或 ID: ")
        sp_id = self._resolve_id(raw)
        if not sp_id or not valid_uuid(sp_id):
            print("❌ 無效的說話者 ID，請重新嘗試。")
            return
        obj = self.manager.get_speaker(sp_id)
        if obj is None:
            print("❌ 查無此說話者。")
            return
        props = obj.properties
        print("\n說話者詳細資訊:")
        print(f"UUID            : {obj.uuid}")
        print(f"名稱            : {props.get('name', '未命名')}")
        print(f"建立時間        : {props.get('create_time', '未知')}")
        print(f"最後活動時間    : {props.get('last_active_time', '未知')}")
        print(f"聲紋數量        : {len(props.get('voiceprint_ids', []))}\n")

    def _action_rename(self) -> None:
        raw = input("請輸入說話者序號或 ID: ")
        sp_id = self._resolve_id(raw)
        if not sp_id or not valid_uuid(sp_id):
            print("❌ 無效的說話者 ID。")
            return
        new_name = input("請輸入新名稱: ").strip()
        if not new_name:
            print("❌ 名稱不可為空。")
            return
        if self.manager.update_speaker_name(sp_id, new_name):
            print("✅ 名稱已更新。")
        else:
            print("❌ 更新失敗，請查看日誌。")

    def _action_transfer(self) -> None:
        src_raw = input("請輸入來源說話者序號或 ID: ")
        src_id = self._resolve_id(src_raw)
        dest_raw = input("請輸入目標說話者序號或 ID: ")
        dest_id = self._resolve_id(dest_raw)
        if not src_id or not dest_id or not (valid_uuid(src_id) and valid_uuid(dest_id)):
            print("❌ 無效的來源或目標 ID。")
            return
        confirm = input("確定要轉移所有聲紋? (y/N): ").lower()
        if confirm != "y":
            print("已取消。")
            return
        if self.manager.transfer_voiceprints(src_id, dest_id):
            print("✅ 聲紋已轉移。")
        else:
            print("❌ 轉移失敗，請查看日誌。")

    def _action_delete(self) -> None:
        raw = input("請輸入要刪除的說話者序號或 ID: ")
        sp_id = self._resolve_id(raw)
        if not sp_id or not valid_uuid(sp_id):
            print("❌ 無效的說話者 ID。")
            return
        confirm = input("⚠️  此操作無法復原，確定刪除? (y/N): ").lower()
        if confirm != "y":
            print("已取消。")
            return
        if self.manager.delete_speaker(sp_id):
            print("✅ 說話者已刪除。")
        else:
            print("❌ 刪除失敗，請查看日誌。")

    def _action_cleanup(self) -> None:
        self.manager.cleanup()
        print("✅ 資料庫清理完成。")

    def run(self) -> None:
        actions = {
            "1": self._action_list,
            "2": self._action_view,
            "3": self._action_rename,
            "4": self._action_transfer,
            "5": self._action_delete,
            "6": self._action_cleanup,
            "0": lambda: sys.exit(0),
        }
        while True:
            print(self.MENU)
            choice = input("請選擇操作 (0-6): ").strip()
            action = actions.get(choice)
            if action:
                action()
            else:
                print("❌ 無效選項，請重新輸入 0-6。\n")

# ---------------------------------------------------------------------------
# 入口點
# ---------------------------------------------------------------------------

def _build_weaviate_client() -> weaviate.WeaviateClient:
    """建立 Weaviate 連線。"""
    try:
        client = weaviate.connect_to_local()
        return client
    except Exception as exc:
        logger.error(f"連線 Weaviate 失敗: {exc}")
        sys.exit(1)

def main() -> None:
    client = _build_weaviate_client()
    manager = SpeakerManager(client)
    cli = SpeakerManagerCLI(manager)
    cli.run()

if __name__ == "__main__":
    main()