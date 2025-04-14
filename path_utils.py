#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路徑處理工具模組
用於解析和查找音訊檔案路徑，支援多種路徑格式和智能檔案搜尋

適用於 Windows 系統中的路徑處理，處理反斜線、引號等特殊字元
"""

import os
import sys
from typing import List, Optional, Tuple, Dict, Any

class PathResult:
    """路徑處理結果類別，封裝搜尋和解析的結果"""
    
    def __init__(self, 
                 is_valid: bool, 
                 path: str = "", 
                 found_files: List[str] = None,
                 tried_paths: List[str] = None,
                 is_file: bool = False,
                 is_dir: bool = False,
                 error_message: str = "") -> None:
        """
        初始化路徑處理結果
        
        Args:
            is_valid: 是否找到有效路徑
            path: 找到的有效路徑（如果有）
            found_files: 搜尋到的所有匹配檔案列表
            tried_paths: 已嘗試的所有路徑
            is_file: 找到的路徑是否為檔案
            is_dir: 找到的路徑是否為目錄
            error_message: 錯誤訊息（如果有錯誤）
        """
        self.is_valid = is_valid
        self.path = path
        self.found_files = found_files or []
        self.tried_paths = tried_paths or []
        self.is_file = is_file
        self.is_dir = is_dir
        self.error_message = error_message
        
    def __bool__(self) -> bool:
        """將結果物件視為布林值時，根據是否找到有效路徑返回"""
        return self.is_valid
    
    def __str__(self) -> str:
        """返回結果的字串表示"""
        if self.is_valid:
            type_str = "檔案" if self.is_file else "目錄" if self.is_dir else "路徑"
            return f"有效{type_str}: {self.path}"
        else:
            return f"無效路徑{f': {self.error_message}' if self.error_message else ''}"


def search_file_in_directory(directory: str, filename: str, verbose: bool = False) -> List[str]:
    """
    在指定目錄及其子目錄中搜尋檔案
    
    Args:
        directory: 要搜尋的起始目錄
        filename: 要尋找的檔案名稱
        verbose: 是否顯示詳細的搜尋過程
        
    Returns:
        List[str]: 找到的所有匹配檔案路徑列表
    """
    found_files = []
    
    if verbose:
        print(f"在 {directory} 中搜尋檔案: {filename}")
    
    try:
        # 確保目錄存在
        if not os.path.exists(directory):
            if verbose:
                print(f"目錄 {directory} 不存在")
            return found_files
        
        # 清理檔名中可能的引號
        clean_filename = filename
        if clean_filename.startswith('"') and clean_filename.endswith('"'):
            clean_filename = clean_filename[1:-1]
        
        # 遞迴搜尋匹配檔案
        for root, _, files in os.walk(directory):
            for file in files:
                if file == clean_filename or clean_filename in file:
                    found_path = os.path.join(root, file)
                    found_files.append(found_path)
                    if verbose:
                        print(f"找到匹配檔案: {found_path}")
        
        if verbose:
            if found_files:
                print(f"共找到 {len(found_files)} 個匹配檔案")
            else:
                print("未找到匹配檔案")
                
        return found_files
    
    except Exception as e:
        if verbose:
            print(f"搜尋檔案時發生錯誤: {str(e)}")
        return found_files


def normalize_path(raw_path: str) -> str:
    """
    標準化路徑，處理反斜線和引號
    
    Args:
        raw_path: 原始路徑字串
        
    Returns:
        str: 標準化後的路徑
    """
    # 移除引號
    path = raw_path
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    
    # 標準化路徑分隔符
    path = os.path.normpath(path)
    
    return path


def get_possible_paths(raw_path: str) -> List[str]:
    """
    生成多種可能的路徑格式
    
    Args:
        raw_path: 原始路徑字串
        
    Returns:
        List[str]: 所有可能的路徑格式列表
    """
    # 清理原始路徑
    clean_path = raw_path
    if clean_path.startswith('"') and clean_path.endswith('"'):
        clean_path = clean_path[1:-1]
    
    # 生成各種可能的路徑格式
    paths = [
        clean_path,                                  # 原始輸入
        os.path.normpath(clean_path),                # 標準化路徑
        os.path.abspath(clean_path),                 # 絕對路徑
        os.path.join(os.getcwd(), clean_path),       # 從當前目錄解析
        # 針對 Windows 的特殊處理
        clean_path.replace('\\', '/'),               # 替換反斜線為正斜線
        os.path.join(os.getcwd(), clean_path.replace('\\', '/')),  # 從當前目錄解析，使用正斜線
    ]
    
    # 如果是單純的檔名，嘗試在常見目錄中尋找
    if os.path.basename(clean_path) == clean_path:
        filename = clean_path
        common_dirs = [
            os.path.join(os.getcwd(), "testFiles", "test_audioFile"),
            os.path.join(os.getcwd(), "testFiles", "audioFile"),
            os.path.join(os.getcwd(), "testFiles"),
        ]
        
        for directory in common_dirs:
            paths.append(os.path.join(directory, filename))
    
    # 移除重複項目
    unique_paths = list(dict.fromkeys(paths))
    
    return unique_paths


def resolve_audio_path(raw_path: str, verbose: bool = False) -> PathResult:
    """
    智能解析音訊檔案路徑，支援多種格式和智能搜尋
    
    Args:
        raw_path: 原始路徑字串
        verbose: 是否顯示詳細處理過程
        
    Returns:
        PathResult: 路徑處理結果
    """
    if verbose:
        print(f"原始輸入路徑: {raw_path}")
    
    # 步驟 1: 提取檔名，便於後續搜尋
    norm_path = normalize_path(raw_path)
    filename = os.path.basename(norm_path.replace('\\', '/'))
    
    if verbose:
        print(f"提取檔名: {filename}")
    
    # 步驟 2: 在測試資料夾中搜尋檔案
    test_dirs = [
        os.path.join(os.getcwd(), "testFiles", "test_audioFile"),
        os.path.join(os.getcwd(), "testFiles", "audioFile"),
    ]
    
    found_files = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            if verbose:
                print(f"掃描資料夾: {test_dir}")
            found = search_file_in_directory(test_dir, filename, verbose=verbose)
            found_files.extend(found)
            if found and verbose:
                print(f"在 {test_dir} 中找到 {len(found)} 個匹配檔案")
    
    # 步驟 3: 如果找到檔案，返回第一個匹配結果
    if found_files:
        valid_path = found_files[0]
        is_file = os.path.isfile(valid_path)
        is_dir = os.path.isdir(valid_path)
        
        if verbose:
            print(f"使用找到的{'檔案' if is_file else '目錄'}: {valid_path}")
        
        return PathResult(
            is_valid=True,
            path=valid_path,
            found_files=found_files,
            is_file=is_file,
            is_dir=is_dir
        )
    
    # 步驟 4: 如果未找到檔案，嘗試直接解析路徑
    if verbose:
        print("在測試資料夾中未找到匹配的檔案，嘗試直接解析路徑...")
    
    # 獲取所有可能的路徑格式
    paths_to_try = get_possible_paths(raw_path)
    
    # 驗證每個路徑
    for path in paths_to_try:
        if verbose:
            print(f"嘗試路徑: {path}")
        
        if os.path.exists(path):
            is_file = os.path.isfile(path)
            is_dir = os.path.isdir(path)
            
            if verbose:
                print(f"  ✓ 路徑有效! ({'檔案' if is_file else '目錄' if is_dir else '未知類型'})")
            
            return PathResult(
                is_valid=True,
                path=path,
                tried_paths=paths_to_try,
                is_file=is_file,
                is_dir=is_dir
            )
        elif verbose:
            print(f"  ✗ 路徑無效")
    
    # 所有嘗試都失敗，返回失敗結果
    error_message = f"找不到 {raw_path}"
    if verbose:
        print(f"所有路徑嘗試均失敗: {error_message}")
    
    return PathResult(
        is_valid=False,
        tried_paths=paths_to_try,
        error_message=error_message
    )


def print_path_suggestions(result: PathResult) -> None:
    """
    輸出路徑建議
    
    Args:
        result: 路徑處理結果
    """
    print("\n建議:")
    print("1. 使用完整的絕對路徑")
    print("2. 將音檔檔名改為簡單格式 (例如: test.wav)")
    print("3. 確認檔案確實存在於您指定的位置")
    print("4. 嘗試使用以下語法:")
    print("   python main_identify_v4_weaviate.py \"./testFiles/test_audioFile/0009/9-5.wav\"")
    print("   或 (使用絕對路徑)")
    print(f"   python main_identify_v4_weaviate.py \"{os.path.join(os.getcwd(), 'testFiles', 'test_audioFile', '0009', '9-5.wav')}\"")
    
    # 嘗試列出可用目錄
    test_dir = os.path.join(os.getcwd(), "testFiles", "test_audioFile")
    if os.path.exists(test_dir):
        print("\n測試資料夾中的可用目錄:")
        for item in sorted(os.listdir(test_dir)):
            item_path = os.path.join(test_dir, item)
            if os.path.isdir(item_path):
                print(f"  - {item}/")


def print_path_result(result: PathResult) -> None:
    """
    輸出路徑處理結果的詳細資訊
    
    Args:
        result: 路徑處理結果
    """
    if result.is_valid:
        print(f"有效路徑: {result.path}")
        if result.is_file:
            print("路徑類型: 檔案")
        elif result.is_dir:
            print("路徑類型: 目錄")
    else:
        print(f"無效的路徑")
        print(f"已嘗試以下路徑:")
        for p in result.tried_paths:
            print(f"  - {p}")
        print_path_suggestions(result)


# 單元測試
if __name__ == "__main__":
    # 測試路徑處理
    test_paths = [
        "test.wav",
        "./test.wav",
        "test_audioFile\\0009\\9-5.wav",
        "./testFiles/test_audioFile/0009/9-5.wav"
    ]
    
    print("=== 路徑處理工具測試 ===")
    for path in test_paths:
        print(f"\n測試路徑: {path}")
        result = resolve_audio_path(path, verbose=True)
        print_path_result(result)
        print("-" * 50)