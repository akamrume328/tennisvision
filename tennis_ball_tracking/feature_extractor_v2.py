import json
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from tqdm import tqdm # tqdmをインポート
import numba
import cv2

warnings.filterwarnings('ignore')

@numba.jit(nopython=True, parallel=True)
def vectorized_rolling_stats(data_matrix: np.ndarray, window: int, 
                            is_interpolated: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ベクトル化された移動統計計算 (Numba JITコンパイル対応、axis引数問題を完全修正)"""
    n_rows, n_cols = data_matrix.shape
    
    ma_results = np.zeros_like(data_matrix)
    std_results = np.zeros_like(data_matrix)
    max_results = np.zeros_like(data_matrix)
    min_results = np.zeros_like(data_matrix)
    
    half_window = window // 2
    
    # --- パディング処理 (手動実装) ---
    padded_data = np.empty((n_rows + 2 * half_window, n_cols), dtype=np.float64)
    padded_data[half_window:half_window + n_rows] = data_matrix
    for i in range(half_window):
        padded_data[i] = data_matrix[0]
        padded_data[n_rows + half_window + i] = data_matrix[-1]

    weights = np.where(is_interpolated, 0.3, 0.8)
    
    padded_weights = np.empty(n_rows + 2 * half_window, dtype=np.float64)
    padded_weights[half_window:half_window + n_rows] = weights
    for i in range(half_window):
        padded_weights[i] = weights[0]
        padded_weights[n_rows + half_window + i] = weights[-1]

    # --- スライディングウィンドウでの計算 (並列化) ---
    for i in numba.prange(n_rows):
        window_start = i
        window_end = i + window
        
        window_data = padded_data[window_start:window_end]
        window_weights = padded_weights[window_start:window_end]
        
        # --- 平均値 (mean) の計算 ---
        weight_sum = np.sum(window_weights)
        if weight_sum > 0:
            weighted_data = window_data * window_weights.reshape(-1, 1)
            ma_results[i, :] = np.sum(weighted_data, axis=0) / weight_sum
        else:
            ma_results[i, :] = np.sum(window_data, axis=0) / window_data.shape[0]

        # ★★★ ここからが今回の修正箇所 ★★★

        # --- 標準偏差 (std) の計算 (手動実装) ---
        # E[X^2] - (E[X])^2 を利用
        mean_val = np.sum(window_data, axis=0) / window_data.shape[0]
        mean_sq_val = np.sum(window_data**2, axis=0) / window_data.shape[0]
        variance = mean_sq_val - mean_val**2
        # 浮動小数点誤差で負になるのを防ぐ
        for j in range(n_cols):
            if variance[j] < 0:
                variance[j] = 0
        std_results[i, :] = np.sqrt(variance)

        # --- 最大値 (max) の計算 (手動実装) ---
        # 各列ごとに最大値を計算するループ
        for j in range(n_cols):
            max_results[i, j] = np.max(window_data[:, j])

        # --- 最小値 (min) の計算 (手動実装) ---
        # 各列ごとに最小値を計算するループ
        for j in range(n_cols):
            min_results[i, j] = np.min(window_data[:, j])
    
    return ma_results, std_results, max_results, min_results



class TennisFeatureExtractor:
    """
    テニス動画の局面アノテーションデータとトラッキング特徴量を統合し、
    機械学習用の特徴量を作成するクラス
    """
    
    def __init__(self, training_data_dir: str = "training_data"):
        self.training_data_dir = Path(training_data_dir)
        self.features_dir = self.training_data_dir / "features"
        self.features_dir.mkdir(exist_ok=True)
        
        self.phase_labels = [
            "point_interval",           # 0: ポイント間
            "rally",                   # 1: ラリー中
            "serve_front_deuce",      # 3: 手前デュースサイドからのサーブ
            "serve_front_ad",         # 4: 手前アドサイドからのサーブ
            "serve_back_deuce",       # 5: 奥デュースサイドからのサーブ
            "serve_back_ad",          # 6: 奥アドサイドからのサーブ
            "changeover"              # 7: チェンジコート間
        ]
        self.label_to_id = {label: idx for idx, label in enumerate(self.phase_labels)}
        
        print(f"特徴量抽出器を初期化しました")
        print(f"データディレクトリ: {self.training_data_dir}")
        print(f"特徴量保存先: {self.features_dir}")
        print(f"対象局面数: {len(self.phase_labels)}")
    
    # データ読み込みメソッド群
    def load_phase_annotations(self, video_name: str = None) -> Dict[str, Any]:
        """局面アノテーションファイルを読み込み"""
        pattern = "phase_annotations_*.json"
        if video_name:
            pattern = f"phase_annotations_{video_name}_*.json"
        
        annotation_files = list(self.training_data_dir.glob(pattern))
        
        if not annotation_files:
            print(f"⚠️  局面アノテーションファイルが見つかりません: {pattern}")
            return {}
        
        all_annotations = {}
        for file_path in annotation_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                video_key = data.get('video_name', file_path.stem)
                all_annotations[video_key] = data
                print(f"✅ 局面アノテーション読み込み: {file_path.name}")
                print(f"   動画: {video_key}, 局面変更数: {len(data.get('phase_changes', []))}")
                
            except Exception as e:
                print(f"❌ 読み込みエラー: {file_path.name} - {e}")
        
        return all_annotations
    
    def load_tracking_features(self, video_name: str = None) -> Dict[str, Dict]:
        """トラッキング特徴量ファイルを読み込み（メタデータ含む）"""
        pattern = "tracking_features_*.json"
        if video_name:
            pattern = f"tracking_features_{video_name}_*.json"
        
        tracking_files = list(self.training_data_dir.glob(pattern))
        
        if not tracking_files:
            print(f"⚠️  トラッキング特徴量ファイルが見つかりません: {pattern}")
            return {}
        
        all_tracking = {}
        for file_path in tracking_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # ファイル名から動画名を推定
                filename_parts = file_path.stem.split('_')
                if len(filename_parts) >= 3:
                    # tracking_features_video_name_timestamp の形式を想定
                    video_key = '_'.join(filename_parts[2:-1])  # タイムスタンプを除去
                else:
                    video_key = file_path.stem
                
                # データ構造をチェック（新形式か旧形式か）
                if isinstance(data, dict) and 'metadata' in data and 'frames' in data:
                    # 新形式：メタデータとフレームデータが分離されている
                    all_tracking[video_key] = data
                    print(f"✅ トラッキング特徴量読み込み（新形式）: {file_path.name}")
                    print(f"   動画: {video_key}, フレーム数: {len(data['frames'])}")
                    if 'frame_skip' in data['metadata']:
                        print(f"   フレームスキップ: {data['metadata']['frame_skip']}")
                elif isinstance(data, list):
                    # 旧形式：フレームデータのみのリスト
                    all_tracking[video_key] = {
                        'metadata': {'frame_skip': 1, 'legacy_format': True},
                        'frames': data
                    }
                    print(f"✅ トラッキング特徴量読み込み（旧形式）: {file_path.name}")
                    print(f"   動画: {video_key}, フレーム数: {len(data)}")
                    print(f"   注意: 旧形式のためフレームスキップ情報なし")
                else:
                    print(f"⚠️  不明なデータ形式: {file_path.name}")
                    print(f"   データ型: {type(data)}")
                    if isinstance(data, dict):
                        print(f"   キー: {list(data.keys())[:10]}")
                    continue
                
            except Exception as e:
                print(f"❌ 読み込みエラー: {file_path.name} - {e}")
        
        return all_tracking
    
    def load_court_coordinates(self, video_name: str = None) -> Dict[str, Dict]:
        """コート座標データを読み込み"""
        pattern = "court_coords_*.json"
        if video_name:
            pattern = f"court_coords_{video_name}_*.json"
        
        court_files = list(self.training_data_dir.glob(pattern))
        
        if not court_files:
            print(f"⚠️  コート座標ファイルが見つかりません: {pattern}")
            return {}
        
        all_court_coords = {}
        for file_path in court_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # ファイル名から動画名を推定
                filename_parts = file_path.stem.split('_')
                if len(filename_parts) >= 3:
                    video_key = '_'.join(filename_parts[2:])  # court_coords_を除く
                else:
                    video_key = file_path.stem
                
                all_court_coords[video_key] = data
                print(f"✅ コート座標読み込み: {file_path.name}")
                print(f"   動画: {video_key}")
                
            except Exception as e:
                print(f"❌ 読み込みエラー: {file_path.name} - {e}")
        
        return all_court_coords
    
    # データマッチング・検証メソッド群
    def match_video_data(self, phase_annotations: Dict, tracking_features: Dict, court_coordinates: Dict = None) -> List[Tuple[str, Dict, Dict, Dict]]:
        """動画名でアノテーションデータ、トラッキングデータ、コート座標をマッチング（データ構造更新・診断強化）"""
        matched_data = []
        
        print("\n=== データマッチング ===")
        print(f"局面アノテーション動画数: {len(phase_annotations)}")
        print(f"トラッキング特徴量動画数: {len(tracking_features)}")
        print(f"コート座標動画数: {len(court_coordinates) if court_coordinates else 0}")
        
        print("\n--- 利用可能なファイル ---")
        print("局面アノテーション:")
        for key in phase_annotations.keys():
            print(f"  - {key}")
        print("トラッキング特徴量:")
        for key in tracking_features.keys():
            print(f"  - {key}")
        if court_coordinates:
            print("コート座標:")
            for key in court_coordinates.keys():
                print(f"  - {key}")
        
        print("\n--- マッチング処理 ---")
        
        for video_name in phase_annotations.keys():
            tracking_data = None
            court_data = None
            tracking_match_type = "なし"
            court_match_type = "なし"
            tracking_matched_key = ""
            court_matched_key = ""
            
            print(f"\n🎯 処理中: {video_name}")
            
            # トラッキングデータのマッチング
            if video_name in tracking_features:
                tracking_data = tracking_features[video_name]
                tracking_match_type = "完全一致"
                tracking_matched_key = video_name
                print(f"  📊 トラッキング: ✅ 完全一致 - {video_name}")
                
                # フレーム数の詳細チェック
                actual_frame_count = self.get_actual_frame_count(tracking_data)
                print(f"      実際のフレーム数: {actual_frame_count}")
                
                if actual_frame_count <= 10:
                    print(f"      ⚠️  フレーム数が異常に少ないです")
            else:
                # 部分マッチを試行
                for tracking_key in tracking_features.keys():
                    if video_name in tracking_key:
                        tracking_data = tracking_features[tracking_key]
                        tracking_match_type = "部分一致(動画名がキーに含まれる)"
                        tracking_matched_key = tracking_key
                        print(f"  📊 トラッキング: ✅ 部分一致 - {video_name} ⊆ {tracking_key}")
                        
                        actual_frame_count = self.get_actual_frame_count(tracking_data)
                        print(f"      実際のフレーム数: {actual_frame_count}")
                        break
                    elif tracking_key in video_name:
                        tracking_data = tracking_features[tracking_key]
                        tracking_match_type = "部分一致(キーが動画名に含まれる)"
                        tracking_matched_key = tracking_key
                        print(f"  📊 トラッキング: ✅ 部分一致 - {tracking_key} ⊆ {video_name}")
                        
                        actual_frame_count = self.get_actual_frame_count(tracking_data)
                        print(f"      実際のフレーム数: {actual_frame_count}")
                        break
                
                if not tracking_data:
                    print(f"  📊 トラッキング: ❌ マッチなし")
            
            # コート座標のマッチング
            if court_coordinates:
                if video_name in court_coordinates:
                    court_data = court_coordinates[video_name]
                    court_match_type = "完全一致"
                    court_matched_key = video_name
                    print(f"  🎾 コート座標: ✅ 完全一致 - {video_name}")
                else:
                    # 部分マッチを試行
                    for court_key in court_coordinates.keys():
                        if video_name in court_key:
                            court_data = court_coordinates[court_key]
                            court_match_type = "部分一致(動画名がキーに含まれる)"
                            court_matched_key = court_key
                            print(f"  🎾 コート座標: ✅ 部分一致 - {video_name} ⊆ {court_key}")
                            break
                        elif court_key in video_name:
                            court_data = court_coordinates[court_key]
                            court_match_type = "部分一致(キーが動画名に含まれる)"
                            court_matched_key = court_key
                            print(f"  🎾 コート座標: ✅ 部分一致 - {court_key} ⊆ {video_name}")
                            break
                    if not court_data:
                        print(f"  🎾 コート座標: ❌ マッチなし")
            else:
                print(f"  🎾 コート座標: ❌ データなし")
            
            # 結果の判定
            if tracking_data:
                matched_data.append((
                    video_name,
                    phase_annotations[video_name],
                    tracking_data,  # 新形式: {'metadata': {...}, 'frames': [...]}
                    court_data
                ))
                
                frame_count = self.get_actual_frame_count(tracking_data)
                print(f"  ✅ マッチング成功:")
                print(f"     局面アノテーション: {video_name}")
                print(f"     トラッキング: {tracking_matched_key} ({tracking_match_type})")
                print(f"     実際のフレーム数: {frame_count}")
                if isinstance(tracking_data, dict) and 'metadata' in tracking_data:
                    frame_skip = tracking_data['metadata'].get('frame_skip', 1)
                    print(f"     フレームスキップ: {frame_skip}")
                if court_data:
                    print(f"     コート座標: {court_matched_key} ({court_match_type})")
                else:
                    print(f"     コート座標: なし")
            else:
                print(f"  ❌ マッチング失敗: トラッキングデータが見つかりません")
        
        print(f"\n=== マッチング結果サマリー ===")
        print(f"✅ 成功したマッチング: {len(matched_data)}")
        print(f"❌ 失敗したマッチング: {len(phase_annotations) - len(matched_data)}")
        
        if matched_data:
            print("\n📋 成功したマッチング一覧:")
            for i, (video_name, _, tracking_data, court_data) in enumerate(matched_data, 1):
                frame_count = self.get_actual_frame_count(tracking_data)
                court_status = "あり" if court_data else "なし"
                print(f"  {i}. {video_name}")
                print(f"     フレーム数: {frame_count}")
                print(f"     コート座標: {court_status}")
        
        print(f"最終マッチング数: {len(matched_data)}")
        return matched_data
    
    def get_actual_frame_count(self, tracking_data) -> int:
        """実際のフレーム数を取得（データ構造に関係なく）"""
        if isinstance(tracking_data, list):
            return len(tracking_data)
        elif isinstance(tracking_data, dict):
            if 'frames' in tracking_data:
                frames = tracking_data['frames']
                return len(frames) if hasattr(frames, '__len__') else 0
            elif 'frame_data' in tracking_data:
                frame_data = tracking_data['frame_data']
                return len(frame_data) if hasattr(frame_data, '__len__') else 0
            else:
                # 数値キーをカウント
                numeric_keys = [k for k in tracking_data.keys() if str(k).isdigit()]
                return len(numeric_keys)
        else:
            return 0

    def validate_tracking_data_consistency(self, tracking_data_dict: Dict) -> Dict:
        """トラッキングデータの一貫性を検証（記録されたメタデータ対応）"""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'frame_count': 0,
            'frame_skip_detected': False,
            'recorded_frame_skip': 1,
            'actual_frame_skip_interval': 1,
            'missing_frames': [],
            'duplicate_frames': [],
            'interpolated_count': 0,
            'metadata_available': False,
            'processing_mode': 'unknown'
        }
        
        # データ構造をチェック
        if isinstance(tracking_data_dict, list):
            # 旧形式
            tracking_data = tracking_data_dict
            validation_result['warnings'].append("旧形式データ: フレームスキップ情報が記録されていません")
        elif isinstance(tracking_data_dict, dict) and 'frames' in tracking_data_dict:
            # 新形式
            tracking_data = tracking_data_dict['frames']
            metadata = tracking_data_dict.get('metadata', {})
            validation_result['metadata_available'] = True
            
            # メタデータからフレームスキップ情報を取得
            recorded_skip = metadata.get('frame_skip', 1)
            processing_mode = metadata.get('processing_mode', 'unknown')
            
            validation_result['recorded_frame_skip'] = recorded_skip
            validation_result['processing_mode'] = processing_mode
            
            if recorded_skip > 1:
                validation_result['frame_skip_detected'] = True
                validation_result['warnings'].append(
                    f"記録されたフレームスキップ: {recorded_skip} (モード: {processing_mode})"
                )
            
            # 他のメタデータ情報も検証
            if 'original_fps' in metadata:
                validation_result['original_fps'] = metadata['original_fps']
            if 'processing_fps' in metadata:
                validation_result['processing_fps'] = metadata['processing_fps']
            if 'total_original_frames' in metadata:
                validation_result['total_original_frames'] = metadata['total_original_frames']
        else:
            validation_result['is_valid'] = False
            validation_result['warnings'].append("不明なデータ形式です")
            return validation_result
        
        validation_result['frame_count'] = len(tracking_data)
        
        if not tracking_data:
            validation_result['is_valid'] = False
            validation_result['warnings'].append("トラッキングデータが空です")
            return validation_result
        
        # フレーム番号の整合性チェック
        frame_numbers = [data.get('frame_number', 0) for data in tracking_data]
        frame_numbers.sort()
        
        # 重複フレームのチェック
        from collections import Counter
        frame_counts = Counter(frame_numbers)
        duplicates = [frame for frame, count in frame_counts.items() if count > 1]
        if duplicates:
            validation_result['duplicate_frames'] = duplicates
            validation_result['warnings'].append(f"重複フレームが検出されました: {duplicates}")
        
        # 実際のフレームスキップの検出と記録値との比較
        intervals = []
        missing_frames = []
        
        for i in range(1, len(frame_numbers)):
            interval = frame_numbers[i] - frame_numbers[i-1]
            if interval > 1:
                intervals.append(interval)
                # 欠損フレームを記録
                for missing in range(frame_numbers[i-1] + 1, frame_numbers[i]):
                    missing_frames.append(missing)
            elif interval == 1:
                intervals.append(1)
        
        if intervals:
            # 最も頻繁な間隔をスキップ間隔として判定
            interval_counts = Counter(intervals)
            most_common_interval = interval_counts.most_common(1)[0][0]
            
            validation_result['actual_frame_skip_interval'] = most_common_interval
            
            # 記録された値と実際の値を比較
            recorded_skip = validation_result['recorded_frame_skip']
            if recorded_skip > 1:
                if most_common_interval == recorded_skip:
                    validation_result['warnings'].append(
                        f"✅ フレームスキップ整合性確認: 記録値({recorded_skip}) = 実測値({most_common_interval})"
                    )
                else:
                    validation_result['warnings'].append(
                        f"⚠️  フレームスキップ不整合: 記録値({recorded_skip}) ≠ 実測値({most_common_interval})"
                    )
            elif most_common_interval > 1:
                validation_result['warnings'].append(
                    f"⚠️  記録なしスキップ検出: 実測間隔({most_common_interval})"
                )
        
        validation_result['missing_frames'] = missing_frames
        
        # 補間フレームのカウント
        interpolated_count = sum(1 for data in tracking_data if data.get('interpolated', False))
        validation_result['interpolated_count'] = interpolated_count
        
        if interpolated_count > 0:
            validation_result['warnings'].append(f"補間フレームが含まれています: {interpolated_count}フレーム")
        
        # データ品質の評価
        ball_detection_rate = sum(1 for data in tracking_data if data.get('ball_detected', 0)) / len(tracking_data)
        validation_result['ball_detection_rate'] = ball_detection_rate
        
        if ball_detection_rate < 0.3:
            validation_result['warnings'].append(f"ボール検出率が低いです: {ball_detection_rate:.1%}")
        
        return validation_result
    
    # データ診断・代替読み込みメソッド群
    def diagnose_tracking_data_structure(self, tracking_data_dict: Dict, video_name: str):
        """トラッキングデータ構造の詳細診断"""
        print(f"\n=== {video_name} トラッキングデータ構造診断 ===")
        
        if isinstance(tracking_data_dict, dict):
            print("データ構造: 辞書型")
            print(f"トップレベルキー: {list(tracking_data_dict.keys())}")
            
            if 'frames' in tracking_data_dict:
                frames_data = tracking_data_dict['frames']
                print(f"frames要素型: {type(frames_data)}")
                print(f"frames要素数: {len(frames_data) if hasattr(frames_data, '__len__') else 'N/A'}")
                
                if isinstance(frames_data, list) and len(frames_data) > 0:
                    sample_frame = frames_data[0]
                    print(f"サンプルフレーム型: {type(sample_frame)}")
                    if isinstance(sample_frame, dict):
                        print(f"フレーム内キー数: {len(sample_frame.keys())}")
                        print(f"フレーム内キー例: {list(sample_frame.keys())[:10]}")
                
            if 'metadata' in tracking_data_dict:
                metadata = tracking_data_dict['metadata']
                print(f"メタデータ: {metadata}")
        
        elif isinstance(tracking_data_dict, list):
            print("データ構造: リスト型")
            print(f"要素数: {len(tracking_data_dict)}")
            if len(tracking_data_dict) > 0:
                sample = tracking_data_dict[0]
                print(f"サンプル要素型: {type(sample)}")
        
        else:
            print(f"データ構造: 不明な型 - {type(tracking_data_dict)}")
    
    def attempt_alternative_data_loading(self, video_name: str) -> List[Dict]:
        """代替データ読み込み試行"""
        print(f"=== {video_name} 代替データ読み込み試行 ===")
        
        # tracking_features_*.jsonファイルを直接再読み込み
        pattern = f"tracking_features_{video_name}_*.json"
        tracking_files = list(self.training_data_dir.glob(pattern))
        
        if not tracking_files:
            print(f"対応するファイルが見つかりません: {pattern}")
            return []
        
        for file_path in tracking_files:
            try:
                print(f"ファイル再読み込み試行: {file_path.name}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                print(f"生データ型: {type(raw_data)}")
                
                # 様々なデータ構造に対応
                if isinstance(raw_data, list):
                    print(f"リスト形式: {len(raw_data)}要素")
                    return raw_data
                    
                elif isinstance(raw_data, dict):
                    print(f"辞書形式キー: {list(raw_data.keys())}")
                    
                    # 'frames'キーを優先
                    if 'frames' in raw_data:
                        frames = raw_data['frames']
                        print(f"framesキー: {type(frames)}, 要素数: {len(frames) if hasattr(frames, '__len__') else 'N/A'}")
                        if isinstance(frames, list):
                            return frames
                    
                    # 'frame_data'キーを試行
                    if 'frame_data' in raw_data:
                        frame_data = raw_data['frame_data']
                        print(f"frame_dataキー: {type(frame_data)}, 要素数: {len(frame_data) if hasattr(frame_data, '__len__') else 'N/A'}")
                        if isinstance(frame_data, list):
                            return frame_data
                    
                    # 数値キー（フレーム番号）を探索
                    numeric_keys = [k for k in raw_data.keys() if str(k).isdigit()]
                    if numeric_keys:
                        print(f"数値キー検出: {len(numeric_keys)}個")
                        # 数値キーから辞書を構築
                        frame_list = []
                        for key in sorted(numeric_keys, key=int):
                            frame_data = raw_data[key]
                            if isinstance(frame_data, dict):
                                frame_data['frame_number'] = int(key)
                                frame_list.append(frame_data)
                        
                        if frame_list:
                            print(f"数値キーから構築: {len(frame_list)}フレーム")
                            return frame_list
                
                print("⚠️  認識可能なデータ構造が見つかりません")
                return []
                
            except Exception as e:
                print(f"代替読み込みエラー: {e}")
                continue
        
        return []
    
    # フレーム正規化・補間メソッド群
    def normalize_frame_numbers(self, tracking_data_dict: Dict) -> List[Dict]:
        """フレーム番号を正規化し、フレームスキップを考慮した展開を行う"""
        print("=== フレーム番号正規化・展開処理 ===")
        
        # データ構造をチェック
        if isinstance(tracking_data_dict, list):
            # 旧形式の場合は従来の処理
            print("⚠️  旧形式データ: フレームスキップ情報が記録されていません")
            return self.legacy_normalize_frame_numbers(tracking_data_dict)
        
        # 新形式の処理：フレームスキップを考慮した展開
        metadata = tracking_data_dict.get('metadata', {})
        frame_skip = metadata.get('frame_skip', 1)
        
        print(f"✅ 記録されたフレームスキップ情報: {frame_skip}")
        
        if frame_skip == 1:
            # スキップなしの場合は通常の正規化のみ
            frames_data = tracking_data_dict.get('frames', [])
            print("フレームスキップなし - 通常の正規化を実行")
            return frames_data
        else:
            # フレームスキップありの場合は展開処理
            print(f"フレームスキップ({frame_skip})を検出 - 元シーケンスへの展開を実行")
            expanded_frames = self.expand_frames_to_original_sequence(tracking_data_dict)
            return expanded_frames
    
    def legacy_normalize_frame_numbers(self, tracking_data: List[Dict]) -> List[Dict]:
        """旧形式データのフレーム番号正規化"""
        if not tracking_data:
            return []
        
        # フレーム番号でソート
        sorted_data = sorted(tracking_data, key=lambda x: x.get('frame_number', 0))
        
        # 連続した番号に正規化
        for i, frame_data in enumerate(tqdm(sorted_data, desc="旧形式フレーム正規化", leave=False)):
            frame_data['original_frame_number'] = frame_data.get('frame_number', 0)
            frame_data['frame_number'] = i
            frame_data['interpolated'] = False
        
        return sorted_data
    
    def expand_frames_to_original_sequence(self, tracking_data_dict: Dict) -> List[Dict]:
        """フレームスキップされたデータを元の動画シーケンスに展開"""
        print("=== フレームシーケンス展開処理 ===")
        
        # メタデータを取得
        metadata = tracking_data_dict['metadata']
        frames_data = tracking_data_dict['frames']
        frame_skip = metadata.get('frame_skip', 1)
        total_original_frames = metadata.get('total_original_frames')
        
        print(f"記録されたフレームスキップ: {frame_skip}")
        print(f"処理済みフレーム数: {len(frames_data)}")
        if total_original_frames:
            print(f"元動画総フレーム数: {total_original_frames}")
        
        # 処理済みフレームを元のフレーム番号でソート
        sorted_frames = sorted(frames_data, key=lambda x: x.get('frame_number', 0))
        
        if not sorted_frames:
            print("処理済みフレームが存在しません")
            return []
        
        # 最初と最後の処理済みフレーム番号を取得
        first_processed_frame = sorted_frames[0].get('frame_number', 0)
        last_processed_frame = sorted_frames[-1].get('frame_number', 0)
        
        print(f"最初の処理フレーム: {first_processed_frame}")
        print(f"最後の処理フレーム: {last_processed_frame}")
        
        # 元動画の総フレーム数を推定（メタデータがない場合）
        if total_original_frames is None:
            estimated_total = last_processed_frame + frame_skip
            total_original_frames = estimated_total
            print(f"推定総フレーム数: {estimated_total}")
        
        # 処理済みフレームのインデックスを作成
        processed_frame_map = {}
        for frame_data in sorted_frames:
            original_frame_num = frame_data.get('frame_number', 0)
            processed_frame_map[original_frame_num] = frame_data
        
        # 元のフレームシーケンスを再構築
        expanded_frames = []
        interpolated_count = 0
        
        for original_frame_num in tqdm(range(total_original_frames), desc="フレームシーケンス展開", leave=False):
            if original_frame_num in processed_frame_map:
                # 処理済みフレームをそのまま使用
                frame_data = processed_frame_map[original_frame_num].copy()
                frame_data['original_frame_number'] = original_frame_num
                frame_data['interpolated'] = False
                expanded_frames.append(frame_data)
            else:
                # 補間フレームを作成
                interpolated_frame = self.create_interpolated_frame_from_skip(
                    original_frame_num, processed_frame_map, frame_skip
                )
                expanded_frames.append(interpolated_frame)
                interpolated_count += 1
        
        print(f"✅ フレーム展開完了:")
        print(f"   展開後総フレーム数: {len(expanded_frames)}")
        print(f"   処理済みフレーム数: {len(sorted_frames)}")
        print(f"   補間フレーム数: {interpolated_count}")
        print(f"   補間率: {interpolated_count/len(expanded_frames)*100:.1f}%")
        
        return expanded_frames
    
    def create_interpolated_frame_from_skip(self, target_frame_num: int, 
                                          processed_frame_map: Dict, frame_skip: int) -> Dict:
        """フレームスキップによる欠損を補間してフレームデータを作成"""
        # 前後の処理済みフレームを探す
        prev_frame_data = None
        next_frame_data = None
        
        # 前の処理済みフレームを探す
        for i in range(target_frame_num - 1, -1, -1):
            if i in processed_frame_map:
                prev_frame_data = processed_frame_map[i]
                break
        
        # 次の処理済みフレームを探す
        max_search_range = target_frame_num + frame_skip * 2
        for i in range(target_frame_num + 1, max_search_range):
            if i in processed_frame_map:
                next_frame_data = processed_frame_map[i]
                break
        
        # 補間フレームを作成
        interpolated_frame = {
            'frame_number': target_frame_num,
            'original_frame_number': target_frame_num,
            'interpolated': True,
            'timestamp': '',
            'ball_detected': 0,
            'ball_x': None,
            'ball_y': None,
            'ball_x_normalized': 0,
            'ball_y_normalized': 0,
            'ball_movement_score': 0,
            'ball_tracking_confidence': 0,
            'ball_velocity_x': 0,
            'ball_velocity_y': 0,
            'ball_speed': 0,
            'player_front_count': 0,
            'player_back_count': 0,
            'total_players': 0,
            'candidate_balls_count': 0,
            'disappeared_count': 0,
            'trajectory_length': 0,
            'prediction_active': 0
        }
        
        # 前後のフレームがある場合は補間
        if prev_frame_data and next_frame_data:
            prev_frame_num = prev_frame_data.get('frame_number', 0)
            next_frame_num = next_frame_data.get('frame_number', 0)
            
            if next_frame_num > prev_frame_num:
                # 線形補間の比率を計算
                ratio = (target_frame_num - prev_frame_num) / (next_frame_num - prev_frame_num)
                ratio = max(0, min(1, ratio))
                
                # 数値データの線形補間
                numeric_keys = [
                    'ball_x', 'ball_y', 'ball_x_normalized', 'ball_y_normalized',
                    'ball_velocity_x', 'ball_velocity_y', 'ball_speed',
                    'ball_movement_score', 'ball_tracking_confidence',
                    'player_front_x', 'player_front_y', 'player_front_x_normalized', 'player_front_y_normalized',
                    'player_back_x', 'player_back_y', 'player_back_x_normalized', 'player_back_y_normalized',
                    'player_front_confidence', 'player_back_confidence',
                    'player_distance', 'player_distance_normalized'
                ]
                
                for key in numeric_keys:
                    prev_val = prev_frame_data.get(key)
                    next_val = next_frame_data.get(key)
                    
                    if prev_val is not None and next_val is not None:
                        interpolated_val = prev_val + (next_val - prev_val) * ratio
                        interpolated_frame[key] = interpolated_val
                    elif prev_val is not None:
                        interpolated_frame[key] = prev_val
                    elif next_val is not None:
                        interpolated_frame[key] = next_val
        
        elif prev_frame_data:
            # 前のフレームのみある場合、値を継承
            for key in ['ball_x', 'ball_y', 'ball_x_normalized', 'ball_y_normalized',
                       'player_front_x', 'player_front_y', 'player_back_x', 'player_back_y']:
                if key in prev_frame_data:
                    interpolated_frame[key] = prev_frame_data[key]
        
        elif next_frame_data:
            # 次のフレームのみある場合、値を継承
            for key in ['ball_x', 'ball_y', 'ball_x_normalized', 'ball_y_normalized',
                       'player_front_x', 'player_front_y', 'player_back_x', 'player_back_y']:
                if key in next_frame_data:
                    interpolated_frame[key] = next_frame_data[key]
        
        return interpolated_frame
    
    # データフレーム処理メソッド群
    def safe_create_dataframe_from_tracking_data(self, tracking_data: List[Dict], video_name: str) -> pd.DataFrame:
        """トラッキングデータから安全にDataFrameを作成（エラー対策強化）"""
        if not tracking_data:
            print("⚠️  空のトラッキングデータ")
            return pd.DataFrame()
        
        try:
            # 方法1: 標準的なDataFrame作成を試行
            if all(isinstance(frame, dict) for frame in tracking_data):
                return pd.DataFrame(tracking_data)
            else:
                print("⚠️  非辞書形式のフレームが含まれています")
                # 辞書形式のフレームのみをフィルタ
                valid_frames = [frame for frame in tracking_data if isinstance(frame, dict)]
                if valid_frames:
                    return pd.DataFrame(valid_frames)
                else:
                    print("❌ 有効な辞書形式フレームが見つかりません")
                    return pd.DataFrame()
                
        except ValueError as e:
            if "Mixing dicts with non-Series" in str(e):
                print(f"⚠️  DataFrame作成エラーを検出: {e}")
                return self.handle_mixed_data_types_error(tracking_data, video_name)
            else:
                print(f"❌ 予期しないDataFrame作成エラー: {e}")
                return pd.DataFrame()
        except Exception as e:
            print(f"❌ DataFrame作成で例外発生: {e}")
            return self.handle_mixed_data_types_error(tracking_data, video_name)
    
    def handle_mixed_data_types_error(self, tracking_data: List[Dict], video_name: str) -> pd.DataFrame:
        """混合データ型エラーの処理（改善版）"""
        print("=== 混合データ型エラー対処中 ===")
        
        try:
            # データ構造を詳細分析
            print(f"フレーム数: {len(tracking_data)}")
            if tracking_data:
                sample_frame = tracking_data[0]
                print(f"サンプルフレーム型: {type(sample_frame)}")
                if isinstance(sample_frame, dict):
                    print(f"サンプルフレームキー数: {len(sample_frame.keys())}")
                    print(f"キー例: {list(sample_frame.keys())[:10]}")
            
            # 改善された正規化処理
            normalized_frames = []
            for i, frame in enumerate(tracking_data):
                if isinstance(frame, dict):
                    normalized_frame = {'frame_number': i}  # フレーム番号を設定
                    
                    for key, value in frame.items():
                        try:
                            # 値の型に応じて適切に処理
                            if value is None:
                                normalized_frame[key] = 0
                            elif isinstance(value, (int, float)):
                                normalized_frame[key] = float(value)
                            elif isinstance(value, bool):
                                normalized_frame[key] = int(value)
                            elif isinstance(value, str):
                                # 数値に変換可能かチェック
                                try:
                                    normalized_frame[key] = float(value)
                                except ValueError:
                                    normalized_frame[key] = 0
                            elif isinstance(value, (list, tuple)):
                                # リストの場合は長さまたは最初の値を使用
                                if len(value) > 0:
                                    first_val = value[0]
                                    if isinstance(first_val, (int, float)):
                                        normalized_frame[key] = float(first_val)
                                    else:
                                        normalized_frame[key] = len(value)
                                else:
                                    normalized_frame[key] = 0
                            elif isinstance(value, dict):
                                # 辞書の場合は展開または要約
                                if 'x' in value and 'y' in value:
                                    normalized_frame[f'{key}_x'] = float(value.get('x', 0))
                                    normalized_frame[f'{key}_y'] = float(value.get('y', 0))
                                elif len(value) == 1:
                                    # 単一キーの場合はその値を使用
                                    single_key = list(value.keys())[0]
                                    single_val = value[single_key]
                                    if isinstance(single_val, (int, float)):
                                        normalized_frame[key] = float(single_val)
                                    else:
                                        normalized_frame[key] = 1
                                else:
                                    normalized_frame[key] = len(value)
                            else:
                                # その他の型は0に設定
                                normalized_frame[key] = 0
                        except Exception as e:
                            print(f"キー '{key}' の処理でエラー: {e}")
                            normalized_frame[key] = 0
                    
                    normalized_frames.append(normalized_frame)
                else:
                    print(f"フレーム {i} は辞書ではありません: {type(frame)}")
            
            if normalized_frames:
                print(f"✅ データ正規化完了: {len(normalized_frames)}フレーム")
                
                # DataFrameの安全な作成
                try:
                    # 全てのフレームで共通のキーセットを作成
                    all_keys = set()
                    for frame in normalized_frames:
                        all_keys.update(frame.keys())
                    
                    # 欠損キーを補完
                    for frame in normalized_frames:
                        for key in all_keys:
                            if key not in frame:
                                frame[key] = 0
                    
                    df = pd.DataFrame(normalized_frames)
                    print(f"DataFrame作成成功: {df.shape}")
                    return df
                    
                except Exception as df_error:
                    print(f"DataFrame作成でエラー: {df_error}")
                    # 最小限のDataFrameを作成
                    basic_df = pd.DataFrame({
                        'frame_number': range(len(normalized_frames)),
                        'ball_detected': [frame.get('ball_detected', 0) for frame in normalized_frames],
                        'ball_x': [frame.get('ball_x', 0) for frame in normalized_frames],
                        'ball_y': [frame.get('ball_y', 0) for frame in normalized_frames]
                    })
                    print(f"基本DataFrame作成: {basic_df.shape}")
                    return basic_df
            else:
                print("❌ 正規化可能なフレームがありません")
                return pd.DataFrame()
                
        except Exception as e2:
            print(f"❌ データ正規化でもエラー: {e2}")
            # 空のDataFrameを返す
            return pd.DataFrame()
    
    # ラベル生成メソッド
    def interpolate_phase_labels(self, phase_changes: List[Dict], total_frames: int, fps: float) -> np.ndarray:
        """局面変更データからフレームごとの局面ラベルを生成"""
        frame_labels = np.full(total_frames, -1, dtype=int)  # -1: ラベル未設定
        
        if not phase_changes:
            return frame_labels
        
        # 局面変更を時間順にソート
        sorted_changes = sorted(phase_changes, key=lambda x: x['frame_number'])
        
        for i, change in enumerate(tqdm(sorted_changes, desc="局面ラベル補間", leave=False)):
            start_frame = int(change['frame_number'])
            phase_label = change['phase']
            
            # ラベルIDに変換
            if phase_label in self.label_to_id:
                label_id = self.label_to_id[phase_label]
            else:
                print(f"⚠️  未知の局面ラベル: {phase_label}")
                continue
            
            # 次の変更まで、または動画終了まで同じラベルを適用
            if i + 1 < len(sorted_changes):
                end_frame = int(sorted_changes[i + 1]['frame_number'])
            else:
                end_frame = total_frames
            
            # フレーム範囲チェック
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))
            
            frame_labels[start_frame:end_frame] = label_id
        
        return frame_labels
    
    # メイン特徴量抽出メソッド
    def extract_features_from_video(self, video_name: str, phase_data: Dict, tracking_data_dict: Dict, court_coords: Dict = None) -> pd.DataFrame:
            """単一動画から特徴量を抽出（処理順序をpredictパイプラインと統一）"""
            print(f"\n--- 特徴量抽出: {video_name} ---")

            # ステップ1: データ検証と正規化
            print("  ステップ1/3: データ検証とフレーム正規化...")
            validation_result = self.validate_tracking_data_consistency(tracking_data_dict)
            if validation_result['frame_count'] <= 10:
                print(f"⚠️  フレーム数が異常に少ないため、代替読み込みを試行します...")
                alternative_data = self.attempt_alternative_data_loading(video_name)
                if alternative_data:
                    print(f"✅ 代替データ読み込み成功: {len(alternative_data)}フレーム")
                    tracking_data_dict = {'metadata': tracking_data_dict.get('metadata', {}), 'frames': alternative_data}

            normalized_tracking_data = self.normalize_frame_numbers(tracking_data_dict)
            if not normalized_tracking_data:
                print("❌ 正規化後のトラッキングデータが空のため、処理を中断します。")
                return pd.DataFrame()
            
            features_df = self.safe_create_dataframe_from_tracking_data(normalized_tracking_data, video_name)
            if features_df.empty:
                print("❌ DataFrame作成に失敗しました。")
                return pd.DataFrame()

            # ステップ2: 特徴量計算（全フレーム対象）
            # この時点では、ラベル付けやフィルタリングを行わず、動画全体のデータで特徴量を計算します。
            print("  ステップ2/3: 特徴量エンジニアリング (全フレーム対象)...")
            
            # 元のフレーム番号を'original_frame_number'として確実に保持
            if 'frame_number' in features_df.columns:
                features_df['original_frame_number'] = features_df['frame_number']
            # 'frame_number'列は後で使うが、一旦インデックスとしてリセット
            features_df['frame_number'] = range(len(features_df))
            
            # 'interpolated'列の存在を保証
            if 'interpolated' not in features_df.columns:
                features_df['interpolated'] = False
            
            # 特徴量生成関数を順番に適用
            features_df = self.handle_missing_values(features_df)
            features_df = self.create_court_features(features_df, court_coords)
            features_df = self.create_temporal_features(features_df)
            features_df = self.create_contextual_features(features_df)
            print("  ステップ2/3: 特徴量エンジニアリング完了")

            # ステップ3: ラベル付与とフィルタリング
            # 全ての特徴量計算が終わった後で、正解ラベルを付与し、不要なフレームを削除します。
            print("  ステップ3/3: 局面ラベルの付与とフィルタリング...")
            
            total_frames = len(features_df)
            fps = phase_data.get('fps', 30.0)
            phase_changes = phase_data.get('phase_changes', [])
            frame_labels = self.interpolate_phase_labels(phase_changes, total_frames, fps)
            
            # ラベル列を追加
            if len(frame_labels) == len(features_df):
                features_df['label'] = frame_labels
            else:
                print(f"⚠️  ラベル数({len(frame_labels)})とフレーム数({len(features_df)})が不一致。小さい方に合わせます。")
                min_len = min(len(frame_labels), len(features_df))
                features_df = features_df.iloc[:min_len]
                features_df['label'] = frame_labels[:min_len]

            features_df['video_name'] = video_name
            
            # ラベルが-1（未定義）のフレームを除外して、最終的なデータセットを作成
            features_df = features_df[features_df['label'] != -1].copy()
            
            print(f"最終的なラベル付きフレーム数: {len(features_df)}")
            if len(features_df) == 0:
                print("⚠️  有効なラベルを持つフレームが存在しませんでした。")
                return pd.DataFrame()
                
            # 最終結果のサマリーを表示
            print("最終的なラベル分布:")
            label_counts = features_df['label'].value_counts().sort_index()
            for label_id, count in label_counts.items():
                phase_name = self.phase_labels[label_id] if label_id < len(self.phase_labels) else f"Unknown_{label_id}"
                print(f"  {label_id} ({phase_name}): {count}フレーム")
            
            print(f"最終特徴量数: {len(features_df.columns)}")
            return features_df
    
    # 特徴量作成メソッド群
    def create_temporal_features(self, features_df: pd.DataFrame, window_sizes: List[int] = [3, 5, 10, 15]) -> pd.DataFrame:
        """時系列特徴量を作成（Numba高速化対応）"""
        temporal_df = features_df.copy()
        
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        target_columns = [col for col in numeric_columns if col not in ['frame_number', 'original_frame_number']]
        
        print(f"時系列特徴量作成対象: {len(target_columns)}特徴量 (Numba高速化対応)")
        
        is_interpolated = features_df.get('interpolated', pd.Series([False] * len(features_df))).values
        
        print("  データをNumPy配列に変換中...")
        data_matrix = features_df[target_columns].values.astype(np.float64)
        data_matrix = np.nan_to_num(data_matrix, nan=0.0)
        
        new_features = {}
        
        for window in tqdm(window_sizes, desc="時系列特徴量(ウィンドウ別)", leave=False):
            print(f"  ウィンドウサイズ {window} の特徴量作成中... (Numba 高速化)")
            
            ma_results, std_results, max_results, min_results = vectorized_rolling_stats(
                data_matrix, window, is_interpolated
            )
            
            for i, col in enumerate(target_columns):
                new_features[f'{col}_ma_{window}'] = ma_results[:, i]
                new_features[f'{col}_std_{window}'] = std_results[:, i]
                new_features[f'{col}_max_{window}'] = max_results[:, i]
                new_features[f'{col}_min_{window}'] = min_results[:, i]
                
                if window <= 5:
                    diff1, diff2 = self.vectorized_diff_features(
                        data_matrix[:, i], is_interpolated
                    )
                    new_features[f'{col}_diff'] = diff1
                    new_features[f'{col}_diff_abs'] = np.abs(diff1)
                    new_features[f'{col}_diff2'] = diff2
                    new_features[f'{col}_diff2_abs'] = np.abs(diff2)
                
                if window == 5:
                    trend_values = self.vectorized_rolling_trend(data_matrix[:, i], window)
                    new_features[f'{col}_trend_{window}'] = trend_values
                    
                    ma_vals = ma_results[:, i]
                    std_vals = std_results[:, i]
                    cv_values = np.divide(std_vals, np.abs(ma_vals), 
                                        out=np.zeros_like(std_vals), where=ma_vals!=0)
                    new_features[f'{col}_cv_{window}'] = cv_values
        
        new_features['data_quality'] = (~is_interpolated).astype(float)
        
        interpolation_kernel = np.ones(10) / 10
        interpolation_ratio = np.convolve(is_interpolated.astype(float), interpolation_kernel, mode='same')
        new_features['interpolation_ratio'] = interpolation_ratio
        
        print("  新しい特徴量をDataFrameに統合中...")
        for feature_name, feature_values in new_features.items():
            temporal_df[feature_name] = feature_values
        
        print(f"時系列特徴量作成完了: {len(new_features)}特徴量追加")
        return temporal_df
    
    def vectorized_diff_features(self, data_array: np.ndarray,
                            is_interpolated: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ベクトル化された差分特徴量計算"""
        diff1 = np.diff(data_array, prepend=data_array[0])

        if np.any(is_interpolated):
            diff_weights = (~is_interpolated).astype(float) * 1.0 + is_interpolated.astype(float) * 0.5
            diff1 = diff1 * diff_weights

        diff2 = np.diff(diff1, prepend=diff1[0])

        return diff1, diff2

    def vectorized_rolling_trend(self, data_array: np.ndarray, window: int) -> np.ndarray:
        """ベクトル化された移動トレンド計算"""
        n = len(data_array)
        trend_values = np.zeros(n)
        half_window = window // 2

        padded_data = np.pad(data_array, half_window, mode='edge')

        x = np.arange(window) - half_window
        x_mean = np.mean(x)
        x_centered = x - x_mean
        x_var = np.sum(x_centered ** 2)

        if x_var == 0:
            return trend_values

        for i in range(n):
            y = padded_data[i:i+window]
            y_mean = np.mean(y)
            y_centered = y - y_mean

            covariance = np.sum(x_centered * y_centered)
            trend_values[i] = covariance / x_var

        return trend_values
    
    def create_contextual_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """コンテキスト特徴量を作成（真のNumPy高速化版）"""
        context_df = features_df.copy()
        
        print("コンテキスト特徴量を作成中... (真のNumPy高速化)")
        
        available_fields = set(context_df.columns)
        data_length = len(context_df)
        
        numeric_data = {}
        for field in available_fields:
            if context_df[field].dtype in [np.number, 'float64', 'float32', 'int64', 'int32']:
                numeric_data[field] = context_df[field].fillna(0).values
        
        ball_activity = self.calculate_ball_activity_vectorized(numeric_data, available_fields, data_length)
        context_df['ball_activity'] = ball_activity
        
        players_interaction, players_confidence_avg = self.calculate_player_features_vectorized(
            numeric_data, available_fields, data_length
        )
        context_df['players_interaction'] = players_interaction
        context_df['players_confidence_avg'] = players_confidence_avg
        
        distance_features = self.calculate_distances_vectorized(numeric_data, available_fields, data_length)
        for feature_name, feature_values in distance_features.items():
            context_df[feature_name] = feature_values
        
        position_features = self.calculate_position_features_vectorized(numeric_data, available_fields, data_length)
        for feature_name, feature_values in position_features.items():
            context_df[feature_name] = feature_values
        
        tracking_quality = self.calculate_tracking_quality_vectorized(numeric_data, available_fields, data_length)
        context_df['tracking_quality'] = tracking_quality
        
        temporal_context_features = self.calculate_temporal_context_vectorized(
            numeric_data, available_fields, data_length
        )
        for feature_name, feature_values in temporal_context_features.items():
            context_df[feature_name] = feature_values
        
        added_features = len(context_df.columns) - len(features_df.columns)
        print(f"  コンテキスト特徴量作成完了: {added_features}特徴量追加")
        return context_df
    
    def calculate_ball_activity_vectorized(self, numeric_data: Dict[str, np.ndarray],
                                        available_fields: set, data_length: int) -> np.ndarray:
        """ベクトル化されたボール活動度計算"""
        components = []
        if 'ball_detected' in available_fields: components.append(numeric_data['ball_detected'])
        if 'ball_movement_score' in available_fields: components.append(numeric_data['ball_movement_score'])
        elif 'ball_speed' in available_fields: components.append(numeric_data['ball_speed'] / 100.0)
        if 'ball_tracking_confidence' in available_fields: components.append(numeric_data['ball_tracking_confidence'])

        if components:
            ball_activity = np.ones(data_length)
            for component in components:
                ball_activity *= component
            return ball_activity
        else:
            return np.zeros(data_length)

    def calculate_player_features_vectorized(self, numeric_data: Dict[str, np.ndarray],
                                        available_fields: set, data_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """ベクトル化されたプレイヤー特徴量計算"""
        player_counts = []
        if 'player_front_count' in available_fields: player_counts.append(numeric_data['player_front_count'])
        if 'player_back_count' in available_fields: player_counts.append(numeric_data['player_back_count'])

        players_interaction = np.mean(player_counts, axis=0) if player_counts else np.zeros(data_length)

        confidence_components = []
        if 'player_front_confidence' in available_fields: confidence_components.append(numeric_data['player_front_confidence'])
        if 'player_back_confidence' in available_fields: confidence_components.append(numeric_data['player_back_confidence'])

        players_confidence_avg = np.mean(confidence_components, axis=0) if confidence_components else np.zeros(data_length)

        return players_interaction, players_confidence_avg

    def calculate_distances_vectorized(self, numeric_data: Dict[str, np.ndarray],
                                    available_fields: set, data_length: int) -> Dict[str, np.ndarray]:
        """ベクトル化された距離計算"""
        distance_features = {}

        ball_pos_available = all(col in available_fields for col in ['ball_x', 'ball_y'])
        front_pos_available = all(col in available_fields for col in ['player_front_x', 'player_front_y'])
        back_pos_available = all(col in available_fields for col in ['player_back_x', 'player_back_y'])

        if ball_pos_available and front_pos_available:
            dist = np.sqrt((numeric_data['ball_x'] - numeric_data['player_front_x'])**2 + (numeric_data['ball_y'] - numeric_data['player_front_y'])**2)
            distance_features['ball_to_front_distance'] = np.where(np.isnan(dist) | np.isinf(dist), 1000, dist)
        else:
            distance_features['ball_to_front_distance'] = np.full(data_length, 1000.0)

        if ball_pos_available and back_pos_available:
            dist = np.sqrt((numeric_data['ball_x'] - numeric_data['player_back_x'])**2 + (numeric_data['ball_y'] - numeric_data['player_back_y'])**2)
            distance_features['ball_to_back_distance'] = np.where(np.isnan(dist) | np.isinf(dist), 1000, dist)
        else:
            distance_features['ball_to_back_distance'] = np.full(data_length, 1000.0)

        if 'ball_to_front_distance' in distance_features and 'ball_to_back_distance' in distance_features:
            front_dist = distance_features['ball_to_front_distance']
            back_dist = distance_features['ball_to_back_distance']
            distance_features['ball_closer_to_front'] = (front_dist < back_dist).astype(int)
        else:
            distance_features['ball_closer_to_front'] = np.zeros(data_length, dtype=int)

        return distance_features

    def calculate_position_features_vectorized(self, numeric_data: Dict[str, np.ndarray],
                                            available_fields: set, data_length: int) -> Dict[str, np.ndarray]:
        """ベクトル化された位置特徴量計算"""
        position_features = {}

        if all(col in available_fields for col in ['ball_x_normalized', 'ball_y_normalized']):
            ball_x_norm = np.nan_to_num(numeric_data['ball_x_normalized'], nan=0.5)
            ball_y_norm = np.nan_to_num(numeric_data['ball_y_normalized'], nan=0.5)

            position_features['ball_in_upper_half'] = (ball_x_norm < 0.5).astype(int)
            position_features['ball_in_left_half'] = (ball_y_norm < 0.5).astype(int)

            in_center = ((ball_x_norm > 0.3) & (ball_x_norm < 0.7) & (ball_y_norm > 0.3) & (ball_y_norm < 0.7))
            position_features['ball_in_center'] = in_center.astype(int)
        else:
            position_features.update({
                'ball_in_upper_half': np.zeros(data_length, dtype=int),
                'ball_in_left_half': np.zeros(data_length, dtype=int),
                'ball_in_center': np.zeros(data_length, dtype=int)
            })

        return position_features

    def calculate_tracking_quality_vectorized(self, numeric_data: Dict[str, np.ndarray],
                                            available_fields: set, data_length: int) -> np.ndarray:
        """ベクトル化されたトラッキング品質計算"""
        quality_components, weights = [], []
        if 'ball_detected' in available_fields: quality_components.append(numeric_data['ball_detected']); weights.append(0.4)
        if 'ball_tracking_confidence' in available_fields: quality_components.append((numeric_data['ball_tracking_confidence'] > 0.5).astype(int)); weights.append(0.3)
        if 'candidate_balls_count' in available_fields: quality_components.append((numeric_data['candidate_balls_count'] > 0).astype(int)); weights.append(0.2)
        if 'disappeared_count' in available_fields: quality_components.append((numeric_data['disappeared_count'] == 0).astype(int)); weights.append(0.1)

        if quality_components:
            quality_array = np.array(quality_components)
            weights_array = np.array(weights).reshape(-1, 1)
            weighted_sum = np.sum(quality_array * weights_array, axis=0)
            total_weight = np.sum(weights_array)
            return weighted_sum / total_weight if total_weight > 0 else np.zeros(data_length)
        else:
            return np.zeros(data_length)

    def calculate_temporal_context_vectorized(self, numeric_data: Dict[str, np.ndarray],
                                            available_fields: set, data_length: int) -> Dict[str, np.ndarray]:
        """ベクトル化された時系列コンテキスト特徴量計算"""
        print("    時系列コンテキスト特徴量を作成中... (真のNumPy高速化)")
        temporal_features = {}

        if 'ball_detected' in available_fields:
            ball_detected = numeric_data['ball_detected']
            for window in [3, 5, 10]:
                kernel = np.ones(window) / window
                temporal_features[f'ball_detection_stability_{window}'] = np.convolve(ball_detected, kernel, mode='same')

        if all(col in available_fields for col in ['ball_x', 'ball_y']):
            x_diff = np.diff(numeric_data['ball_x'], prepend=numeric_data['ball_x'][0])
            y_diff = np.diff(numeric_data['ball_y'], prepend=numeric_data['ball_y'][0])
            movement_distance = np.sqrt(x_diff**2 + y_diff**2)
            temporal_features['ball_movement_distance'] = movement_distance
            for window in [3, 5]:
                temporal_features[f'ball_movement_stability_{window}'] = self.fast_rolling_std_vectorized(movement_distance, window)

        for player in ['front', 'back']:
            if all(f'player_{player}_{coord}' in available_fields for coord in ['x', 'y']):
                x_diff = np.diff(numeric_data[f'player_{player}_x'], prepend=numeric_data[f'player_{player}_x'][0])
                y_diff = np.diff(numeric_data[f'player_{player}_y'], prepend=numeric_data[f'player_{player}_y'][0])
                movement_dist = np.sqrt(x_diff**2 + y_diff**2)
                temporal_features[f'player_{player}_movement_distance'] = movement_dist
                for window in [5, 10]:
                    kernel = np.ones(window) / window
                    temporal_features[f'player_{player}_activity_{window}'] = np.convolve(movement_dist, kernel, mode='same')

        movement_cols = [k for k in temporal_features if 'movement_distance' in k]
        if movement_cols:
            movement_matrix = np.array([temporal_features[col] for col in movement_cols])
            scene_dynamics = np.mean(movement_matrix, axis=0)
            temporal_features['scene_dynamics'] = scene_dynamics
            for window in [5, 10]:
                kernel = np.ones(window) / window
                temporal_features[f'scene_dynamics_ma_{window}'] = np.convolve(scene_dynamics, kernel, mode='same')
                temporal_features[f'scene_dynamics_std_{window}'] = self.fast_rolling_std_vectorized(scene_dynamics, window)

        if 'ball_movement_distance' in temporal_features:
            movement_dist = temporal_features['ball_movement_distance']
            kernel = np.ones(5) / 5
            movement_ma = np.convolve(movement_dist, kernel, mode='same')
            spike_mask = movement_dist > movement_ma * 2
            temporal_features['ball_movement_spike'] = spike_mask.astype(int)
            for window in [10, 20]:
                kernel = np.ones(window)
                temporal_features[f'ball_events_frequency_{window}'] = np.convolve(spike_mask.astype(float), kernel, mode='same')

        return temporal_features

    def fast_rolling_std_vectorized(self, values: np.ndarray, window: int) -> np.ndarray:
        """完全ベクトル化された高速移動標準偏差計算"""
        n= len(values)

        padded_values = np.pad(values, window//2, mode='edge')
        kernel = np.ones(window)
        moving_sum = np.convolve(padded_values, kernel, mode='valid')
        moving_mean = moving_sum / window
        moving_sum_sq = np.convolve(padded_values**2, kernel, mode='valid')
        moving_mean_sq = moving_sum_sq / window
        variance = np.clip(moving_mean_sq - moving_mean**2, 0, None)

        if len(variance) > n:
            variance = variance[:n]
        return np.sqrt(variance)
    
    def create_court_features(self, features_df: pd.DataFrame, court_coords: Dict) -> pd.DataFrame:
            """コート座標を使用した特徴量を作成（NumPy高速化版）"""
            if not court_coords:
                print("コート座標データがないため、コート特徴量はスキップします")
                return features_df
            
            court_df = features_df.copy()
            print("コート座標特徴量を作成中... (NumPy高速化)")
            
            # --- ボール位置のコート座標系変換 ---
            if all(col in court_df.columns for col in ['ball_x', 'ball_y']):
                ball_court_coords = self.transform_to_court_coordinates(
                    court_df['ball_x'].values, 
                    court_df['ball_y'].values, 
                    court_coords
                )
                
                court_df['ball_court_x'] = ball_court_coords['x']
                court_df['ball_court_y'] = ball_court_coords['y']
                
                ball_court_x = ball_court_coords['x']
                ball_court_y = ball_court_coords['y']
                
                # --- ★★★ ここからが復元する特徴量計算 ★★★ ---
                
                # コート上の位置に基づく特徴量（ベクトル化）
                court_df['ball_in_court'] = ((ball_court_x >= 0) & (ball_court_x <= 1) &
                                            (ball_court_y >= 0) & (ball_court_y <= 1)).astype(int)
                
                # コート上の領域特徴量（ベクトル化）
                court_df['ball_in_front_court'] = (ball_court_y > 0.5).astype(int)
                court_df['ball_in_back_court'] = (ball_court_y <= 0.5).astype(int)
                court_df['ball_in_left_court'] = (ball_court_x <= 0.5).astype(int)
                court_df['ball_in_right_court'] = (ball_court_x > 0.5).astype(int)
                
                # 各種距離の計算（NumPy配列で一括計算）
                net_y = 0.5
                court_df['ball_distance_to_net'] = np.abs(ball_court_y - net_y)
                court_df['ball_distance_to_left_line'] = ball_court_x
                court_df['ball_distance_to_right_line'] = 1 - ball_court_x
                court_df['ball_distance_to_sideline'] = np.minimum(
                    court_df['ball_distance_to_left_line'],
                    court_df['ball_distance_to_right_line']
                )
                court_df['ball_distance_to_front_baseline'] = 1 - ball_court_y
                court_df['ball_distance_to_back_baseline'] = ball_court_y
                court_df['ball_distance_to_baseline'] = np.minimum(
                    court_df['ball_distance_to_front_baseline'],
                    court_df['ball_distance_to_back_baseline']
                )
            
            # --- プレイヤー位置のコート座標系変換 ---
            for player in ['front', 'back']:
                x_col, y_col = f'player_{player}_x', f'player_{player}_y'
                if x_col in court_df.columns and y_col in court_df.columns:
                    player_court_coords = self.transform_to_court_coordinates(
                        court_df[x_col].values, court_df[y_col].values, court_coords
                    )
                    court_df[f'player_{player}_court_x'] = player_court_coords['x']
                    court_df[f'player_{player}_court_y'] = player_court_coords['y']
                    
                    player_court_x, player_court_y = player_court_coords['x'], player_court_coords['y']
                    
                    court_df[f'player_{player}_in_court'] = ((player_court_x >= 0) & (player_court_x <= 1) &
                                                            (player_court_y >= 0) & (player_court_y <= 1)).astype(int)
                    court_df[f'player_{player}_distance_to_net'] = np.abs(player_court_y - 0.5)
            
            # --- プレイヤー間の関係特徴量（コート座標系） ---
            if all(col in court_df.columns for col in ['player_front_court_x', 'player_back_court_x']):
                front_x, front_y = court_df['player_front_court_x'].values, court_df['player_front_court_y'].values
                back_x, back_y = court_df['player_back_court_x'].values, court_df['player_back_court_y'].values
                
                court_df['players_court_distance'] = np.sqrt((front_x - back_x)**2 + (front_y - back_y)**2)
                court_df['players_correct_sides'] = (front_y > back_y).astype(int)
            
            # --- ボールとプレイヤーの関係特徴量（コート座標系） ---
            if 'ball_court_x' in court_df.columns:
                ball_court_x, ball_court_y = court_df['ball_court_x'].values, court_df['ball_court_y'].values
                for player in ['front', 'back']:
                    if f'player_{player}_court_x' in court_df.columns:
                        player_x, player_y = court_df[f'player_{player}_court_x'].values, court_df[f'player_{player}_court_y'].values
                        court_df[f'ball_to_{player}_court_distance'] = np.sqrt((ball_court_x - player_x)**2 + (ball_court_y - player_y)**2)
            
            print(f"コート特徴量作成完了: {len(court_df.columns) - len(features_df.columns)}特徴量追加")
            return court_df
        
    # コート座標系計算メソッド群
    def calculate_court_geometry(self, court_coords: Dict) -> Dict:
        """コート座標から幾何学的情報を計算"""
        # 基本的なコート情報を計算
        top_left = np.array(court_coords['top_left_corner'])
        top_right = np.array(court_coords['top_right_corner'])
        bottom_left = np.array(court_coords['bottom_left_corner'])
        bottom_right = np.array(court_coords['bottom_right_corner'])
        
        # コートの幅と高さ
        court_width_top = np.linalg.norm(top_right - top_left)
        court_width_bottom = np.linalg.norm(bottom_right - bottom_left)
        court_height_left = np.linalg.norm(bottom_left - top_left)
        court_height_right = np.linalg.norm(bottom_right - top_right)
        
        court_info = {
            'width_top': court_width_top,
            'width_bottom': court_width_bottom,
            'height_left': court_height_left,
            'height_right': court_height_right,
            'avg_width': (court_width_top + court_width_bottom) / 2,
            'avg_height': (court_height_left + court_height_right) / 2
        }
        
        return court_info
    
    def transform_to_court_coordinates(self, x_coords: np.ndarray, y_coords: np.ndarray, 
                                     court_coords: Dict) -> Dict[str, np.ndarray]:
        """画像座標をコート座標系に変換（NumPy高速化版）"""
        tl, tr, bl, br = (np.array(court_coords[k]) for k in ['top_left_corner', 'top_right_corner', 'bottom_left_corner', 'bottom_right_corner'])
        
        court_x = np.full_like(x_coords, -1.0, dtype=float)
        court_y = np.full_like(y_coords, -1.0, dtype=float)
        valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords))
        if not np.any(valid_mask): return {'x': court_x, 'y': court_y}

        valid_x, valid_y = x_coords[valid_mask], y_coords[valid_mask]
        
        try:
            top_ratio = (valid_x - tl[0]) / (tr[0] - tl[0]) if (tr[0] - tl[0]) != 0 else np.full_like(valid_x, 0.5)
            bottom_ratio = (valid_x - bl[0]) / (br[0] - bl[0]) if (br[0] - bl[0]) != 0 else np.full_like(valid_x, 0.5)
            y_ratio = (valid_y - (top_ratio * tr[1] + (1 - top_ratio) * tl[1])) / \
                      ((bottom_ratio * br[1] + (1 - bottom_ratio) * bl[1]) - (top_ratio * tr[1] + (1 - top_ratio) * tl[1]))

            valid_court_x = y_ratio * bottom_ratio + (1 - y_ratio) * top_ratio
            valid_court_y = y_ratio
            
            court_x[valid_mask] = np.clip(valid_court_x, -0.5, 1.5)
            court_y[valid_mask] = np.clip(valid_court_y, -0.5, 1.5)
        except (ZeroDivisionError, ValueError):
            pass
        
        return {'x': court_x, 'y': court_y}
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値を処理（完全ベクトル化版）"""
        df_cleaned = df.copy()
        position_columns = ['ball_x', 'ball_y', 'player_front_x', 'player_front_y', 'player_back_x', 'player_back_y']
        
        for col in position_columns:
            if col in df_cleaned.columns:
                values = df_cleaned[col].values
                mask = pd.isna(values)
                if np.any(mask):
                    valid_indices = np.where(~mask)[0]
                    if len(valid_indices) > 0:
                        first_valid = values[valid_indices[0]]
                        values[:valid_indices[0]] = first_valid
                        for i in range(1, len(values)):
                            if mask[i]: values[i] = values[i-1]
                    else:
                        values[:] = 0
                    df_cleaned[col] = values
        
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in position_columns:
                df_cleaned[col] = np.nan_to_num(df_cleaned[col].values, nan=0)
        
        return df_cleaned
    
    # メイン処理・統合メソッド群
    def extract_all_features(self, video_name: str = None) -> pd.DataFrame:
        """すべての動画から特徴量を抽出してDataFrameを統合"""
        print("=== 特徴量抽出開始 ===")
        
        # データを読み込み
        print("1. データファイル読み込み中...")
        phase_annotations = self.load_phase_annotations(video_name)
        tracking_features = self.load_tracking_features(video_name)
        court_coordinates = self.load_court_coordinates(video_name)
        
        if not phase_annotations:
            print("❌ 局面アノテーションデータが見つかりません")
            return pd.DataFrame()
        
        if not tracking_features:
            print("❌ トラッキング特徴量データが見つかりません")
            return pd.DataFrame()
        
        # データをマッチング
        print("\n2. データマッチング中...")
        matched_data = self.match_video_data(phase_annotations, tracking_features, court_coordinates)
        
        if not matched_data:
            print("❌ マッチするデータが見つかりません")
            return pd.DataFrame()
        
        # 各動画から特徴量を抽出
        print("\n3. 特徴量抽出中...")
        all_features = []
        total_videos = len(matched_data)
        print(f"処理対象動画数: {total_videos}")
        
        for i, (video_name, phase_data, tracking_data, court_coords) in enumerate(tqdm(matched_data, desc="動画別特徴量抽出"), 1):
            print(f"\n--- 動画処理進捗: [{i}/{total_videos}] - {video_name} ---")
            
            try:
                features_df = self.extract_features_from_video(
                    video_name, phase_data, tracking_data, court_coords
                )
                
                if not features_df.empty:
                    all_features.append(features_df)
                    print(f"✅ 特徴量抽出完了: {len(features_df)}行")
                else:
                    print(f"⚠️  特徴量が空です")
                    
            except Exception as e:
                print(f"❌ 特徴量抽出エラー: {e}")
                continue
        
        # 特徴量を統合
        if all_features:
            print(f"\n4. 特徴量統合中... ({len(all_features)}動画)")
            combined_features = pd.concat(all_features, ignore_index=True)
            
            print(f"✅ 統合完了:")
            print(f"   総行数: {len(combined_features)}")
            print(f"   総特徴量数: {len(combined_features.columns)}")
            print(f"   動画数: {combined_features['video_name'].nunique()}")
            
            # ラベル分布を表示
            print("\n📊 全体ラベル分布:")
            label_counts = combined_features['label'].value_counts().sort_index()
            for label_id, count in label_counts.items():
                phase_name = self.phase_labels[label_id] if label_id < len(self.phase_labels) else f"Unknown_{label_id}"
                percentage = count / len(combined_features) * 100
                print(f"   {label_id} ({phase_name}): {count}フレーム ({percentage:.1f}%)")
            
            return combined_features
        else:
            print("❌ 抽出できた特徴量がありません")
            return pd.DataFrame()
    
    def save_features(self, features_df: pd.DataFrame, filename: str = None) -> str:
        """特徴量をファイルに保存"""
        if features_df.empty:
            print("⚠️  保存する特徴量が空です")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tennis_features_{timestamp}.csv"
        
        output_path = self.features_dir / filename
        
        try:
            features_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ 特徴量保存完了: {output_path}")
            print(f"   ファイルサイズ: {output_path.stat().st_size / (1024*1024):.2f} MB")
            return str(output_path)
        except Exception as e:
            print(f"❌ 保存エラー: {e}")
            return ""
    
    def save_feature_info(self, features_df: pd.DataFrame, filename: str = None) -> str:
        """特徴量の詳細情報をJSONファイルに保存"""
        if features_df.empty:
            print("⚠️  保存する特徴量が空です")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tennis_features_info_{timestamp}.json"
        
        output_path = self.features_dir / filename
        
        try:
            # 特徴量情報を作成
            feature_info = {
                'creation_time': datetime.now().isoformat(),
                'total_frames': len(features_df),
                'total_features': len(features_df.columns),
                'videos': features_df['video_name'].nunique(),
                'video_list': features_df['video_name'].unique().tolist(),
                'phase_labels': self.phase_labels,
                'label_distribution': features_df['label'].value_counts().sort_index().to_dict(),
                'feature_columns': features_df.columns.tolist(),
                'feature_types': {
                    'temporal': [col for col in features_df.columns if any(suffix in col for suffix in ['_ma_', '_std_', '_max_', '_min_', '_diff', '_trend_', '_cv_'])],
                    'contextual': [col for col in features_df.columns if any(keyword in col for keyword in ['activity', 'interaction', 'confidence', 'distance', 'movement', 'stability', 'dynamics', 'quality'])],
                    'court': [col for col in features_df.columns if 'court' in col],
                    'basic': [col for col in features_df.columns if col.startswith(('ball_', 'player_')) and not any(suffix in col for suffix in ['_ma_', '_std_', '_max_', '_min_', '_diff', '_trend_', '_cv_', '_court'])]
                },
                'data_quality': {
                    'interpolated_frames': int(features_df.get('interpolated', pd.Series([False] * len(features_df))).sum()),
                    'interpolation_rate': float(features_df.get('interpolated', pd.Series([False] * len(features_df))).mean()),
                    'missing_values': features_df.isnull().sum().to_dict()
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(feature_info, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 特徴量情報保存完了: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 情報保存エラー: {e}")
            return ""
    
    def run_feature_extraction(self, video_name: str = None, save_results: bool = True) -> Tuple[pd.DataFrame, str, str]:
        """特徴量抽出の全プロセスを実行"""
        print("🎾 テニス動画特徴量抽出プロセス開始")
        print("=" * 50)
        
        # 特徴量抽出
        features_df = self.extract_all_features(video_name)
        
        if features_df.empty:
            print("❌ 特徴量抽出に失敗しました")
            return pd.DataFrame(), "", ""
        
        feature_file = ""
        info_file = ""
        
        if save_results:
            print("\n5. ファイル保存中...")
            feature_file = self.save_features(features_df)
            info_file = self.save_feature_info(features_df)
        
        print("\n🎉 特徴量抽出プロセス完了!")
        print("=" * 50)
        
        return features_df, feature_file, info_file
    
    def analyze_features(self, features_df: pd.DataFrame):
        """特徴量の統計分析を表示"""
        if features_df.empty:
            print("⚠️  分析する特徴量が空です")
            return
        
        print("\n📈 特徴量分析結果")
        print("=" * 30)
        
        # 基本統計
        print(f"📊 基本統計:")
        print(f"   総フレーム数: {len(features_df):,}")
        print(f"   特徴量数: {len(features_df.columns)}")
        print(f"   動画数: {features_df['video_name'].nunique()}")
        
        # 動画別統計
        print(f"\n🎬 動画別フレーム数:")
        video_counts = features_df['video_name'].value_counts()
        for video, count in video_counts.items():
            print(f"   {video}: {count:,}フレーム")
        
        # ラベル分布
        print(f"\n🏷️  ラベル分布:")
        label_counts = features_df['label'].value_counts().sort_index()
        for label_id, count in label_counts.items():
            phase_name = self.phase_labels[label_id] if label_id < len(self.phase_labels) else f"Unknown_{label_id}"
            percentage = count / len(features_df) * 100
            print(f"   {label_id} ({phase_name}): {count:,}フレーム ({percentage:.1f}%)")
        
        # データ品質
        if 'interpolated' in features_df.columns:
            interpolated_count = features_df['interpolated'].sum()
            interpolation_rate = interpolated_count / len(features_df) * 100
            print(f"\n🔧 データ品質:")
            print(f"   補間フレーム数: {interpolated_count:,}")
            print(f"   補間率: {interpolation_rate:.1f}%")
        
        # 欠損値
        missing_counts = features_df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if len(missing_cols) > 0:
            print(f"\n⚠️  欠損値:")
            for col, count in missing_cols.items():
                rate = count / len(features_df) * 100
                print(f"   {col}: {count:,} ({rate:.1f}%)")
        else:
            print(f"\n✅ 欠損値: なし")

if __name__ == "__main__":
    # 特徴量抽出器のインスタンスを作成
    # training_dataディレクトリは、このスクリプトと同じ階層にあるか、
    # もしくは適切なパスを指定してください。
    # 例: extractor = TennisFeatureExtractor(training_data_dir="path/to/your/training_data")
    extractor = TennisFeatureExtractor(training_data_dir="training_data")

    # 特定の動画のみを処理する場合 (例: "video1")
    # video_to_process = "your_video_name_here" # 必要に応じて動画名を指定
    video_to_process = None # Noneの場合、全動画を処理

    # 特徴量抽出を実行
    # save_results=True にすると、抽出された特徴量がCSVファイルとして保存されます。
    # Falseにすると保存されません。
    features_dataframe, saved_feature_path, saved_info_path = extractor.run_feature_extraction(
        video_name=video_to_process,
        save_results=True
    )

    if not features_dataframe.empty:
        print(f"\n--- 特徴量抽出成功 ---")
        if saved_feature_path:
            print(f"特徴量データが保存されました: {saved_feature_path}")
        if saved_info_path:
            print(f"特徴量情報が保存されました: {saved_info_path}")

        # 抽出された特徴量の分析
        extractor.analyze_features(features_dataframe)
    else:
        print("\n--- 特徴量抽出失敗 ---")
        print("処理できるデータがなかったか、エラーが発生しました。ログを確認してください。")

    print("\nメイン処理完了")
