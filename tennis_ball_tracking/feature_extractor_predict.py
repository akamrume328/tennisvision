import json
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from tqdm import tqdm
import numba # ★ Numbaをインポート

warnings.filterwarnings('ignore')

class TennisInferenceFeatureExtractor:
    """
    テニス動画のトラッキング特徴量から、推論用の特徴量を作成するクラス
    （局面アノテーションデータは使用しない）
    """
    
    def __init__(self, inference_data_dir: str = "inference_data", court_data_dir: Optional[str] = None):
        self.inference_data_dir = Path(inference_data_dir)
        self.court_data_dir = Path(court_data_dir) if court_data_dir else None
        self.features_dir = Path("./training_data/predict_features")
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"推論用特徴量抽出器を初期化しました")
        print(f"データディレクトリ (トラッキング): {self.inference_data_dir}")
        if self.court_data_dir:
            print(f"データディレクトリ (コート座標): {self.court_data_dir}")
        else:
            print(f"データディレクトリ (コート座標): 指定なし (トラッキングデータディレクトリ '{self.inference_data_dir}' 内を探索)")
        print(f"特徴量保存先: {self.features_dir}")
    
    def load_tracking_features(self, video_name: str = None) -> Dict[str, Dict]:
        """トラッキング特徴量ファイルを読み込み（メタデータ含む）"""
        pattern = "tracking_features_*.json"
        if video_name:
            pattern = f"tracking_features_{video_name}_*.json"
        
        tracking_files = list(self.inference_data_dir.glob(pattern))
        
        if not tracking_files:
            print(f"⚠️  トラッキング特徴量ファイルが見つかりません: {pattern} in {self.inference_data_dir}")
            return {}
        
        all_tracking = {}
        for file_path in tracking_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                filename_parts = file_path.stem.split('_')
                if len(filename_parts) >= 3:
                    video_key = '_'.join(filename_parts[2:-1])
                else:
                    video_key = file_path.stem
                
                if isinstance(data, dict) and 'metadata' in data and 'frames' in data:
                    all_tracking[video_key] = data
                    print(f"✅ トラッキング特徴量読み込み（新形式）: {file_path.name}")
                    print(f"   動画: {video_key}, フレーム数: {len(data['frames'])}")
                    if 'frame_skip' in data['metadata']:
                        print(f"   フレームスキップ: {data['metadata']['frame_skip']}")
                elif isinstance(data, list):
                    all_tracking[video_key] = {
                        'metadata': {'frame_skip': 1, 'legacy_format': True},
                        'frames': data
                    }
                    print(f"✅ トラッキング特徴量読み込み（旧形式）: {file_path.name}")
                    print(f"   動画: {video_key}, フレーム数: {len(data)}")
                    print(f"   注意: 旧形式のためフレームスキップ情報なし")
                else:
                    print(f"⚠️  不明なデータ形式: {file_path.name}")
                    continue
                
            except Exception as e:
                print(f"❌ 読み込みエラー: {file_path.name} - {e}")
        
        return all_tracking
    
    def load_court_coordinates(self, video_name: str = None) -> Dict[str, Dict]:
        """コート座標データを読み込み"""
        pattern = "court_coords_*.json"
        if video_name:
            pattern = f"court_coords_{video_name}_*.json"
        
        data_source_dir = self.court_data_dir if self.court_data_dir else self.inference_data_dir
        
        court_files = list(data_source_dir.glob(pattern))
        
        if not court_files:
            print(f"⚠️  コート座標ファイルが見つかりません: {pattern} in {data_source_dir}")
            return {}
        
        all_court_coords = {}
        for file_path in court_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                filename_parts = file_path.stem.split('_')
                if len(filename_parts) >= 3:
                    video_key = '_'.join(filename_parts[2:])
                else:
                    video_key = file_path.stem
                
                all_court_coords[video_key] = data
                print(f"✅ コート座標読み込み: {file_path.name}")
                print(f"   動画: {video_key}")
                
            except Exception as e:
                print(f"❌ 読み込みエラー: {file_path.name} - {e}")
        
        return all_court_coords
    
    def match_video_data(self, tracking_features: Dict, court_coordinates: Dict = None) -> List[Tuple[str, Dict, Dict]]:
        """動画名でトラッキングデータ、コート座標をマッチング"""
        matched_data = []
        
        print("\n=== データマッチング (推論用) ===")
        print(f"トラッキング特徴量動画数: {len(tracking_features)}")
        print(f"コート座標動画数: {len(court_coordinates) if court_coordinates else 0}")
        
        print("\n--- 利用可能なファイル ---")
        print("トラッキング特徴量:")
        for key in tracking_features.keys():
            print(f"  - {key}")
        if court_coordinates:
            print("コート座標:")
            for key in court_coordinates.keys():
                print(f"  - {key}")
        
        print("\n--- マッチング処理 ---")
        
        for video_name in tracking_features.keys():
            tracking_data = tracking_features[video_name]
            court_data = None
            court_match_type = "なし"
            court_matched_key = ""
            
            print(f"\n🎯 処理中: {video_name}")
            
            if court_coordinates:
                if video_name in court_coordinates:
                    court_data = court_coordinates[video_name]
                    court_match_type = "完全一致"
                    court_matched_key = video_name
                    print(f"  🎾 コート座標: ✅ 完全一致 - {video_name}")
                else:
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
            
            matched_data.append((
                video_name,
                tracking_data,
                court_data
            ))
            
            frame_count = self.get_actual_frame_count(tracking_data)
            print(f"  ✅ マッチング成功:")
            print(f"     トラッキング: {video_name}")
            print(f"     実際のフレーム数: {frame_count}")
            if isinstance(tracking_data, dict) and 'metadata' in tracking_data:
                frame_skip = tracking_data['metadata'].get('frame_skip', 1)
                print(f"     フレームスキップ: {frame_skip}")
            if court_data:
                print(f"     コート座標: {court_matched_key} ({court_match_type})")
            else:
                print(f"     コート座標: なし")
        
        print(f"\n=== マッチング結果サマリー ===")
        print(f"✅ マッチング数: {len(matched_data)}")
        
        if matched_data:
            print("\n📋 マッチング一覧:")
            for i, (video_name, tracking_data, court_data) in enumerate(matched_data, 1):
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
        
        if isinstance(tracking_data_dict, list):
            tracking_data = tracking_data_dict
            validation_result['warnings'].append("旧形式データ: フレームスキップ情報が記録されていません")
        elif isinstance(tracking_data_dict, dict) and 'frames' in tracking_data_dict:
            tracking_data = tracking_data_dict['frames']
            metadata = tracking_data_dict.get('metadata', {})
            validation_result['metadata_available'] = True
            
            recorded_skip = metadata.get('frame_skip', 1)
            processing_mode = metadata.get('processing_mode', 'unknown')
            
            validation_result['recorded_frame_skip'] = recorded_skip
            validation_result['processing_mode'] = processing_mode
            
            if recorded_skip > 1:
                validation_result['frame_skip_detected'] = True
                validation_result['warnings'].append(
                    f"記録されたフレームスキップ: {recorded_skip} (モード: {processing_mode})"
                )
            
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
        
        frame_numbers = [data.get('frame_number', 0) for data in tracking_data]
        frame_numbers.sort()
        
        from collections import Counter
        frame_counts = Counter(frame_numbers)
        duplicates = [frame for frame, count in frame_counts.items() if count > 1]
        if duplicates:
            validation_result['duplicate_frames'] = duplicates
            validation_result['warnings'].append(f"重複フレームが検出されました: {duplicates}")
        
        intervals = []
        missing_frames = []
        
        for i in range(1, len(frame_numbers)):
            interval = frame_numbers[i] - frame_numbers[i-1]
            if interval > 1:
                intervals.append(interval)
                for missing in range(frame_numbers[i-1] + 1, frame_numbers[i]):
                    missing_frames.append(missing)
            elif interval == 1:
                intervals.append(1)
        
        if intervals:
            interval_counts = Counter(intervals)
            most_common_interval = interval_counts.most_common(1)[0][0]
            
            validation_result['actual_frame_skip_interval'] = most_common_interval
            
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
        
        interpolated_count = sum(1 for data in tracking_data if data.get('interpolated', False))
        validation_result['interpolated_count'] = interpolated_count
        
        if interpolated_count > 0:
            validation_result['warnings'].append(f"補間フレームが含まれています: {interpolated_count}フレーム")
        
        if len(tracking_data) > 0:
            ball_detection_rate = sum(1 for data in tracking_data if data.get('ball_detected', 0)) / len(tracking_data)
            validation_result['ball_detection_rate'] = ball_detection_rate
            if ball_detection_rate < 0.3:
                validation_result['warnings'].append(f"ボール検出率が低いです: {ball_detection_rate:.1%}")
        else:
            validation_result['ball_detection_rate'] = 0

        return validation_result
    
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
        
        pattern = f"tracking_features_{video_name}_*.json"
        tracking_files = list(self.inference_data_dir.glob(pattern))
        
        if not tracking_files:
            print(f"対応するファイルが見つかりません: {pattern}")
            return []
        
        for file_path in tracking_files:
            try:
                print(f"ファイル再読み込み試行: {file_path.name}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                print(f"生データ型: {type(raw_data)}")
                
                if isinstance(raw_data, list):
                    print(f"リスト形式: {len(raw_data)}要素")
                    return raw_data
                elif isinstance(raw_data, dict):
                    print(f"辞書形式キー: {list(raw_data.keys())}")
                    
                    if 'frames' in raw_data:
                        frames = raw_data['frames']
                        if isinstance(frames, list): return frames
                    
                    if 'frame_data' in raw_data:
                        frame_data = raw_data['frame_data']
                        if isinstance(frame_data, list): return frame_data
                    
                    numeric_keys = [k for k in raw_data.keys() if str(k).isdigit()]
                    if numeric_keys:
                        frame_list = []
                        for key in sorted(numeric_keys, key=int):
                            frame_data = raw_data[key]
                            if isinstance(frame_data, dict):
                                frame_data['frame_number'] = int(key)
                                frame_list.append(frame_data)
                        if frame_list: return frame_list
                
                print("⚠️  認識可能なデータ構造が見つかりません")
                return []
                
            except Exception as e:
                print(f"代替読み込みエラー: {e}")
                continue
        
        return []
    
    def normalize_frame_numbers(self, tracking_data_dict: Dict) -> List[Dict]:
        """フレーム番号を正規化し、フレームスキップを考慮した展開を行う"""
        print("=== フレーム番号正規化・展開処理 ===")
        
        if isinstance(tracking_data_dict, list):
            print("⚠️  旧形式データ: フレームスキップ情報が記録されていません")
            return self.legacy_normalize_frame_numbers_fast(tracking_data_dict)
        
        metadata = tracking_data_dict.get('metadata', {})
        frame_skip = metadata.get('frame_skip', 1)
        
        print(f"✅ 記録されたフレームスキップ情報: {frame_skip}")
        
        if frame_skip == 1:
            frames_data = tracking_data_dict.get('frames', [])
            print("フレームスキップなし - 通常の正規化を実行")
            return frames_data
        else:
            print(f"フレームスキップ({frame_skip})を検出 - 元シーケンスへの展開を実行")
            expanded_frames = self.expand_frames_to_original_sequence_fast(tracking_data_dict)
            return expanded_frames
    
    def legacy_normalize_frame_numbers_fast(self, tracking_data: List[Dict]) -> List[Dict]:
        """旧形式データのフレーム番号正規化（NumPy高速化版）"""
        if not tracking_data:
            return []
        
        frame_numbers = np.array([frame.get('frame_number', 0) for frame in tracking_data])
        sorted_indices = np.argsort(frame_numbers)
        
        sorted_data = [tracking_data[i] for i in sorted_indices]
        
        for i, frame_data in enumerate(sorted_data):
            frame_data['original_frame_number'] = frame_data.get('frame_number', 0)
            frame_data['frame_number'] = i
            frame_data['interpolated'] = False
        
        return sorted_data
    
    def expand_frames_to_original_sequence_fast(self, tracking_data_dict: Dict) -> List[Dict]:
        """フレームシーケンス展開処理（NumPy高速化版）"""
        print("=== フレームシーケンス展開処理 (NumPy高速化) ===")
        
        metadata = tracking_data_dict['metadata']
        frames_data = tracking_data_dict['frames']
        frame_skip = metadata.get('frame_skip', 1)
        total_original_frames = metadata.get('total_original_frames')
        
        print(f"記録されたフレームスキップ: {frame_skip}")
        print(f"処理済みフレーム数: {len(frames_data)}")
        
        if not frames_data:
            return []
        
        frame_numbers = np.array([frame.get('frame_number', 0) for frame in frames_data])
        sorted_indices = np.argsort(frame_numbers)
        sorted_frames = [frames_data[i] for i in sorted_indices]
        sorted_frame_numbers = frame_numbers[sorted_indices]
        
        last_frame = sorted_frame_numbers[-1]
        
        if total_original_frames is None:
            total_original_frames = last_frame + frame_skip
        
        print(f"推定総フレーム数: {total_original_frames}")
        
        processed_frame_map = {frame_num: frame for frame_num, frame in zip(sorted_frame_numbers, sorted_frames)}
        
        all_frame_numbers = np.arange(total_original_frames)
        
        expanded_frames = []
        interpolated_count = 0
        
        for frame_num in all_frame_numbers:
            if frame_num in processed_frame_map:
                frame_data = processed_frame_map[frame_num].copy()
                frame_data['original_frame_number'] = frame_num
                frame_data['interpolated'] = False
                expanded_frames.append(frame_data)
            else:
                interpolated_frame = self.create_interpolated_frame_from_skip_fast(
                    frame_num, processed_frame_map, sorted_frame_numbers, frame_skip
                )
                expanded_frames.append(interpolated_frame)
                interpolated_count += 1
        
        print(f"✅ 展開完了: 総{len(expanded_frames)}フレーム, 補間{interpolated_count}フレーム")
        return expanded_frames
    
    def create_interpolated_frame_from_skip_fast(self, target_frame_num: int, 
                                               processed_frame_map: Dict, 
                                               sorted_frame_numbers: np.ndarray, 
                                               frame_skip: int) -> Dict:
        """補間フレーム作成（NumPy高速化版）"""
        prev_mask = sorted_frame_numbers < target_frame_num
        next_mask = sorted_frame_numbers > target_frame_num
        
        prev_frame_data = None
        next_frame_data = None
        
        if np.any(prev_mask):
            prev_frame_num = sorted_frame_numbers[prev_mask][-1]
            prev_frame_data = processed_frame_map[prev_frame_num]
        
        if np.any(next_mask):
            next_frame_num = sorted_frame_numbers[next_mask][0]
            next_frame_data = processed_frame_map[next_frame_num]
        
        interpolated_frame = {
            'frame_number': target_frame_num,
            'original_frame_number': target_frame_num,
            'interpolated': True,
            'timestamp': '',
            'ball_detected': 0, 'ball_x': None, 'ball_y': None, 'ball_x_normalized': 0,
            'ball_y_normalized': 0, 'ball_movement_score': 0, 'ball_tracking_confidence': 0,
            'ball_velocity_x': 0, 'ball_velocity_y': 0, 'ball_speed': 0,
            'player_front_count': 0, 'player_back_count': 0, 'total_players': 0,
            'candidate_balls_count': 0, 'disappeared_count': 0, 'trajectory_length': 0,
            'prediction_active': 0
        }
        
        if prev_frame_data and next_frame_data:
            prev_frame_num = prev_frame_data.get('frame_number', 0)
            next_frame_num = next_frame_data.get('frame_number', 0)
            
            if next_frame_num > prev_frame_num:
                ratio = (target_frame_num - prev_frame_num) / (next_frame_num - prev_frame_num)
                ratio = np.clip(ratio, 0, 1)
                
                numeric_keys = [
                    'ball_x', 'ball_y', 'ball_x_normalized', 'ball_y_normalized', 'ball_velocity_x', 
                    'ball_velocity_y', 'ball_speed', 'ball_movement_score', 'ball_tracking_confidence',
                    'player_front_x', 'player_front_y', 'player_front_x_normalized', 'player_front_y_normalized',
                    'player_back_x', 'player_back_y', 'player_back_x_normalized', 'player_back_y_normalized',
                    'player_front_confidence', 'player_back_confidence', 'player_distance', 'player_distance_normalized'
                ]
                
                prev_values = np.array([prev_frame_data.get(key, 0) or 0 for key in numeric_keys], dtype=float)
                next_values = np.array([next_frame_data.get(key, 0) or 0 for key in numeric_keys], dtype=float)
                
                interpolated_values = prev_values + (next_values - prev_values) * ratio
                
                for key, value in zip(numeric_keys, interpolated_values):
                    interpolated_frame[key] = value
        
        elif prev_frame_data:
            inherit_keys = ['ball_x', 'ball_y', 'ball_x_normalized', 'ball_y_normalized',
                           'player_front_x', 'player_front_y', 'player_back_x', 'player_back_y']
            for key in inherit_keys:
                if key in prev_frame_data and prev_frame_data[key] is not None:
                    interpolated_frame[key] = prev_frame_data[key]
        
        return interpolated_frame

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
            
            ma_results, std_results, max_results, min_results = TennisInferenceFeatureExtractor.vectorized_rolling_stats(
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

    @staticmethod
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

    def extract_features_from_video(self, video_name: str, tracking_data_dict: Dict, court_coords: Dict = None) -> pd.DataFrame:
        """単一動画から特徴量を抽出"""
        print(f"\n--- 推論用特徴量抽出: {video_name} ---")
        
        print("  ステップ1/4: データ検証中...")
        validation_result = self.validate_tracking_data_consistency(tracking_data_dict)
        for key, value in validation_result.items():
            if key == 'warnings' and value:
                for warning in value: print(f"  📋 {warning}")
            elif key in ['missing_frames', 'duplicate_frames']:
                if value:
                    print(f"  {key}: {len(value)}個のフレームを検出")
            elif key != 'warnings':
                print(f"  {key}: {value}")

        if validation_result['frame_count'] <= 10:
            print(f"⚠️  異常にフレーム数が少ないです: {validation_result['frame_count']}")
            self.diagnose_tracking_data_structure(tracking_data_dict, video_name)
            alternative_data = self.attempt_alternative_data_loading(video_name)
            if alternative_data and len(alternative_data) > validation_result['frame_count']:
                print(f"✅ 代替データ読み込み成功: {len(alternative_data)}フレーム")
                current_metadata = tracking_data_dict.get('metadata', {'frame_skip': 1, 'legacy_format': True})
                tracking_data_dict = {'metadata': current_metadata, 'frames': alternative_data}
        
        print("  ステップ2/4: フレーム番号正規化中...")
        normalized_tracking_data = self.normalize_frame_numbers(tracking_data_dict)
        
        if not normalized_tracking_data:
            print("⚠️  正規化後のトラッキングデータが空です")
            return pd.DataFrame()
        
        print("  ステップ3/4: DataFrame作成中...")
        features_df = self.safe_create_dataframe_from_tracking_data(normalized_tracking_data, video_name)
        
        if features_df.empty:
            print("❌ DataFrame作成に失敗しました")
            return pd.DataFrame()
        if 'frame_number' in features_df.columns:
            features_df['original_frame_number'] = features_df['frame_number']

        features_df['frame_number'] = range(len(features_df))
        features_df['video_name'] = video_name
        if 'interpolated' not in features_df.columns:
            features_df['interpolated'] = False
        
        print("  ステップ4/4: 特徴量エンジニアリング中...")
        features_df = self.handle_missing_values(features_df)
        features_df = self.create_court_features(features_df, court_coords)
        features_df = self.create_temporal_features(features_df)
        features_df = self.create_contextual_features(features_df)
        
        print(f"最終特徴量数: {len(features_df.columns)}")
        return features_df
    
    def safe_create_dataframe_from_tracking_data(self, tracking_data: List[Dict], video_name: str) -> pd.DataFrame:
        """トラッキングデータから安全にDataFrameを作成"""
        if not tracking_data:
            return pd.DataFrame()
        try:
            return pd.DataFrame(tracking_data)
        except Exception as e:
            print(f"❌ DataFrame作成で例外発生: {e}")
            return self.handle_mixed_data_types_error_fast(tracking_data, video_name)
    
    def handle_mixed_data_types_error_fast(self, tracking_data: List[Dict], video_name: str) -> pd.DataFrame:
        """混合データ型エラーの処理（NumPy高速化版）"""
        print("=== 混合データ型エラー対処中 (NumPy高速化) ===")
        all_keys = set()
        valid_frames = [frame for frame in tracking_data if isinstance(frame, dict)]
        if not valid_frames: return pd.DataFrame()
        for frame in valid_frames: all_keys.update(frame.keys())
        
        normalized_data = {}
        for key in all_keys:
            values = []
            for frame in valid_frames:
                value = frame.get(key)
                if value is None: values.append(0.0)
                elif isinstance(value, (int, float, bool)): values.append(float(value))
                elif isinstance(value, str):
                    try: values.append(float(value))
                    except ValueError: values.append(0.0)
                elif isinstance(value, (list, tuple)): values.append(float(value[0]) if len(value) > 0 and isinstance(value[0], (int, float)) else float(len(value)))
                elif isinstance(value, dict): values.append(np.sqrt(float(value.get('x',0))**2 + float(value.get('y',0))**2) if 'x' in value and 'y' in value else float(len(value)))
                else: values.append(0.0)
            normalized_data[key] = np.array(values, dtype=float)
        
        if 'frame_number' not in normalized_data:
            normalized_data['frame_number'] = np.arange(len(valid_frames), dtype=float)
        
        return pd.DataFrame(normalized_data)

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
    
    def calculate_court_geometry(self, court_coords: Dict) -> Dict:
        """コート座標から幾何学的情報を計算"""
        corners = {k: np.array(v) for k, v in court_coords.items() if 'corner' in k}
        return {
            'avg_width': (np.linalg.norm(corners['top_right'] - corners['top_left']) + np.linalg.norm(corners['bottom_right'] - corners['bottom_left'])) / 2,
            'avg_height': (np.linalg.norm(corners['bottom_left'] - corners['top_left']) + np.linalg.norm(corners['bottom_right'] - corners['top_right'])) / 2,
        }

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
    
    def extract_all_features(self, video_name: str = None) -> pd.DataFrame:
        """すべての動画から特徴量を抽出してDataFrameを統合"""
        print("=== 推論用特徴量抽出開始 ===")
        
        print("1. データファイル読み込み中...")
        tracking_features = self.load_tracking_features(video_name)
        court_coordinates = self.load_court_coordinates(video_name)
        
        if not tracking_features:
            print("❌ トラッキング特徴量データが見つかりません")
            return pd.DataFrame()
        
        print("\n2. データマッチング中...")
        matched_data = self.match_video_data(tracking_features, court_coordinates)
        
        if not matched_data:
            print("❌ マッチするデータが見つかりません")
            return pd.DataFrame()
        
        print("\n3. 特徴量抽出中...")
        all_features_list = []
        total_videos = len(matched_data)
        print(f"処理対象動画数: {total_videos}")
        
        for i, (vid_name, track_data, court_data) in enumerate(matched_data, 1):
            print(f"\n--- 動画処理進捗: [{i}/{total_videos}] - {vid_name} ---")
            try:
                features_df = self.extract_features_from_video(
                    vid_name, track_data, court_data
                )
                
                if not features_df.empty:
                    all_features_list.append(features_df)
                    print(f"✅ 特徴量抽出完了: {len(features_df)}行")
                else:
                    print(f"⚠️  特徴量が空です: {vid_name}")
            except Exception as e:
                print(f"❌ 特徴量抽出エラー ({vid_name}): {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if all_features_list:
            print(f"\n4. 特徴量統合中... ({len(all_features_list)}動画)")
            combined_features = pd.concat(all_features_list, ignore_index=True)
            print("✅ 統合完了:")
            print(f"   総行数: {len(combined_features)}")
            print(f"   総特徴量数: {len(combined_features.columns)}")
            if 'video_name' in combined_features.columns:
                print(f"   動画数: {combined_features['video_name'].nunique()}")
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
            filename = f"tennis_inference_features_{timestamp}.csv"
        
        output_path = self.features_dir / filename
        
        try:
            features_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ 特徴量保存完了: {output_path}")
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
            filename = f"tennis_inference_features_info_{timestamp}.json"
        
        output_path = self.features_dir / filename
        
        try:
            info = {
                'creation_time': datetime.now().isoformat(),
                'total_frames': len(features_df),
                'total_features': len(features_df.columns),
                'videos': features_df['video_name'].nunique() if 'video_name' in features_df.columns else 0,
                'video_list': features_df['video_name'].unique().tolist() if 'video_name' in features_df.columns else [],
                'feature_columns': features_df.columns.tolist(),
                'data_quality': {
                    'interpolated_frames': int(features_df.get('interpolated', pd.Series([False])).sum()),
                    'interpolation_rate': float(features_df.get('interpolated', pd.Series([False])).mean()),
                    'missing_values_summary': features_df.isnull().sum()[features_df.isnull().sum() > 0].to_dict()
                }
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            print(f"✅ 推論用特徴量情報保存完了: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"❌ 情報保存エラー: {e}")
            return ""
    
    def run_feature_extraction(self, video_name: str = None, save_results: bool = True) -> Tuple[pd.DataFrame, str, str]:
        """特徴量抽出の全プロセスを実行"""
        print("🎾 テニス動画 推論用特徴量抽出プロセス開始")
        print("=" * 50)
        
        features_df = self.extract_all_features(video_name)
        
        if features_df.empty:
            print("❌ 推論用特徴量抽出に失敗しました")
            return pd.DataFrame(), "", ""
        
        feature_file, info_file = "", ""
        if save_results:
            print("\n5. ファイル保存中...")
            feature_file = self.save_features(features_df)
            info_file = self.save_feature_info(features_df)
        
        print("\n🎉 推論用特徴量抽出プロセス完了!")
        print("=" * 50)
        
        return features_df, feature_file, info_file
    
    def analyze_features(self, features_df: pd.DataFrame):
        """特徴量の統計分析を表示"""
        if features_df.empty:
            print("⚠️  分析する特徴量が空です")
            return
        
        print("\n📈 推論用特徴量分析結果")
        print("=" * 30)
        
        print(f"📊 基本統計:")
        print(f"   総フレーム数: {len(features_df):,}")
        print(f"   特徴量数: {len(features_df.columns)}")
        if 'video_name' in features_df.columns:
            print(f"   動画数: {features_df['video_name'].nunique()}")
        
            print(f"\n🎬 動画別フレーム数:")
            video_counts = features_df['video_name'].value_counts()
            for video, count in video_counts.items():
                print(f"   {video}: {count:,}フレーム")

        if 'interpolated' in features_df.columns:
            interpolated_count = features_df['interpolated'].sum()
            interpolation_rate = interpolated_count / len(features_df) * 100 if len(features_df) > 0 else 0
            print(f"\n🔧 データ品質:")
            print(f"   補間フレーム数: {interpolated_count:,}")
            print(f"   補間率: {interpolation_rate:.1f}%")
        
        missing_counts = features_df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if not missing_cols.empty:
            print(f"\n⚠️  欠損値:")
            for col, count in missing_cols.items():
                print(f"   {col}: {count:,} ({count / len(features_df) * 100:.1f}%)")
        else:
            print(f"\n✅ 欠損値: なし")


if __name__ == "__main__":
    default_input_dir = "./training_data" 
    
    if not Path(default_input_dir).exists():
        print(f"⚠️  デフォルトの入力ディレクトリ '{default_input_dir}' が見つかりません。")
        # Example of how to handle this case, e.g., by exiting or asking for input
        # exit()
        
    extractor = TennisInferenceFeatureExtractor(inference_data_dir=default_input_dir)
    video_to_process = None 

    features_dataframe, saved_feature_path, saved_info_path = extractor.run_feature_extraction(
        video_name=video_to_process,
        save_results=True
    )

    if not features_dataframe.empty:
        print(f"\n--- 推論用特徴量抽出成功 ---")
        if saved_feature_path:
            print(f"推論用特徴量データが保存されました: {saved_feature_path}")
        if saved_info_path:
            print(f"推論用特徴量情報が保存されました: {saved_info_path}")

        print("\n--- 特徴量分析開始 (オプション) ---")
        extractor.analyze_features(features_dataframe)
        print("--- 特徴量分析完了 ---")
    else:
        print("\n--- 推論用特徴量抽出失敗 ---")
        print("処理できるデータがなかったか、エラーが発生しました。ログを確認してください。")

    print("\nメイン処理完了")