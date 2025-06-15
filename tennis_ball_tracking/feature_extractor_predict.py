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

warnings.filterwarnings('ignore')

class TennisInferenceFeatureExtractor: # クラス名を変更
    """
    テニス動画のトラッキング特徴量から、推論用の特徴量を作成するクラス
    （局面アノテーションデータは使用しない）
    """
    
    def __init__(self, inference_data_dir: str = "inference_data", court_data_dir: Optional[str] = None): # court_data_dir 引数を追加
        self.inference_data_dir = Path(inference_data_dir)
        self.court_data_dir = Path(court_data_dir) if court_data_dir else None # court_data_dir を初期化
        self.features_dir = Path("./training_data/predict_features") # 保存先ディレクトリ名を指定のパスに変更
        self.features_dir.mkdir(parents=True, exist_ok=True) # parents=True を追加して親ディレクトリも作成
        
        # phase_labels と label_to_id は推論時には不要なため削除
        
        print(f"推論用特徴量抽出器を初期化しました") # print文を変更
        print(f"データディレクトリ (トラッキング): {self.inference_data_dir}") # print文を変更
        if self.court_data_dir:
            print(f"データディレクトリ (コート座標): {self.court_data_dir}")
        else:
            print(f"データディレクトリ (コート座標): 指定なし (トラッキングデータディレクトリ '{self.inference_data_dir}' 内を探索)")
        print(f"特徴量保存先: {self.features_dir}") # print文を変更
    
    # データ読み込みメソッド群
    # load_phase_annotations メソッドは推論時には不要なため削除
    
    def load_tracking_features(self, video_name: str = None) -> Dict[str, Dict]:
        """トラッキング特徴量ファイルを読み込み（メタデータ含む）"""
        pattern = "tracking_features_*.json"
        if video_name:
            pattern = f"tracking_features_{video_name}_*.json"
        
        tracking_files = list(self.inference_data_dir.glob(pattern)) # 参照ディレクトリを変更
        
        if not tracking_files:
            print(f"⚠️  トラッキング特徴量ファイルが見つかりません: {pattern} in {self.inference_data_dir}") # メッセージにディレクトリパス追加
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
        
        # 読み込み元のディレクトリを決定
        data_source_dir = self.court_data_dir if self.court_data_dir else self.inference_data_dir
        
        court_files = list(data_source_dir.glob(pattern))
        
        if not court_files:
            print(f"⚠️  コート座標ファイルが見つかりません: {pattern} in {data_source_dir}") # メッセージにディレクトリパス追加
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
    def match_video_data(self, tracking_features: Dict, court_coordinates: Dict = None) -> List[Tuple[str, Dict, Dict]]: # phase_annotations を削除
        """動画名でトラッキングデータ、コート座標をマッチング（データ構造更新・診断強化）"""
        matched_data = []
        
        print("\n=== データマッチング (推論用) ===") # print文を変更
        # print(f"局面アノテーション動画数: {len(phase_annotations)}") # 削除
        print(f"トラッキング特徴量動画数: {len(tracking_features)}")
        print(f"コート座標動画数: {len(court_coordinates) if court_coordinates else 0}")
        
        print("\n--- 利用可能なファイル ---")
        # print("局面アノテーション:") # 削除
        # for key in phase_annotations.keys(): # 削除
        #     print(f"  - {key}") # 削除
        print("トラッキング特徴量:")
        for key in tracking_features.keys():
            print(f"  - {key}")
        if court_coordinates:
            print("コート座標:")
            for key in court_coordinates.keys():
                print(f"  - {key}")
        
        print("\n--- マッチング処理 ---")
        
        for video_name in tracking_features.keys(): # tracking_features を基準に変更
            tracking_data = tracking_features[video_name] # tracking_data を直接使用
            court_data = None
            # tracking_match_type = "なし" # 削除
            court_match_type = "なし"
            # tracking_matched_key = "" # 削除
            court_matched_key = ""
            
            print(f"\n🎯 処理中: {video_name}")
            
            # トラッキングデータのマッチング処理は不要 (tracking_features を基準とするため)
            
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
                        if video_name in court_key: # video_name が court_key に含まれる場合
                            court_data = court_coordinates[court_key]
                            court_match_type = "部分一致(動画名がキーに含まれる)"
                            court_matched_key = court_key
                            print(f"  🎾 コート座標: ✅ 部分一致 - {video_name} ⊆ {court_key}")
                            break
                        elif court_key in video_name: # court_key が video_name に含まれる場合
                            court_data = court_coordinates[court_key]
                            court_match_type = "部分一致(キーが動画名に含まれる)"
                            court_matched_key = court_key
                            print(f"  🎾 コート座標: ✅ 部分一致 - {court_key} ⊆ {video_name}")
                            break
                    if not court_data:
                        print(f"  🎾 コート座標: ❌ マッチなし")
            else:
                print(f"  🎾 コート座標: ❌ データなし")
            
            # 結果の判定 (tracking_data は常に存在すると仮定)
            matched_data.append((
                video_name,
                # phase_annotations[video_name], # 削除
                tracking_data,
                court_data
            ))
            
            frame_count = self.get_actual_frame_count(tracking_data)
            print(f"  ✅ マッチング成功:")
            # print(f"     局面アノテーション: {video_name}") # 削除
            print(f"     トラッキング: {video_name}") # tracking_matched_key と tracking_match_type を削除
            print(f"     実際のフレーム数: {frame_count}")
            if isinstance(tracking_data, dict) and 'metadata' in tracking_data:
                frame_skip = tracking_data['metadata'].get('frame_skip', 1)
                print(f"     フレームスキップ: {frame_skip}")
            if court_data:
                print(f"     コート座標: {court_matched_key} ({court_match_type})")
            else:
                print(f"     コート座標: なし")
            # else: # 削除 (tracking_data は常に存在)
            #     print(f"  ❌ マッチング失敗: トラッキングデータが見つかりません") # 削除
        
        print(f"\n=== マッチング結果サマリー ===")
        print(f"✅ マッチング数: {len(matched_data)}") # 表現を変更
        # print(f"❌ 失敗したマッチング: {len(phase_annotations) - len(matched_data)}") # 削除
        
        if matched_data:
            print("\n📋 マッチング一覧:")
            for i, (video_name, tracking_data, court_data) in enumerate(matched_data, 1): # phase_data を削除
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
    
    # メイン特徴量抽出メソッド
    def extract_features_from_video(self, video_name: str, tracking_data_dict: Dict, court_coords: Dict = None) -> pd.DataFrame: # phase_data を削除
        """単一動画から特徴量を抽出（記録されたフレームスキップ情報対応）"""
        print(f"\n--- 推論用特徴量抽出: {video_name} ---") # print文を変更
        
        # データの一貫性を検証
        print("  ステップ1/4: データ検証中...") # ステップ数を変更
        validation_result = self.validate_tracking_data_consistency(tracking_data_dict)
        
        print("  === データ検証結果 ===")
        print(f"データ品質: {'良好' if validation_result['is_valid'] else '問題あり'}")
        print(f"フレーム数: {validation_result['frame_count']}")
        print(f"メタデータ: {'あり' if validation_result['metadata_available'] else 'なし（旧形式）'}")
        
        if validation_result['metadata_available']:
            print(f"処理モード: {validation_result['processing_mode']}")
            print(f"記録されたフレームスキップ: {validation_result['recorded_frame_skip']}")
            print(f"実測フレームスキップ: {validation_result['actual_frame_skip_interval']}")
            
            # メタデータの詳細情報
            if isinstance(tracking_data_dict, dict) and 'metadata' in tracking_data_dict:
                metadata = tracking_data_dict['metadata']
                if 'original_fps' in metadata and metadata['original_fps']:
                    print(f"元FPS: {metadata['original_fps']}")
                if 'processing_fps' in metadata and metadata['processing_fps']:
                    print(f"処理FPS: {metadata['processing_fps']}")
                if 'total_original_frames' in metadata and metadata['total_original_frames']:
                    print(f"元動画総フレーム数: {metadata['total_original_frames']}")
                if 'processing_efficiency' in metadata:
                    print(f"処理効率: {metadata['processing_efficiency']}")
        
        if validation_result['interpolated_count'] > 0:
            print(f"既存補間フレーム数: {validation_result['interpolated_count']}")
        
        if validation_result['warnings']:
            for warning in validation_result['warnings']:
                print(f"📋 {warning}")
        
        # フレーム数が異常に少ない場合の診断
        if validation_result['frame_count'] <= 10:
            print(f"⚠️  異常にフレーム数が少ないです: {validation_result['frame_count']}")
            self.diagnose_tracking_data_structure(tracking_data_dict, video_name)
            
            # 緊急対処：元のJSONファイルから直接読み込み試行
            alternative_data = self.attempt_alternative_data_loading(video_name)
            if alternative_data and len(alternative_data) > validation_result['frame_count']:
                print(f"✅ 代替データ読み込み成功: {len(alternative_data)}フレーム")
                # メタデータを保持しつつフレームデータを更新
                current_metadata = tracking_data_dict.get('metadata', {'frame_skip': 1, 'legacy_format': True})
                tracking_data_dict = {
                    'metadata': current_metadata,
                    'frames': alternative_data
                }
                validation_result['frame_count'] = len(alternative_data) # フレーム数を更新
        
        # フレーム番号を正規化（記録されたスキップ情報を使用）
        print("  ステップ2/4: フレーム番号正規化中...") # ステップ数を変更
        normalized_tracking_data = self.normalize_frame_numbers(tracking_data_dict)
        
        # 基本情報
        total_frames = len(normalized_tracking_data)
        # fps = phase_data.get('fps', 30.0) # phase_data からではなく metadata から取得
        metadata = tracking_data_dict.get('metadata', {})
        fps = metadata.get('original_fps', metadata.get('processing_fps', 30.0)) # original_fps優先、なければprocessing_fps、デフォルト30
        # phase_changes = phase_data.get('phase_changes', []) # 削除
        
        print(f"正規化後フレーム数: {total_frames}, FPS (推定): {fps}") # print文を変更
        # print(f"局面変更数: {len(phase_changes)}") # 削除
        print(f"コート座標: {'あり' if court_coords else 'なし'}")
        
        # フレームデータが空の場合のチェック
        if not normalized_tracking_data:
            print("⚠️  正規化後のトラッキングデータが空です")
            return pd.DataFrame()
        
        # フレームごとの局面ラベルを生成する処理は削除
        # print("  ステップ3/5: 局面ラベル生成中...") # 削除
        # frame_labels = self.interpolate_phase_labels(phase_changes, total_frames, fps) # 削除
        
        # トラッキングデータをDataFrameに変換（エラー対策強化）
        print("  ステップ3/4: DataFrame作成中...") # ステップ数を変更
        features_df = self.safe_create_dataframe_from_tracking_data(normalized_tracking_data, video_name)
        
        if features_df.empty:
            print("❌ DataFrame作成に失敗しました")
            return pd.DataFrame()
        
        print(f"DataFrame作成成功: {len(features_df)}行, {len(features_df.columns)}列")
        
        # フレーム番号の整合性を確保
        if 'frame_number' in features_df.columns:
            # フレーム番号を連続番号に正規化
            features_df['original_frame_number'] = features_df['frame_number']
            features_df['frame_number'] = range(len(features_df))
        else: # もし frame_number がなければ作成
            features_df['frame_number'] = range(len(features_df))
            features_df['original_frame_number'] = range(len(features_df))


        # ラベルを追加する処理は削除
        # label_length = min(len(frame_labels), len(features_df)) # 削除
        # features_df = features_df.iloc[:label_length].copy() # 削除
        # features_df['label'] = frame_labels[:label_length] # 削除
        features_df['video_name'] = video_name
        
        # 補間フレームの情報を保持
        if 'interpolated' not in features_df.columns:
            features_df['interpolated'] = False
        
        # ラベル未設定フレームを除外する処理は削除
        # labeled_frames = features_df['label'] != -1 # 削除
        # features_df = features_df[labeled_frames].copy() # 削除
        
        # print(f"ラベル付きフレーム数: {len(features_df)}") # 削除
        
        if len(features_df) == 0: # ラベルではなくDF自体の行数で判定
            print("⚠️  処理可能なフレームが存在しません") # メッセージ変更
            return pd.DataFrame()
        
        # ラベル分布を表示する処理は削除
        interpolated_count = features_df['interpolated'].sum() if 'interpolated' in features_df.columns else 0
        if interpolated_count > 0:
            print(f"補間フレーム含有率: {interpolated_count}/{len(features_df)} ({interpolated_count/len(features_df)*100:.1f}%)")
        
        # 欠損値処理
        print("  ステップ4/4: 特徴量エンジニアリング中...") # ステップ数を変更
        print("    サブステップ4.1/4.4: 欠損値処理...") # ステップ数を変更
        features_df = self.handle_missing_values(features_df)
        print("    サブステップ4.1/4.4: 欠損値処理完了") # ステップ数を変更
        
        # コート座標特徴量を作成
        print("    サブステップ4.2/4.4: コート特徴量作成...") # ステップ数を変更
        features_df = self.create_court_features(features_df, court_coords)
        print("    サブステップ4.2/4.4: コート特徴量作成完了") # ステップ数を変更
        
        # 時系列特徴量を作成（補間データを考慮)
        print("    サブステップ4.3/4.4: 時系列特徴量作成...") # ステップ数を変更
        features_df = self.create_temporal_features(features_df)
        print("    サブステップ4.3/4.4: 時系列特徴量作成完了") # ステップ数を変更
        
        # コンテキスト特徴量を作成
        print("    サブステップ4.4/4.4: コンテキスト特徴量作成...") # ステップ数を変更
        features_df = self.create_contextual_features(features_df)
        print("    サブステップ4.4/4.4: コンテキスト特徴量作成完了") # ステップ数を変更
        
        print(f"  ステップ4/4: 特徴量エンジニアリング完了") # ステップ数を変更
        print(f"最終特徴量数: {len(features_df.columns)}")
        
        return features_df
    
    # 特徴量作成メソッド群
    def create_temporal_features(self, features_df: pd.DataFrame, window_sizes: List[int] = [3, 5, 10, 15]) -> pd.DataFrame:
        """時系列特徴量を作成（補間データを考慮）"""
        temporal_df = features_df.copy()
        
        # 数値列のみを対象
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        
        # フレーム番号とラベルは除外 (ラベルは推論時ないので削除)
        target_columns = [col for col in numeric_columns if col not in ['frame_number', 'original_frame_number']] # 'label' を削除
        
        print(f"時系列特徴量作成対象: {len(target_columns)}特徴量")
        
        # 補間フレームの処理
        is_interpolated = features_df.get('interpolated', pd.Series([False] * len(features_df)))
        
        for window in tqdm(window_sizes, desc="時系列特徴量(ウィンドウ別)", leave=False):
            print(f"  ウィンドウサイズ {window} の特徴量作成中...")
            
            for col in tqdm(target_columns, desc=f"特徴量 (win={window})", leave=False):
                base_values = features_df[col]
                
                # 補間データを考慮した重み付き移動平均
                if is_interpolated.any():
                    # 補間フレームに低い重みを与える
                    weights = (~is_interpolated).astype(float) * 0.8 + is_interpolated.astype(float) * 0.3
                    
                    # 重み付き移動平均を計算
                    weighted_values = base_values * weights
                    temporal_df[f'{col}_ma_{window}'] = weighted_values.rolling(
                        window=window, center=True, min_periods=1
                    ).sum() / weights.rolling(window=window, center=True, min_periods=1).sum()
                else:
                    # 通常の移動平均
                    temporal_df[f'{col}_ma_{window}'] = base_values.rolling(
                        window=window, center=True, min_periods=1
                    ).mean()
                
                # 移動標準偏差（補間フレームの影響を軽減）
                temporal_df[f'{col}_std_{window}'] = base_values.rolling(
                    window=window, center=True, min_periods=1
                ).std().fillna(0)
                
                # 移動最大値・最小値
                temporal_df[f'{col}_max_{window}'] = base_values.rolling(
                    window=window, center=True, min_periods=1
                ).max()
                
                temporal_df[f'{col}_min_{window}'] = base_values.rolling(
                    window=window, center=True, min_periods=1
                ).min()
                
                # 小さなウィンドウでのみ作成する特徴量
                if window <= 5:
                    # 1次差分（変化率）- 補間フレームを考慮
                    diff = base_values.diff().fillna(0)
                    
                    # 補間フレーム間の差分は重みを下げる
                    if is_interpolated.any():
                        diff_weights = (~is_interpolated).astype(float) * 1.0 + is_interpolated.astype(float) * 0.5
                        diff = diff * diff_weights
                    
                    temporal_df[f'{col}_diff'] = diff
                    temporal_df[f'{col}_diff_abs'] = np.abs(diff)
                    
                    # 2次差分（加速度）
                    temporal_df[f'{col}_diff2'] = temporal_df[f'{col}_diff'].diff().fillna(0)
                    temporal_df[f'{col}_diff2_abs'] = np.abs(temporal_df[f'{col}_diff2'])
                
                # 中サイズウィンドウでのみ作成
                if window == 5:
                    # トレンド（線形回帰の傾き近似）
                    temporal_df[f'{col}_trend_{window}'] = base_values.rolling(
                        window=window, center=True, min_periods=1
                    ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True)
                    
                    # 変動係数（CV: Coefficient of Variation）
                    mean_vals = temporal_df[f'{col}_ma_{window}']
                    std_vals = temporal_df[f'{col}_std_{window}']
                    temporal_df[f'{col}_cv_{window}'] = np.where(
                        mean_vals != 0, std_vals / np.abs(mean_vals), 0
                    )
        
        # データ品質フラグの追加
        temporal_df['data_quality'] = (~is_interpolated).astype(float)
        temporal_df['interpolation_ratio'] = is_interpolated.rolling(
            window=10, center=True, min_periods=1
        ).mean()
        
        print(f"時系列特徴量作成完了: {len(temporal_df.columns) - len(features_df.columns)}特徴量追加")
        return temporal_df
    
    def create_contextual_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """コンテキスト特徴量を作成（時系列対応強化・フィールド存在チェック付き）"""
        context_df = features_df.copy()
        
        print("コンテキスト特徴量を作成中...")
        
        # 利用可能なフィールドをチェック
        available_fields = set(context_df.columns)
        
        # ボール関連の複合特徴量（存在するフィールドのみ使用）
        ball_activity_components = []
        
        if 'ball_detected' in available_fields:
            ball_activity_components.append(context_df['ball_detected'])
        
        if 'ball_movement_score' in available_fields:
            ball_activity_components.append(context_df['ball_movement_score'])
        elif 'ball_speed' in available_fields:
            # ball_movement_scoreがない場合はball_speedで代用
            ball_activity_components.append(context_df['ball_speed'] / 100.0)  # 正規化
        
        if 'ball_tracking_confidence' in available_fields:
            ball_activity_components.append(context_df['ball_tracking_confidence'])
        
        # ボール活動度を計算（利用可能なコンポーネントから）
        if ball_activity_components:
            if len(ball_activity_components) == 1:
                context_df['ball_activity'] = ball_activity_components[0]
            else:
                # 複数コンポーネントの積
                ball_activity = ball_activity_components[0]
                for component in ball_activity_components[1:]:
                    ball_activity = ball_activity * component
                context_df['ball_activity'] = ball_activity
        else:
            context_df['ball_activity'] = 0
        
        # プレイヤー関連の複合特徴量
        player_interaction_components = []
        
        if 'player_front_count' in available_fields:
            player_interaction_components.append(context_df['player_front_count'])
        
        if 'player_back_count' in available_fields:
            player_interaction_components.append(context_df['player_back_count'])
        
        if player_interaction_components:
            context_df['players_interaction'] = sum(player_interaction_components) / len(player_interaction_components)
        else:
            context_df['players_interaction'] = 0
        
        # プレイヤー信頼度平均
        confidence_components = []
        if 'player_front_confidence' in available_fields:
            confidence_components.append(context_df['player_front_confidence'].fillna(0))
        if 'player_back_confidence' in available_fields:
            confidence_components.append(context_df['player_back_confidence'].fillna(0))
        
        if confidence_components:
            context_df['players_confidence_avg'] = sum(confidence_components) / len(confidence_components)
        else:
            context_df['players_confidence_avg'] = 0
        
        # ボールとプレイヤーの位置関係（フィールド存在チェック付き）
        ball_pos_available = all(col in available_fields for col in ['ball_x', 'ball_y'])
        front_pos_available = all(col in available_fields for col in ['player_front_x', 'player_front_y'])
        back_pos_available = all(col in available_fields for col in ['player_back_x', 'player_back_y'])
        
        if ball_pos_available and front_pos_available:
            # ボール-手前プレイヤー距離
            context_df['ball_to_front_distance'] = np.sqrt(
                (context_df['ball_x'].fillna(0) - context_df['player_front_x'].fillna(0))**2 + 
                (context_df['ball_y'].fillna(0) - context_df['player_front_y'].fillna(0))**2
            )
            # NaNや無効値を大きな値で置換
            context_df['ball_to_front_distance'] = context_df['ball_to_front_distance'].fillna(1000)
        else:
            context_df['ball_to_front_distance'] = 1000
        
        if ball_pos_available and back_pos_available:
            # ボール-奥プレイヤー距離
            context_df['ball_to_back_distance'] = np.sqrt(
                (context_df['ball_x'].fillna(0) - context_df['player_back_x'].fillna(0))**2 + 
                (context_df['ball_y'].fillna(0) - context_df['player_back_y'].fillna(0))**2
            )
            context_df['ball_to_back_distance'] = context_df['ball_to_back_distance'].fillna(1000)
        else:
            context_df['ball_to_back_distance'] = 1000
        
        # どちらのプレイヤーにボールが近いか
        if 'ball_to_front_distance' in context_df.columns and 'ball_to_back_distance' in context_df.columns:
            context_df['ball_closer_to_front'] = (
                context_df['ball_to_front_distance'] < context_df['ball_to_back_distance']
            ).astype(int)
        else:
            context_df['ball_closer_to_front'] = 0
        
        # ボールの画面上の位置（正規化座標を使用）
        if all(col in available_fields for col in ['ball_x_normalized', 'ball_y_normalized']):
            context_df['ball_in_upper_half'] = (context_df['ball_x_normalized'].fillna(0.5) < 0.5).astype(int)
            context_df['ball_in_left_half'] = (context_df['ball_y_normalized'].fillna(0.5) < 0.5).astype(int)
            context_df['ball_in_center'] = (
                (context_df['ball_x_normalized'].fillna(0) > 0.3) & 
                (context_df['ball_x_normalized'].fillna(0) < 0.7) &
                (context_df['ball_y_normalized'].fillna(0) > 0.3) & 
                (context_df['ball_y_normalized'].fillna(0) < 0.7)
            ).astype(int)
        else:
            context_df['ball_in_upper_half'] = 0
            context_df['ball_in_left_half'] = 0
            context_df['ball_in_center'] = 0
        
        # トラッキング品質指標（存在するフィールドのみ使用）
        quality_components = []
        weights = []
        
        if 'ball_detected' in available_fields:
            quality_components.append(context_df['ball_detected'])
            weights.append(0.4)
        
        if 'ball_tracking_confidence' in available_fields:
            quality_components.append((context_df['ball_tracking_confidence'].fillna(0) > 0.5).astype(int))
            weights.append(0.3)
        
        if 'candidate_balls_count' in available_fields:
            quality_components.append((context_df['candidate_balls_count'].fillna(0) > 0).astype(int))
            weights.append(0.2)
        
        if 'disappeared_count' in available_fields:
            quality_components.append((context_df['disappeared_count'].fillna(1) == 0).astype(int))
            weights.append(0.1)
        
        if quality_components:
            # 重み付き平均でトラッキング品質を計算
            weighted_sum = sum(comp * weight for comp, weight in zip(quality_components, weights))
            total_weight = sum(weights)
            context_df['tracking_quality'] = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            context_df['tracking_quality'] = 0
        
        # 時系列対応のコンテキスト特徴量を追加
        print("  時系列コンテキスト特徴量を作成中...")
        
        # ボール検出の安定性（連続フレームでの検出率）
        if 'ball_detected' in available_fields:
            for window in [3, 5, 10]:
                context_df[f'ball_detection_stability_{window}'] = context_df['ball_detected'].rolling(
                    window=window, center=True, min_periods=1
                ).mean()
        
        # ボール位置の変動（安定性の指標）
        if all(col in available_fields for col in ['ball_x', 'ball_y']):
            # ボール位置の移動距離
            ball_x_diff = context_df['ball_x'].fillna(method='ffill').diff().fillna(0)
            ball_y_diff = context_df['ball_y'].fillna(method='ffill').diff().fillna(0)
            context_df['ball_movement_distance'] = np.sqrt(ball_x_diff**2 + ball_y_diff**2)
            
            # ボール移動の安定性
            for window in [3, 5]:
                context_df[f'ball_movement_stability_{window}'] = context_df['ball_movement_distance'].rolling(
                    window=window, center=True, min_periods=1
                ).std().fillna(0)
        else:
            context_df['ball_movement_distance'] = 0
            for window in [3, 5]:
                context_df[f'ball_movement_stability_{window}'] = 0
        
        # プレイヤー位置の変動
        for player in ['front', 'back']:
            x_col = f'player_{player}_x'
            y_col = f'player_{player}_y'
            
            if x_col in available_fields and y_col in available_fields:
                # プレイヤー移動距離
                x_diff = context_df[x_col].fillna(method='ffill').diff().fillna(0)
                y_diff = context_df[y_col].fillna(method='ffill').diff().fillna(0)
                context_df[f'player_{player}_movement_distance'] = np.sqrt(x_diff**2 + y_diff**2)
                
                # プレイヤー活動レベル（移動の激しさ）
                for window in [5, 10]:
                    context_df[f'player_{player}_activity_{window}'] = context_df[f'player_{player}_movement_distance'].rolling(
                        window=window, center=True, min_periods=1
                    ).mean()
            else:
                context_df[f'player_{player}_movement_distance'] = 0
                for window in [5, 10]:
                    context_df[f'player_{player}_activity_{window}'] = 0
        
        # 全体的な動きの激しさ（シーンの動的レベル）
        movement_cols = [col for col in context_df.columns if 'movement_distance' in col and col in context_df.columns]
        if movement_cols:
            context_df['scene_dynamics'] = context_df[movement_cols].mean(axis=1)
            
            # シーン動的レベルの時系列特徴量
            for window in [5, 10]:
                context_df[f'scene_dynamics_ma_{window}'] = context_df['scene_dynamics'].rolling(
                    window=window, center=True, min_periods=1
                ).mean()
                
                context_df[f'scene_dynamics_std_{window}'] = context_df['scene_dynamics'].rolling(
                    window=window, center=True, min_periods=1
                ).std().fillna(0)
        else:
            context_df['scene_dynamics'] = 0
            for window in [5, 10]:
                context_df[f'scene_dynamics_ma_{window}'] = 0
                context_df[f'scene_dynamics_std_{window}'] = 0
        
        # イベント検出（急激な変化の検出）
        if 'ball_movement_distance' in context_df.columns:
            # ボール移動の急激な変化（ヒット、バウンドなどのイベント）
            ball_movement_ma = context_df['ball_movement_distance'].rolling(
                window=5, center=True, min_periods=1
            ).mean()
            
            context_df['ball_movement_spike'] = (
                context_df['ball_movement_distance'] > ball_movement_ma * 2
            ).astype(int)
            
            # イベント頻度
            for window in [10, 20]:
                context_df[f'ball_events_frequency_{window}'] = context_df['ball_movement_spike'].rolling(
                    window=window, center=True, min_periods=1
                ).sum()
        else:
            context_df['ball_movement_spike'] = 0
            for window in [10, 20]:
                context_df[f'ball_events_frequency_{window}'] = 0
        
        print(f"  コンテキスト特徴量作成完了: {len(context_df.columns) - len(features_df.columns)}特徴量追加")
        return context_df
    
    def create_court_features(self, features_df: pd.DataFrame, court_coords: Dict) -> pd.DataFrame:
        """コート座標を使用した特徴量を作成"""
        if not court_coords:
            print("コート座標データがないため、コート特徴量はスキップします")
            return features_df
        
        court_df = features_df.copy()
        print("コート座標特徴量を作成中...")
        
        # コート座標から基本情報を計算
        court_info = self.calculate_court_geometry(court_coords)
        
        # ボール位置のコート座標系変換
        if all(col in court_df.columns for col in ['ball_x', 'ball_y']):
            ball_court_coords = self.transform_to_court_coordinates(
                court_df['ball_x'].values, 
                court_df['ball_y'].values, 
                court_coords
            )
            
            court_df['ball_court_x'] = ball_court_coords['x']
            court_df['ball_court_y'] = ball_court_coords['y']
            
            # コート上の位置に基づく特徴量
            court_df['ball_in_court'] = (
                (court_df['ball_court_x'] >= 0) & 
                (court_df['ball_court_x'] <= 1) &
                (court_df['ball_court_y'] >= 0) & 
                (court_df['ball_court_y'] <= 1)
            ).astype(int)
            
            # コート上の領域特徴量
            court_df['ball_in_front_court'] = (court_df['ball_court_y'] > 0.5).astype(int)
            court_df['ball_in_back_court'] = (court_df['ball_court_y'] <= 0.5).astype(int)
            court_df['ball_in_left_court'] = (court_df['ball_court_x'] <= 0.5).astype(int)
            court_df['ball_in_right_court'] = (court_df['ball_court_x'] > 0.5).astype(int)
            
            # ネットからの距離
            net_y = 0.5  # コート座標系ではネットはy=0.5
            court_df['ball_distance_to_net'] = np.abs(court_df['ball_court_y'] - net_y)
            
            # サイドラインからの距離
            court_df['ball_distance_to_left_line'] = court_df['ball_court_x']
            court_df['ball_distance_to_right_line'] = 1 - court_df['ball_court_x']
            court_df['ball_distance_to_sideline'] = np.minimum(
                court_df['ball_distance_to_left_line'],
                court_df['ball_distance_to_right_line']
            )
            
            # ベースラインからの距離
            court_df['ball_distance_to_front_baseline'] = 1 - court_df['ball_court_y']
            court_df['ball_distance_to_back_baseline'] = court_df['ball_court_y']
            court_df['ball_distance_to_baseline'] = np.minimum(
                court_df['ball_distance_to_front_baseline'],
                court_df['ball_distance_to_back_baseline']
            )
        
        # プレイヤー位置のコート座標系変換
        for player in ['front', 'back']:
            x_col = f'player_{player}_x'
            y_col = f'player_{player}_y'
            
            if x_col in court_df.columns and y_col in court_df.columns:
                player_court_coords = self.transform_to_court_coordinates(
                    court_df[x_col].values,
                    court_df[y_col].values,
                    court_coords
                )
                
                court_df[f'player_{player}_court_x'] = player_court_coords['x']
                court_df[f'player_{player}_court_y'] = player_court_coords['y']
                
                # プレイヤーのコート上位置特徴量
                court_df[f'player_{player}_in_court'] = (
                    (court_df[f'player_{player}_court_x'] >= 0) & 
                    (court_df[f'player_{player}_court_x'] <= 1) &

                    (court_df[f'player_{player}_court_y'] >= 0) & 
                    (court_df[f'player_{player}_court_y'] <= 1)
                ).astype(int)
                
                # ネットからの距離
                court_df[f'player_{player}_distance_to_net'] = np.abs(
                    court_df[f'player_{player}_court_y'] - net_y
                )
        
        # プレイヤー間の関係特徴量（コート座標系）
        if all(col in court_df.columns for col in ['player_front_court_x', 'player_front_court_y', 
                                                  'player_back_court_x', 'player_back_court_y']):
            # プレイヤー間距離（コート座標系）
            court_df['players_court_distance'] = np.sqrt(
                (court_df['player_front_court_x'] - court_df['player_back_court_x'])**2 +
                (court_df['player_front_court_y'] - court_df['player_back_court_y'])**2
            )
            
            # プレイヤーが正しい側にいるかの判定
            court_df['players_correct_sides'] = (
                court_df['player_front_court_y'] > court_df['player_back_court_y']
            ).astype(int)
        
        # コート座標系でのボール-プレイヤー関係
        if all(col in court_df.columns for col in ['ball_court_x', 'ball_court_y']):
            for player in ['front', 'back']:
                if all(col in court_df.columns for col in [f'player_{player}_court_x', f'player_{player}_court_y']):
                    # コート座標系でのボール-プレイヤー距離
                    court_df[f'ball_to_{player}_court_distance'] = np.sqrt(
                        (court_df['ball_court_x'] - court_df[f'player_{player}_court_x'])**2 +
                        (court_df['ball_court_y'] - court_df[f'player_{player}_court_y'])**2
                    )
        
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
            'avg_height': (court_height_left + court_height_right) /  2
        }
        
        return court_info
    
    def transform_to_court_coordinates(self, x_coords: np.ndarray, y_coords: np.ndarray, 
                                     court_coords: Dict) -> Dict[str, np.ndarray]:
        """画像座標をコート座標系（0-1の正規化座標）に変換"""
        # コート四隅の座標
        top_left = np.array(court_coords['top_left_corner'])
        top_right = np.array(court_coords['top_right_corner'])
        bottom_left = np.array(court_coords['bottom_left_corner'])
        bottom_right = np.array(court_coords['bottom_right_corner'])
        
        # バイリニア補間による座標変換
        court_x = np.zeros_like(x_coords, dtype=float)
        court_y = np.zeros_like(y_coords, dtype=float)
        
        for i in range(len(x_coords)):
            x, y = x_coords[i], y_coords[i]
            
            # None値の処理
            if pd.isna(x) or pd.isna(y) or x == 0 or y == 0:
                court_x[i] = -1  # コート外を示す値
                court_y[i] = -1
                continue
            
            # コート四隅を基準にした正規化座標を計算
            # 簡易的な透視変換近似
            try:
                # 上辺と下辺での位置を計算
                if top_right[0] != top_left[0]:
                    top_ratio = (x - top_left[0]) / (top_right[0] - top_left[0])
                else:
                    top_ratio = 0.5
                
                if bottom_right[0] != bottom_left[0]:
                    bottom_ratio = (x - bottom_left[0]) / (bottom_right[0] - bottom_left[0])
                else:
                    bottom_ratio = 0.5
                
                # y方向の比率を計算
                total_height = max(bottom_left[1] - top_left[1], bottom_right[1] - top_right[1], 1)
                y_ratio = (y - top_left[1]) / total_height if total_height > 0 else 0
                
                # バイリニア補間でx座標を計算
                court_x[i] = np.clip(top_ratio * (1 - y_ratio) + bottom_ratio * y_ratio, -0.5, 1.5)
                court_y[i] = np.clip(y_ratio, -0.5, 1.5)
                
            except (ZeroDivisionError, ValueError):
                court_x[i] = -1
                court_y[i] = -1
        
        return {'x': court_x, 'y': court_y}
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値を処理"""
        df_cleaned = df.copy()
        
        # None値を適切な値に置換
        none_columns = ['ball_x', 'ball_y', 'player_front_x', 'player_front_y', 
                       'player_back_x', 'player_back_y']
        
        for col in none_columns:
            if col in df_cleaned.columns:
                # None/NaNを前の値で埋める（前向き補間）
                df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
                # それでも残るNaNを0で埋める
                df_cleaned[col] = df_cleaned[col].fillna(0)
        
        # 数値列の残りのNaNを0で埋める
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(0)
        
        return df_cleaned
    
    # メイン処理・統合メソッド群
    def extract_all_features(self, video_name: str = None) -> pd.DataFrame:
        """すべての動画から特徴量を抽出してDataFrameを統合"""
        print("=== 推論用特徴量抽出開始 ===") # print文を変更
        
        # データを読み込み
        print("1. データファイル読み込み中...")
        # phase_annotations = self.load_phase_annotations(video_name) # 削除
        tracking_features = self.load_tracking_features(video_name)
        court_coordinates = self.load_court_coordinates(video_name)
        
        # if not phase_annotations: # 削除
        #     print("❌ 局面アノテーションデータが見つかりません") # 削除
        #     return pd.DataFrame() # 削除
        
        if not tracking_features:
            print("❌ トラッキング特徴量データが見つかりません")
            return pd.DataFrame()
        
        # データをマッチング
        print("\n2. データマッチング中...")
        matched_data = self.match_video_data(tracking_features, court_coordinates) # phase_annotations を削除
        
        if not matched_data:
            print("❌ マッチするデータが見つかりません")
            return pd.DataFrame()
        
        # 各動画から特徴量を抽出
        print("\n3. 特徴量抽出中...")
        all_features_list = [] # 変数名を変更 (all_features -> all_features_list)
        total_videos = len(matched_data)
        print(f"処理対象動画数: {total_videos}")
        
        for i, (vid_name, track_data, court_data) in enumerate(tqdm(matched_data, desc="動画別特徴量抽出"), 1): # 変数名を変更
            print(f"\n--- 動画処理進捗: [{i}/{total_videos}] - {vid_name} ---") # 変数名を変更
            
            try:
                features_df = self.extract_features_from_video(
                    vid_name, track_data, court_data # phase_data を削除、変数名を変更
                )
                
                if not features_df.empty:
                    all_features_list.append(features_df) # 変数名を変更
                    print(f"✅ 特徴量抽出完了: {len(features_df)}行")
                else:
                    print(f"⚠️  特徴量が空です: {vid_name}") # メッセージに動画名追加
                    
            except Exception as e:
                print(f"❌ 特徴量抽出エラー ({vid_name}): {e}") # メッセージに動画名追加
                import traceback # traceback をインポート
                traceback.print_exc() # スタックトレースを出力
                continue
        
        # 特徴量を統合
        if all_features_list: # 変数名を変更
            print(f"\n4. 特徴量統合中... ({len(all_features_list)}動画)") # 変数名を変更
            combined_features = pd.concat(all_features_list, ignore_index=True) # 変数名を変更
            
            print(f"✅ 統合完了:")
            print(f"   総行数: {len(combined_features)}")
            print(f"   総特徴量数: {len(combined_features.columns)}")
            if 'video_name' in combined_features.columns: # video_name が存在するか確認
                print(f"   動画数: {combined_features['video_name'].nunique()}")
            
            # ラベル分布を表示する処理は削除
            
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
            filename = f"tennis_inference_features_{timestamp}.csv" # ファイル名を変更
        
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
            filename = f"tennis_inference_features_info_{timestamp}.json" # ファイル名を変更
        
        output_path = self.features_dir / filename
        
        try:
            # 特徴量情報を作成
            feature_info = {
                'creation_time': datetime.now().isoformat(),
                'total_frames': len(features_df),
                'total_features': len(features_df.columns),
                'videos': features_df['video_name'].nunique() if 'video_name' in features_df.columns else 0, # video_name が存在するか確認
                'video_list': features_df['video_name'].unique().tolist() if 'video_name' in features_df.columns else [], # video_name が存在するか確認
                # 'phase_labels': self.phase_labels, # 削除
                # 'label_distribution': features_df['label'].value_counts().sort_index().to_dict(), # 削除
                'feature_columns': features_df.columns.tolist(),
                'feature_types': { # 推論時には簡略化も検討可能だが、一旦維持
                    'temporal': [col for col in features_df.columns if any(suffix in col for suffix in ['_ma_', '_std_', '_max_', '_min_', '_diff', '_trend_', '_cv_'])],
                    'contextual': [col for col in features_df.columns if any(keyword in col for keyword in ['activity', 'interaction', 'confidence', 'distance', 'movement', 'stability', 'dynamics', 'quality'])],
                    'court': [col for col in features_df.columns if 'court' in col],
                    'basic': [col for col in features_df.columns if col.startswith(('ball_', 'player_')) and not any(suffix in col for suffix in ['_ma_', '_std_', '_max_', '_min_', '_diff', '_trend_', '_cv_', '_court'])]
                },
                'data_quality': {
                    'interpolated_frames': int(features_df.get('interpolated', pd.Series([False] * len(features_df))).sum()), # デフォルト値を修正
                    'interpolation_rate': float(features_df.get('interpolated', pd.Series([False] * len(features_df))).mean()), # デフォルト値を修正
                    'missing_values_summary': features_df.isnull().sum()[features_df.isnull().sum() > 0].to_dict() # missing_values を missing_values_summary に変更し、内容を調整
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(feature_info, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 推論用特徴量情報保存完了: {output_path}") # print文を変更
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 情報保存エラー: {e}")
            return ""
    
    def run_feature_extraction(self, video_name: str = None, save_results: bool = True) -> Tuple[pd.DataFrame, str, str]:
        """特徴量抽出の全プロセスを実行"""
        print("🎾 テニス動画 推論用特徴量抽出プロセス開始") # print文を変更
        print("=" * 50)
        
        # 特徴量抽出
        features_df = self.extract_all_features(video_name)
        
        if features_df.empty:
            print("❌ 推論用特徴量抽出に失敗しました") # print文を変更
            return pd.DataFrame(), "", ""
        
        feature_file = ""
        info_file = ""
        
        if save_results:
            print("\n5. ファイル保存中...")
            feature_file = self.save_features(features_df)
            info_file = self.save_feature_info(features_df)
        
        print("\n🎉 推論用特徴量抽出プロセス完了!") # print文を変更
        print("=" * 50)
        
        return features_df, feature_file, info_file
    
    def analyze_features(self, features_df: pd.DataFrame):
        """特徴量の統計分析を表示"""
        if features_df.empty:
            print("⚠️  分析する特徴量が空です")
            return
        
        print("\n📈 推論用特徴量分析結果") # print文を変更
        print("=" * 30)
        
        # 基本統計
        print(f"📊 基本統計:")
        print(f"   総フレーム数: {len(features_df):,}")
        print(f"   特徴量数: {len(features_df.columns)}")
        if 'video_name' in features_df.columns: # video_name が存在するか確認
            print(f"   動画数: {features_df['video_name'].nunique()}")
        
        # 動画別統計
        if 'video_name' in features_df.columns: # video_name が存在するか確認
            print(f"\n🎬 動画別フレーム数:")
            video_counts = features_df['video_name'].value_counts()
            for video, count in video_counts.items():
                print(f"   {video}: {count:,}フレーム")
        
        # ラベル分布の表示は削除
        
        # データ品質
        if 'interpolated' in features_df.columns:
            interpolated_count = features_df['interpolated'].sum()
            interpolation_rate = interpolated_count / len(features_df) * 100 if len(features_df) > 0 else 0 # ゼロ除算を回避
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
    # inference_data_dir は、推論対象のトラッキングデータとコート座標データが含まれるディレクトリを指定します。
    # 例: extractor = TennisInferenceFeatureExtractor(inference_data_dir="path/to/your/inference_data")
    # ここでは、スクリプトと同じ階層にある "inference_data_example" をデフォルトとしています。
    # このディレクトリが存在しない場合は、適宜作成またはパスを変更してください。
    
    # 例: カレントディレクトリに 'my_inference_input' という名前のフォルダがある場合
    # extractor = TennisInferenceFeatureExtractor(inference_data_dir="my_inference_input")
    
    # デフォルトの入力データディレクトリ (スクリプトと同じ階層を想定)
    default_input_dir = "./training_data" 
    
    # もしデフォルトのディレクトリがなければ、ユーザーにパス入力を促すことも検討できます
    if not Path(default_input_dir).exists():
        print(f"⚠️  デフォルトの入力ディレクトリ '{default_input_dir}' が見つかりません。")
        print(f"    サンプルデータ用に '{default_input_dir}' を作成するか、")
        print(f"    適切な inference_data_dir を指定して TennisInferenceFeatureExtractor を初期化してください。")
        # ここで処理を終了するか、ユーザーに入力を求めるなどの処理を追加できます。
        # 例: exit() または input_dir = input("推論データディレクトリのパスを入力してください: ")
        
    extractor = TennisInferenceFeatureExtractor(inference_data_dir=default_input_dir)

    # 特定の動画のみを処理する場合 (例: "your_video_name_without_extension")
    # video_to_process = "your_inference_video_name_here" 
    # video_to_process に動画名を指定すると、その動画のデータのみが処理されます。
    # None の場合、inference_data_dir 内のすべての動画データが処理対象となります。
    video_to_process = None 

    # 特徴量抽出を実行
    # save_results=True にすると、抽出された特徴量がCSVファイルおよび情報JSONファイルとして保存されます。
    # False にすると保存されません。
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

        # 抽出された特徴量の分析 (オプション)
        # 大量のデータの場合、分析に時間がかかることがあります。
        print("\n--- 特徴量分析開始 (オプション) ---")
        extractor.analyze_features(features_dataframe)
        print("--- 特徴量分析完了 ---")
    else:
        print("\n--- 推論用特徴量抽出失敗 ---")
        print("処理できるデータがなかったか、エラーが発生しました。ログを確認してください。")
        print(f"入力データディレクトリ: {extractor.inference_data_dir}")
        print(f"特徴量保存先ディレクトリ: {extractor.features_dir}")


    print("\nメイン処理完了")
