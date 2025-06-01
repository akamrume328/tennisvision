import json
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class TennisFeatureExtractor:
    """
    テニス動画の局面アノテーションデータとトラッキング特徴量を統合し、
    機械学習用の特徴量を作成するクラス
    """
    
    def __init__(self, training_data_dir: str = "training_data"):
        self.training_data_dir = Path(training_data_dir)
        self.phase_labels = [
            "point_interval",           # 0: ポイント間
            "rally",                   # 1: ラリー中
            "serve_preparation",       # 2: サーブ準備
            "serve_front_deuce",      # 3: 手前デュースサイドからのサーブ
            "serve_front_ad",         # 4: 手前アドサイドからのサーブ
            "serve_back_deuce",       # 5: 奥デュースサイドからのサーブ
            "serve_back_ad",          # 6: 奥アドサイドからのサーブ
            "changeover"              # 7: チェンジコート間
        ]
        self.label_to_id = {label: idx for idx, label in enumerate(self.phase_labels)}
        
        print(f"特徴量抽出器を初期化しました")
        print(f"データディレクトリ: {self.training_data_dir}")
        print(f"対象局面数: {len(self.phase_labels)}")
    
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
    
    def load_tracking_features(self, video_name: str = None) -> Dict[str, List[Dict]]:
        """トラッキング特徴量ファイルを読み込み"""
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
                    video_key = '_'.join(filename_parts[2:-2])  # tracking_features_を除く、日時を除く
                else:
                    video_key = file_path.stem
                
                all_tracking[video_key] = data
                print(f"✅ トラッキング特徴量読み込み: {file_path.name}")
                print(f"   動画: {video_key}, フレーム数: {len(data)}")
                
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
    
    def match_video_data(self, phase_annotations: Dict, tracking_features: Dict, court_coordinates: Dict = None) -> List[Tuple[str, Dict, List[Dict], Dict]]:
        """動画名でアノテーションデータ、トラッキングデータ、コート座標をマッチング"""
        matched_data = []
        
        print("\n=== データマッチング ===")
        print(f"局面アノテーション動画数: {len(phase_annotations)}")
        print(f"トラッキング特徴量動画数: {len(tracking_features)}")
        print(f"コート座標動画数: {len(court_coordinates) if court_coordinates else 0}")
        
        for video_name in phase_annotations.keys():
            tracking_data = None
            court_data = None
            
            # トラッキングデータのマッチング
            if video_name in tracking_features:
                tracking_data = tracking_features[video_name]
                print(f"✅ トラッキングマッチング: {video_name}")
            else:
                # 部分マッチを試行
                for tracking_key in tracking_features.keys():
                    if video_name in tracking_key or tracking_key in video_name:
                        tracking_data = tracking_features[tracking_key]
                        print(f"✅ トラッキング部分マッチング: {video_name} <-> {tracking_key}")
                        break
            
            # コート座標のマッチング
            if court_coordinates:
                if video_name in court_coordinates:
                    court_data = court_coordinates[video_name]
                    print(f"✅ コート座標マッチング: {video_name}")
                else:
                    # 部分マッチを試行
                    for court_key in court_coordinates.keys():
                        if video_name in court_key or court_key in video_name:
                            court_data = court_coordinates[court_key]
                            print(f"✅ コート座標部分マッチング: {video_name} <-> {court_key}")
                            break
                    
                    if not court_data:
                        print(f"⚠️  コート座標なし: {video_name}")
            
            # トラッキングデータがある場合のみ追加
            if tracking_data:
                matched_data.append((
                    video_name,
                    phase_annotations[video_name],
                    tracking_data,
                    court_data
                ))
            else:
                print(f"⚠️  トラッキングデータなし: {video_name}")
        
        print(f"最終マッチング数: {len(matched_data)}")
        return matched_data
    
    def interpolate_phase_labels(self, phase_changes: List[Dict], total_frames: int, fps: float) -> np.ndarray:
        """局面変更データからフレームごとの局面ラベルを生成"""
        frame_labels = np.full(total_frames, -1, dtype=int)  # -1: ラベル未設定
        
        if not phase_changes:
            return frame_labels
        
        # 局面変更を時間順にソート
        sorted_changes = sorted(phase_changes, key=lambda x: x['frame_number'])
        
        for i, change in enumerate(sorted_changes):
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
    
    def create_temporal_features(self, features_df: pd.DataFrame, window_sizes: List[int] = [3, 5, 10, 15]) -> pd.DataFrame:
        """時系列特徴量を作成（移動平均、移動標準偏差、加速度など）"""
        temporal_df = features_df.copy()
        
        # 数値列のみを対象
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        
        # フレーム番号とラベルは除外
        target_columns = [col for col in numeric_columns if col not in ['frame_number', 'label']]
        
        print(f"時系列特徴量作成対象: {len(target_columns)}特徴量")
        
        for window in window_sizes:
            print(f"  ウィンドウサイズ {window} の特徴量作成中...")
            
            for col in target_columns:
                base_values = features_df[col]
                
                # 移動平均
                temporal_df[f'{col}_ma_{window}'] = base_values.rolling(
                    window=window, center=True, min_periods=1
                ).mean()
                
                # 移動標準偏差
                temporal_df[f'{col}_std_{window}'] = base_values.rolling(
                    window=window, center=True, min_periods=1
                ).std().fillna(0)
                
                # 移動最大値
                temporal_df[f'{col}_max_{window}'] = base_values.rolling(
                    window=window, center=True, min_periods=1
                ).max()
                
                # 移動最小値
                temporal_df[f'{col}_min_{window}'] = base_values.rolling(
                    window=window, center=True, min_periods=1
                ).min()
                
                # 小さなウィンドウでのみ作成する特徴量
                if window <= 5:
                    # 1次差分（変化率）
                    temporal_df[f'{col}_diff'] = base_values.diff().fillna(0)
                    temporal_df[f'{col}_diff_abs'] = np.abs(temporal_df[f'{col}_diff'])
                    
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
        
        print(f"時系列特徴量作成完了: {len(temporal_df.columns) - len(features_df.columns)}特徴量追加")
        return temporal_df
    
    def create_contextual_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """コンテキスト特徴量を作成（時系列対応強化）"""
        context_df = features_df.copy()
        
        # ボール関連の複合特徴量
        context_df['ball_activity'] = (
            context_df['ball_detected'] * context_df['ball_movement_score'] * 
            context_df['ball_tracking_confidence']
        )
        
        # プレイヤー関連の複合特徴量
        context_df['players_interaction'] = (
            context_df['player_front_count'] + context_df['player_back_count']
        ) / 2.0
        
        context_df['players_confidence_avg'] = (
            context_df['player_front_confidence'] + context_df['player_back_confidence']
        ) / 2.0
        
        # ボールとプレイヤーの位置関係
        if all(col in context_df.columns for col in ['ball_x', 'player_front_x', 'player_back_x']):
            # ボール-手前プレイヤー距離
            context_df['ball_to_front_distance'] = np.sqrt(
                (context_df['ball_x'] - context_df['player_front_x']).fillna(0)**2 + 
                (context_df['ball_y'] - context_df['player_front_y']).fillna(0)**2
            ).fillna(1000)  # ボールまたはプレイヤーが検出されない場合は大きな値
            
            # ボール-奥プレイヤー距離
            context_df['ball_to_back_distance'] = np.sqrt(
                (context_df['ball_x'] - context_df['player_back_x']).fillna(0)**2 + 
                (context_df['ball_y'] - context_df['player_back_y']).fillna(0)**2
            ).fillna(1000)
            
            # どちらのプレイヤーにボールが近いか
            context_df['ball_closer_to_front'] = (
                context_df['ball_to_front_distance'] < context_df['ball_to_back_distance']
            ).astype(int)
        
        # ボールの画面上の位置（正規化座標を使用）
        if all(col in context_df.columns for col in ['ball_x_normalized', 'ball_y_normalized']):
            context_df['ball_in_upper_half'] = (context_df['ball_y_normalized'] < 0.5).astype(int)
            context_df['ball_in_left_half'] = (context_df['ball_x_normalized'] < 0.5).astype(int)
            context_df['ball_in_center'] = (
                (context_df['ball_x_normalized'] > 0.3) & 
                (context_df['ball_x_normalized'] < 0.7) &
                (context_df['ball_y_normalized'] > 0.3) & 
                (context_df['ball_y_normalized'] < 0.7)
            ).astype(int)
        
        # トラッキング品質指標
        context_df['tracking_quality'] = (
            context_df['ball_detected'] * 0.4 +
            (context_df['ball_tracking_confidence'] > 0.5).astype(int) * 0.3 +
            (context_df['candidate_balls_count'] > 0).astype(int) * 0.2 +
            (context_df['disappeared_count'] == 0).astype(int) * 0.1
        )
        
        # 時系列対応のコンテキスト特徴量を追加
        print("時系列コンテキスト特徴量を作成中...")
        
        # ボール検出の安定性（連続フレームでの検出率）
        for window in [3, 5, 10]:
            context_df[f'ball_detection_stability_{window}'] = context_df['ball_detected'].rolling(
                window=window, center=True, min_periods=1
            ).mean()
        
        # ボール位置の変動（安定性の指標）
        if 'ball_x' in context_df.columns and 'ball_y' in context_df.columns:
            # ボール位置の移動距離
            ball_x_diff = context_df['ball_x'].diff().fillna(0)
            ball_y_diff = context_df['ball_y'].diff().fillna(0)
            context_df['ball_movement_distance'] = np.sqrt(ball_x_diff**2 + ball_y_diff**2)
            
            # ボール移動の安定性
            for window in [3, 5]:
                context_df[f'ball_movement_stability_{window}'] = context_df['ball_movement_distance'].rolling(
                    window=window, center=True, min_periods=1
                ).std().fillna(0)
        
        # プレイヤー位置の変動
        for player in ['front', 'back']:
            x_col = f'player_{player}_x'
            y_col = f'player_{player}_y'
            
            if x_col in context_df.columns and y_col in context_df.columns:
                # プレイヤー移動距離
                x_diff = context_df[x_col].diff().fillna(0)
                y_diff = context_df[y_col].diff().fillna(0)
                context_df[f'player_{player}_movement_distance'] = np.sqrt(x_diff**2 + y_diff**2)
                
                # プレイヤー活動レベル（移動の激しさ）
                for window in [5, 10]:
                    context_df[f'player_{player}_activity_{window}'] = context_df[f'player_{player}_movement_distance'].rolling(
                        window=window, center=True, min_periods=1
                    ).mean()
        
        # 全体的な動きの激しさ（シーンの動的レベル）
        movement_cols = [col for col in context_df.columns if 'movement_distance' in col]
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
        
        print(f"コンテキスト特徴量作成完了: {len(context_df.columns) - len(features_df.columns)}特徴量追加")
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
    
    def extract_features_from_video(self, video_name: str, phase_data: Dict, tracking_data: List[Dict], court_coords: Dict = None) -> pd.DataFrame:
        """単一動画から特徴量を抽出（コート座標対応）"""
        print(f"\n--- 特徴量抽出: {video_name} ---")
        
        # 基本情報
        total_frames = len(tracking_data)
        fps = phase_data.get('fps', 30.0)
        phase_changes = phase_data.get('phase_changes', [])
        
        print(f"フレーム数: {total_frames}, FPS: {fps}")
        print(f"局面変更数: {len(phase_changes)}")
        print(f"コート座標: {'あり' if court_coords else 'なし'}")
        
        # フレームごとの局面ラベルを生成
        frame_labels = self.interpolate_phase_labels(phase_changes, total_frames, fps)
        
        # トラッキングデータをDataFrameに変換
        features_df = pd.DataFrame(tracking_data)
        
        # ラベルを追加
        features_df['label'] = frame_labels[:len(features_df)]
        features_df['video_name'] = video_name
        
        # ラベル未設定フレームを除外
        labeled_frames = features_df['label'] != -1
        features_df = features_df[labeled_frames].copy()
        
        print(f"ラベル付きフレーム数: {len(features_df)}")
        
        if len(features_df) == 0:
            print("⚠️  ラベル付きフレームが存在しません")
            return pd.DataFrame()
        
        # ラベル分布を表示
        label_counts = features_df['label'].value_counts().sort_index()
        print("ラベル分布:")
        for label_id, count in label_counts.items():
            phase_name = self.phase_labels[label_id] if label_id < len(self.phase_labels) else f"Unknown_{label_id}"
            print(f"  {label_id} ({phase_name}): {count}フレーム")
        
        # 欠損値処理
        features_df = self.handle_missing_values(features_df)
        
        # コート座標特徴量を作成
        features_df = self.create_court_features(features_df, court_coords)
        
        # 時系列特徴量を作成
        features_df = self.create_temporal_features(features_df)
        
        # コンテキスト特徴量を作成
        features_df = self.create_contextual_features(features_df)
        
        print(f"最終特徴量数: {len(features_df.columns)}")
        
        return features_df
    
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
    
    def process_all_videos(self) -> pd.DataFrame:
        """全ての動画データを処理して統合特徴量データセットを作成"""
        print("=== 全動画データ処理開始 ===")
        
        # データ読み込み
        phase_annotations = self.load_phase_annotations()
        tracking_features = self.load_tracking_features()
        court_coordinates = self.load_court_coordinates()
        
        if not phase_annotations or not tracking_features:
            print("❌ 必要なデータファイルが不足しています")
            return pd.DataFrame()
        
        # データマッチング
        matched_data = self.match_video_data(phase_annotations, tracking_features, court_coordinates)
        
        if not matched_data:
            print("❌ マッチングするデータがありません")
            return pd.DataFrame()
        
        # 各動画から特徴量を抽出
        all_features = []
        
        for video_name, phase_data, tracking_data, court_coords in matched_data:
            try:
                video_features = self.extract_features_from_video(video_name, phase_data, tracking_data, court_coords)
                if not video_features.empty:
                    all_features.append(video_features)
                    print(f"✅ {video_name}: {len(video_features)}フレーム処理完了")
                else:
                    print(f"⚠️  {video_name}: 処理可能なフレームなし")
            except Exception as e:
                print(f"❌ {video_name} 処理エラー: {e}")
        
        if not all_features:
            print("❌ 処理可能な動画データがありません")
            return pd.DataFrame()
        
        # 全データを統合
        combined_df = pd.concat(all_features, ignore_index=True)
        
        print(f"\n=== 統合結果 ===")
        print(f"総フレーム数: {len(combined_df)}")
        print(f"動画数: {combined_df['video_name'].nunique()}")
        print(f"特徴量数: {len(combined_df.columns)}")
        
        # コート特徴量の統計
        court_features = [col for col in combined_df.columns if 'court' in col or 'distance_to' in col]
        if court_features:
            print(f"コート関連特徴量数: {len(court_features)}")
        
        # 最終的なラベル分布
        print("\n全体のラベル分布:")
        total_label_counts = combined_df['label'].value_counts().sort_index()
        for label_id, count in total_label_counts.items():
            phase_name = self.phase_labels[label_id] if label_id < len(self.phase_labels) else f"Unknown_{label_id}"
            percentage = (count / len(combined_df)) * 100
            print(f"  {label_id} ({phase_name}): {count}フレーム ({percentage:.1f}%)")
        
        return combined_df
    
    def save_dataset(self, features_df: pd.DataFrame, output_dir: str = None) -> Dict[str, str]:
        """データセットを保存"""
        if output_dir is None:
            output_dir = self.training_data_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ファイルパス
        csv_path = output_dir / f"tennis_features_dataset_{timestamp}.csv"
        json_path = output_dir / f"tennis_features_metadata_{timestamp}.json"
        
        # CSVファイルとして保存
        features_df.to_csv(csv_path, index=False)
        
        # メタデータを保存
        metadata = {
            'creation_time': timestamp,
            'total_samples': len(features_df),
            'feature_count': len(features_df.columns),
            'video_count': features_df['video_name'].nunique(),
            'phase_labels': self.phase_labels,
            'label_distribution': features_df['label'].value_counts().to_dict(),
            'feature_columns': list(features_df.columns),
            'videos_included': list(features_df['video_name'].unique())
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== データセット保存完了 ===")
        print(f"📊 CSVファイル: {csv_path}")
        print(f"📋 メタデータ: {json_path}")
        print(f"📈 サンプル数: {len(features_df)}")
        print(f"🎯 特徴量数: {len(features_df.columns)}")
        
        return {
            'csv_path': str(csv_path),
            'metadata_path': str(json_path),
            'sample_count': len(features_df),
            'feature_count': len(features_df.columns)
        }
    
    def analyze_feature_importance(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """特徴量の重要度を分析（簡易版）"""
        # 数値特徴量のみ抽出
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_features if col not in ['label', 'frame_number']]
        
        if len(feature_columns) == 0:
            print("分析可能な数値特徴量がありません")
            return pd.DataFrame()
        
        # 各特徴量の基本統計
        importance_stats = []
        
        for feature in feature_columns:
            feature_data = features_df[feature]
            
            # 基本統計
            stats = {
                'feature': feature,
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'min': feature_data.min(),
                'max': feature_data.max(),
                'non_zero_ratio': (feature_data != 0).mean(),
                'missing_ratio': feature_data.isna().mean()
            }
            
            # 局面間での分散（簡易的な重要度指標）
            label_means = features_df.groupby('label')[feature].mean()
            if len(label_means) > 1:
                stats['label_variance'] = label_means.var()
            else:
                stats['label_variance'] = 0
            
            importance_stats.append(stats)
        
        importance_df = pd.DataFrame(importance_stats)
        importance_df = importance_df.sort_values('label_variance', ascending=False)
        
        print("\n=== 特徴量重要度分析（TOP 20） ===")
        print(importance_df.head(20)[['feature', 'label_variance', 'mean', 'std', 'non_zero_ratio']].to_string(index=False))
        
        return importance_df

def main():
    """メイン関数 - 特徴量抽出の実行"""
    print("=== テニス動画特徴量抽出ツール ===")
    print("局面アノテーション、トラッキングデータ、コート座標から機械学習用特徴量を作成します")
    
    # 特徴量抽出器を初期化
    extractor = TennisFeatureExtractor()
    
    # データ存在チェック
    annotation_files = list(extractor.training_data_dir.glob("phase_annotations_*.json"))
    tracking_files = list(extractor.training_data_dir.glob("tracking_features_*.json"))
    court_files = list(extractor.training_data_dir.glob("court_coords_*.json"))
    
    print(f"\n=== データファイル確認 ===")
    print(f"局面アノテーション: {len(annotation_files)}ファイル")
    print(f"トラッキング特徴量: {len(tracking_files)}ファイル")
    print(f"コート座標: {len(court_files)}ファイル")
    
    if not annotation_files or not tracking_files:
        print("\n❌ 必要なデータファイルが不足しています")
        print("必要なファイル:")
        print("- phase_annotations_*.json (data_collector.pyで作成)")
        print("- tracking_features_*.json (balltracking.pyで作成)")
        print("オプション:")
        print("- court_coords_*.json (コート座標データ)")
        return
    
    try:
        # 全動画データを処理
        features_dataset = extractor.process_all_videos()
        
        if features_dataset.empty:
            print("❌ 特徴量データセットの作成に失敗しました")
            return
        
        # 特徴量重要度分析
        extractor.analyze_feature_importance(features_dataset)
        
        # データセットを保存
        save_info = extractor.save_dataset(features_dataset)
        
        print(f"\n🎉 特徴量抽出完了！")
        print(f"次のステップ: train_phase_model.py で機械学習モデルを訓練してください")
        print(f"使用するファイル: {save_info['csv_path']}")
        
    except Exception as e:
        print(f"❌ 処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
