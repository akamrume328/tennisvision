import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import deque
from typing import List, Tuple, Optional
import csv
import os
from datetime import datetime
import json
import glob
from pathlib import Path

class TennisBallTracker:
    def __init__(self, model_path: str, imgsz: int = 1920, save_training_data: bool = False):
        """
        テニスボールトラッカーの初期化
        
        Args:
            model_path: YOLOv8モデルのパス
            imgsz: 推論時の画像サイズ（デフォルト: 640）
            save_training_data: 学習用データを保存するかどうか
        """
        self.model = YOLO(model_path)
        self.imgsz = imgsz  # 推論時の画像サイズ
        self.tennis_ball_class_id = 2  # tennis_ballのクラスID
        self.player_front_class_id = 0  # player_frontのクラスID
        self.player_back_class_id = 1   # player_backのクラスID
        
        # プレイヤー検出用の信頼度閾値
        self.player_confidence_threshold = 0.3
        
        # トラッキング関連のパラメータ
        self.max_disappeared = 15  # ボールが見えなくなってから削除するまでのフレーム数（増加）
        self.max_distance = 200  # 最大マッチング距離（ピクセル）（増加）
        self.min_movement_threshold = 3  # 最小移動距離（静止判定用）（減少）
        self.physics_check_frames = 5  # 物理的チェックに使用するフレーム数
        self.max_velocity_change = 80  # 最大速度変化（物理的制約）（増加）
        
        # 動的な検出閾値
        self.base_confidence_threshold = 0.2  # 基本信頼度閾値（減少）
        self.high_confidence_threshold = 0.4  # 高信頼度閾値
        
        # 現在追跡中のボール
        self.active_ball = None
        self.ball_trajectory = deque(maxlen=50)  # 軌跡を保存（最大50ポイント）
        self.disappeared_count = 0
        
        # 候補ボールの管理
        self.candidate_balls = {}  # ID: {position_history, last_seen, movement_score, predicted_pos}
        self.next_id = 0
        
        # 時系列データ記録用
        self.time_series_data = []
        self.frame_number = 0
        self.start_time = datetime.now()
        
        # 学習用データ保存関連
        self.save_training_data = save_training_data
        self.training_features = []  # 学習用特徴量データ
        self.training_data_dir = Path("training_data")
        
        if self.save_training_data:
            self.training_data_dir.mkdir(exist_ok=True)
            print(f"学習用データを保存します: {self.training_data_dir}")
    
    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """2点間の距離を計算"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_velocity(self, positions: List[Tuple[int, int]]) -> Tuple[float, float]:
        """位置リストから速度を計算"""
        if len(positions) < 2:
            return 0.0, 0.0
        
        # 最新の2点から速度を計算
        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
        return dx, dy
    
    def is_physically_valid_movement(self, new_pos: Tuple[int, int], 
                                   position_history: List[Tuple[int, int]]) -> bool:
        """物理的に妥当な移動かどうかをチェック"""
        if len(position_history) < 2:
            return True
        
        # 前の速度を計算
        prev_velocity = self.calculate_velocity(list(position_history))
        
        # 新しい位置での速度を計算
        temp_history = list(position_history) + [new_pos]
        new_velocity = self.calculate_velocity(temp_history)
        
        # 速度変化をチェック
        velocity_change = math.sqrt(
            (new_velocity[0] - prev_velocity[0])**2 + 
            (new_velocity[1] - prev_velocity[1])**2
        )
        
        return velocity_change <= self.max_velocity_change
    
    def calculate_movement_score(self, position_history: List[Tuple[int, int]]) -> float:
        """ボールの動きの活発さをスコア化"""
        if len(position_history) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(position_history)):
            total_distance += self.calculate_distance(position_history[i-1], position_history[i])
        
        return total_distance / len(position_history) if len(position_history) > 0 else 0.0
    
    def update_candidate_balls(self, detections: List[Tuple[int, int, float]]):
        """候補ボールを更新"""
        matched_candidates = set()
        
        for detection in detections:
            x, y, confidence = detection
            pos = (x, y)
            
            # 既存の候補ボールとマッチング
            best_match_id = None
            best_distance = float('inf')
            
            for ball_id, ball_info in self.candidate_balls.items():
                if ball_info['last_seen'] > 8:  # 長時間見えないボールは除外（緩和）
                    continue
                
                # 予測を考慮したマッチング距離を計算
                distance = self.calculate_prediction_match_distance(pos, ball_info)
                
                # 見失い期間中は距離制約を緩和
                max_dist = self.max_distance
                if ball_info['last_seen'] > 0:
                    max_dist = self.max_distance * (1 + ball_info['last_seen'] * 0.3)
                
                if (distance < max_dist and 
                    distance < best_distance and
                    self.is_physically_valid_movement(pos, ball_info['position_history'])):
                    best_distance = distance
                    best_match_id = ball_id
            
            if best_match_id is not None:
                # 既存のボールを更新
                self.candidate_balls[best_match_id]['position_history'].append(pos)
                self.candidate_balls[best_match_id]['last_seen'] = 0
                self.candidate_balls[best_match_id]['movement_score'] = self.calculate_movement_score(
                    self.candidate_balls[best_match_id]['position_history']
                )
                # 予測位置を更新
                self.candidate_balls[best_match_id]['predicted_pos'] = self.predict_next_position(
                    list(self.candidate_balls[best_match_id]['position_history'])
                )
                matched_candidates.add(best_match_id)
            else:
                # 新しいボールとして追加
                self.candidate_balls[self.next_id] = {
                    'position_history': deque([pos], maxlen=20),
                    'last_seen': 0,
                    'movement_score': 0.0,
                    'predicted_pos': None
                }
                matched_candidates.add(self.next_id)
                self.next_id += 1
        
        # 見えなくなったボールの処理
        to_remove = []
        for ball_id in self.candidate_balls:
            if ball_id not in matched_candidates:
                self.candidate_balls[ball_id]['last_seen'] += 1
                
                # 予測位置を更新（見失い中も予測を継続）
                if len(self.candidate_balls[ball_id]['position_history']) >= 2:
                    self.candidate_balls[ball_id]['predicted_pos'] = self.predict_next_position(
                        list(self.candidate_balls[ball_id]['position_history'])
                    )
                
                if self.candidate_balls[ball_id]['last_seen'] > self.max_disappeared:
                    to_remove.append(ball_id)
        
        # 古いボールを削除
        for ball_id in to_remove:
            del self.candidate_balls[ball_id]
    
    def select_active_ball(self):
        """最も活発に動いているボールをアクティブボールとして選択"""
        if not self.candidate_balls:
            return None
        
        # 現在のアクティブボールが有効な場合は継続
        if (self.active_ball is not None and 
            self.active_ball in self.candidate_balls and
            self.candidate_balls[self.active_ball]['last_seen'] <= 5):  # 継続条件を緩和
            return self.active_ball
        
        # 最小移動距離を満たし、最も活発に動いているボールを選択
        best_ball_id = None
        best_score = 0.0
        
        for ball_id, ball_info in self.candidate_balls.items():
            # 選択条件を緩和
            if (len(ball_info['position_history']) >= 2 and  # 必要履歴数を減少
                ball_info['movement_score'] >= self.min_movement_threshold and
                ball_info['last_seen'] <= 3):  # 見失い許容フレーム数を緩和
                
                # 継続性ボーナス（既存のトラッキング対象に優先度を与える）
                score = ball_info['movement_score']
                if ball_id == self.active_ball:
                    score *= 1.5  # 継続ボーナス
                
                if score > best_score:
                    best_score = score
                    best_ball_id = ball_id
        
        return best_ball_id
    
    def record_frame_data(self, detections: List[Tuple[int, int, float]], 
                         player_detections: List[Tuple[int, int, int, int, int, float]]) -> dict:
        """フレームごとのデータを記録"""
        self.frame_number += 1
        current_time = datetime.now()
        
        # 基本情報
        frame_data = {
            'frame_number': self.frame_number,
            'timestamp': current_time.isoformat(),
            'elapsed_time_seconds': (current_time - self.start_time).total_seconds(),
            'detections_count': len(detections),
            'candidate_balls_count': len(self.candidate_balls),
            'active_ball_id': self.active_ball,
            'disappeared_count': self.disappeared_count,
            'trajectory_length': len(self.ball_trajectory)
        }
        
        # アクティブボール情報
        if self.active_ball is not None and self.active_ball in self.candidate_balls:
            ball_info = self.candidate_balls[self.active_ball]
            if len(ball_info['position_history']) > 0:
                current_pos = ball_info['position_history'][-1]
                frame_data.update({
                    'ball_x': current_pos[0],
                    'ball_y': current_pos[1],
                    'ball_movement_score': ball_info['movement_score'],
                    'ball_last_seen': ball_info['last_seen'],
                    'ball_tracking_status': 'tracking' if ball_info['last_seen'] == 0 else 'predicting'
                })
                
                # 速度情報
                if len(ball_info['position_history']) >= 2:
                    velocity = self.calculate_velocity(list(ball_info['position_history']))
                    frame_data.update({
                        'ball_velocity_x': velocity[0],
                        'ball_velocity_y': velocity[1],
                        'ball_speed': math.sqrt(velocity[0]**2 + velocity[1]**2)
                    })
                else:
                    frame_data.update({
                        'ball_velocity_x': 0,
                        'ball_velocity_y': 0,
                        'ball_speed': 0
                    })
                
                # 予測位置
                predicted_pos = self.predict_next_position(list(ball_info['position_history']))
                if predicted_pos:
                    frame_data.update({
                        'predicted_x': predicted_pos[0],
                        'predicted_y': predicted_pos[1]
                    })
                else:
                    frame_data.update({
                        'predicted_x': None,
                        'predicted_y': None
                    })
            else:
                frame_data.update({
                    'ball_x': None, 'ball_y': None, 'ball_movement_score': 0,
                    'ball_last_seen': 0, 'ball_tracking_status': 'none',
                    'ball_velocity_x': 0, 'ball_velocity_y': 0, 'ball_speed': 0,
                    'predicted_x': None, 'predicted_y': None
                })
        else:
            frame_data.update({
                'ball_x': None, 'ball_y': None, 'ball_movement_score': 0,
                'ball_last_seen': 0, 'ball_tracking_status': 'none',
                'ball_velocity_x': 0, 'ball_velocity_y': 0, 'ball_speed': 0,
                'predicted_x': None, 'predicted_y': None
            })
        
        # プレイヤー情報
        frame_data['players_detected'] = len(player_detections)
        player_front_count = sum(1 for _, _, _, _, class_id, _ in player_detections if class_id == self.player_front_class_id)
        player_back_count = sum(1 for _, _, _, _, class_id, _ in player_detections if class_id == self.player_back_class_id)
        frame_data.update({
            'player_front_count': player_front_count,
            'player_back_count': player_back_count
        })
        
        # 最高信頼度の検出情報
        if detections:
            best_detection = max(detections, key=lambda x: x[2])
            frame_data.update({
                'best_detection_x': best_detection[0],
                'best_detection_y': best_detection[1],
                'best_detection_confidence': best_detection[2]
            })
        else:
            frame_data.update({
                'best_detection_x': None,
                'best_detection_y': None,
                'best_detection_confidence': None
            })
        
        # 動的閾値情報
        frame_data['confidence_threshold'] = self.get_dynamic_confidence_threshold()
        
        # 候補ボールの詳細情報（最大3つまで）
        sorted_candidates = sorted(
            self.candidate_balls.items(),
            key=lambda x: x[1]['movement_score'],
            reverse=True
        )
        
        for i in range(min(3, len(sorted_candidates))):
            ball_id, ball_info = sorted_candidates[i]
            prefix = f'candidate_{i+1}_'
            if len(ball_info['position_history']) > 0:
                pos = ball_info['position_history'][-1]
                frame_data.update({
                    f'{prefix}id': ball_id,
                    f'{prefix}x': pos[0],
                    f'{prefix}y': pos[1],
                    f'{prefix}movement_score': ball_info['movement_score'],
                    f'{prefix}last_seen': ball_info['last_seen']
                })
            else:
                frame_data.update({
                    f'{prefix}id': None,
                    f'{prefix}x': None,
                    f'{prefix}y': None,
                    f'{prefix}movement_score': 0,
                    f'{prefix}last_seen': 0
                })
        
        # 足りない候補ボール情報を埋める
        for i in range(len(sorted_candidates), 3):
            prefix = f'candidate_{i+1}_'
            frame_data.update({
                f'{prefix}id': None,
                f'{prefix}x': None,
                f'{prefix}y': None,
                f'{prefix}movement_score': 0,
                f'{prefix}last_seen': 0
            })
        
        self.time_series_data.append(frame_data)
        return frame_data
    
    def save_time_series_data(self, output_path: str):
        """時系列データをCSVファイルに保存"""
        if not self.time_series_data:
            print("保存するデータがありません")
            return
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = self.time_series_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(self.time_series_data)
        
        print(f"時系列データを保存しました: {output_path}")
        print(f"総フレーム数: {len(self.time_series_data)}")
    
    def save_training_features(self, video_name: str):
        """学習用特徴量データを保存"""
        if not self.training_features:
            print("保存する学習用データがありません")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.training_data_dir / f"tracking_features_{video_name}_{timestamp}.json"
        
        # データを保存
        save_data = {
            'metadata': {
                'video_name': video_name,
                'total_frames': len(self.training_features),
                'created_at': timestamp,
                'model_path': str(self.model.model_path) if hasattr(self.model, 'model_path') else 'unknown',
                'imgsz': self.imgsz,
                'data_type': 'tracking_features'
            },
            'features': self.training_features
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"学習用特徴量データを保存しました: {output_file}")
            print(f"保存されたフレーム数: {len(self.training_features)}")
        except Exception as e:
            print(f"学習用データの保存に失敗: {e}")

    def extract_training_features(self, player_detections: List[Tuple[int, int, int, int, int, float]], 
                                 court_coordinates: dict = None) -> dict:
        """学習用の包括的な特徴量を抽出（data_collector独立版）"""
        features = {
            'timestamp': datetime.now().isoformat(),
            'frame_number': self.frame_number,
            'video_timestamp': self.frame_number / 30.0  # 30fps想定
        }
        
        # ボール位置特徴量
        if self.active_ball is not None and self.active_ball in self.candidate_balls:
            ball_info = self.candidate_balls[self.active_ball]
            if len(ball_info['position_history']) > 0:
                pos = ball_info['position_history'][-1]
                features.update({
                    'ball_x': pos[0],
                    'ball_y': pos[1],
                    'ball_x_normalized': pos[0] / 1920.0,
                    'ball_y_normalized': pos[1] / 1080.0,
                    'ball_detected': 1,
                    'ball_movement_score': ball_info['movement_score'],
                    'ball_tracking_confidence': 1.0 if ball_info['last_seen'] == 0 else max(0.1, 1.0 - ball_info['last_seen'] * 0.1)
                })
                
                # ボール速度と物理量
                if len(ball_info['position_history']) >= 2:
                    velocity = self.calculate_velocity(list(ball_info['position_history']))
                    speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
                    features.update({
                        'ball_velocity_x': velocity[0],
                        'ball_velocity_y': velocity[1],
                        'ball_speed': speed,
                        'ball_velocity_x_normalized': velocity[0] / 100.0,
                        'ball_velocity_y_normalized': velocity[1] / 100.0,
                        'ball_speed_normalized': speed / 100.0
                    })
                else:
                    features.update({
                        'ball_velocity_x': 0, 'ball_velocity_y': 0, 'ball_speed': 0,
                        'ball_velocity_x_normalized': 0, 'ball_velocity_y_normalized': 0,
                        'ball_speed_normalized': 0
                    })
            else:
                features.update({
                    'ball_x': None, 'ball_y': None, 'ball_detected': 0,
                    'ball_x_normalized': 0, 'ball_y_normalized': 0,
                    'ball_movement_score': 0, 'ball_tracking_confidence': 0,
                    'ball_velocity_x': 0, 'ball_velocity_y': 0, 'ball_speed': 0,
                    'ball_velocity_x_normalized': 0, 'ball_velocity_y_normalized': 0,
                    'ball_speed_normalized': 0
                })
        else:
            features.update({
                'ball_x': None, 'ball_y': None, 'ball_detected': 0,
                'ball_x_normalized': 0, 'ball_y_normalized': 0,
                'ball_movement_score': 0, 'ball_tracking_confidence': 0,
                'ball_velocity_x': 0, 'ball_velocity_y': 0, 'ball_speed': 0,
                'ball_velocity_x_normalized': 0, 'ball_velocity_y_normalized': 0,
                'ball_speed_normalized': 0
            })
        
        # プレイヤー特徴量
        front_players = [det for det in player_detections if det[4] == self.player_front_class_id]
        back_players = [det for det in player_detections if det[4] == self.player_back_class_id]
        
        features.update({
            'player_front_count': len(front_players),
            'player_back_count': len(back_players),
            'total_players': len(player_detections)
        })
        
        # プレイヤー位置情報（最も信頼度の高いプレイヤー）
        if front_players:
            best_front = max(front_players, key=lambda x: x[5])
            x1, y1, x2, y2, _, conf = best_front
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            features.update({
                'player_front_x': center_x,
                'player_front_y': center_y,
                'player_front_x_normalized': center_x / 1920.0,
                'player_front_y_normalized': center_y / 1080.0,
                'player_front_confidence': conf,
                'player_front_width': width,
                'player_front_height': height,
                'player_front_area': width * height,
                'player_front_area_normalized': (width * height) / (1920.0 * 1080.0)
            })
        else:
            features.update({
                'player_front_x': None, 'player_front_y': None,
                'player_front_x_normalized': 0, 'player_front_y_normalized': 0,
                'player_front_confidence': 0, 'player_front_width': 0,
                'player_front_height': 0, 'player_front_area': 0,
                'player_front_area_normalized': 0
            })
        
        if back_players:
            best_back = max(back_players, key=lambda x: x[5])
            x1, y1, x2, y2, _, conf = best_back
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            features.update({
                'player_back_x': center_x,
                'player_back_y': center_y,
                'player_back_x_normalized': center_x / 1920.0,
                'player_back_y_normalized': center_y / 1080.0,
                'player_back_confidence': conf,
                'player_back_width': width,
                'player_back_height': height,
                'player_back_area': width * height,
                'player_back_area_normalized': (width * height) / (1920.0 * 1080.0)
            })
        else:
            features.update({
                'player_back_x': None, 'player_back_y': None,
                'player_back_x_normalized': 0, 'player_back_y_normalized': 0,
                'player_back_confidence': 0, 'player_back_width': 0,
                'player_back_height': 0, 'player_back_area': 0,
                'player_back_area_normalized': 0
            })
        
        # プレイヤー間の関係性
        if front_players and back_players:
            front_center = ((front_players[0][0] + front_players[0][2]) / 2,
                           (front_players[0][1] + front_players[0][3]) / 2)
            back_center = ((back_players[0][0] + back_players[0][2]) / 2,
                          (back_players[0][1] + back_players[0][3]) / 2)
            
            distance = math.sqrt((front_center[0] - back_center[0])**2 + 
                               (front_center[1] - back_center[1])**2)
            
            # 相対位置
            dx = front_center[0] - back_center[0]
            dy = front_center[1] - back_center[1]
            
            features.update({
                'player_distance': distance,
                'player_distance_normalized': distance / 1920.0,
                'player_relative_x': dx,
                'player_relative_y': dy,
                'player_relative_x_normalized': dx / 1920.0,
                'player_relative_y_normalized': dy / 1080.0
            })
        else:
            features.update({
                'player_distance': 0, 'player_distance_normalized': 0,
                'player_relative_x': 0, 'player_relative_y': 0,
                'player_relative_x_normalized': 0, 'player_relative_y_normalized': 0
            })
        
        # トラッキング状態情報
        features.update({
            'candidate_balls_count': len(self.candidate_balls),
            'disappeared_count': self.disappeared_count,
            'trajectory_length': len(self.ball_trajectory),
            'max_movement_score': max([info['movement_score'] for info in self.candidate_balls.values()]) if self.candidate_balls else 0,
            'prediction_active': 1 if (self.active_ball and 
                                     self.active_ball in self.candidate_balls and
                                     self.candidate_balls[self.active_ball]['last_seen'] > 0) else 0
        })
        
        # 画面領域情報（ボールの位置による）
        if features['ball_detected']:
            ball_x, ball_y = features['ball_x'], features['ball_y']
            # 画面を9分割した領域（1-9）
            region_x = min(2, ball_x // (1920 // 3))
            region_y = min(2, ball_y // (1080 // 3))
            screen_region = int(region_y * 3 + region_x + 1)
            
            features.update({
                'ball_screen_region': screen_region,
                'ball_left_side': 1 if ball_x < 960 else 0,
                'ball_top_half': 1 if ball_y < 540 else 0
            })
        else:
            features.update({
                'ball_screen_region': 0,
                'ball_left_side': 0,
                'ball_top_half': 0
            })
        
        return features

    def load_court_coordinates(self) -> dict:
        """コート座標ファイルを検索してロード（data_collector独立版）"""
        # コート座標は使用しない（data_collectorで別途管理）
        return {}

    def _is_ball_in_court(self, ball_pos: Tuple[int, int], court_coordinates: dict) -> int:
        """ボールがコート内にあるかどうかを判定（data_collector独立版）"""
        # コート判定は行わない（data_collectorで別途実装）
        return 0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """フレームを処理してテニスボールとプレイヤーを追跡"""
        # 動的な信頼度閾値を取得
        confidence_threshold = self.get_dynamic_confidence_threshold()
        
        # YOLOv8で検出（imgszを指定）
        results = self.model(frame, imgsz=self.imgsz, verbose=False)
        
        # テニスボールの検出結果を抽出
        detections = []
        player_detections = []  # プレイヤー検出結果を格納
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if class_id == self.tennis_ball_class_id and confidence > confidence_threshold:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        detections.append((center_x, center_y, confidence))
                    
                    elif (class_id in [self.player_front_class_id, self.player_back_class_id] and 
                          confidence > self.player_confidence_threshold):
                        player_detections.append((x1, y1, x2, y2, class_id, confidence))
        
        # 候補ボールを更新
        self.update_candidate_balls(detections)
        
        # アクティブボールを選択
        active_ball_id = self.select_active_ball()
        
        if active_ball_id is not None:
            self.active_ball = active_ball_id
            self.disappeared_count = 0
            
            # 軌跡を更新
            current_pos = self.candidate_balls[active_ball_id]['position_history'][-1]
            self.ball_trajectory.append(current_pos)
        else:
            self.disappeared_count += 1
            if self.disappeared_count > self.max_disappeared:
                self.active_ball = None
                self.ball_trajectory.clear()
        
        # フレームデータを記録
        self.record_frame_data(detections, player_detections)
        
        # 学習用データ保存（コート座標なしで保存）
        if self.save_training_data:
            training_features = self.extract_training_features(player_detections, {})
            self.training_features.append(training_features)
        
        # 結果を描画（ボールとプレイヤー両方）
        result_frame = self.draw_tracking_results(frame, detections, player_detections)
        
        return result_frame
    
    def draw_tracking_results(self, frame: np.ndarray, detections: List[Tuple[int, int, float]], 
                            player_detections: List[Tuple[int, int, int, int, int, float]]) -> np.ndarray:
        """追跡結果を描画（ボールとプレイヤー）"""
        result_frame = frame.copy()
        
        # プレイヤーを描画
        self.draw_players(result_frame, player_detections)
        
        # 全ての検出結果を薄い色で描画
        for x, y, conf in detections:
            cv2.circle(result_frame, (x, y), 8, (100, 100, 100), 2)
            cv2.putText(result_frame, f'{conf:.2f}', (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # 候補ボールの予測位置を描画
        for ball_id, ball_info in self.candidate_balls.items():
            if ball_info['predicted_pos'] is not None and ball_info['last_seen'] > 0:
                pred_pos = ball_info['predicted_pos']
                cv2.circle(result_frame, pred_pos, 6, (255, 165, 0), 2)  # オレンジ色で予測位置
                cv2.putText(result_frame, 'PRED', (pred_pos[0]+10, pred_pos[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
        
        # アクティブボールがある場合
        if self.active_ball is not None and self.active_ball in self.candidate_balls:
            # アクティブボールを強調表示
            ball_info = self.candidate_balls[self.active_ball]
            if len(ball_info['position_history']) > 0:
                current_pos = ball_info['position_history'][-1]
                cv2.circle(result_frame, current_pos, 12, (0, 255, 0), 3)
                
                # トラッキング状態に応じてテキストを変更
                status_text = 'TRACKING' if ball_info['last_seen'] == 0 else f'PREDICTING ({ball_info["last_seen"]})'
                cv2.putText(result_frame, status_text, (current_pos[0]+15, current_pos[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 軌跡を描画
        if len(self.ball_trajectory) > 1:
            points = np.array(list(self.ball_trajectory), np.int32)
            
            # 軌跡を線で描画（古い点ほど薄く）
            for i in range(1, len(points)):
                alpha = i / len(points)  # 透明度
                thickness = max(1, int(3 * alpha))
                color_intensity = int(255 * alpha)
                color = (0, color_intensity, 0)
                
                cv2.line(result_frame, tuple(points[i-1]), tuple(points[i]), color, thickness)
            
            # 軌跡上の点を描画
            for i, point in enumerate(points):
                alpha = (i + 1) / len(points)
                radius = max(2, int(4 * alpha))
                color_intensity = int(255 * alpha)
                color = (0, color_intensity, 0)
                cv2.circle(result_frame, tuple(point), radius, color, -1)
        
        # 情報を表示
        confidence_threshold = self.get_dynamic_confidence_threshold()
        info_text = [
            f"Candidate balls: {len(self.candidate_balls)}",
            f"Active ball: {'Yes' if self.active_ball else 'No'}",
            f"Trajectory points: {len(self.ball_trajectory)}",
            f"Players detected: {len(player_detections)}",
            f"Confidence threshold: {confidence_threshold:.2f}",
            f"Disappeared count: {self.disappeared_count}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(result_frame, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_frame
    
    def draw_players(self, frame: np.ndarray, player_detections: List[Tuple[int, int, int, int, int, float]]):
        """プレイヤーを描画"""
        for x1, y1, x2, y2, class_id, confidence in player_detections:
            # クラスIDに応じて色とラベルを設定
            if class_id == self.player_front_class_id:
                color = (255, 0, 0)  # 青色でplayer_front
                label = "Player Front"
            elif class_id == self.player_back_class_id:
                color = (0, 0, 255)  # 赤色でplayer_back
                label = "Player Back"
            else:
                continue
            
            # バウンディングボックスを描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # ラベルと信頼度を描画
            label_text = f"{label}: {confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # ラベル背景を描画
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # ラベルテキストを描画
            cv2.putText(frame, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def predict_next_position(self, position_history: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """次のフレームでのボール位置を予測"""
        if len(position_history) < 2:
            return None
        
        # 直近の速度を計算
        velocity = self.calculate_velocity(list(position_history))
        
        # 物理的な加速度を考慮（重力の影響）
        gravity_acceleration = 2  # ピクセル/フレーム²（下向き）
        
        # 次の位置を予測
        last_pos = position_history[-1]
        predicted_x = last_pos[0] + velocity[0]
        predicted_y = last_pos[1] + velocity[1] + gravity_acceleration
        
        return (int(predicted_x), int(predicted_y))
    
    def get_dynamic_confidence_threshold(self) -> float:
        """検出が少ない場合に信頼度閾値を動的に調整"""
        # アクティブボールがトラッキング中の場合は低い閾値を使用
        if self.active_ball is not None and self.disappeared_count > 3:
            return self.base_confidence_threshold
        else:
            return self.high_confidence_threshold
    
    def calculate_prediction_match_distance(self, detection_pos: Tuple[int, int], 
                                          ball_info: dict) -> float:
        """予測位置を考慮したマッチング距離を計算"""
        position_history = ball_info['position_history']
        
        # 通常のマッチング距離
        last_pos = position_history[-1]
        normal_distance = self.calculate_distance(detection_pos, last_pos)
        
        # 予測位置との距離
        predicted_pos = self.predict_next_position(list(position_history))
        if predicted_pos is not None:
            prediction_distance = self.calculate_distance(detection_pos, predicted_pos)
            # 予測距離と通常距離の最小値を使用
            return min(normal_distance, prediction_distance)
        
        return normal_distance

    def get_tracking_data(self) -> dict:
        """局面判断用のトラッキングデータを取得"""
        tracking_data = {
            'timestamp': cv2.getTickCount() / cv2.getTickFrequency(),
            'active_ball': None,
            'ball_trajectory': list(self.ball_trajectory),
            'ball_velocity': None,
            'ball_position': None,
            'ball_confidence': None,
            'prediction_active': False,
            'players': []
        }
        
        # アクティブボール情報
        if self.active_ball is not None and self.active_ball in self.candidate_balls:
            ball_info = self.candidate_balls[self.active_ball]
            if len(ball_info['position_history']) > 0:
                tracking_data['ball_position'] = ball_info['position_history'][-1]
                tracking_data['active_ball'] = self.active_ball
                tracking_data['prediction_active'] = ball_info['last_seen'] > 0
                
                # 速度計算
                if len(ball_info['position_history']) >= 2:
                    tracking_data['ball_velocity'] = self.calculate_velocity(
                        list(ball_info['position_history'])
                    )
        
        return tracking_data
    
    def process_frame_for_analysis(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """フレーム処理と局面判断用データを同時に返す"""
        result_frame = self.process_frame(frame)
        tracking_data = self.get_tracking_data()
        return result_frame, tracking_data

def main():
    """メイン関数 - テスト用"""
    # モデルのパスを設定（適切なパスに変更してください）
    model_path = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/models/weights/best_5_31.pt"  # または他の適切なモデルパス
    
    # 推論時の画像サイズを選択
    print("推論時の画像サイズを選択してください:")
    print("1. 1920 (高精度・低速)")
    print("2. 1280 (バランス)")
    print("3. 640 (高速・低精度)")
    
    imgsz_options = {1: 1920, 2: 1280, 3: 640}
    
    while True:
        try:
            imgsz_choice = int(input("画像サイズを選択 (1, 2, または 3): "))
            if imgsz_choice in imgsz_options:
                inference_imgsz = imgsz_options[imgsz_choice]
                break
            else:
                print("1, 2, または 3 を入力してください。")
        except ValueError:
            print("数字を入力してください。")
    
    print(f"選択された画像サイズ: {inference_imgsz}")
    
    # 処理モードの選択
    print("\n処理モードを選択してください:")
    print("1. 動画保存モード（結果を動画ファイルに保存）")
    print("2. リアルタイム表示モード（リアルタイム表示のみ）")
    
    while True:
        try:
            mode = int(input("モードを選択 (1 または 2): "))
            if mode in [1, 2]:
                break
            else:
                print("1 または 2 を入力してください。")
        except ValueError:
            print("数字を入力してください。")
    
    save_video = (mode == 1)
    show_realtime = True  # 両モードで表示
    
    if save_video:
        print("選択されたモード: 動画保存モード")
    else:
        print("選択されたモード: リアルタイム表示モード")
    
    # 時系列データ保存オプション
    print("\n時系列データ出力を行いますか？")
    print("1. はい（CSVファイルに保存）")
    print("2. いいえ")
    
    while True:
        try:
            save_data_choice = int(input("選択 (1 または 2): "))
            if save_data_choice in [1, 2]:
                break
            else:
                print("1 または 2 を入力してください。")
        except ValueError:
            print("数字を入力してください。")
    
    save_time_series = (save_data_choice == 1)
    
    if save_time_series:
        print("時系列データをCSVファイルに保存します")
    else:
        print("時系列データは保存しません")
    
    # 学習用データ保存オプションを追加
    print("\n学習用データ保存を行いますか？")
    print("1. はい（特徴量データを保存）")
    print("2. いいえ")
    
    while True:
        try:
            save_training_choice = int(input("選択 (1 または 2): "))
            if save_training_choice in [1, 2]:
                break
            else:
                print("1 または 2 を入力してください。")
        except ValueError:
            print("数字を入力してください。")
    
    save_training_data = (save_training_choice == 1)
    
    if save_training_data:
        print("学習用データを保存します")
        print("注意: コート座標ファイル(court_coords_*.json)がtraining_dataフォルダにあることを確認してください")
    else:
        print("学習用データは保存しません")
    
    # トラッカーを初期化（imgszを指定）
    tracker = TennisBallTracker(model_path, imgsz=inference_imgsz, save_training_data=save_training_data)
    
    print(f"推論画像サイズ: {inference_imgsz}")
    
    # ビデオファイルまたはカメラを開く
    video_path = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/raw/output_segment_000.mp4"  # または適切なビデオパス
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: ビデオファイルを開けませんでした")
        return
    
    # ビデオの情報を取得
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 出力ファイルパスの設定
    output_dir = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/output"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 出力ビデオの設定（保存モードの場合のみ）
    out = None
    output_video_path = None
    if save_video:
        output_video_path = os.path.join(output_dir, f"tennis_tracking_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 時系列データ出力パス
    csv_output_path = None
    if save_time_series:
        csv_output_path = os.path.join(output_dir, f"tracking_data_{timestamp}.csv")
    
    print(f"処理開始 - FPS: {fps}, 解像度: {width}x{height}")
    if save_video:
        print(f"動画出力ファイル: {output_video_path}")
    if save_time_series:
        print(f"CSVデータ出力ファイル: {csv_output_path}")
    print("リアルタイム表示中... 'q'キーで終了")
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # フレームを処理
            result_frame = tracker.process_frame(frame)
            
            # 動画保存（保存モードの場合のみ）
            if save_video and out is not None:
                out.write(result_frame)
            
            # リアルタイム表示
            if show_realtime:
                cv2.imshow('Tennis Ball Tracking', result_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("ユーザーによって処理が中断されました")
                    break
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"処理済みフレーム: {frame_count}")
    
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # 時系列データを保存
        if save_time_series and csv_output_path:
            tracker.save_time_series_data(csv_output_path)
        
        # 学習用データを保存
        if save_training_data:
            video_name = Path(video_path).stem if video_path else "unknown"
            tracker.save_training_features(video_name)
        
        if save_video:
            print(f"動画処理完了 - 出力ファイル: {output_video_path}")
        else:
            print("処理完了 - リアルタイム表示モード")
        
        print(f"総処理フレーム数: {frame_count}")

if __name__ == "__main__":
    main()