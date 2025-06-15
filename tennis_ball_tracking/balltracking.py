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

class BallTracker:
    def __init__(self, model_path: str = "yolov8n.pt", imgsz: int = 640, 
                 save_training_data: bool = False, data_dir: str = "training_data",
                 frame_skip: int = 1):
        """
        テニスボールトラッカーの初期化
        
        Args:
            model_path: YOLOv8モデルのパス
            imgsz: 推論時の画像サイズ（デフォルト: 640）
            save_training_data: 学習用データを保存するかどうか
            data_dir: 学習データ保存先ディレクトリ
            frame_skip: フレームスキップ設定（1=全フレーム処理、2=2フレームに1回、3=3フレームに1回）
        """
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        
        # フレームスキップ設定
        self.frame_skip = frame_skip
        if frame_skip == 1:
            print(f"🖥️  全フレーム処理モード")
        else:
            print(f"🔄 フレームスキップモード: {frame_skip}フレームに1回処理")
        
        self.player_front_class_id = 0  # player_frontのクラスID
        self.player_back_class_id = 1   # player_backのクラスID
        self.tennis_ball_class_id = 2  # tennis_ballのクラスID
        
        # プレイヤー検出用の信頼度閾値
        self.player_confidence_threshold = 0.3
        
        # トラッキング関連のパラメータ
        self.max_disappeared = 15  # ボールが見えなくなってから削除するまでのフレーム数
        self.max_distance = 200  # 最大マッチング距離（ピクセル）
        self.min_movement_threshold = 3  # 最小移動距離（静止判定用）
        self.physics_check_frames = 5  # 物理的チェックに使用するフレーム数
        self.max_velocity_change = 80  # 最大速度変化（物理的制約）
        
        # 動的な検出閾値
        self.base_confidence_threshold = 0.2  # 基本信頼度閾値
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
        
        # フレーム補間用
        self.last_processed_frame = None
        self.last_ball_position = None
        
        # 学習用データ保存関連
        self.save_training_data = save_training_data
        self.training_features = []  # 学習用特徴量データ
        self.training_data_dir = Path(data_dir)
        
        if self.save_training_data:
            self.training_data_dir.mkdir(exist_ok=True)
            print(f"学習用データを保存します: {self.training_data_dir}")
    
    def should_process_frame(self, frame_count: int) -> bool:
        """フレームを処理すべきかどうかを判定"""
        return frame_count % self.frame_skip == 0
    
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
                if ball_info['last_seen'] > 8:  # 長時間見えないボールは除外
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
                current_ball_info = self.candidate_balls[best_match_id]
                prev_pos = None
                if len(current_ball_info['position_history']) > 0:
                    prev_pos = current_ball_info['position_history'][-1]

                current_ball_info['position_history'].append(pos)
                
                # 静止判定
                moved_significantly_this_frame = True
                if prev_pos:
                    movement_this_frame = self.calculate_distance(pos, prev_pos)
                    if movement_this_frame < self.min_movement_threshold:
                        moved_significantly_this_frame = False
                
                if not moved_significantly_this_frame:
                    current_ball_info['last_seen'] += 1
                else:
                    current_ball_info['last_seen'] = 0

                current_ball_info['movement_score'] = self.calculate_movement_score(
                    current_ball_info['position_history']
                )
                # 予測位置を更新
                current_ball_info['predicted_pos'] = self.predict_next_position(
                    list(current_ball_info['position_history'])
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
                         player_detections: List[Tuple[int, int, int, int, int, float]], 
                         original_frame_number: int = None, is_lightweight: bool = False) -> dict:
        """統合されたフレームデータ記録メソッド"""
        self.frame_number += 1
        current_time = datetime.now()
        
        # 元のフレーム番号を使用（指定されない場合は処理フレーム番号を使用）
        if original_frame_number is not None:
            actual_frame_number = original_frame_number
        else:
            actual_frame_number = self.frame_number
        
        # 基本情報
        frame_data = {
            'frame_number': actual_frame_number,
            'processed_frame_number': self.frame_number,
            'timestamp': current_time.isoformat(),
            'elapsed_time_seconds': (current_time - self.start_time).total_seconds(),
            'detections_count': len(detections),
            'candidate_balls_count': len(self.candidate_balls),
            'active_ball_id': self.active_ball,
            'disappeared_count': self.disappeared_count,
            'trajectory_length': len(self.ball_trajectory),
            'interpolated': False
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
                
                # 予測位置（軽量版では省略する場合がある）
                if not is_lightweight:
                    predicted_pos = self.predict_next_position(list(ball_info['position_history']))
                    if predicted_pos:
                        frame_data.update({
                            'predicted_x': predicted_pos[0],
                            'predicted_y': predicted_pos[1]
                        })
                    else:
                        frame_data.update({'predicted_x': None, 'predicted_y': None})
                else:
                    frame_data.update({'predicted_x': None, 'predicted_y': None})
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
        front_players = [det for det in player_detections if det[4] == self.player_front_class_id]
        back_players = [det for det in player_detections if det[4] == self.player_back_class_id]
        
        frame_data.update({
            'players_detected': len(player_detections),
            'player_front_count': len(front_players),
            'player_back_count': len(back_players)
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
        if not is_lightweight:
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
    
    def create_interpolated_frame_data(self, frame_number: int, interpolated_pos: Tuple[int, int]) -> dict:
        """補間フレームデータを作成"""
        return {
            'frame_number': frame_number,
            'processed_frame_number': None,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time_seconds': None,
            'detections_count': 0,
            'candidate_balls_count': 0,
            'active_ball_id': None,
            'disappeared_count': 0,
            'trajectory_length': 0,
            'ball_x': interpolated_pos[0],
            'ball_y': interpolated_pos[1],
            'ball_movement_score': 0,
            'ball_last_seen': 0,
            'ball_tracking_status': 'interpolated',
            'ball_velocity_x': 0,
            'ball_velocity_y': 0,
            'ball_speed': 0,
            'predicted_x': None,
            'predicted_y': None,
            'players_detected': 0,
            'player_front_count': 0,
            'player_back_count': 0,
            'best_detection_x': None,
            'best_detection_y': None,
            'best_detection_confidence': None,
            'confidence_threshold': 0.0,
            'interpolated': True
        }
    
    def process_skipped_frames_interpolation(self, current_ball_pos: Optional[Tuple[int, int]], frame_number: int):
        """スキップしたフレームの補間データを生成"""
        if self.last_ball_position is None or current_ball_pos is None:
            self.last_ball_position = current_ball_pos
            return
        
        # スキップしたフレーム数分の補間データを生成
        num_skipped = self.frame_skip - 1
        
        for i in range(num_skipped):
            skipped_frame_number = frame_number - num_skipped + i
            
            # 線形補間
            t = (i + 1) / (num_skipped + 1)
            interpolated_x = int(self.last_ball_position[0] + t * (current_ball_pos[0] - self.last_ball_position[0]))
            interpolated_y = int(self.last_ball_position[1] + t * (current_ball_pos[1] - self.last_ball_position[1]))
            
            interpolated_data = self.create_interpolated_frame_data(
                skipped_frame_number, (interpolated_x, interpolated_y)
            )
            self.time_series_data.append(interpolated_data)
        
        self.last_ball_position = current_ball_pos
    
    def extract_tracking_features(self, player_detections: List[Tuple[int, int, int, int, int, float]], 
                                 original_frame_number: int = None) -> dict:
        """ボール・プレイヤー検出の基本特徴量を抽出"""
        features = {
            'timestamp': datetime.now().isoformat(),
            'frame_number': original_frame_number or self.frame_number
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
                    'ball_tracking_confidence': 1.0 if ball_info['last_seen'] == 0 else 0.5
                })
                
                # ボール速度
                if len(ball_info['position_history']) >= 2:
                    velocity = self.calculate_velocity(list(ball_info['position_history']))
                    features.update({
                        'ball_velocity_x': velocity[0],
                        'ball_velocity_y': velocity[1],
                        'ball_speed': math.sqrt(velocity[0]**2 + velocity[1]**2),
                        'ball_velocity_x_normalized': velocity[0] / 100.0,
                        'ball_velocity_y_normalized': velocity[1] / 100.0
                    })
                else:
                    features.update({
                        'ball_velocity_x': 0, 'ball_velocity_y': 0, 'ball_speed': 0,
                        'ball_velocity_x_normalized': 0, 'ball_velocity_y_normalized': 0
                    })
            else:
                features.update({
                    'ball_x': None, 'ball_y': None, 'ball_detected': 0,
                    'ball_x_normalized': 0, 'ball_y_normalized': 0,
                    'ball_movement_score': 0, 'ball_tracking_confidence': 0,
                    'ball_velocity_x': 0, 'ball_velocity_y': 0, 'ball_speed': 0,
                    'ball_velocity_x_normalized': 0, 'ball_velocity_y_normalized': 0
                })
        else:
            features.update({
                'ball_x': None, 'ball_y': None, 'ball_detected': 0,
                'ball_x_normalized': 0, 'ball_y_normalized': 0,
                'ball_movement_score': 0, 'ball_tracking_confidence': 0,
                'ball_velocity_x': 0, 'ball_velocity_y': 0, 'ball_speed': 0,
                'ball_velocity_x_normalized': 0, 'ball_velocity_y_normalized': 0
            })
        
        # プレイヤー特徴量
        front_players = [det for det in player_detections if det[4] == self.player_front_class_id]
        back_players = [det for det in player_detections if det[4] == self.player_back_class_id]
        
        features.update({
            'player_front_count': len(front_players),
            'player_back_count': len(back_players),
            'total_players': len(player_detections)
        })
        
        # 最も信頼度の高いプレイヤーの位置
        if front_players:
            best_front = max(front_players, key=lambda x: x[5])
            x1, y1, x2, y2, _, conf = best_front
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            features.update({
                'player_front_x': center_x,
                'player_front_y': center_y,
                'player_front_x_normalized': center_x / 1920.0,
                'player_front_y_normalized': center_y / 1080.0,
                'player_front_confidence': conf,
                'player_front_width': x2 - x1,
                'player_front_height': y2 - y1
            })
        else:
            features.update({
                'player_front_x': None, 'player_front_y': None,
                'player_front_x_normalized': 0, 'player_front_y_normalized': 0,
                'player_front_confidence': 0,
                'player_front_width': 0, 'player_front_height': 0
            })
        
        if back_players:
            best_back = max(back_players, key=lambda x: x[5])
            x1, y1, x2, y2, _, conf = best_back
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            features.update({
                'player_back_x': center_x,
                'player_back_y': center_y,
                'player_back_x_normalized': center_x / 1920.0,
                'player_back_y_normalized': center_y / 1080.0,
                'player_back_confidence': conf,
                'player_back_width': x2 - x1,
                'player_back_height': y2 - y1
            })
        else:
            features.update({
                'player_back_x': None, 'player_back_y': None,
                'player_back_x_normalized': 0, 'player_back_y_normalized': 0,
                'player_back_confidence': 0,
                'player_back_width': 0, 'player_back_height': 0
            })
        
        # プレイヤー間距離
        if front_players and back_players:
            front_center = ((front_players[0][0] + front_players[0][2]) / 2,
                           (front_players[0][1] + front_players[0][3]) / 2)
            back_center = ((back_players[0][0] + back_players[0][2]) / 2,
                          (back_players[0][1] + back_players[0][3]) / 2)
            player_distance = math.sqrt(
                (front_center[0] - back_center[0])**2 + 
                (front_center[1] - back_center[1])**2
            )
            features['player_distance'] = player_distance
            features['player_distance_normalized'] = player_distance / 1920.0
        else:
            features['player_distance'] = 0
            features['player_distance_normalized'] = 0
        
        # トラッキング状態特徴量
        features.update({
            'candidate_balls_count': len(self.candidate_balls),
            'disappeared_count': self.disappeared_count,
            'trajectory_length': len(self.ball_trajectory),
            'prediction_active': 1 if (self.active_ball and 
                                     self.active_ball in self.candidate_balls and
                                     self.candidate_balls[self.active_ball]['last_seen'] > 0) else 0
        })
        
        return features
    
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
    
    def process_frame_core(self, frame: np.ndarray, original_frame_number: int = None, 
                          is_lightweight: bool = False) -> Tuple[np.ndarray, bool]:
        """コアフレーム処理ロジック（統合版）"""
        # 動的な信頼度を取得
        confidence_threshold = self.get_dynamic_confidence_threshold()
        
        # YOLOv8で検出
        results = self.model(frame, imgsz=self.imgsz, verbose=False)
        
        # 検出結果を抽出
        detections = []
        player_detections = []
        
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
        
        # トラッキング更新
        self.update_candidate_balls(detections)
        
        # アクティブボール選択
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
        
        # データ記録
        self.record_frame_data(detections, player_detections, original_frame_number, is_lightweight)
        
        # 学習用データ保存
        if self.save_training_data:
            tracking_features = self.extract_tracking_features(player_detections, original_frame_number)
            self.training_features.append(tracking_features)
        
        # 結果描画（軽量版では省略）
        if not is_lightweight:
            result_frame = self.draw_tracking_results(frame, detections, player_detections)
            return result_frame, True
        else:
            return frame, True
    
    def process_frame_optimized(self, frame: np.ndarray, frame_count: int, 
                               training_data_only: bool = False) -> Tuple[np.ndarray, bool]:
        """最適化されたフレーム処理（統合版）"""
        processed = False
        
        if self.should_process_frame(frame_count):
            # 処理対象フレーム
            result_frame, processed = self.process_frame_core(
                frame, frame_count, is_lightweight=training_data_only
            )
            
            # フレームスキップ補間（学習データモード時）
            if training_data_only and frame_count > 0 and self.frame_skip > 1:
                current_ball_pos = None
                if (self.active_ball is not None and 
                    self.active_ball in self.candidate_balls and
                    len(self.candidate_balls[self.active_ball]['position_history']) > 0):
                    current_ball_pos = self.candidate_balls[self.active_ball]['position_history'][-1]
                
                self.process_skipped_frames_interpolation(current_ball_pos, frame_count)
            
            return result_frame, processed
        else:
            # スキップフレーム
            if training_data_only:
                return frame, False
            else:
                # 軽量な表示用処理
                result_frame = frame.copy()
                height, width = frame.shape[:2]
                skip_info = f"SKIP ({frame_count % self.frame_skip}/{self.frame_skip})"
                cv2.putText(result_frame, skip_info, 
                           (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                return result_frame, False
    
    def save_time_series_data(self, output_path: str):
        """時系列データをCSVファイルに保存"""
        if not self.time_series_data:
            print("保存する時系列データがありません")
            return
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = self.time_series_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.time_series_data)
            
            total_frames = len(self.time_series_data)
            processed_frames = sum(1 for data in self.time_series_data if not data.get('interpolated', False))
            interpolated_frames = total_frames - processed_frames
            
            print(f"時系列データを保存しました: {output_path}")
            print(f"総フレーム数: {total_frames}")
            print(f"実処理フレーム: {processed_frames}")
            print(f"補間フレーム: {interpolated_frames}")
        except Exception as e:
            print(f"CSV保存エラー: {e}")
    
    def save_tracking_features_with_video_info(self, video_name: str, video_fps: float = None, 
                                              total_video_frames: int = None):
        """動画情報を含めてトラッキング特徴量を保存"""
        if not self.training_features:
            print("保存する特徴量データがありません")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tracking_features_{video_name}_{timestamp}.json"
        output_path = self.training_data_dir / filename
        
        # 詳細なメタデータを含む構造で保存
        data_with_metadata = {
            'metadata': {
                'video_name': video_name,
                'frame_skip': self.frame_skip,
                'total_frames_processed': len(self.training_features),
                'creation_time': timestamp,
                'model_path': "best_5_31.pt",
                'inference_size': self.imgsz,
                'processing_mode': 'frame_skip' if self.frame_skip > 1 else 'full_frame',
                'original_fps': video_fps,
                'processing_fps': video_fps / self.frame_skip if video_fps and self.frame_skip > 1 else video_fps,
                'total_original_frames': total_video_frames,
                'frame_skip_ratio': f"1/{self.frame_skip}" if self.frame_skip > 1 else "1/1",
                'processing_efficiency': f"{100/self.frame_skip:.1f}%" if self.frame_skip > 1 else "100%"
            },
            'frames': self.training_features
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(data_with_metadata, f, indent=2)
            print(f"検出・トラッキング特徴量を保存しました: {output_path}")
            print(f"フレームスキップ設定: {self.frame_skip} (処理効率: {100/self.frame_skip:.1f}%)")
            print(f"処理フレーム数: {len(self.training_features)}")
        except Exception as e:
            print(f"特徴量保存エラー: {e}")
    
def main():
    """メイン関数 - 対話式の選択メニュー"""
    # モデルのパスを設定
    model_path = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/models/weights/best_5_31.pt"
    
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
    
    # フレームスキップ設定の選択
    print("\nフレームスキップ設定を選択してください:")
    print("1. 全フレーム処理（スキップなし・高精度）")
    print("2. 2フレームに1回処理（2倍高速）")
    print("3. 3フレームに1回処理（3倍高速）")
    print("4. 4フレームに1回処理（4倍高速）")
    print("5. 5フレームに1回処理（5倍高速）")
    print("6. 6フレームに1回処理（6倍高速）")
    print("7. 10フレームに1回処理（10倍高速）")
    
    frame_skip_options = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 10}
    
    while True:
        try:
            skip_choice = int(input(f"フレームスキップを選択 (1-{len(frame_skip_options)}): "))
            if skip_choice in frame_skip_options:
                frame_skip = frame_skip_options[skip_choice]
                break
            else:
                print(f"1 から {len(frame_skip_options)} の間の数字を入力してください。")
        except ValueError:
            print("数字を入力してください。")
    
    if frame_skip == 1:
        print("選択されたモード: 全フレーム処理（最高精度）")
    else:
        print(f"選択されたモード: {frame_skip}フレームに1回処理（約{frame_skip}倍高速）")
    
    # 処理モードの選択
    print("\n出力モードを選択してください:")
    print("1. 動画保存モード（結果を動画ファイルに保存）")
    print("2. リアルタイム表示モード（リアルタイム表示のみ）")
    print("3. 学習用データ出力モード（高速・データのみ出力）")
    
    while True:
        try:
            mode = int(input("モードを選択 (1, 2, または 3): "))
            if mode in [1, 2, 3]:
                break
            else:
                print("1, 2, または 3 を入力してください。")
        except ValueError:
            print("数字を入力してください。")
    
    save_video = (mode == 1)
    show_realtime = (mode in [1, 2])  # モード1,2で表示
    training_data_only = (mode == 3)  # モード3で学習データのみ
    
    if save_video:
        print("選択されたモード: 動画保存モード")
    elif show_realtime:
        print("選択されたモード: リアルタイム表示モード")
    else:
        print("選択されたモード: 学習用データ出力モード（高速処理）")
    
    # 時系列データ保存オプション（学習データのみモードでは自動で有効)
    if training_data_only:
        save_time_series = True
        print("学習用データ出力モードでは時系列データも自動保存されます")
    else:
        print("\n時系列データ出力を行いますか？")
        print("1. はい（CSVファイルに保存・補間データも含む）")
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
        print("時系列データをCSVファイルに保存します（補間データも含む）")
    elif not training_data_only:
        print("時系列データは保存しません")
    
    # 学習用データ保存オプション（学習データのみモードでは自動で有効）
    if training_data_only:
        save_training_data = True
        print("学習用データ出力モードでは学習用データが自動保存されます")
    else:
        print("\n学習用データ保存を行いますか？")
        print("1. はい（特徴量データを保存）")
        print("2. いいえ")
        
        while True:
            try:
                save_training_choice = int(input("選択 (1 または 2): "))
                if save_training_choice in [1, 2]:
                    break
                else:
                    print("1, 2 を入力してください。")
            except ValueError:
                print("数字を入力してください。")
    
        save_training_data = (save_training_choice == 1)
    
    if save_training_data:
        print("学習用データを保存します")
    elif not training_data_only:
        print("学習用データは保存しません")

    # 動画ファイルの選択
    print("\n処理する動画ファイルを選択してください:")
    video_dir = Path("C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/raw")
    video_files = sorted([f for f in video_dir.glob("*.mp4")])

    if not video_files:
        print(f"エラー: {video_dir} にMP4ファイルが見つかりません。")
        while True:
            video_path_input = input("動画ファイルのフルパスを入力してください: ")
            if Path(video_path_input).is_file() and video_path_input.lower().endswith(".mp4"):
                video_path = video_path_input
                break
            else:
                print("無効なファイルパスまたはMP4ファイルではありません。再入力してください。")
    else:
        print("利用可能な動画ファイル:")
        for i, f_path in enumerate(video_files):
            print(f"{i + 1}. {f_path.name}")
        print(f"{len(video_files) + 1}. 別のパスを直接入力する")

        while True:
            try:
                video_choice = int(input(f"動画を選択 (1-{len(video_files) + 1}): "))
                if 1 <= video_choice <= len(video_files):
                    video_path = str(video_files[video_choice - 1])
                    break
                elif video_choice == len(video_files) + 1:
                    while True:
                        video_path_input = input("動画ファイルのフルパスを入力してください: ")
                        if Path(video_path_input).is_file() and video_path_input.lower().endswith(".mp4"):
                            video_path = video_path_input
                            break
                        else:
                            print("無効なファイルパスまたはMP4ファイルではありません。再入力してください。")
                    break
                else:
                    print(f"1 から {len(video_files) + 1} の間の数字を入力してください。")
            except ValueError:
                print("数字を入力してください。")
    
    print(f"選択された動画ファイル: {video_path}")

    # トラッカーを初期化（フレームスキップ設定を含める）
    tracker = BallTracker(model_path, imgsz=inference_imgsz, 
                          save_training_data=save_training_data, frame_skip=frame_skip)
    
    print(f"推論画像サイズ: {inference_imgsz}")
    print(f"フレームスキップ: {frame_skip}フレームに1回処理")
    
    # ビデオファイルを開く
    # video_path = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/raw/output_segment_000.mp4" # 修正: ユーザー選択を使用
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: ビデオファイルを開けませんでした")
        return
    
    # ビデオの情報を取得
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
    if total_frames > 0:
        print(f"総フレーム数: {total_frames}")
    if save_video:
        print(f"動画出力ファイル: {output_video_path}")
    if save_time_series:
        print(f"CSVデータ出力ファイル: {csv_output_path}")
    
    if training_data_only:
        print("学習用データ出力モード - 高速処理中...")
        print("注意: 画面表示は行われません。進捗を確認してください。")
    else:
        print("リアルタイム表示中... 'q'キーで終了")
    
    frame_count = 0
    processed_frame_count = 0
    start_time = datetime.now()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # フレームスキップ対応の処理
            result_frame, was_processed = tracker.process_frame_optimized(
                frame, frame_count, training_data_only
            )
            
            if was_processed:
                processed_frame_count += 1
            
            # 動画保存（保存モードの場合のみ）
            if save_video and out is not None and not training_data_only:
                out.write(result_frame)
            
            # リアルタイム表示
            if show_realtime and not training_data_only:
                cv2.imshow('Tennis Ball Tracking', result_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("ユーザーによって処理が中断されました")
                    break
            
            frame_count += 1
            
            # 進捗表示（フレームスキップ考慮）
            if training_data_only:
                if frame_count % 200 == 0:  # 200フレームごとに表示
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    fps_current = frame_count / elapsed_time if elapsed_time > 0 else 0
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    processing_rate = (processed_frame_count / frame_count * 100) if frame_count > 0 else 0
                    print(f"処理中... フレーム: {frame_count}/{total_frames if total_frames > 0 else '?'} "
                          f"({progress:.1f}%) | 処理速度: {fps_current:.1f} FPS | "
                          f"実処理数: {processed_frame_count} ({processing_rate:.1f}%) | 軌跡: {len(tracker.ball_trajectory)}")
            elif frame_count % 100 == 0:
                processing_rate = (processed_frame_count / frame_count * 100) if frame_count > 0 else 0
                print(f"表示フレーム: {frame_count} | 処理フレーム: {processed_frame_count} ({processing_rate:.1f}%)")

    finally:
        # 最終統計の表示（フレームスキップ考慮）
        elapsed_time = (datetime.now() - start_time).total_seconds()
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        processing_rate = (processed_frame_count / frame_count * 100) if frame_count > 0 else 0
        expected_rate = (100 / frame_skip) if frame_skip > 0 else 100
        
        print(f"\n=== 処理完了統計 ===")
        print(f"総フレーム数: {frame_count}")
        print(f"実処理フレーム数: {processed_frame_count}")
        print(f"実際の処理率: {processing_rate:.1f}%")
        print(f"期待処理率: {expected_rate:.1f}%（{frame_skip}フレームに1回）")
        print(f"平均処理速度: {avg_fps:.1f} FPS")
        print(f"処理時間: {elapsed_time:.1f}秒")
        if frame_skip > 1:
            print(f"スピードアップ: 約{frame_skip}倍")
        
        if tracker.save_training_data:
            feature_count = len(tracker.training_features)
            time_series_count = len(tracker.time_series_data)
            print(f"学習用特徴量: {feature_count}件")
            print(f"時系列データ: {time_series_count}件")
            print(f"軌跡ポイント: {len(tracker.ball_trajectory)}件")
        
        # 時系列データを保存
        if save_time_series and csv_output_path:
            tracker.save_time_series_data(csv_output_path)
        
        # 学習用データを保存
        if save_training_data:
            video_name = Path(video_path).stem
            tracker.save_tracking_features_with_video_info(video_name, fps, total_frames)
        
        cap.release()
        if out is not None:
            out.release()
        if show_realtime and not training_data_only:
            cv2.destroyAllWindows()
    
    return True

if __name__ == "__main__":
    main()