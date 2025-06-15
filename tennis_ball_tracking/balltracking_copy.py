"""
テニスボールトラッキングシステム
YOLOv8を使用したボールとプレイヤーの検出・追跡
"""

import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
import csv
import os
from datetime import datetime
import json
from pathlib import Path

class Config:
    """設定クラス"""
    
    # クラスID定義
    TENNIS_BALL_CLASS_ID = 2
    PLAYER_FRONT_CLASS_ID = 0
    PLAYER_BACK_CLASS_ID = 1
    
    # 信頼度閾値
    BASE_CONFIDENCE_THRESHOLD = 0.2
    HIGH_CONFIDENCE_THRESHOLD = 0.4
    PLAYER_CONFIDENCE_THRESHOLD = 0.3
    
    # トラッキングパラメータ
    MAX_DISAPPEARED = 15
    MAX_DISTANCE = 200
    MIN_MOVEMENT_THRESHOLD = 3
    PHYSICS_CHECK_FRAMES = 5
    MAX_VELOCITY_CHANGE = 80
    
    # 表示設定
    TRAJECTORY_MAX_LENGTH = 50
    GRAVITY_ACCELERATION = 2

class BallCandidate:
    """ボール候補を管理するクラス"""
    
    def __init__(self, position: Tuple[int, int]):
        self.position_history = deque([position], maxlen=20)
        self.last_seen = 0
        self.movement_score = 0.0
        self.predicted_pos = None
    
    def update_position(self, new_pos: Tuple[int, int]):
        """位置を更新"""
        prev_pos = self.position_history[-1] if self.position_history else None
        self.position_history.append(new_pos)
        
        # 移動判定
        if prev_pos:
            movement = self._calculate_distance(new_pos, prev_pos)
            if movement < Config.MIN_MOVEMENT_THRESHOLD:
                self.last_seen += 1
            else:
                self.last_seen = 0
        
        self.movement_score = self._calculate_movement_score()
        self.predicted_pos = self._predict_next_position()
    
    def increment_last_seen(self):
        """見失いカウントを増加"""
        self.last_seen += 1
        if len(self.position_history) >= 2:
            self.predicted_pos = self._predict_next_position()
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """2点間の距離を計算"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_movement_score(self) -> float:
        """動きの活発さをスコア化"""
        if len(self.position_history) < 2:
            return 0.0
        
        total_distance = 0.0
        positions = list(self.position_history)
        for i in range(1, len(positions)):
            total_distance += self._calculate_distance(positions[i-1], positions[i])
        
        return total_distance / len(positions)
    
    def _predict_next_position(self) -> Optional[Tuple[int, int]]:
        """次のフレームでのボール位置を予測"""
        if len(self.position_history) < 2:
            return None
        
        positions = list(self.position_history)
        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
        
        # 重力の影響を考慮
        last_pos = positions[-1]
        predicted_x = last_pos[0] + dx
        predicted_y = last_pos[1] + dy + Config.GRAVITY_ACCELERATION
        
        return (int(predicted_x), int(predicted_y))

class FrameDataRecorder:
    """フレームデータの記録と管理クラス"""
    
    def __init__(self):
        self.time_series_data = []
        self.skipped_frames_data = []
        self.frame_number = 0
        self.start_time = datetime.now()
    
    def record_frame_data(self, candidate_balls: Dict[int, BallCandidate], 
                         active_ball_id: Optional[int], disappeared_count: int,
                         detections: List[Tuple[int, int, float]], 
                         player_detections: List[Tuple[int, int, int, int, int, float]],
                         trajectory_length: int) -> dict:
        """フレームごとのデータを記録"""
        self.frame_number += 1
        current_time = datetime.now()
        
        frame_data = {
            'frame_number': self.frame_number,
            'timestamp': current_time.isoformat(),
            'elapsed_time_seconds': (current_time - self.start_time).total_seconds(),
            'detections_count': len(detections),
            'candidate_balls_count': len(candidate_balls),
            'active_ball_id': active_ball_id,
            'disappeared_count': disappeared_count,
            'trajectory_length': trajectory_length
        }
        
        # アクティブボール情報
        self._add_ball_data(frame_data, candidate_balls, active_ball_id)
        
        # プレイヤー情報
        self._add_player_data(frame_data, player_detections)
        
        # 検出情報
        self._add_detection_data(frame_data, detections)
        
        # 候補ボール詳細情報
        self._add_candidate_details(frame_data, candidate_balls)
        
        self.time_series_data.append(frame_data)
        return frame_data
    
    def _add_ball_data(self, frame_data: dict, candidate_balls: Dict[int, BallCandidate], 
                      active_ball_id: Optional[int]):
        """アクティブボール情報を追加"""
        if active_ball_id is not None and active_ball_id in candidate_balls:
            ball = candidate_balls[active_ball_id]
            if ball.position_history:
                current_pos = ball.position_history[-1]
                frame_data.update({
                    'ball_x': current_pos[0],
                    'ball_y': current_pos[1],
                    'ball_movement_score': ball.movement_score,
                    'ball_last_seen': ball.last_seen,
                    'ball_tracking_status': 'tracking' if ball.last_seen == 0 else 'predicting'
                })
                
                # 速度情報
                if len(ball.position_history) >= 2:
                    positions = list(ball.position_history)
                    dx = positions[-1][0] - positions[-2][0]
                    dy = positions[-1][1] - positions[-2][1]
                    speed = math.sqrt(dx**2 + dy**2)
                    frame_data.update({
                        'ball_velocity_x': dx,
                        'ball_velocity_y': dy,
                        'ball_speed': speed
                    })
                
                # 予測位置
                if ball.predicted_pos:
                    frame_data.update({
                        'predicted_x': ball.predicted_pos[0],
                        'predicted_y': ball.predicted_pos[1]
                    })
        
        # データが無い場合のデフォルト値
        default_ball_data = {
            'ball_x': None, 'ball_y': None, 'ball_movement_score': 0,
            'ball_last_seen': 0, 'ball_tracking_status': 'none',
            'ball_velocity_x': 0, 'ball_velocity_y': 0, 'ball_speed': 0,
            'predicted_x': None, 'predicted_y': None
        }
        
        for key, value in default_ball_data.items():
            if key not in frame_data:
                frame_data[key] = value
    
    def _add_player_data(self, frame_data: dict, player_detections: List[Tuple[int, int, int, int, int, float]]):
        """プレイヤー情報を追加"""
        frame_data['players_detected'] = len(player_detections)
        frame_data['player_front_count'] = sum(1 for det in player_detections if det[4] == Config.PLAYER_FRONT_CLASS_ID)
        frame_data['player_back_count'] = sum(1 for det in player_detections if det[4] == Config.PLAYER_BACK_CLASS_ID)
    
    def _add_detection_data(self, frame_data: dict, detections: List[Tuple[int, int, float]]):
        """検出情報を追加"""
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
    
    def _add_candidate_details(self, frame_data: dict, candidate_balls: Dict[int, BallCandidate]):
        """候補ボールの詳細情報を追加"""
        sorted_candidates = sorted(candidate_balls.items(), key=lambda x: x[1].movement_score, reverse=True)
        
        for i in range(3):  # 最大3つの候補
            prefix = f'candidate_{i+1}_'
            if i < len(sorted_candidates):
                ball_id, ball = sorted_candidates[i]
                if ball.position_history:
                    pos = ball.position_history[-1]
                    frame_data.update({
                        f'{prefix}id': ball_id,
                        f'{prefix}x': pos[0],
                        f'{prefix}y': pos[1],
                        f'{prefix}movement_score': ball.movement_score,
                        f'{prefix}last_seen': ball.last_seen
                    })
                    continue
            
            # デフォルト値
            frame_data.update({
                f'{prefix}id': None,
                f'{prefix}x': None,
                f'{prefix}y': None,
                f'{prefix}movement_score': 0,
                f'{prefix}last_seen': 0
            })
    
    def save_time_series_data(self, output_path: str):
        """時系列データをCSVファイルに保存"""
        if not self.time_series_data and not self.skipped_frames_data:
            print("保存するデータがありません")
            return
        
        all_data = self.time_series_data + self.skipped_frames_data
        all_data.sort(key=lambda x: x.get('frame_number', 0))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            if all_data:
                fieldnames = all_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_data)
        
        print(f"時系列データを保存: {output_path}")
        print(f"処理フレーム数: {len(self.time_series_data)}")
        print(f"補間フレーム数: {len(self.skipped_frames_data)}")

class FeatureExtractor:
    """特徴量抽出クラス"""
    
    def __init__(self):
        self.training_features = []
    
    def extract_tracking_features(self, frame_number: int, candidate_balls: Dict[int, BallCandidate],
                                active_ball_id: Optional[int], 
                                player_detections: List[Tuple[int, int, int, int, int, float]]) -> dict:
        """基本的なトラッキング特徴量を抽出"""
        features = {
            'timestamp': datetime.now().isoformat(),
            'frame_number': frame_number
        }
        
        # ボール特徴量
        self._extract_ball_features(features, candidate_balls, active_ball_id)
        
        # プレイヤー特徴量
        self._extract_player_features(features, player_detections)
        
        return features
    
    def _extract_ball_features(self, features: dict, candidate_balls: Dict[int, BallCandidate], 
                              active_ball_id: Optional[int]):
        """ボール特徴量を抽出"""
        if active_ball_id is not None and active_ball_id in candidate_balls:
            ball = candidate_balls[active_ball_id]
            if ball.position_history:
                pos = ball.position_history[-1]
                features.update({
                    'ball_x': pos[0],
                    'ball_y': pos[1],
                    'ball_x_normalized': pos[0] / 1920.0,
                    'ball_y_normalized': pos[1] / 1080.0,
                    'ball_detected': 1,
                    'ball_movement_score': ball.movement_score,
                    'ball_tracking_confidence': 1.0 if ball.last_seen == 0 else 0.5
                })
                
                # 速度計算
                if len(ball.position_history) >= 2:
                    positions = list(ball.position_history)
                    dx = positions[-1][0] - positions[-2][0]
                    dy = positions[-1][1] - positions[-2][1]
                    speed = math.sqrt(dx**2 + dy**2)
                    features.update({
                        'ball_velocity_x': dx,
                        'ball_velocity_y': dy,
                        'ball_speed': speed,
                        'ball_velocity_x_normalized': dx / 100.0,
                        'ball_velocity_y_normalized': dy / 100.0
                    })
        
        # デフォルト値設定
        default_values = {
            'ball_x': None, 'ball_y': None, 'ball_detected': 0,
            'ball_x_normalized': 0, 'ball_y_normalized': 0,
            'ball_movement_score': 0, 'ball_tracking_confidence': 0,
            'ball_velocity_x': 0, 'ball_velocity_y': 0, 'ball_speed': 0,
            'ball_velocity_x_normalized': 0, 'ball_velocity_y_normalized': 0
        }
        
        for key, value in default_values.items():
            if key not in features:
                features[key] = value
    
    def _extract_player_features(self, features: dict, 
                                player_detections: List[Tuple[int, int, int, int, int, float]]):
        """プレイヤー特徴量を抽出"""
        front_players = [det for det in player_detections if det[4] == Config.PLAYER_FRONT_CLASS_ID]
        back_players = [det for det in player_detections if det[4] == Config.PLAYER_BACK_CLASS_ID]
        
        features.update({
            'player_front_count': len(front_players),
            'player_back_count': len(back_players),
            'total_players': len(player_detections)
        })
        # 各プレイヤーの位置情報
        self._add_player_position_features(features, front_players, 'front')
        self._add_player_position_features(features, back_players, 'back')
        
        # プレイヤー間距離
        self._add_player_distance_features(features, front_players, back_players)
    
    def _add_player_position_features(self, features: dict, players: list, player_type: str):
        """プレイヤー位置特徴量を追加"""
        if players:
            best_player = max(players, key=lambda x: x[5])
            x1, y1, x2, y2, _, conf = best_player
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            features.update({
                f'player_{player_type}_x': center_x,
                f'player_{player_type}_y': center_y,
                f'player_{player_type}_x_normalized': center_x / 1920.0,
                f'player_{player_type}_y_normalized': center_y / 1080.0,
                f'player_{player_type}_confidence': conf,
                f'player_{player_type}_width': x2 - x1,
                f'player_{player_type}_height': y2 - y1
            })
        else:
            features.update({
                f'player_{player_type}_x': None,
                f'player_{player_type}_y': None,
                f'player_{player_type}_x_normalized': 0,
                f'player_{player_type}_y_normalized': 0,
                f'player_{player_type}_confidence': 0,
                f'player_{player_type}_width': 0,
                f'player_{player_type}_height': 0
            })
    
    def _add_player_distance_features(self, features: dict, front_players: list, back_players: list):
        """プレイヤー間距離特徴量を追加"""
        if front_players and back_players:
            front_center = ((front_players[0][0] + front_players[0][2]) / 2,
                           (front_players[0][1] + front_players[0][3]) / 2)
            back_center = ((back_players[0][0] + back_players[0][2]) / 2,
                          (back_players[0][1] + back_players[0][3]) / 2)
            
            distance = math.sqrt((front_center[0] - back_center[0])**2 + 
                               (front_center[1] - back_center[1])**2)
            
            features.update({
                'player_distance': distance,
                'player_distance_normalized': distance / 1920.0
            })
        else:
            features.update({
                'player_distance': 0,
                'player_distance_normalized': 0
            })
    
    def save_features(self, video_name: str, training_data_dir: Path):
        """特徴量を保存"""
        if not self.training_features:
            print("保存する特徴量データがありません")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tracking_features_{video_name}_{timestamp}.json"
        output_path = training_data_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(self.training_features, f, indent=2)
        
        print(f"検出・トラッキング特徴量を保存: {output_path}")
        print(f"総フレーム数: {len(self.training_features)}")

class TennisBallTracker:
    """メインのテニスボールトラッカークラス"""
    
    def __init__(self, model_path: str, imgsz: int = 1920, 
                 save_training_data: bool = False, frame_skip: int = 1):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.frame_skip = frame_skip
        
        # ボール管理
        self.candidate_balls: Dict[int, BallCandidate] = {}
        self.active_ball: Optional[int] = None
        self.next_id = 0
        self.disappeared_count = 0
        self.ball_trajectory = deque(maxlen=Config.TRAJECTORY_MAX_LENGTH)
        
        # データ記録
        self.data_recorder = FrameDataRecorder()
        self.feature_extractor = FeatureExtractor() if save_training_data else None
        self.training_data_dir = Path("training_data")
        
        # フレーム補間用
        self.last_ball_position = None
        
        if save_training_data:
            self.training_data_dir.mkdir(exist_ok=True)
            print(f"学習用データを保存します: {self.training_data_dir}")
        
        if frame_skip > 1:
            print(f"フレームスキップ機能が有効: {frame_skip}フレームに1回処理")
    
    def get_dynamic_confidence_threshold(self) -> float:
        """動的な信頼度閾値を取得"""
        if self.active_ball is not None and self.disappeared_count > 3:
            return Config.BASE_CONFIDENCE_THRESHOLD
        return Config.HIGH_CONFIDENCE_THRESHOLD
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, float]], 
                                                       List[Tuple[int, int, int, int, int, float]]]:
        """オブジェクト検出"""
        confidence_threshold = self.get_dynamic_confidence_threshold()
        results = self.model(frame, imgsz=self.imgsz, verbose=False)
        
        detections = []
        player_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if (class_id == Config.TENNIS_BALL_CLASS_ID and 
                        confidence > confidence_threshold):
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        detections.append((center_x, center_y, confidence))
                    
                    elif (class_id in [Config.PLAYER_FRONT_CLASS_ID, Config.PLAYER_BACK_CLASS_ID] and 
                          confidence > Config.PLAYER_CONFIDENCE_THRESHOLD):
                        player_detections.append((x1, y1, x2, y2, class_id, confidence))
        
        return detections, player_detections
    
    def update_candidate_balls(self, detections: List[Tuple[int, int, float]]):
        """候補ボールを更新"""
        matched_candidates = set()
        
        # 検出結果と既存候補をマッチング
        for detection in detections:
            x, y, confidence = detection
            pos = (x, y)
            
            best_match_id = self._find_best_match(pos)
            
            if best_match_id is not None:
                self.candidate_balls[best_match_id].update_position(pos)
                matched_candidates.add(best_match_id)
            else:
                # 新しい候補として追加
                self.candidate_balls[self.next_id] = BallCandidate(pos)
                matched_candidates.add(self.next_id)
                self.next_id += 1
        
        # 見えなくなった候補の処理
        self._update_unmatched_candidates(matched_candidates)
    
    def _find_best_match(self, detection_pos: Tuple[int, int]) -> Optional[int]:
        """最適なマッチング候補を見つける"""
        best_match_id = None
        best_distance = float('inf')
        
        for ball_id, ball in self.candidate_balls.items():
            if ball.last_seen > 8:
                continue
            
            distance = self._calculate_match_distance(detection_pos, ball)
            max_dist = Config.MAX_DISTANCE * (1 + ball.last_seen * 0.3)
            
            if (distance < max_dist and distance < best_distance and
                self._is_physically_valid_movement(detection_pos, ball)):
                best_distance = distance
                best_match_id = ball_id
        
        return best_match_id
    
    def _calculate_match_distance(self, detection_pos: Tuple[int, int], 
                                 ball: BallCandidate) -> float:
        """マッチング距離を計算"""
        if not ball.position_history:
            return float('inf')
        
        last_pos = ball.position_history[-1]
        normal_distance = self._calculate_distance(detection_pos, last_pos)
        
        # 予測位置も考慮
        if ball.predicted_pos:
            prediction_distance = self._calculate_distance(detection_pos, ball.predicted_pos)
            return min(normal_distance, prediction_distance)
        
        return normal_distance
    
    def _is_physically_valid_movement(self, new_pos: Tuple[int, int], 
                                     ball: BallCandidate) -> bool:
        """物理的に妥当な移動かチェック"""
        if len(ball.position_history) < 2:
            return True
        
        positions = list(ball.position_history)
        
        # 前の速度
        prev_velocity = self._calculate_velocity(positions)
        
        # 新しい速度
        temp_positions = positions + [new_pos]
        new_velocity = self._calculate_velocity(temp_positions)
        
        # 速度変化をチェック
        velocity_change = math.sqrt(
            (new_velocity[0] - prev_velocity[0])**2 + 
            (new_velocity[1] - prev_velocity[1])**2
        )
        
        return velocity_change <= Config.MAX_VELOCITY_CHANGE
    
    def _calculate_velocity(self, positions: List[Tuple[int, int]]) -> Tuple[float, float]:
        """速度を計算"""
        if len(positions) < 2:
            return 0.0, 0.0
        
        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
        return dx, dy
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """2点間の距離を計算"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _update_unmatched_candidates(self, matched_candidates: set):
        """マッチしなかった候補を更新"""
        to_remove = []
        
        for ball_id in self.candidate_balls:
            if ball_id not in matched_candidates:
                self.candidate_balls[ball_id].increment_last_seen()
                
                if self.candidate_balls[ball_id].last_seen > Config.MAX_DISAPPEARED:
                    to_remove.append(ball_id)
        
        for ball_id in to_remove:
            del self.candidate_balls[ball_id]
    
    def select_active_ball(self) -> Optional[int]:
        """アクティブボールを選択"""
        # 現在のアクティブボールが有効な場合は継続
        if (self.active_ball is not None and 
            self.active_ball in self.candidate_balls and
            self.candidate_balls[self.active_ball].last_seen <= 5):
            return self.active_ball
        
        # 最適な候補を選択
        best_ball_id = None
        best_score = 0.0
        
        for ball_id, ball in self.candidate_balls.items():
            if (len(ball.position_history) >= 2 and
                ball.movement_score >= Config.MIN_MOVEMENT_THRESHOLD and
                ball.last_seen <= 3):
                
                score = ball.movement_score
                if ball_id == self.active_ball:
                    score *= 1.5  # 継続ボーナス
                
                if score > best_score:
                    best_score = score
                    best_ball_id = ball_id
        
        return best_ball_id
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """フレームを処理"""
        # オブジェクト検出
        detections, player_detections = self.detect_objects(frame)
        
        # 候補ボールを更新
        self.update_candidate_balls(detections)
        
        # アクティブボールを選択
        active_ball_id = self.select_active_ball()
        
        if active_ball_id is not None:
            self.active_ball = active_ball_id
            self.disappeared_count = 0
            
            # 軌跡を更新
            if self.candidate_balls[active_ball_id].position_history:
                current_pos = self.candidate_balls[active_ball_id].position_history[-1]
                self.ball_trajectory.append(current_pos)
        else:
            self.disappeared_count += 1
            if self.disappeared_count > Config.MAX_DISAPPEARED:
                self.active_ball = None
                self.ball_trajectory.clear()
        
        # データを記録
        self.data_recorder.record_frame_data(
            self.candidate_balls, self.active_ball, self.disappeared_count,
            detections, player_detections, len(self.ball_trajectory)
        )
        
        # 学習用特徴量を抽出
        if self.feature_extractor:
            features = self.feature_extractor.extract_tracking_features(
                self.data_recorder.frame_number, self.candidate_balls, 
                self.active_ball, player_detections
            )
            self.feature_extractor.training_features.append(features)
        
        # 結果を描画
        return self.draw_tracking_results(frame, detections, player_detections)
    
    def draw_tracking_results(self, frame: np.ndarray, 
                            detections: List[Tuple[int, int, float]], 
                            player_detections: List[Tuple[int, int, int, int, int, float]]) -> np.ndarray:
        """追跡結果を描画"""
        result_frame = frame.copy()
        
        # プレイヤーを描画
        self._draw_players(result_frame, player_detections)
        
        # 全ての検出結果を薄い色で描画
        for x, y, conf in detections:
            cv2.circle(result_frame, (x, y), 8, (100, 100, 100), 2)
            cv2.putText(result_frame, f'{conf:.2f}', (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # 候補ボールの予測位置を描画
        for ball_id, ball in self.candidate_balls.items():
            if ball.predicted_pos and ball.last_seen > 0:
                pred_pos = ball.predicted_pos
                cv2.circle(result_frame, pred_pos, 6, (255, 165, 0), 2)
                cv2.putText(result_frame, 'PRED', (pred_pos[0]+10, pred_pos[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
        
        # アクティブボールを描画
        if (self.active_ball is not None and 
            self.active_ball in self.candidate_balls and
            self.candidate_balls[self.active_ball].position_history):
            
            ball = self.candidate_balls[self.active_ball]
            current_pos = ball.position_history[-1]
            cv2.circle(result_frame, current_pos, 12, (0, 255, 0), 3)
            
            status_text = 'TRACKING' if ball.last_seen == 0 else f'PREDICTING ({ball.last_seen})'
            cv2.putText(result_frame, status_text, (current_pos[0]+15, current_pos[1]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 軌跡を描画
        self._draw_trajectory(result_frame)
        
        # 情報を表示
        self._draw_info(result_frame, len(player_detections))
        
        return result_frame
    
    def _draw_players(self, frame: np.ndarray, 
                     player_detections: List[Tuple[int, int, int, int, int, float]]):
        """プレイヤーを描画"""
        for x1, y1, x2, y2, class_id, confidence in player_detections:
            if class_id == Config.PLAYER_FRONT_CLASS_ID:
                color = (255, 0, 0)
                label = "Player Front"
            elif class_id == Config.PLAYER_BACK_CLASS_ID:
                color = (0, 0, 255)
                label = "Player Back"
            else:
                continue
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{label}: {confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_trajectory(self, frame: np.ndarray):
        """軌跡を描画"""
        if len(self.ball_trajectory) > 1:
            points = np.array(list(self.ball_trajectory), np.int32)
            
            # 軌跡を線で描画
            for i in range(1, len(points)):
                alpha = i / len(points)
                thickness = max(1, int(3 * alpha))
                color_intensity = int(255 * alpha)
                color = (0, color_intensity, 0)
                cv2.line(frame, tuple(points[i-1]), tuple(points[i]), color, thickness)
            
            # 軌跡上の点を描画
            for i, point in enumerate(points):
                alpha = (i + 1) / len(points)
                radius = max(2, int(4 * alpha))
                color_intensity = int(255 * alpha)
                color = (0, color_intensity, 0)
                cv2.circle(frame, tuple(point), radius, color, -1)
    
    def _draw_info(self, frame: np.ndarray, players_count: int):
        """情報を表示"""
        confidence_threshold = self.get_dynamic_confidence_threshold()
        info_text = [
            f"Candidate balls: {len(self.candidate_balls)}",
            f"Active ball: {'Yes' if self.active_ball else 'No'}",
            f"Trajectory points: {len(self.ball_trajectory)}",
            f"Players detected: {players_count}",
            f"Confidence threshold: {confidence_threshold:.2f}",
            f"Disappeared count: {self.disappeared_count}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

class VideoProcessor:
    """動画処理管理クラス"""
    
    def __init__(self, tracker: TennisBallTracker):
        self.tracker = tracker
    
    def process_video(self, video_path: str, output_settings: dict):
        """動画を処理"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: ビデオファイルを開けませんでした")
            return
        
        # ビデオ情報取得
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 出力設定
        output_video_writer = self._setup_video_writer(output_settings, fps, width, height)
        
        print(f"処理開始 - FPS: {fps}, 解像度: {width}x{height}")
        if total_frames > 0:
            print(f"総フレーム数: {total_frames}")
        
        frame_count = 0
        start_time = datetime.now()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # フレーム処理
                if output_settings.get('training_data_only', False):
                    # 学習データのみモード
                    if self.tracker.should_process_frame(frame_count):
                        _ = self.tracker.process_frame(frame)
                else:
                    # 通常モード
                    result_frame, was_processed = self.tracker.process_frame_with_skip(frame, frame_count)
                    
                    # 動画保存
                    if output_video_writer:
                        output_video_writer.write(result_frame)
                    
                    # リアルタイム表示
                    if output_settings.get('show_realtime', False):
                        cv2.imshow('Tennis Ball Tracking', result_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("ユーザーによって処理が中断されました")
                            break
                
                frame_count += 1
                
                # 進捗表示
                self._show_progress(frame_count, total_frames, start_time, output_settings)
        
        finally:
            cap.release()
            if output_video_writer:
                output_video_writer.release()
            if output_settings.get('show_realtime', False):
                cv2.destroyAllWindows()
            
            self._finalize_processing(video_path, output_settings, frame_count, start_time)
    
    def _setup_video_writer(self, output_settings: dict, fps: int, width: int, height: int):
        """ビデオライターを設定"""
        if output_settings.get('save_video', False):
            output_dir = output_settings.get('output_dir', 'output')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"tennis_tracking_{timestamp}.mp4")
            
            os.makedirs(output_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            print(f"動画出力ファイル: {output_path}")
            return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        return None
    
    def _show_progress(self, frame_count: int, total_frames: int, start_time: datetime, 
                      output_settings: dict):
        """進捗を表示"""
        if output_settings.get('training_data_only', False):
            if frame_count % (50 * self.tracker.frame_skip) == 0:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                fps_current = frame_count / elapsed_time if elapsed_time > 0 else 0
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"処理中... フレーム: {frame_count}/{total_frames if total_frames > 0 else '?'} "
                      f"({progress:.1f}%) | 表示速度: {fps_current:.1f} FPS")
        elif frame_count % 100 == 0:
            processed_count = frame_count // self.tracker.frame_skip
            print(f"表示フレーム: {frame_count} | 処理フレーム: {processed_count}")
    
    def _finalize_processing(self, video_path: str, output_settings: dict, 
                           frame_count: int, start_time: datetime):
        """処理の最終化"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # データ保存
        if output_settings.get('save_time_series', False):
            csv_path = output_settings.get('csv_output_path')
            if csv_path:
                self.tracker.save_time_series_data(csv_path)
        
        if output_settings.get('save_training_data', False):
            video_name = Path(video_path).stem
            self.tracker.save_tracking_features(video_name)
        
        # 結果表示
        print(f"\n=== 処理完了 ===")
        processed_frame_count = frame_count // self.tracker.frame_skip
        print(f"総表示フレーム数: {frame_count}")
        print(f"実際の処理フレーム数: {processed_frame_count}")
        if self.tracker.frame_skip > 1:
            print(f"フレームスキップ効果: 約{self.tracker.frame_skip}倍の処理速度向上")
        print(f"処理時間: {processing_time:.2f}秒")
        print(f"平均表示速度: {frame_count / processing_time:.2f} FPS")

def get_user_preferences() -> Tuple[str, int, int, dict]:
    """ユーザーの設定を取得"""
    model_path = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/models/weights/best_5_31.pt"
    
    # 画像サイズ選択
    print("推論時の画像サイズを選択してください:")
    print("1. 1920 (高精度・低速)")
    print("2. 1280 (バランス)")
    print("3. 640 (高速・低精度)")
    
    imgsz_options = {1: 1920, 2: 1280, 3: 640}
    while True:
        try:
            choice = int(input("画像サイズを選択 (1-3): "))
            if choice in imgsz_options:
                inference_imgsz = imgsz_options[choice]
                break
            print("1, 2, または 3 を入力してください。")
        except ValueError:
            print("数字を入力してください。")
    
    # フレームスキップ選択
    print("\nフレーム処理モードを選択してください:")
    print("1. 全フレーム処理（通常モード）")
    print("2. 2フレームに1回処理（2倍高速・補間あり）")
    print("3. 3フレームに1回処理（3倍高速・補間あり）")
    print("4. 5フレームに1回処理（5倍高速・補間あり）")
    
    frame_skip_options = {1: 1, 2: 2, 3: 3, 4: 5}
    while True:
        try:
            choice = int(input("処理モードを選択 (1-4): "))
            if choice in frame_skip_options:
                frame_skip = frame_skip_options[choice]
                break
            print("1, 2, 3, または 4 を入力してください。")
        except ValueError:
            print("数字を入力してください。")
    
    # 出力モード選択
    print("\n出力モードを選択してください:")
    print("1. 動画保存モード（結果を動画ファイルに保存）")
    print("2. リアルタイム表示モード（リアルタイム表示のみ）")
    print("3. 学習用データ出力モード（高速・データのみ出力）")
    
    while True:
        try:
            mode = int(input("モードを選択 (1-3): "))
            if mode in [1, 2, 3]:
                break
            print("1, 2, または 3 を入力してください。")
        except ValueError:
            print("数字を入力してください。")
    
    # 出力設定を構築
    output_settings = {
        'save_video': (mode == 1),
        'show_realtime': (mode in [1, 2]),
        'training_data_only': (mode == 3),
        'save_training_data': (mode == 3),
        'save_time_series': (mode == 3),
        'output_dir': "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/output"
    }
    
    # 追加設定
    if mode != 3:
        # 時系列データ保存
        choice = int(input("\n時系列データ出力を行いますか？ (1: はい, 2: いいえ): "))
        output_settings['save_time_series'] = (choice == 1)
        
        # 学習用データ保存
        choice = int(input("学習用データ保存を行いますか？ (1: はい, 2: いいえ): "))
        output_settings['save_training_data'] = (choice == 1)
    
    # CSVパス設定
    if output_settings['save_time_series']:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_settings['csv_output_path'] = os.path.join(
            output_settings['output_dir'], f"tracking_data_{timestamp}.csv"
        )
    
    return model_path, inference_imgsz, frame_skip, output_settings

def main():
    """メイン関数"""
    # ユーザー設定を取得
    model_path, inference_imgsz, frame_skip, output_settings = get_user_preferences()
    
    # トラッカー初期化
    tracker = TennisBallTracker(
        model_path=model_path,
        imgsz=inference_imgsz,
        save_training_data=output_settings['save_training_data'],
        frame_skip=frame_skip
    )
    
    # 動画処理
    video_path = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/raw/output6.mp4"
    processor = VideoProcessor(tracker)
    processor.process_video(video_path, output_settings)

if __name__ == "__main__":
    main()