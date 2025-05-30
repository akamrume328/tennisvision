import numpy as np
import cv2
from collections import deque
import math

class TennisBallTracker:
    def __init__(self, max_distance=200, history_length=8, prediction_frames=5):
        """
        テニスボール専用の軌道予測追跡器（単一ボール最適化版）
        """
        self.tracks = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.history_length = history_length
        self.prediction_frames = prediction_frames
        self.frame_count = 0
        self.lost_tracks = {}
        
        # 単一ボール制約用パラメータ
        self.single_ball_mode = True  # ラリー中は一つのボールのみ
        self.max_active_tracks = 1  # 同時にアクティブなトラック数の上限
        self.detection_confidence_boost = 0.2  # 単一検出時の信頼度ブースト
        
        # 検出不安定対応のパラメータ
        self.min_track_length = 2  # さらに短縮（単一ボールなので早期表示可能）
        self.confidence_threshold = 0.05  # さらに低い閾値（単一ボールなので緩く）
        
    class Track:
        def __init__(self, track_id, center, score, class_id):
            self.id = track_id
            self.centers = deque(maxlen=25)  # より長い履歴
            self.centers.append(center)
            self.scores = deque(maxlen=25)
            self.scores.append(score)
            self.class_id = class_id
            self.last_seen = 0
            self.velocity = (0, 0)
            self.acceleration = (0, 0)
            self.predicted_center = center
            self.lost_count = 0
            self.max_lost_frames = 120  # より長期間の再追跡（4秒）
            self.creation_frame = 0
            
            # 物理的妥当性チェック用のパラメータ（単一ボール用に調整）
            self.consecutive_invalid_moves = 0
            self.max_invalid_moves = 2  # より厳格に（誤検出を早期除外）
            self.teleport_threshold = 200  # 少し緩和（単一ボールなので追跡を優先）
            self.max_speed_change = 40  # 少し緩化
            self.direction_change_threshold = math.pi * 0.8  # 少し緩化
            
            # 単一ボール用の品質指標
            self.continuity_score = 1.0  # 軌跡の連続性スコア
            self.detection_consistency = 1.0  # 検出の一貫性
            
        def update(self, center, score, frame_count):
            """新しい検出で軌道を更新（物理的妥当性チェック付き）"""
            # 物理的妥当性をチェック
            if len(self.centers) > 0:
                if not self._is_physically_valid_move(center):
                    self.consecutive_invalid_moves += 1
                    # 連続して不正な動きが続く場合は更新を拒否
                    if self.consecutive_invalid_moves >= self.max_invalid_moves:
                        return False  # 更新失敗を示す
                    # 軽微な不正動きの場合は警告程度で続行
                else:
                    self.consecutive_invalid_moves = 0  # 正常な動きなのでリセット
            
            self.centers.append(center)
            self.scores.append(score)
            self.last_seen = frame_count
            self.lost_count = 0
            self._calculate_motion()
            return True  # 更新成功を示す
            
        def _calculate_motion(self):
            """速度と加速度を計算（テニスボール用に最適化）"""
            if len(self.centers) < 2:
                return
                
            # 速度計算（直近2点から）
            curr = self.centers[-1]
            prev = self.centers[-2]
            self.velocity = (curr[0] - prev[0], curr[1] - prev[1])
            
            # 加速度計算（より多くの点から安定した計算）
            if len(self.centers) >= 4:
                # 直近3点の平均速度変化を使用
                velocities = []
                for i in range(len(self.centers) - 1):
                    v = (self.centers[i+1][0] - self.centers[i][0], 
                         self.centers[i+1][1] - self.centers[i][1])
                    velocities.append(v)
                
                if len(velocities) >= 2:
                    recent_v = velocities[-1]
                    prev_v = velocities[-2]
                    self.acceleration = (recent_v[0] - prev_v[0], recent_v[1] - prev_v[1])
        
        def predict_position(self, steps_ahead=1):
            """テニスボール用の物理法則予測（安定性向上版）"""
            if len(self.centers) == 0:
                return self.predicted_center
                
            curr_pos = self.centers[-1]
            
            # 検出履歴が少ない場合は保守的な予測
            if len(self.centers) < 3:
                # 履歴が少ない場合は単純な線形予測のみ
                if len(self.centers) >= 2:
                    # 直近の速度で予測（減衰あり）
                    decay = 0.9 ** steps_ahead  # 予測距離に応じて減衰
                    pred_x = curr_pos[0] + self.velocity[0] * steps_ahead * decay
                    pred_y = curr_pos[1] + self.velocity[1] * steps_ahead * decay
                else:
                    # 履歴が1点のみの場合は現在位置を返す
                    pred_x, pred_y = curr_pos
                
                self.predicted_center = (int(pred_x), int(pred_y))
                return self.predicted_center
            
            # 速度の信頼性チェック
            speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
            
            # 異常に高速な場合は予測を制限
            max_reasonable_speed = 50  # 1フレームあたりの最大移動距離
            if speed > max_reasonable_speed:
                # 速度を制限
                scale = max_reasonable_speed / speed
                limited_velocity = (self.velocity[0] * scale, self.velocity[1] * scale)
            else:
                limited_velocity = self.velocity
            
            # 重力と空気抵抗の設定（検出履歴に応じて調整）
            confidence_factor = min(1.0, len(self.centers) / 5.0)  # 履歴が多いほど信頼度高
            
            # テニスボール用の重力（履歴が少ない時は弱く）
            gravity = 0.3 + 0.5 * confidence_factor  # 0.3〜0.8の範囲
            
            # 空気抵抗による減速効果（履歴が少ない時は強く）
            air_resistance = 0.95 + 0.03 * confidence_factor  # 0.95〜0.98の範囲
            
            # 加速度の制限
            max_acceleration = 5.0  # 最大加速度
            limited_acceleration = (
                max(-max_acceleration, min(max_acceleration, self.acceleration[0])),
                max(-max_acceleration, min(max_acceleration, self.acceleration[1]))
            )
            
            # 予測計算（保守的）
            effective_velocity = (
                limited_velocity[0] * (air_resistance ** steps_ahead),
                limited_velocity[1] * (air_resistance ** steps_ahead)
            )
            
            # 予測距離の制限
            max_prediction_distance = 100  # 最大予測距離
            prediction_distance = math.sqrt(
                (effective_velocity[0] * steps_ahead)**2 + 
                (effective_velocity[1] * steps_ahead)**2
            )
            
            if prediction_distance > max_prediction_distance:
                scale = max_prediction_distance / prediction_distance
                effective_velocity = (
                    effective_velocity[0] * scale,
                    effective_velocity[1] * scale
                )
                limited_acceleration = (
                    limited_acceleration[0] * scale,
                    limited_acceleration[1] * scale
                )
            
            pred_x = curr_pos[0] + effective_velocity[0] * steps_ahead + 0.5 * limited_acceleration[0] * steps_ahead**2
            pred_y = curr_pos[1] + effective_velocity[1] * steps_ahead + 0.5 * (limited_acceleration[1] + gravity) * steps_ahead**2
            
            # 画面境界内に制限（オプション - 必要に応じて）
            # pred_x = max(0, min(frame_width, pred_x))
            # pred_y = max(0, min(frame_height, pred_y))
            
            self.predicted_center = (int(pred_x), int(pred_y))
            return self.predicted_center
        
        def get_trajectory_confidence(self):
            """軌道の信頼度（単一ボール最適化版）"""
            if len(self.centers) < 2:
                return 0.3  # 単一ボールなので初期信頼度を上げる
            elif len(self.centers) < 3:
                return 0.5
            
            # 基本信頼度（履歴の長さベース）
            base_confidence = min(0.9, len(self.centers) * 0.08)
            
            # 単一ボール特有のボーナス
            single_ball_bonus = 0.2  # 単一ボール前提でのボーナス
            
            # 連続性ボーナス
            continuity_bonus = self.continuity_score * 0.15
            
            # 速度の一貫性をチェック（より緩く）
            if len(self.centers) >= 3:
                speeds = []
                for i in range(len(self.centers) - 1):
                    dx = self.centers[i+1][0] - self.centers[i][0]
                    dy = self.centers[i+1][1] - self.centers[i][1]
                    speed = math.sqrt(dx*dx + dy*dy)
                    speeds.append(speed)
                
                if speeds:
                    max_speed = max(speeds)
                    
                    if max_speed <= 70:  # 閾値をさらに緩和
                        speed_variance = np.var(speeds)
                        speed_consistency = max(0.3, 1.0 / (1.0 + speed_variance / 60))
                        
                        confidence = min(0.95, base_confidence + single_ball_bonus + 
                                       continuity_bonus + speed_consistency)
                        return confidence
                    else:
                        return max(0.4, base_confidence + single_ball_bonus * 0.5)
            
            return max(0.4, base_confidence + single_ball_bonus)
        
        def update_quality_metrics(self):
            """単一ボール用の品質指標を更新"""
            if len(self.centers) >= 3:
                # 連続性スコア: 急激な変化が少ないほど高い
                direction_changes = 0
                speed_changes = 0
                
                for i in range(2, len(self.centers)):
                    # 方向変化をチェック
                    v1 = (self.centers[i-1][0] - self.centers[i-2][0], 
                          self.centers[i-1][1] - self.centers[i-2][1])
                    v2 = (self.centers[i][0] - self.centers[i-1][0], 
                          self.centers[i][1] - self.centers[i-1][1])
                    
                    if (v1[0] != 0 or v1[1] != 0) and (v2[0] != 0 or v2[1] != 0):
                        angle1 = math.atan2(v1[1], v1[0])
                        angle2 = math.atan2(v2[1], v2[0])
                        angle_diff = abs(angle2 - angle1)
                        if angle_diff > math.pi:
                            angle_diff = 2 * math.pi - angle_diff
                        
                        if angle_diff > math.pi / 3:  # 60度以上の変化
                            direction_changes += 1
                    
                    # 速度変化をチェック
                    speed1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    speed2 = math.sqrt(v2[0]**2 + v2[1]**2)
                    if abs(speed2 - speed1) > 20:  # 大きな速度変化
                        speed_changes += 1
                
                # 連続性スコアを計算
                total_segments = len(self.centers) - 2
                if total_segments > 0:
                    self.continuity_score = max(0.0, 1.0 - 
                                              (direction_changes + speed_changes) / (total_segments * 2))
                else:
                    self.continuity_score = 1.0

        def _is_physically_valid_move(self, new_center):
            """新しい位置への移動が物理的に妥当かチェック"""
            if len(self.centers) == 0:
                return True
                
            current_pos = self.centers[-1]
            distance = math.sqrt((new_center[0] - current_pos[0])**2 + 
                               (new_center[1] - current_pos[1])**2)
            
            # 1. テレポートチェック（あまりに遠い移動）
            if distance > self.teleport_threshold:
                return False
            
            # 2. 速度変化チェック（履歴が2点以上ある場合）
            if len(self.centers) >= 2:
                prev_pos = self.centers[-2]
                prev_distance = math.sqrt((current_pos[0] - prev_pos[0])**2 + 
                                        (current_pos[1] - prev_pos[1])**2)
                
                speed_change = abs(distance - prev_distance)
                if speed_change > self.max_speed_change:
                    return False
            
            # 3. 方向転換チェック（履歴が3点以上ある場合）
            if len(self.centers) >= 2:
                prev_pos = self.centers[-2]
                
                # 前の移動ベクトル
                prev_vector = (current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])
                # 新しい移動ベクトル
                new_vector = (new_center[0] - current_pos[0], new_center[1] - current_pos[1])
                
                # ベクトルの角度差を計算
                if (prev_vector[0] != 0 or prev_vector[1] != 0) and (new_vector[0] != 0 or new_vector[1] != 0):
                    prev_angle = math.atan2(prev_vector[1], prev_vector[0])
                    new_angle = math.atan2(new_vector[1], new_vector[0])
                    angle_diff = abs(new_angle - prev_angle)
                    
                    # 角度差を0〜πの範囲に正規化
                    if angle_diff > math.pi:
                        angle_diff = 2 * math.pi - angle_diff
                    
                    # 急激な方向転換をチェック
                    if angle_diff > self.direction_change_threshold:
                        return False
            
            return True
        
        def _has_oscillating_pattern(self):
            """振動パターン（往復運動）を検出"""
            if len(self.centers) < 6:  # 最低6点必要（3往復を検出するため）
                return False
                
            # 直近6点での振動パターンをチェック
            recent_centers = list(self.centers)[-6:]
            
            # 各軸での振動をチェック
            x_coords = [pos[0] for pos in recent_centers]
            y_coords = [pos[1] for pos in recent_centers]
            
            # X軸での振動パターン検出
            x_oscillation = self._detect_oscillation(x_coords)
            y_oscillation = self._detect_oscillation(y_coords)
            
            return x_oscillation or y_oscillation
        
        def _detect_oscillation(self, coords):
            """座標列での振動パターンを検出"""
            if len(coords) < 6:
                return False
                
            # 方向変化を記録
            direction_changes = 0
            for i in range(1, len(coords) - 1):
                # 前後の方向をチェック
                dir1 = 1 if coords[i] > coords[i-1] else -1
                dir2 = 1 if coords[i+1] > coords[i] else -1
                
                if dir1 != dir2:  # 方向変化
                    direction_changes += 1
            
            # 短期間で多くの方向変化がある場合は振動パターン
            return direction_changes >= 3
        
        def is_tennis_ball_trajectory(self):
            """テニスボールらしい軌道かどうかを判定（物理的妥当性強化版）"""
            if len(self.centers) < 2:
                return True
            
            # 基本的な速度チェック
            speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
            if speed < 1:
                return len(self.centers) >= 2
            elif speed > 80:
                return False
            
            # 振動パターンチェック
            if self._has_oscillating_pattern():
                return False
            
            # 連続不正動きチェック
            if self.consecutive_invalid_moves >= 2:
                return False
            
            # 軌跡の一貫性チェック（総移動距離 vs 直線距離）
            if len(self.centers) >= 4:
                total_distance = 0
                for i in range(1, len(self.centers)):
                    dx = self.centers[i][0] - self.centers[i-1][0]
                    dy = self.centers[i][1] - self.centers[i-1][1]
                    total_distance += math.sqrt(dx*dx + dy*dy)
                
                # 開始点と終了点の直線距離
                start_pos = self.centers[0]
                end_pos = self.centers[-1]
                straight_distance = math.sqrt(
                    (end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2
                )
                
                # 効率性比率（直線距離/総移動距離）
                if straight_distance > 0 and total_distance > 0:
                    efficiency_ratio = straight_distance / total_distance
                    # あまりに非効率な軌跡（ジグザグすぎる）は除外
                    if efficiency_ratio < 0.3 and len(self.centers) >= 6:
                        return False
            
            # 既存の物理的妥当性チェック
            if len(self.centers) >= 4:
                recent_directions = []
                recent_speeds = []
                
                for i in range(len(self.centers) - 3):
                    dx1 = self.centers[i+1][0] - self.centers[i][0]
                    dy1 = self.centers[i+1][1] - self.centers[i][1]
                    dx2 = self.centers[i+2][0] - self.centers[i+1][0]
                    dy2 = self.centers[i+2][1] - self.centers[i+1][1]
                    
                    speed1 = math.sqrt(dx1*dx1 + dy1*dy1)
                    speed2 = math.sqrt(dx2*dx2 + dy2*dy2)
                    recent_speeds.extend([speed1, speed2])
                    
                    if dx1 != 0 or dy1 != 0:
                        angle_change = math.atan2(dy2, dx2) - math.atan2(dy1, dx1)
                        recent_directions.append(abs(angle_change))
                
                # 急激な方向転換をチェック（より厳格に）
                if recent_directions and max(recent_directions) > math.pi / 2:
                    return len(self.centers) >= 8  # より長い履歴が必要
                
                # 速度の急激な変化をチェック
                if recent_speeds:
                    speed_variance = np.var(recent_speeds)
                    if speed_variance > 200:
                        return len(self.centers) >= 8
            
            return True

    def update(self, detections):
        """
        テニスボール検出結果で追跡を更新（単一ボール最適化版）
        
        Args:
            detections: [(center_x, center_y, score, class_id), ...]
                       tennis_ballクラスのもののみ処理
        
        Returns:
            tracks: [Track, ...] アクティブな追跡結果
        """
        self.frame_count += 1
        
        # 単一ボール制約の適用
        if self.single_ball_mode and len(detections) > 1:
            # 複数検出がある場合、最も信頼度の高いものを選択
            best_detection = max(detections, key=lambda x: x[2])  # scoreで選択
            detections = [best_detection]
        
        tennis_ball_detections = []
        for detection in detections:
            if len(detection) >= 4:
                tennis_ball_detections.append(detection)
        
        # 既存トラックの予測
        for track in self.tracks.values():
            track.predict_position()
        
        for track in self.lost_tracks.values():
            track.predict_position()
            track.lost_count += 1
        
        # 単一ボール用の距離調整
        current_max_distance = self.max_distance
        
        # 単一ボールなので、より積極的なマッチング
        if len(tennis_ball_detections) == 1:
            current_max_distance = self.max_distance * 3.0  # 大幅に距離を緩和
        elif len(tennis_ball_detections) == 0:
            # 検出なしの場合でも既存トラックは維持
            pass
        
        matched_tracks = set()
        matched_detections = set()
        
        # マッチング処理（単一ボール最適化）
        for i, detection in enumerate(tennis_ball_detections):
            center = (detection[0], detection[1])
            score = detection[2]
            best_track = None
            best_distance = float('inf')
            best_source = None
            
            # アクティブトラックから検索
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                pred_pos = track.predicted_center
                distance = math.sqrt((center[0] - pred_pos[0])**2 + (center[1] - pred_pos[1])**2)
                
                if distance < current_max_distance and distance < best_distance:
                    best_distance = distance
                    best_track = track
                    best_source = 'active'
            
            # 見失ったトラックからも検索（より積極的に）
            if best_track is None or best_distance > current_max_distance * 0.5:
                for track_id, track in self.lost_tracks.items():
                    if track_id in matched_tracks:
                        continue
                    
                    pred_pos = track.predicted_center
                    distance = math.sqrt((center[0] - pred_pos[0])**2 + (center[1] - pred_pos[1])**2)
                    
                    # 単一ボールなので非常に緩い再追跡条件
                    max_retrack_distance = current_max_distance * 4.0
                    if distance < max_retrack_distance and distance < best_distance:
                        best_distance = distance
                        best_track = track
                        best_source = 'lost'
            
            # マッチング処理
            if best_track is not None:
                update_success = best_track.update(center, score, self.frame_count)
                if update_success:
                    # 品質指標を更新
                    best_track.update_quality_metrics()
                    
                    matched_tracks.add(best_track.id)
                    matched_detections.add(i)
                    
                    # 見失ったトラックの復活
                    if best_source == 'lost' and best_track.id in self.lost_tracks:
                        self.tracks[best_track.id] = self.lost_tracks[best_track.id]
                        del self.lost_tracks[best_track.id]
        
        # 新しいトラックの作成（単一ボール制約を考慮）
        for i, detection in enumerate(tennis_ball_detections):
            if i not in matched_detections:
                # 単一ボールモードでは、既存のアクティブトラックがある場合は新規作成を慎重に
                if len(self.tracks) >= self.max_active_tracks:
                    # 既存トラックの品質と比較
                    center = (detection[0], detection[1])
                    score = detection[2]
                    
                    # 既存トラックの中で最も品質の低いものを見つける
                    worst_track = None
                    worst_confidence = float('inf')
                    
                    for track in self.tracks.values():
                        confidence = track.get_trajectory_confidence()
                        if confidence < worst_confidence:
                            worst_confidence = confidence
                            worst_track = track
                    
                    # 新しい検出の方が有望な場合、既存の最悪トラックを置き換え
                    if worst_track and (score > worst_track.scores[-1] + 0.1 or 
                                      worst_confidence < 0.3):
                        # 既存トラックを削除
                        del self.tracks[worst_track.id]
                        
                        # 新しいトラックを作成
                        center = (detection[0], detection[1])
                        score = detection[2]
                        class_id = detection[3] if len(detection) > 3 else 0
                        new_track = self.Track(self.next_id, center, score, class_id)
                        self.tracks[self.next_id] = new_track
                        self.next_id += 1
                else:
                    # アクティブトラック数が上限未満なので新規作成
                    center = (detection[0], detection[1])
                    score = detection[2]
                    class_id = detection[3] if len(detection) > 3 else 0
                    new_track = self.Track(self.next_id, center, score, class_id)
                    self.tracks[self.next_id] = new_track
                    self.next_id += 1
        
        # トラック状態の更新
        tracks_to_move_to_lost = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                tracks_to_move_to_lost.append(track_id)
        
        for track_id in tracks_to_move_to_lost:
            track = self.tracks[track_id]
            track.lost_count = 1
            self.lost_tracks[track_id] = track
            del self.tracks[track_id]
        
        # 長時間見失ったトラックの削除
        tracks_to_remove = []
        for track_id, track in self.lost_tracks.items():
            if track.lost_count > track.max_lost_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.lost_tracks[track_id]
        
        # 有効トラックの選択（単一ボール最適化）
        valid_tracks = []
        
        # アクティブトラックから最も品質の高いものを選択
        if self.single_ball_mode and self.tracks:
            best_track = None
            best_quality = -1
            
            for track in self.tracks.values():
                if (track.is_tennis_ball_trajectory() and 
                    track.get_trajectory_confidence() > self.confidence_threshold and  
                    len(track.centers) >= self.min_track_length and
                    track.consecutive_invalid_moves < track.max_invalid_moves):
                    
                    # 品質スコアを計算
                    confidence = track.get_trajectory_confidence()
                    continuity = track.continuity_score
                    length_bonus = min(0.3, len(track.centers) * 0.02)
                    
                    quality = confidence + continuity + length_bonus
                    
                    if quality > best_quality:
                        best_quality = quality
                        best_track = track
            
            if best_track:
                valid_tracks.append(best_track)
        else:
            # 通常モード（複数トラック許可）
            for track in self.tracks.values():
                if (track.is_tennis_ball_trajectory() and 
                    track.get_trajectory_confidence() > self.confidence_threshold and  
                    len(track.centers) >= self.min_track_length and
                    track.consecutive_invalid_moves < track.max_invalid_moves):
                    valid_tracks.append(track)
        
        return valid_tracks

    def get_predicted_positions(self, steps_ahead=3):
        """全トラック予測位置を取得（安定性重視）"""
        predictions = {}
        
        # アクティブトラック（信頼度が高いもののみ予測表示）
        for track_id, track in self.tracks.items():
            confidence = track.get_trajectory_confidence()
            if confidence > 0.3:  # 信頼度が高い場合のみ予測表示
                predictions[track_id] = track.predict_position(min(steps_ahead, 3))  # 予測距離を制限
        
        # 見失ったトラック（さらに厳格に）
        for track_id, track in self.lost_tracks.items():
            if track.lost_count <= 10 and track.get_trajectory_confidence() > 0.4:  # より厳しい条件
                predictions[f"lost_{track_id}"] = track.predict_position(1)  # 1フレーム先のみ
        
        return predictions
    
    def get_all_tracks_info(self):
        """デバッグ用：全トラック情報を取得"""
        info = {
            'active_tracks': len(self.tracks),
            'lost_tracks': len(self.lost_tracks),
            'total_frame_count': self.frame_count
        }
        return info
