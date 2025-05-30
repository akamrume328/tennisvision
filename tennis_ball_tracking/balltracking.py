import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import deque
from typing import List, Tuple, Optional

class TennisBallTracker:
    def __init__(self, model_path: str):
        """
        テニスボールトラッカーの初期化
        
        Args:
            model_path: YOLOv8モデルのパス
        """
        self.model = YOLO(model_path)
        self.tennis_ball_class_id = 2  # tennis_ballのクラスID
        
        # トラッキング関連のパラメータ
        self.max_disappeared = 15  # ボールが見えなくなってから削除するまでのフレーム数（増加）
        self.max_distance = 200  # 最大マッチング距離（ピクセル）（増加）
        self.min_movement_threshold = 3  # 最小移動距離（静止判定用）（減少）
        self.physics_check_frames = 5  # 物理的チェックに使用するフレーム数
        self.max_velocity_change = 80  # 最大速度変化（物理的制約）（増加）
        
        # 動的な検出閾値
        self.base_confidence_threshold = 0.3  # 基本信頼度閾値（減少）
        self.high_confidence_threshold = 0.5  # 高信頼度閾値
        
        # 現在追跡中のボール
        self.active_ball = None
        self.ball_trajectory = deque(maxlen=50)  # 軌跡を保存（最大50ポイント）
        self.disappeared_count = 0
        
        # 候補ボールの管理
        self.candidate_balls = {}  # ID: {position_history, last_seen, movement_score, predicted_pos}
        self.next_id = 0
    
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
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """フレームを処理してテニスボールを追跡"""
        # 動的な信頼度閾値を取得
        confidence_threshold = self.get_dynamic_confidence_threshold()
        
        # YOLOv8で検出
        results = self.model(frame, verbose=False)
        
        # テニスボールの検出結果を抽出
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id == self.tennis_ball_class_id and confidence > confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        detections.append((center_x, center_y, confidence))
        
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
        
        # 結果を描画
        result_frame = self.draw_tracking_results(frame, detections)
        
        return result_frame
    
    def draw_tracking_results(self, frame: np.ndarray, detections: List[Tuple[int, int, float]]) -> np.ndarray:
        """追跡結果を描画"""
        result_frame = frame.copy()
        
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
            f"Confidence threshold: {confidence_threshold:.2f}",
            f"Disappeared count: {self.disappeared_count}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(result_frame, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_frame

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

def main():
    """メイン関数 - テスト用"""
    # モデルのパスを設定（適切なパスに変更してください）
    model_path = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/models/weights/best_5_25.pt"  # または他の適切なモデルパス
    
    # 処理モードの選択
    print("処理モードを選択してください:")
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
    
    # トラッカーを初期化
    tracker = TennisBallTracker(model_path)
    
    # ビデオファイルまたはカメラを開く
    video_path = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/raw/output4.mp4"  # または適切なビデオパス
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: ビデオファイルを開けませんでした")
        return
    
    # ビデオの情報を取得
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 出力ビデオの設定（保存モードの場合のみ）
    out = None
    output_path = None
    if save_video:
        output_path = "../data/output/tennis_ball_tracking_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"処理開始 - FPS: {fps}, 解像度: {width}x{height}")
    if save_video:
        print(f"出力ファイル: {output_path}")
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
        
        if save_video:
            print(f"処理完了 - 出力ファイル: {output_path}")
        else:
            print("処理完了 - リアルタイム表示モード")

if __name__ == "__main__":
    main()