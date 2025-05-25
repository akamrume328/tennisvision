import cv2
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1" # __pycache__ を作成しないようにする
import sys
from ultralytics import YOLO
from tqdm import tqdm # tqdm をインポート
import numpy as np # numpy をインポート

# モデルのロード (グローバル)
MODEL_PATH = r"C:\\Users\\akama\\AppData\\Local\\Programs\\Python\\Python310\\python_file\\projects\\tennisvision\\models\\weights\\best_5_25.pt"
model = YOLO(MODEL_PATH)
IMG_SIZE = 1920 # ★★★ 推論時の入力画像サイズを指定 (例: 640, 1280, 1920) ★★★
# CONF_THRESHOLD はモデル推論時に直接指定するため、ここでは不要

# プロジェクトのルートディレクトリをsys.pathに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# from src.tracking import ObjectTracker, draw_tracking_results # 既存のトラッカーを削除

# --- 簡易トラッキングクラス (preannotation.py からコピー) ---
class SimpleTracker:
    def __init__(self, max_distance=100, max_age=30): # 値は適宜調整
        self.tracks = {}  # {track_id: {'bbox': [x, y, w, h], 'class_id': int, 'age': int, 'confidence': float}}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_age = max_age
    
    def calculate_distance(self, bbox1, bbox2):
        """2つのバウンディングボックスの中心点間の距離を計算"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def update(self, detections):
        """
        検出結果でトラックを更新
        detections: list of {'bbox': [x, y, w, h], 'class_id': int, 'confidence': float}
        returns: list of {'bbox': [x, y, w, h], 'class_id': int, 'confidence': float, 'track_id': int}
        """
        matched_tracks_output = []
        used_detections_indices = set()
        
        # 既存のトラックと新しい検出をマッチング
        for track_id, track_data in list(self.tracks.items()):
            best_match_detection = None
            min_distance = self.max_distance
            best_detection_idx = -1
            
            for i, det in enumerate(detections):
                if i in used_detections_indices:
                    continue
                # クラスが異なる場合はマッチングしない (オプション)
                # if det['class_id'] != track_data['class_id']:
                #     continue
                    
                distance = self.calculate_distance(track_data['bbox'], det['bbox'])
                if distance < min_distance:
                    min_distance = distance
                    best_match_detection = det
                    best_detection_idx = i
            
            if best_match_detection:
                self.tracks[track_id] = {
                    'bbox': best_match_detection['bbox'],
                    'class_id': best_match_detection['class_id'],
                    'age': 0,
                    'confidence': best_match_detection['confidence']
                }
                matched_tracks_output.append({**self.tracks[track_id], 'track_id': track_id})
                used_detections_indices.add(best_detection_idx)
            else:
                self.tracks[track_id]['age'] += 1
        
        # 古いトラックを削除
        self.tracks = {tid: tdata for tid, tdata in self.tracks.items() if tdata['age'] < self.max_age}
        
        # マッチしなかった新しい検出に新しいIDを割り当て
        for i, det in enumerate(detections):
            if i not in used_detections_indices:
                new_track_id = self.next_id
                self.next_id += 1
                self.tracks[new_track_id] = {
                    'bbox': det['bbox'],
                    'class_id': det['class_id'],
                    'age': 0,
                    'confidence': det['confidence']
                }
                matched_tracks_output.append({**self.tracks[new_track_id], 'track_id': new_track_id})
        
        return matched_tracks_output

# --- 描画関数 (preannotation.py からコピーし調整) ---
def draw_simple_tracking_results(frame, tracks, line_thickness=2, font_scale=0.5):
    """SimpleTrackerからのトラッキング結果をフレームに描画"""
    for track in tracks:
        x, y, w, h = map(int, track['bbox']) # 座標を整数に
        track_id = track['track_id']
        confidence = track['confidence']
        class_id = track['class_id']
        
        # model.names を使用してクラス名を取得
        class_name = model.names.get(class_id, 'Unknown') 
        
        # クラスに応じた色を設定 (例)
        color = (0, 255, 0) # デフォルトは緑
        if "ball" in class_name.lower():
            color = (0, 0, 255)  # ボールは赤
        elif "player" in class_name.lower(): # player_front, player_back などを想定
            color = (255, 0, 0)  # プレイヤーは青

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, line_thickness)
        
        label = f'ID:{track_id} {class_name} {confidence:.2f}'
        
        # ラベルの背景とテキストを描画
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
        cv2.rectangle(frame, (x, y - label_height - baseline), (x + label_width, y), color, cv2.FILLED)
        cv2.putText(frame, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), line_thickness)
    
    return frame


def initialize_resources(enable_tracking=True):
    """テスト用のパスとトラッカーを初期化します."""
    video_path = "../data/raw/output3.mp4"
    # ファイル名サフィックスをトラッキングの有無に応じて変更
    # enable_tracking はファイル名にのみ影響し、SimpleTracker は常に有効
    output_video_filename_suffix = "simple_tracked" if enable_tracking else "detected_only_placeholder" 
    output_video_path = f"data/processed/output_1920_{output_video_filename_suffix}.mp4"
    
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # SimpleTracker を初期化
    tracker = SimpleTracker(max_distance=150, max_age=20) # パラメータは適宜調整
    return video_path, output_video_path, tracker

def detect_and_draw_on_video(video_path, output_video_path, current_tracker):
    """動画から物体を検出し、SimpleTrackerで追跡し、結果を描画して保存します."""
    if not os.path.exists(video_path):
        print(f"エラー: テスト動画ファイルが見つかりません: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("エラー: 動画ファイルを開けませんでした。")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    with tqdm(total=total_frames, desc="動画処理中 (SimpleTracker)") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOモデルで推論 (conf はここで指定)
            results = model(frame, imgsz=IMG_SIZE, conf=0.25, verbose=False) # confを適切に設定

            detections_for_tracker = []
            if results and results[0] and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes.cls) if boxes.cls is not None else 0):
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])
                    
                    # バウンディングボックスを xywh 形式 (左上x, 左上y, 幅, 高さ) で取得
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    x_abs, y_abs, w_abs, h_abs = x1, y1, (x2 - x1), (y2 - y1)
                    
                    detections_for_tracker.append({
                        'bbox': [x_abs, y_abs, w_abs, h_abs],
                        'class_id': class_id,
                        'confidence': confidence
                    })
            
            # SimpleTrackerで追跡
            tracked_objects = current_tracker.update(detections_for_tracker)
            
            # 描画
            frame_with_tracks = draw_simple_tracking_results(frame.copy(), tracked_objects)

            out.write(frame_with_tracks)
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if not os.path.exists(output_video_path):
        print(f"エラー: 出力動画ファイルが作成されませんでした: {output_video_path}")
    elif os.path.getsize(output_video_path) == 0:
        print("エラー: 出力動画ファイルが空です。")
    else:
        print(f"処理完了。出力動画: {output_video_path}")

if __name__ == '__main__':
    # enable_tracking は initialize_resources でのファイル名生成にのみ影響
    # SimpleTracker は常に使用されます。
    tracking_enabled_for_filename = True 

    print(f"モデルパス: {MODEL_PATH}")
    print(f"推論時の入力画像サイズ (imgsz): {IMG_SIZE}")
    print(f"ファイル名にトラッキング情報を含む: {tracking_enabled_for_filename}")
    print(f"使用トラッカー: SimpleTracker (距離ベース)")

    video_path, output_video_path, tracker_instance = initialize_resources(enable_tracking=tracking_enabled_for_filename)
    detect_and_draw_on_video(video_path, output_video_path, tracker_instance)
