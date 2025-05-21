import cv2
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1" # __pycache__ を作成しないようにする
import sys
from ultralytics import YOLO
from tqdm import tqdm # tqdm をインポート

# モデルのロード (グローバル)
MODEL_PATH = r"C:\\Users\\akama\\AppData\\Local\\Programs\\Python\\Python310\\python_file\\projects\\tennisvision\\models\\weights\\best.pt"
model = YOLO(MODEL_PATH)

# プロジェクトのルートディレクトリをsys.pathに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.tracking import ObjectTracker, draw_tracking_results

def initialize_resources(): # 関数名を変更し、detector を削除
    """テスト用のパスとトラッカーを初期化します."""
    video_path = "../data/raw/output.mp4"
    output_video_path = "data/processed/output_with_detections.mp4"
    
    # ディレクトリが存在しない場合に作成
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # グローバル 'model' を使用して ObjectTracker を初期化
    tracker = ObjectTracker(model) 
    return video_path, output_video_path, tracker

def detect_and_draw_on_video(video_path, output_video_path, current_tracker): # detector 引数を削除
    """動画から物体を検出し、結果を描画して保存します."""
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 総フレーム数を取得

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # frame_count = 0 # tqdm を使うので不要
    # max_frames_to_process = 50  # 処理するフレーム数を制限 <- この行をコメントアウト

    # tqdm を使用して進捗バーを表示
    with tqdm(total=total_frames, desc="動画処理中") as pbar:
        while cap.isOpened(): # frame_count < max_frames_to_process の条件を削除
            ret, frame = cap.read()
            if not ret:
                break

            # detector.detect_objects(frame) の呼び出しを削除
            # ObjectTracker が内部でグローバルモデルを使用して検出と追跡を行うと想定
            tracked_objects = current_tracker.track_objects(frame)
            
            frame_with_detections = draw_tracking_results(frame.copy(), tracked_objects)

            out.write(frame_with_detections)
            # frame_count += 1 # tqdm がカウント
            # print(f"処理中フレーム: {frame_count}") # tqdm が表示するので削除
            pbar.update(1) # 進捗バーを更新


    cap.release()
    out.release()
    cv2.destroyAllWindows() # ウィンドウを閉じる処理を追加

    if not os.path.exists(output_video_path):
        print(f"エラー: 出力動画ファイルが作成されませんでした: {output_video_path}")
    elif os.path.getsize(output_video_path) == 0:
        print("エラー: 出力動画ファイルが空です。")
    else:
        print(f"処理完了。出力動画: {output_video_path}")

    # (オプション) テスト後に生成された動画ファイルを削除する場合はコメントアウトを解除
    # if os.path.exists(output_video_path):
    #     os.remove(output_video_path)

if __name__ == '__main__':
    # detector を削除し、tracker_instance を受け取るように変更
    video_path, output_video_path, tracker_instance = initialize_resources()
    # detector を渡さないように変更
    detect_and_draw_on_video(video_path, output_video_path, tracker_instance)
