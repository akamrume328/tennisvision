import cv2
from ultralytics import YOLO
from simple_trajectory_tracker import TennisBallTracker
import numpy as np
import random
import math
from tqdm import tqdm

# --- 設定 ---
VIDEO_PATH = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/raw/output3.mp4"
MODEL_PATH = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/models/weights/best_5_26.pt"

# テニスボール専用追跡の設定（検出不安定対応）
MAX_TRACKING_DISTANCE = 350  # より大きな追跡距離
HISTORY_LENGTH = 12  # より長い履歴
PREDICTION_FRAMES = 8  # より長い予測

# 表示設定
DRAW_BOX = True
DRAW_TRACK_ID = True
DRAW_CLASS_NAME = True
DRAW_SCORE = True
DRAW_TRAJECTORY = True
DRAW_PREDICTION = True
DRAW_LOST_PREDICTIONS = True
DRAW_PLAYERS = True  # プレイヤーの表示（追跡なし）

OUTPUT_VIDEO_PATH = "output_tracked_tennis_ball_only.mp4"

def get_color(idx):
    """トラックIDに基づいて色を生成する"""
    random.seed(idx)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def get_player_color(class_name):
    """プレイヤークラスに基づいて固定色を返す"""
    if class_name == "player_front":
        return (0, 255, 0)  # 緑色
    elif class_name == "player_back":
        return (255, 0, 0)  # 青色
    else:
        return (128, 128, 128)  # グレー

def get_tennis_ball_color():
    """テニスボール用の固定色（追跡オフ時）"""
    return (0, 255, 255)  # 黄色

def main():
    # --- ユーザー入力による処理モード選択 ---
    while True:
        user_input_display = input("リアルタイムで処理結果を表示しますか？ (y/n): ").lower()
        if user_input_display in ['y', 'yes']:
            REALTIME_DISPLAY = True
            break
        elif user_input_display in ['n', 'no']:
            REALTIME_DISPLAY = False
            break
        else:
            print("無効な入力です。'y' または 'n' で答えてください。")

    while True:
        user_input_save = input("処理結果を動画ファイルとして保存しますか？ (y/n): ").lower()
        if user_input_save in ['y', 'yes']:
            SAVE_VIDEO = True
            break
        elif user_input_save in ['n', 'no']:
            SAVE_VIDEO = False
            break
        else:
            print("無効な入力です。'y' または 'n' で答えてください。")

    # 追跡機能のオン/オフ選択
    while True:
        user_input_tracking = input("追跡機能を使用しますか？ (y/n): ").lower()
        if user_input_tracking in ['y', 'yes']:
            ENABLE_TRACKING = True
            print("追跡機能がオンになりました。")
            break
        elif user_input_tracking in ['n', 'no']:
            ENABLE_TRACKING = False
            print("追跡機能がオフになりました。検出ボックスのみ表示します。")
            break
        else:
            print("無効な入力です。'y' または 'n' で答えてください。")

    # 表示モードの選択（追跡オンの場合のみ）
    TRAJECTORY_ONLY_MODE = False
    if ENABLE_TRACKING:
        while True:
            user_input_mode = input("表示モードを選択してください - (1) 通常表示, (2) 軌跡のみ表示: ").lower()
            if user_input_mode in ['1', 'normal', 'n']:
                TRAJECTORY_ONLY_MODE = False
                print("通常表示モードが選択されました。")
                break
            elif user_input_mode in ['2', 'trajectory', 't']:
                TRAJECTORY_ONLY_MODE = True
                print("軌跡のみ表示モードが選択されました。")
                break
            else:
                print("無効な入力です。'1' または '2' で答えてください。")
    
    # 軌跡のみモードの場合の設定調整
    if TRAJECTORY_ONLY_MODE:
        DRAW_BOX = False
        DRAW_TRACK_ID = False
        DRAW_CLASS_NAME = False
        DRAW_SCORE = False
        DRAW_PREDICTION = False
        DRAW_LOST_PREDICTIONS = False
        DRAW_PLAYERS = False
        print("軌跡のみモード: 動画の上に軌跡のみを表示します。")
    
    # --- ユーザー入力ここまで ---

    # 1. YOLOv8 モデルのロード
    try:
        model = YOLO(MODEL_PATH)
        class_names = model.names
        print(f"モデルロード完了。検出可能クラス: {class_names}")
        
        # クラス名とIDの対応を確認
        tennis_ball_class_id = None
        player_front_class_id = None
        player_back_class_id = None
        
        for class_id, class_name in class_names.items():
            if class_name.lower() == "tennis_ball":
                tennis_ball_class_id = class_id
            elif class_name.lower() == "player_front":
                player_front_class_id = class_id
            elif class_name.lower() == "player_back":
                player_back_class_id = class_id
        
        print(f"テニスボールクラスID: {tennis_ball_class_id}")
        print(f"プレイヤー前クラスID: {player_front_class_id}")
        print(f"プレイヤー後クラスID: {player_back_class_id}")
        
    except Exception as e:
        print(f"エラー: YOLOモデルのロードに失敗しました: {e}")
        return

    # 2. テニスボール専用追跡器の初期化（追跡オンの場合のみ）
    tracker = None
    if ENABLE_TRACKING:
        tracker = TennisBallTracker(
            max_distance=MAX_TRACKING_DISTANCE,
            history_length=HISTORY_LENGTH,
            prediction_frames=PREDICTION_FRAMES
        )

    # 3. 動画ファイルの読み込み
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けません: {VIDEO_PATH}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 総フレーム数を取得

    # (オプション) 動画書き出し設定
    out_video = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))


    # 4. 動画のフレームごとの処理ループ
    frame_id_counter = 0
    frame_original = None  # 元フレーム保存用の初期化
    
    # 進捗バーの制御
    frame_iterable = range(total_frames)
    if not REALTIME_DISPLAY and SAVE_VIDEO: # リアルタイム表示でなく、かつ動画保存する場合のみ進捗バーを表示
        frame_iterable = tqdm(frame_iterable, desc="Processing video")
    elif not REALTIME_DISPLAY and not SAVE_VIDEO: # 何もしないが、処理は実行する場合
        print("リアルタイム表示も動画保存も行いません。処理のみ実行します。")
        frame_iterable = tqdm(frame_iterable, desc="Processing (no display/save)")


    for _ in frame_iterable:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id_counter += 1
        
        # 元フレームを保存（軌跡のみモード用）
        frame_original = frame.copy()

        # 軌跡のみモードでは元の動画フレームをそのまま使用（黒背景にしない）
        # if TRAJECTORY_ONLY_MODE:
        #     # 元のフレームサイズを保持しつつ黒背景を作成
        #     frame = np.zeros_like(frame)

        # a. YOLOv8 で物体検出を実行（検出精度向上）
        try:
            results = model(frame_original, verbose=False, conf=0.03)  # 信頼度閾値を下げて検出を増やす
        except Exception as e:
            print(f"YOLO推論エラー: {e}")
            continue

        # b. 検出結果を分類
        tennis_ball_detections = []
        player_detections = []
        
        if results and results[0].boxes:
            for box in results[0].boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = class_names.get(class_id, f"class_{class_id}")

                    # テニスボール
                    if class_name.lower() == "tennis_ball":
                        if ENABLE_TRACKING:
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            tennis_ball_detections.append((center_x, center_y, score, class_id))
                        else:
                            tennis_ball_detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'score': score,
                                'class_id': class_id,
                                'class_name': class_name
                            })
                    
                    # プレイヤー（軌跡のみモードでは処理しない）
                    elif class_name.lower() in ["player_front", "player_back"] and not TRAJECTORY_ONLY_MODE:
                        player_detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'score': score,
                            'class_id': class_id,
                            'class_name': class_name
                        })
                except Exception as e:
                    print(f"検出結果処理エラー: {e}")
                    continue

        # c. 追跡またはボックス表示の処理
        active_tracks = []
        if ENABLE_TRACKING and tracker:
            try:
                active_tracks = tracker.update(tennis_ball_detections)
            except Exception as e:
                print(f"追跡エラー: {e}")
                active_tracks = []

        # d. プレイヤーの描画（軌跡のみモードでは非表示）
        if DRAW_PLAYERS and not TRAJECTORY_ONLY_MODE:
            for player in player_detections:
                x1, y1, x2, y2 = player['bbox']
                score = player['score']
                class_name = player['class_name']
                
                color = get_player_color(class_name)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name}: {score:.2f}"
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y1, label_size[1])
                
                cv2.rectangle(frame, 
                             (x1, label_y - label_size[1] - 5), 
                             (x1 + label_size[0] + 5, label_y + base_line), 
                             color, cv2.FILLED)
                cv2.putText(frame, label, (x1 + 2, label_y - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # e. テニスボールの描画
        if ENABLE_TRACKING:
            # 軌跡のみモードでテニスボールの検出ボックスも表示
            if TRAJECTORY_ONLY_MODE:
                # テニスボールの検出ボックス表示
                for detection in tennis_ball_detections:
                    center_x, center_y, score, class_id = detection
                    # 検出ボックスのサイズを推定（テニスボール用）
                    box_size = 30  # テニスボール用のボックスサイズ
                    x1 = center_x - box_size // 2
                    y1 = center_y - box_size // 2
                    x2 = center_x + box_size // 2
                    y2 = center_y + box_size // 2
                    
                    # 検出ボックスを描画（軌跡色と区別するため白色）
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    
                    # 検出スコアを表示（小さく）
                    score_text = f"{score:.2f}"
                    cv2.putText(frame, score_text, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 軌跡の描画
            if DRAW_TRAJECTORY or TRAJECTORY_ONLY_MODE:
                for track in active_tracks:
                    color = get_color(track.id)
                    
                    # 軌跡線の描画
                    if len(track.centers) > 1:
                        points = list(track.centers)
                        for i in range(1, len(points)):
                            if TRAJECTORY_ONLY_MODE:
                                # 軌跡のみモード：シンプルで目立つ軌跡線
                                thickness = max(2, int(4 * (i / len(points))))
                                cv2.line(frame, points[i-1], points[i], color, thickness)
                            else:
                                # 通常モード
                                thickness = max(1, int(3 * (i / len(points))))
                                cv2.line(frame, points[i-1], points[i], color, thickness)
                    
                    # 最新位置の表示
                    if track.centers:
                        latest_pos = track.centers[-1]
                        if TRAJECTORY_ONLY_MODE:
                            # 軌跡のみモード：シンプルな円のみ
                            cv2.circle(frame, latest_pos, 8, color, -1)
                            cv2.circle(frame, latest_pos, 10, (255, 255, 255), 2)
                            
                            # トラックIDを表示（軌跡のみモードでも表示）
                            cv2.putText(frame, f"ID:{track.id}", 
                                       (latest_pos[0] + 15, latest_pos[1] - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            cv2.circle(frame, latest_pos, 10, color, -1)
                            cv2.circle(frame, latest_pos, 12, (255, 255, 255), 2)

            # 予測位置の描画（軌跡のみモードでは非表示）
            if DRAW_PREDICTION and not TRAJECTORY_ONLY_MODE and tracker:
                predictions = tracker.get_predicted_positions(5)
                for track_key, pred_pos in predictions.items():
                    # アクティブトラックの予測
                    if isinstance(track_key, int):
                        color = get_color(track_key)
                        cv2.circle(frame, pred_pos, 15, color, 3)
                        cv2.putText(frame, "PRED", (pred_pos[0]-20, pred_pos[1]-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # 現在位置から予測位置への矢印
                        for track in active_tracks:
                            if track.id == track_key and track.centers:
                                curr_pos = track.centers[-1]
                                cv2.arrowedLine(frame, curr_pos, pred_pos, color, 3, tipLength=0.3)
                    
                    # 見失ったトラックの予測（点線で表示）
                    elif DRAW_LOST_PREDICTIONS and track_key.startswith("lost_"):
                        track_id = int(track_key.replace("lost_", ""))
                        color = get_color(track_id)
                        # 点線円で見失ったトラックの予測を表示
                        for angle in range(0, 360, 20):
                            x = pred_pos[0] + int(15 * math.cos(math.radians(angle)))
                            y = pred_pos[1] + int(15 * math.sin(math.radians(angle)))
                            cv2.circle(frame, (x, y), 2, color, -1)
                        cv2.putText(frame, "LOST", (pred_pos[0]-25, pred_pos[1]+25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 追跡結果の詳細描画（軌跡のみモードでは非表示）
            if not TRAJECTORY_ONLY_MODE:
                for track in active_tracks:
                    if not track.centers:
                        continue
                        
                    current_pos = track.centers[-1]
                    color = get_color(track.id)
                    
                    # バウンディングボックス
                    if DRAW_BOX:
                        box_size = 25
                        x1, y1 = current_pos[0] - box_size//2, current_pos[1] - box_size//2
                        x2, y2 = current_pos[0] + box_size//2, current_pos[1] + box_size//2
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                    # ラベル情報
                    label_parts = []
                    if DRAW_TRACK_ID:
                        label_parts.append(f"Ball-{track.id}")
                    if DRAW_CLASS_NAME:
                        label_parts.append("TennisBall")
                    if DRAW_SCORE and track.scores:
                        label_parts.append(f"{track.scores[-1]:.2f}")
                    
                    speed = math.sqrt(track.velocity[0]**2 + track.velocity[1]**2)
                    confidence = track.get_trajectory_confidence()
                    label_parts.append(f"v:{speed:.1f}")
                    label_parts.append(f"conf:{confidence:.2f}")
                    
                    label = " ".join(label_parts)
                    if label:
                        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        label_pos = (current_pos[0] - label_size[0]//2, current_pos[1] - 40)
                        cv2.rectangle(frame, 
                                     (label_pos[0]-5, label_pos[1] - label_size[1]-5), 
                                     (label_pos[0] + label_size[0]+5, label_pos[1] + base_line+5), 
                                     color, cv2.FILLED)
                        cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        else:
            # 追跡オフ：テニスボールのボックスのみ表示（軌跡のみモードでは何も表示しない）
            if not TRAJECTORY_ONLY_MODE:
                for ball in tennis_ball_detections:
                    x1, y1, x2, y2 = ball['bbox']
                    score = ball['score']
                    class_name = ball['class_name']
                    
                    color = get_tennis_ball_color()
                    
                    # バウンディングボックス
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # ラベル
                    label = f"{class_name}: {score:.2f}"
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_y = max(y1, label_size[1])
                    
                    cv2.rectangle(frame, 
                                 (x1, label_y - label_size[1] - 5), 
                                 (x1 + label_size[0] + 5, label_y + base_line), 
                                 color, cv2.FILLED)
                    cv2.putText(frame, label, (x1 + 2, label_y - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # f. デバッグ情報の表示
        if TRAJECTORY_ONLY_MODE:
            # 軌跡のみモード：最小限の情報のみ（画面右上に小さく表示）
            debug_text = f"Frame: {frame_id_counter} | Tracks: {len(active_tracks)} | Detections: {len(tennis_ball_detections)}"
            text_size = cv2.getTextSize(debug_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(frame, debug_text, (frame_width - text_size[0] - 10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # モード表示（画面右上）
            mode_text = "TRAJECTORY + DETECTION"
            mode_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(frame, mode_text, (frame_width - mode_size[0] - 10, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # 凡例表示（画面左下）
            legend_y_start = frame_height - 60
            cv2.putText(frame, "Legend:", (10, legend_y_start), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, "White Box: Detection", (10, legend_y_start + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, "Colored Line: Trajectory", (10, legend_y_start + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, "Colored Circle: Track Center", (10, legend_y_start + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        elif ENABLE_TRACKING and tracker:
            tracker_info = tracker.get_all_tracks_info()
            debug_text = f"Frame: {frame_id_counter} | TRACKING ON | Tennis Balls: {tracker_info['active_tracks']} | Lost: {tracker_info['lost_tracks']} | Players: {len(player_detections)}"
            cv2.putText(frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            mode_text = "TRACKING: ON"
            cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            debug_text = f"Frame: {frame_id_counter} | TRACKING OFF | Tennis Ball Detections: {len(tennis_ball_detections)} | Players: {len(player_detections)}"
            cv2.putText(frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # g. 処理されたフレームを表示またはファイルに書き出し
        if REALTIME_DISPLAY:
            window_title = "Tennis Ball Trajectory Only" if TRAJECTORY_ONLY_MODE else "YOLOv8 + TrajectoryTrack - Tennis Ball Tracking"
            cv2.imshow(window_title, frame)
        
        if SAVE_VIDEO and out_video:
            out_video.write(frame)

        if REALTIME_DISPLAY:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("処理を中断しました。")
                break

    # 5. 終了処理
    cap.release()
    if SAVE_VIDEO and out_video:
       out_video.release()
       print(f"処理が完了しました。出力ファイル: {OUTPUT_VIDEO_PATH}")
    
    if REALTIME_DISPLAY or SAVE_VIDEO: # ウィンドウを開いたか、ファイル操作があった可能性がある場合
        cv2.destroyAllWindows()
    
    if not REALTIME_DISPLAY and not SAVE_VIDEO:
        print("処理が完了しました。(表示・保存なし)")

if __name__ == "__main__":
    # デバッグ情報を追加
    print(f"動画パス: {VIDEO_PATH}")
    print(f"モデルパス: {MODEL_PATH}")
    
    # パスの存在確認
    import os
    if not os.path.exists(VIDEO_PATH):
        print(f"エラー: 動画ファイルが見つかりません: {VIDEO_PATH}")
        exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイルが見つかりません: {MODEL_PATH}")
        exit(1)
    
    try:
        main()
    except Exception as e:
        print(f"プログラム実行エラー: {e}")
        import traceback
        traceback.print_exc()
