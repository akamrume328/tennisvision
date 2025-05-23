import cv2
from ultralytics import YOLO
import torch # YOLOが内部で使用する可能性があるため残すが、Norfairは直接は不要
import numpy as np

# Norfairをインポート
# pip install norfair を実行してください
import norfair
from norfair import Detection, Tracker, Paths, Video
from norfair.distances import iou # iou_xyxy から iou に変更

# ファインチューニング済みのYOLOv8モデルのパスを指定
MODEL_PATH = "../models/weights/best.pt"  # COCOデータセットで学習済みのモデル。テニスボールは 'sports ball' (id 32)
VIDEO_SOURCE = "../data/raw/output.mp4" # ビデオファイルを使用する場合
 # カメラを使用する場合

# 追跡したいオブジェクトのクラスID (COCOデータセットの場合、'sports ball' は 32)
# カスタムモデルでテニスボールのみを学習した場合、クラスIDは 0 になることが多いです。
TENNIS_BALL_CLASS_ID = 5
CONFIDENCE_THRESHOLD = 0.3 # 検出の信頼度閾値

# Norfairトラッカーの初期化
# distance_function: 検出と追跡対象間の距離を計算する関数
# distance_threshold: この閾値より距離が遠い場合、新しい追跡対象として扱われる
tracker = Tracker(
    distance_function=iou, # iou_xyxy から iou に変更
    distance_threshold=0.7, # IoUベースなので0から1の範囲。0.7は一般的な値
    hit_counter_max=15, # 何フレーム連続で検出されなかったら追跡を終了するか
    initialization_delay=3 # 何フレーム連続で検出されたら追跡を開始するか
)

def yolo_detections_to_norfair_detections(yolo_results, frame_shape):
    """YOLOv8の検出結果をNorfairのDetectionオブジェクトのリストに変換する"""
    norfair_detections = []
    if yolo_results and yolo_results[0].boxes is not None:
        for box in yolo_results[0].boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            
            if cls_id == TENNIS_BALL_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
                xyxy = box.xyxy.squeeze().cpu().numpy() # [x1, y1, x2, y2]
                
                # NorfairのDetectionオブジェクトを作成
                # points: バウンディングボックスの座標 (Norfairの距離関数が期待する形式)
                # data: 任意の追加情報 (スコア、クラスIDなど)
                detection_points = np.array([[xyxy[0], xyxy[1]], [xyxy[2], xyxy[3]]]) # 左上と右下の2点
                # iou_xyxy を使う場合、Detectionのpointsは [[x1,y1],[x2,y2]] ではなく、
                # 1Dの [x1,y1,x2,y2] 形式の numpy 配列を期待することが多い。
                # Norfairのドキュメントやiou_xyxyの実装を確認する必要がある。
                # ここでは、一般的な [x1, y1, x2, y2] を渡すように試みる。
                # NorfairのDetectionはpointsとしてnumpy配列を期待する。
                # iou_xyxyは2つの [x1,y1,x2,y2] 形式の配列を比較する。
                # Detectionのpointsは、そのオブジェクトの表現。
                # 追記：NorfairのDetectionのpointsは、そのオブジェクトの「状態」を表す。
                # iou_xyxyを使う場合、Detection(points=np.array([x1,y1,x2,y2])) のようにする。
                norfair_detections.append(Detection(points=xyxy, scores=np.array([conf]), label=cls_id))
                
    return norfair_detections

def main():
    # モデルのロード
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"エラー: YOLOモデルのロードに失敗しました - {e}")
        print(f"モデルパスを確認してください: {MODEL_PATH}")
        return

    # ビデオキャプチャの開始
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"エラー: ビデオソースを開けませんでした - {VIDEO_SOURCE}")
        return

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("ビデオの終端に達したか、フレームの読み込みに失敗しました。")
            break

        # YOLOv8でオブジェクト検出を実行
        results = model(frame, verbose=False) # verbose=Falseでログ出力を抑制

        # YOLO検出結果をNorfair形式に変換
        norfair_detections = yolo_detections_to_norfair_detections(results, frame.shape)

        # Norfairトラッカーを更新
        # tracked_objects には、現在追跡中の各オブジェクトの情報 (ID、推定位置など) が含まれる
        tracked_objects = tracker.update(detections=norfair_detections, period=1) # periodはフレーム間の時間間隔の推定に使う

        annotated_frame = frame.copy()
        
        # Norfairの描画ユーティリティを使用 (オプション)
        # Paths.draw(annotated_frame, tracked_objects) # 軌跡を描画
        # norfair.draw_points(annotated_frame, norfair_detections) # 検出点を描画
        # norfair.draw_tracked_objects(annotated_frame, tracked_objects, color_by_label=True, id_size=2, id_thickness=2)
        norfair.draw_tracked_boxes(annotated_frame, tracked_objects, color_by_label=True, id_size=2, id_thickness=2)


        # 結果を表示
        cv2.imshow("YOLOv8 + Norfair Tennis Ball Tracking", annotated_frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
