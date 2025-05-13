# main.py

import os
# from src.config import Config
# from src.detection import load_model, run_inference
# from src.tracking import track_objects
# from scripts.preprocess_data import preprocess_data
# from scripts.train import train_model

#test
from ultralytics import YOLO
import cv2
from utils import visualize_detections
def main():
    # # Load configuration settings
    # config = Config()

    # # Step 1: Data Preprocessing
    # print("Starting data preprocessing...")
    # preprocess_data(config.raw_data_path, config.processed_data_path)

    # # Step 2: Model Training
    # print("Starting model training...")
    # model = load_model(config.model_weights_path)
    # train_model(model, config.processed_data_path, config.training_params)

    # # Step 3: Object Tracking
    # print("Starting object tracking...")
    # video_path = os.path.join(config.raw_data_path, "sample_video.mp4")  # Example video path
    # track_objects(video_path, model)


    #test
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        print("モデルファイルが正しくダウンロードされているか、パスが正しいか確認してください。")
        return

    # 検出を実行したい画像のパス (適宜変更してください)
    # 例として、インターネット上の画像URLを指定することも可能です
    # image_path = "https://ultralytics.com/images/bus.jpg"
    image_path = r"C:\Users\akama\AppData\Local\Programs\Python\Python310\python_file\projects\tennisvision\data\raw\tennistest.jpg" 
    # ここにご自身の画像パスを指定してください

    try:
        # 画像に対して推論を実行
        results = model(image_path)
    except Exception as e:
        print(f"推論中にエラーが発生しました: {e}")
        print(f"画像パス '{image_path}' が正しいか、画像ファイルが存在するか確認してください。")
        return

    # 元の画像を読み込み
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"画像の読み込みに失敗しました: {image_path}")
            return
    except Exception as e:
        print(f"cv2.imread でエラーが発生しました: {e}")
        return

    # 検出結果を取得
    # results[0].boxes.data は [x1, y1, x2, y2, confidence, class_id] の形式のテンソル
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            detections.append([x1, y1, x2, y2, class_id, confidence])

    # 検出結果を画像に描画 (utils.pyの関数を利用)
    # visualize_detections 関数が YOLO の出力形式に合うように調整が必要な場合があります
    # YOLOのクラス名はモデルの `names` 属性から取得できます
    # 例: class_name = model.names[class_id]
    
    # utils.visualize_detections をYOLOの出力に合わせて調整するか、以下のように直接描画します
    for det in detections:
        x1, y1, x2, y2, class_id, conf = det
        label = f"{model.names[class_id]}: {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # 結果を表示
    cv2.imshow("YOLOv8 Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()