import cv2
import pandas as pd
from datetime import datetime
import shutil # 必要に応じて中間ディレクトリのクリーンアップに使用
from pathlib import Path

# モジュールが同じディレクトリにあるか、PYTHONPATH経由でアクセス可能であると仮定
from balltracking import BallTracker
from feature_extractor_predict import TennisInferenceFeatureExtractor
from predict_lstm_model import TennisLSTMPredictor # LSTMPredictorPlaceholder から変更
from overlay_predictions import PredictionOverlay
from court_calibrator import CourtCalibrator # 新規インポート
from typing import Optional # Optional をインポート

# argparse.Namespaceの代わりに使用するシンプルなクラス
class PipelineArgs:
    def __init__(self, video_path, output_dir, frame_skip, imgsz, yolo_model, lstm_model, overlay_mode):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_skip = frame_skip
        self.imgsz = imgsz
        self.yolo_model = yolo_model
        self.lstm_model = lstm_model
        self.overlay_mode = overlay_mode

def run_pipeline(args):
    video_path = Path(args.video_path)
    video_stem = video_path.stem
    output_dir = Path(args.output_dir)

    calibration_output_dir = output_dir / "00_calibration_data" # 新しいディレクトリ
    tracking_output_dir = output_dir / "01_tracking_data"
    features_output_dir = output_dir / "02_extracted_features"
    predictions_output_dir = output_dir / "03_lstm_predictions"
    final_output_dir = output_dir / "04_final_output"

    for p_dir in [calibration_output_dir, tracking_output_dir, features_output_dir, predictions_output_dir, final_output_dir]:
        p_dir.mkdir(parents=True, exist_ok=True)

    print(f"パイプライン開始: {video_path}")
    print(f"フレームスキップ: {args.frame_skip}")
    print(f"出力ディレクトリ: {output_dir}")

    # --- ステップ 1: コートキャリブレーション ---
    print("\n--- ステップ 1: コートキャリブレーション ---")
    calibrator = CourtCalibrator() # 引数なしで初期化

    # calibrate_and_save の代わりに calibrate と save_to_file を使用
    calibration_json_path: Optional[Path] = None
    cap_calib = None # try-finally のためにここで定義
    try:
        cap_calib = cv2.VideoCapture(str(video_path))
        if not cap_calib.isOpened():
            print(f"エラー: コートキャリブレーション用のビデオを開けませんでした {video_path}")
        else:
            ret_calib, first_frame_calib = cap_calib.read()
            if not ret_calib:
                print(f"エラー: コートキャリブレーション用の最初のフレームを読み込めませんでした {video_path}")
            else:
                print("コートキャリブレーションを開始します。ウィンドウの指示に従ってください。")
                # CourtCalibrator.calibrate は bool を返す
                calibration_successful = calibrator.calibrate(first_frame_calib, cap_calib)
                
                if calibration_successful:
                    # 保存ファイル名を生成 (特徴量抽出器のパターン court_coords_{video_stem}_*.json に合わせる)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"court_coords_{video_stem}_{timestamp}.json"
                    target_calibration_json_path = calibration_output_dir / output_filename
                    
                    # CourtCalibrator.save_to_file は bool を返す
                    save_successful = calibrator.save_to_file(str(target_calibration_json_path))
                    if save_successful:
                        calibration_json_path = target_calibration_json_path
                        # print(f"コートキャリブレーションデータを保存しました: {calibration_json_path}") # このメッセージは後続の処理で表示される
                    else:
                        print(f"エラー: コートキャリブレーションデータの保存に失敗しました: {target_calibration_json_path}")
                else:
                    print("コートキャリブレーションがユーザーによってキャンセルされたか、失敗しました。")
    except Exception as e:
        print(f"コートキャリブレーターの実行中に予期せぬエラーが発生しました: {e}")
        # calibration_json_path は None のまま
    finally:
        if cap_calib is not None and cap_calib.isOpened():
            cap_calib.release()


    court_data_source_dir_for_feature_extraction: Optional[str] = None
    if not calibration_json_path:
        print("警告: コートキャリブレーションに失敗またはスキップされました。")
        # TennisInferenceFeatureExtractor は court_data_dir が None の場合、
        # inference_data_dir (トラッキングデータと同じ場所) からコート座標を探すフォールバック動作をします。
        court_data_source_dir_for_feature_extraction = None # または tracking_output_dir を指すように設定も可能
    else:
        print(f"コートキャリブレーションデータを保存しました: {calibration_json_path}")
        # TennisInferenceFeatureExtractor はディレクトリを期待するので、保存されたファイルの親ディレクトリを渡す
        court_data_source_dir_for_feature_extraction = str(calibration_json_path.parent)

    # --- ステップ 2: ボールトラッキング ---
    print("\n--- ステップ 2: ボールトラッキング ---")
    tracker = BallTracker(
        model_path=args.yolo_model,
        imgsz=args.imgsz,
        save_training_data=True,
        data_dir=str(tracking_output_dir),
        frame_skip=args.frame_skip
    )
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"エラー: ビデオを開けませんでした {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"ビデオ情報: FPS={video_fps}, 総フレーム数={total_video_frames}")

    # tqdm をインポート (進捗表示用)
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("tqdmが見つかりません。進捗バーなしで実行します。`pip install tqdm`でインストールできます。")

    frame_iterable = range(total_video_frames) if total_video_frames > 0 else iter(lambda: cap.read()[0], False)
    if use_tqdm and total_video_frames > 0:
        frame_iterable = tqdm(frame_iterable, desc=f"トラッキング中 {video_stem}", total=total_video_frames)
    
    current_frame_idx = 0
    for _ in frame_iterable: # total_video_framesが0の場合、cap.read()でループ制御
        ret, frame = cap.read()
        if not ret:
            if total_video_frames > 0 and current_frame_idx < total_video_frames:
                 print(f"警告: フレーム {current_frame_idx} の読み込みに失敗しましたが、まだ総フレーム数に達していません。")
            break
        
        tracker.process_frame_optimized(frame, current_frame_idx, training_data_only=True)
        current_frame_idx += 1
    
    actual_processed_frames = current_frame_idx # 実際に読み込まれたフレーム数
    cap.release()
    
    # total_video_framesが0または不正確だった場合、actual_processed_framesで更新
    if total_video_frames == 0 or abs(total_video_frames - actual_processed_frames) > 5 : # 5フレーム以上の誤差がある場合
        print(f"警告: OpenCVの総フレーム数 ({total_video_frames}) と実読み込みフレーム数 ({actual_processed_frames}) が異なります。実読み込みフレーム数を使用します。")
        total_video_frames = actual_processed_frames

    tracker.save_tracking_features_with_video_info(video_stem, video_fps, total_video_frames)
    
    tracking_json_files = sorted(list(tracking_output_dir.glob(f"tracking_features_{video_stem}_*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tracking_json_files:
        print(f"エラー: トラッキングJSON ({video_stem}) が {tracking_output_dir} に見つかりません")
        return
    tracking_json_path = tracking_json_files[0]
    print(f"トラッキングデータを保存しました: {tracking_json_path}")

    # --- ステップ 3: 特徴量抽出 ---
    print("\n--- ステップ 3: 特徴量抽出 ---")
    feature_extractor = TennisInferenceFeatureExtractor(
        inference_data_dir=str(tracking_output_dir),
        court_data_dir=court_data_source_dir_for_feature_extraction # コート座標データのディレクトリを渡す
    )
    feature_extractor.features_dir = features_output_dir
    
    _, features_csv_path_str, _ = feature_extractor.run_feature_extraction(
        video_name=video_stem, save_results=True
    )
    if not features_csv_path_str:
        print("エラー: 特徴量抽出に失敗しました。")
        return
    features_csv_path = Path(features_csv_path_str)
    print(f"抽出された特徴量を保存しました: {features_csv_path}")

    # --- ステップ 4: LSTM予測 ---
    print("\n--- ステップ 4: LSTM予測 ---")
    # TennisLSTMPredictor は __init__ で predictions_output_dir を内部的に設定するため、
    # output_dir 引数は不要。ただし、デフォルトの保存先が ./training_data/predictions のため、
    # パイプラインの predictions_output_dir を使うようにインスタンス化時に渡す。
    predictor = TennisLSTMPredictor(
        models_dir=str(Path(args.lstm_model).parent), # 親ディレクトリを渡す (select_model_files用だが今回は使わない)
        input_features_dir=str(features_output_dir) # 同上
    )
    # TennisLSTMPredictor の predictions_output_dir をパイプラインの指定に合わせる
    predictor.predictions_output_dir = predictions_output_dir
    
    # args.lstm_model は特定のモデルセットのディレクトリパスを期待
    # features_csv_path は Path オブジェクト
    prediction_csv_path = predictor.run_prediction_for_file(
        model_set_path=Path(args.lstm_model), 
        feature_csv_path=features_csv_path
    )
    
    if not prediction_csv_path:
        print("エラー: LSTM予測に失敗しました。")
        return
    # prediction_csv_path は既に Path オブジェクトなので変換不要
    print(f"予測結果を保存しました: {prediction_csv_path}")

    # --- ステップ 5: 予測結果のオーバーレイ ---
    print("\n--- ステップ 5: 予測結果のオーバーレイ ---")
    overlay_processor = PredictionOverlay(
        predictions_csv_dir=str(predictions_output_dir),
        input_video_dir=str(video_path.parent),
        output_video_dir=str(final_output_dir)
    )

    predictions_df = overlay_processor.load_predictions(prediction_csv_path)
    if predictions_df is None:
        print("エラー: オーバーレイ用の予測読み込みに失敗しました。")
        return

    if args.overlay_mode == "ffmpeg":
        print("FFmpegを使用して予測をオーバーレイし、ファイルに保存します...")
        success = overlay_processor.process_video(video_path, predictions_df, prediction_csv_path.name)
        if success:
            print(f"FFmpegオーバーレイビデオが {final_output_dir} に保存されました")
        else:
            print("FFmpegオーバーレイ処理に失敗しました。")
    elif args.overlay_mode == "realtime":
        print("ビデオと予測をリアルタイムで表示します...")
        overlay_processor.display_video_with_predictions_realtime(video_path, predictions_df)
    else:
        print(f"不明なオーバーレイモード: {args.overlay_mode}")

    print("\nパイプラインが完了しました。")

if __name__ == "__main__":
    print("テニスビデオ分析パイプラインへようこそ！")
    print("いくつかの情報を入力してください。デフォルト値を使用する場合はEnterキーを押してください。")

    video_path_str = ""
    raw_data_dir = Path("../data/raw")
    video_files = []

    if raw_data_dir.exists() and raw_data_dir.is_dir():
        print(f"\n利用可能なビデオファイル ({raw_data_dir}):")
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"] # 一般的なビデオ拡張子
        for i, file_path in enumerate(raw_data_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                video_files.append(file_path)
                print(f"  {len(video_files)}. {file_path.name}")
        
        if video_files:
            while True:
                try:
                    choice = input(f"ビデオを選択してください (番号を入力、または 'm' で手動入力): ").strip().lower()
                    if choice == 'm':
                        break # 手動入力へ
                    selected_index = int(choice) - 1
                    if 0 <= selected_index < len(video_files):
                        video_path_str = str(video_files[selected_index].resolve())
                        print(f"選択されたビデオ: {video_path_str}")
                        break
                    else:
                        print("無効な番号です。リストから選択するか、'm' を入力してください。")
                except ValueError:
                    print("無効な入力です。番号を入力するか、'm' を入力してください。")
        else:
            print(f"{raw_data_dir} にビデオファイルが見つかりませんでした。手動でパスを入力してください。")
    else:
        print(f"ディレクトリ {raw_data_dir} が見つかりません。手動でビデオパスを入力してください。")

    if not video_path_str: # 選択されなかった場合、または手動入力を選択した場合
        while not video_path_str:
            video_path_str = input("入力ビデオファイルのパスを入力してください: ").strip()
            if not video_path_str:
                print("ビデオパスは必須です。")
            elif not Path(video_path_str).exists():
                print(f"エラー: 指定されたビデオパス '{video_path_str}' が存在しません。正しいパスを入力してください。")
                video_path_str = "" # 無効なパスの場合は再入力を促す

    output_dir_str = input("出力ディレクトリ (デフォルト: ./tennis_pipeline_output): ").strip() or "./tennis_pipeline_output"
    
    frame_skip_str = input("フレームスキップ (デフォルト: 1): ").strip() or "1"
    try:
        frame_skip_int = int(frame_skip_str)
        if frame_skip_int < 1:
            print("フレームスキップは1以上である必要があります。デフォルトの1を使用します。")
            frame_skip_int = 1
    except ValueError:
        print("無効なフレームスキップ値です。デフォルトの1を使用します。")
        frame_skip_int = 1

    imgsz_str = input("YOLOモデル推論時の画像サイズ (デフォルト: 640): ").strip() or "640"
    try:
        imgsz_int = int(imgsz_str)
    except ValueError:
        print("無効な画像サイズです。デフォルトの640を使用します。")
        imgsz_int = 640

    yolo_model_str = ""
    models_weights_dir = Path("../models/weights")
    pt_files = []

    if models_weights_dir.exists() and models_weights_dir.is_dir():
        print(f"\n利用可能なYOLOモデルファイル ({models_weights_dir}):")
        for i, file_path in enumerate(models_weights_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() == ".pt":
                pt_files.append(file_path)
                print(f"  {len(pt_files)}. {file_path.name}")
        
        if pt_files:
            while True:
                try:
                    choice = input(f"YOLOモデルを選択してください (番号を入力、または 'm' で手動入力、デフォルト: yolov8n.pt): ").strip().lower()
                    if not choice and "yolov8n.pt" in [f.name for f in pt_files]: # デフォルト選択
                        default_model_path = models_weights_dir / "yolov8n.pt"
                        if default_model_path.exists():
                             yolo_model_str = str(default_model_path.resolve())
                             print(f"デフォルトモデルを選択: {yolo_model_str}")
                             break
                        # デフォルトがリストにあっても存在しない場合は手動へ
                        print("デフォルトのyolov8n.ptが見つかりません。手動で入力してください。")
                        choice = 'm' # 手動入力へフォールバック

                    if choice == 'm':
                        break # 手動入力へ
                    selected_index = int(choice) - 1
                    if 0 <= selected_index < len(pt_files):
                        yolo_model_str = str(pt_files[selected_index].resolve())
                        print(f"選択されたYOLOモデル: {yolo_model_str}")
                        break
                    else:
                        print("無効な番号です。リストから選択するか、'm' を入力してください。")
                except ValueError:
                    print("無効な入力です。番号を入力するか、'm' を入力してください。")
        else:
            print(f"{models_weights_dir} に .pt ファイルが見つかりませんでした。手動でパスを入力してください。")
    else:
        print(f"ディレクトリ {models_weights_dir} が見つかりません。手動でYOLOモデルパスを入力してください。")

    if not yolo_model_str: # 選択されなかった場合、または手動入力を選択した場合
        yolo_model_str = input("YOLOv8モデルのパス (デフォルト: yolov8n.pt): ").strip() or "yolov8n.pt"
        if not Path(yolo_model_str).exists() and yolo_model_str != "yolov8n.pt":
             print(f"警告: 指定されたYOLOモデルパス '{yolo_model_str}' が存在しません。")
        elif yolo_model_str == "yolov8n.pt" and not Path(yolo_model_str).exists():
             # デフォルト名で存在しない場合も警告（ただし、ユーザーが明示的にデフォルトを使った場合）
             print(f"警告: デフォルトのYOLOモデル 'yolov8n.pt' がカレントディレクトリまたは指定パスに見つかりません。")

    lstm_model_str = ""
    lstm_models_base_dir = Path("./training_data/lstm_models")
    lstm_model_folders = []

    if lstm_models_base_dir.exists() and lstm_models_base_dir.is_dir():
        print(f"\n利用可能なLSTMモデルフォルダ ({lstm_models_base_dir}):")
        for item in lstm_models_base_dir.iterdir():
            if item.is_dir(): # ディレクトリのみをリストアップ
                lstm_model_folders.append(item)
                print(f"  {len(lstm_model_folders)}. {item.name}")
        
        if lstm_model_folders:
            while True:
                try:
                    choice = input(f"LSTMモデルフォルダを選択してください (番号を入力、または 'm' で手動入力): ").strip().lower()
                    if choice == 'm':
                        break # 手動入力へ
                    selected_index = int(choice) - 1
                    if 0 <= selected_index < len(lstm_model_folders):
                        lstm_model_str = str(lstm_model_folders[selected_index].resolve())
                        print(f"選択されたLSTMモデル: {lstm_model_str}")
                        break
                    else:
                        print("無効な番号です。リストから選択するか、'm' を入力してください。")
                except ValueError:
                    print("無効な入力です。番号を入力するか、'm' を入力してください。")
        else:
            print(f"{lstm_models_base_dir} にLSTMモデルフォルダが見つかりませんでした。手動でパスを入力してください。")
    else:
        print(f"ディレクトリ {lstm_models_base_dir} が見つかりません。手動でLSTMモデルパスを入力してください。")

    if not lstm_model_str: # リストから選択されなかった場合、または手動入力を選択した場合
        while not lstm_model_str:
            lstm_model_str = input("LSTMモデルのパス (ディレクトリを入力してください): ").strip()
            if not lstm_model_str:
                print("LSTMモデルのパスは必須です。")
            elif not Path(lstm_model_str).exists():
                 print(f"警告: 指定されたLSTMモデルパス '{lstm_model_str}' が存在しません。")
                 # 存在しないパスも許可するが警告は表示（LSTMPredictorPlaceholderでも再度チェックされる）


    overlay_mode_str = ""
    valid_overlay_modes = ["ffmpeg", "realtime"]
    while overlay_mode_str not in valid_overlay_modes:
        overlay_mode_str = input(f"予測オーバーレイモード ({'/'.join(valid_overlay_modes)}, デフォルト: ffmpeg): ").strip().lower() or "ffmpeg"
        if overlay_mode_str not in valid_overlay_modes:
            print(f"無効なオーバーレイモードです。{', '.join(valid_overlay_modes)} のいずれかを入力してください。")

    # PipelineArgsオブジェクトを作成
    pipeline_args = PipelineArgs(
        video_path=video_path_str,
        output_dir=output_dir_str,
        frame_skip=frame_skip_int,
        imgsz=imgsz_int,
        yolo_model=yolo_model_str,
        lstm_model=lstm_model_str,
        overlay_mode=overlay_mode_str
    )
    
    run_pipeline(pipeline_args)
