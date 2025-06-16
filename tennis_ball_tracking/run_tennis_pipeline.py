import cv2
import pandas as pd
from datetime import datetime
import shutil # 必要に応じて中間ディレクトリのクリーンアップに使用
from pathlib import Path
import numpy as np # numpyをインポート
import time # 処理時間計測のためにインポート

# モジュールが同じディレクトリにあるか、PYTHONPATH経由でアクセス可能であると仮定
from balltracking import BallTracker
from feature_extractor_predict import TennisInferenceFeatureExtractor
from predict_lstm_model import TennisLSTMPredictor # LSTMPredictorPlaceholder から変更
from overlay_predictions import PredictionOverlay
from court_calibrator import CourtCalibrator # 新規インポート
from hmm_postprocessor import HMMSupervisedPostprocessor # HMM後処理用にインポート
from typing import Optional # Optional をインポート
from cut_long_intervals import cut_video_by_point_interval # FFmpeg版をインポート
from cut_non_rally_segments import cut_non_rally_segments # Rally区間抽出用にインポート

# argparse.Namespaceの代わりに使用するシンプルなクラス
class PipelineArgs:
    def __init__(self, video_path, output_dir, frame_skip, imgsz, yolo_model, lstm_model, hmm_model_path, overlay_mode, cut_interval_mode, cut_interval_threshold, extract_rally_mode, rally_buffer_seconds): # rally抽出パラメータを追加
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_skip = frame_skip
        self.imgsz = imgsz
        self.yolo_model = yolo_model
        self.lstm_model = lstm_model
        self.hmm_model_path = hmm_model_path # HMMモデルパスを追加
        self.overlay_mode = overlay_mode
        self.cut_interval_mode = cut_interval_mode # インターバルカットモードを追加
        self.cut_interval_threshold = cut_interval_threshold # インターバルカットの閾値を追加
        self.extract_rally_mode = extract_rally_mode # Rally区間抽出モードを追加
        self.rally_buffer_seconds = rally_buffer_seconds # Rally前後のバッファ秒数を追加

def run_pipeline(args):
    pipeline_start_time = time.time() # パイプライン全体の開始時刻

    video_path = Path(args.video_path)
    video_stem = video_path.stem
    output_dir = Path(args.output_dir)

    calibration_output_dir = output_dir / "00_calibration_data"
    tracking_output_dir = output_dir / "01_tracking_data"
    features_output_dir = output_dir / "02_extracted_features"
    predictions_output_dir = output_dir / "03_lstm_predictions"
    hmm_output_dir = output_dir / "03a_hmm_processed_predictions" # HMM処理結果用ディレクトリ
    final_output_dir = output_dir / "04_final_output"
    cut_video_output_dir = output_dir / "05_cut_video" # インターバルカットされたビデオ用ディレクトリ
    rally_extract_output_dir = output_dir / "06_rally_extract" # Rally区間抽出されたビデオ用ディレクトリ

    for p_dir in [calibration_output_dir, tracking_output_dir, features_output_dir, predictions_output_dir, hmm_output_dir, final_output_dir, cut_video_output_dir, rally_extract_output_dir]:
        p_dir.mkdir(parents=True, exist_ok=True)

    print(f"パイプライン開始: {video_path}")
    print(f"フレームスキップ: {args.frame_skip}")
    print(f"出力ディレクトリ: {output_dir}")

    # --- ステップ 1: コートキャリブレーション ---
    print("\n--- ステップ 1: コートキャリブレーション ---")
    step_1_start_time = time.time()
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

    step_1_end_time = time.time()
    print(f"ステップ 1 (コートキャリブレーション) 処理時間: {step_1_end_time - step_1_start_time:.2f} 秒")


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
    step_2_start_time = time.time()
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
    
    step_2_end_time = time.time()
    print(f"ステップ 2 (ボールトラッキング) 処理時間: {step_2_end_time - step_2_start_time:.2f} 秒")

    tracking_json_files = sorted(list(tracking_output_dir.glob(f"tracking_features_{video_stem}_*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tracking_json_files:
        print(f"エラー: トラッキングJSON ({video_stem}) が {tracking_output_dir} に見つかりません")
        return
    tracking_json_path = tracking_json_files[0]
    print(f"トラッキングデータを保存しました: {tracking_json_path}")

    # --- ステップ 3: 特徴量抽出 ---
    print("\n--- ステップ 3: 特徴量抽出 ---")
    step_3_start_time = time.time()
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
    step_3_end_time = time.time()
    print(f"ステップ 3 (特徴量抽出) 処理時間: {step_3_end_time - step_3_start_time:.2f} 秒")

    # --- ステップ 4: LSTM予測 ---
    print("\n--- ステップ 4: LSTM予測 ---")
    step_4_start_time = time.time()
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
    step_4_end_time = time.time()
    print(f"ステップ 4 (LSTM予測) 処理時間: {step_4_end_time - step_4_start_time:.2f} 秒")


    # --- ステップ 4.5: HMMによる後処理 ---
    hmm_processed_csv_path = prediction_csv_path # HMM処理しない場合のデフォルト
    step_4_5_start_time = time.time()
    hmm_processing_done = False # HMM処理が実際に行われたかどうかのフラグ
    if args.hmm_model_path:
        print("\n--- ステップ 4.5: HMMによる後処理 ---")
        hmm_postprocessor = HMMSupervisedPostprocessor(verbose=True, random_state=42)
        
        print(f"HMMモデル読み込み: {args.hmm_model_path}")
        if not hmm_postprocessor.load_hmm_model(Path(args.hmm_model_path)):
            print(f"警告: HMMモデルの読み込みに失敗しました ({args.hmm_model_path})。HMM後処理をスキップします。")
        else:
            print(f"入力CSVデータ読み込み (HMM用): {prediction_csv_path.name}")
            # LSTM予測結果の列名を指定 (通常は 'predicted_phase')
            # TennisLSTMPredictor.run_prediction_for_file が出力するCSVの列名に合わせる
            lstm_pred_col_name = 'predicted_phase' # 必要に応じて変更
            if not hmm_postprocessor.load_data(data_csv_path=prediction_csv_path,
                                               pred_col_name=lstm_pred_col_name,
                                               true_col_name=None, # 推論時は真ラベルなし
                                               metadata_json_path=None): # モデルからラベル情報を使用
                print("警告: HMM処理用のデータ読み込みに失敗しました。HMM後処理をスキップします。")
            else:
                if hmm_postprocessor.valid_observations_int is None:
                    print("警告: HMM処理対象の観測シーケンスが読み込めませんでした。HMM後処理をスキップします。")
                else:
                    print("HMMによる平滑化を開始...")
                    smoothed_sequence_int = hmm_postprocessor.smooth()
                    if smoothed_sequence_int is None:
                        print("警告: HMMによる平滑化に失敗しました。HMM後処理をスキップします。")
                    else:
                        print("平滑化結果をDataFrameに追加...")
                        if not hmm_postprocessor.int_to_label:
                            print("警告: ラベル->整数 マッピング (int_to_label) がHMMモデルから復元されていません。HMM後処理をスキップします。")
                        else:
                            smoothed_sequence_labels = np.array([hmm_postprocessor.int_to_label.get(s, "UNKNOWN_STATE") for s in smoothed_sequence_int])
                            if not hmm_postprocessor.add_smoothed_results_to_df(smoothed_sequence_labels):
                                print("警告: 平滑化結果のDataFrameへの追加に失敗しました。HMM後処理をスキップします。")
                            else:
                                print("HMM処理結果を保存...")
                                # save_results は保存先のベースディレクトリを引数に取る
                                saved_hmm_csv_path = hmm_postprocessor.save_results(
                                    hmm_postprocessor.df_loaded,
                                    prediction_csv_path, # 元のCSVパスを渡してファイル名を生成
                                    output_base_dir=hmm_output_dir
                                )
                                if saved_hmm_csv_path:
                                    hmm_processed_csv_path = saved_hmm_csv_path
                                    print(f"HMM後処理済み予測結果を保存しました: {hmm_processed_csv_path}")
                                else:
                                    print("警告: HMM処理結果の保存に失敗しました。元のLSTM予測を使用します。")
                            hmm_processing_done = True # HMM処理が試みられた（成功・失敗問わず）
    else:
        print("\nℹ️ HMMモデルパスが指定されていないため、HMM後処理はスキップされました。")
    
    step_4_5_end_time = time.time()
    if args.hmm_model_path or hmm_processing_done: # HMMパスが指定されたか、実際に処理が試みられた場合のみ時間表示
        print(f"ステップ 4.5 (HMM後処理) 処理時間: {step_4_5_end_time - step_4_5_start_time:.2f} 秒")


    # --- ステップ 5: 予測結果のオーバーレイ (インターバルカットモードまたはRally抽出モードが無効な場合のみ) ---
    if not args.cut_interval_mode and not args.extract_rally_mode:
        print("\n--- ステップ 5: 予測結果のオーバーレイ ---")
        step_5_start_time = time.time()
        # オーバーレイ処理には、HMM処理後のCSV (hmm_processed_csv_path) を使用する
        # HMM処理がスキップされた場合は、元のLSTM予測CSV (prediction_csv_path) が使われる
        overlay_input_csv_path_for_step5 = hmm_processed_csv_path # ステップ5専用の変数としておく
        
        # PredictionOverlay の predictions_csv_dir は、使用するCSVファイルの親ディレクトリを指定
        overlay_processor = PredictionOverlay(
            predictions_csv_dir=str(overlay_input_csv_path_for_step5.parent),
            input_video_dir=str(video_path.parent),
            output_video_dir=str(final_output_dir),
            video_fps=video_fps, 
            total_frames=total_video_frames,
            frame_skip=args.frame_skip 
        )

        predictions_df = overlay_processor.load_predictions(overlay_input_csv_path_for_step5)
        if predictions_df is None:
            print("エラー: オーバーレイ用の予測読み込みに失敗しました。ステップ5をスキップします。")
            # return # パイプラインを停止させずに続行も可能
        else:
            if args.overlay_mode == "ffmpeg":
                print("FFmpegを使用して予測をオーバーレイし、ファイルに保存します...")
                # process_video が出力パスを返すように変更したと仮定、または内部で last_processed_output_path を設定
                success_overlay, processed_video_path = overlay_processor.process_video(video_path, predictions_df, overlay_input_csv_path_for_step5.name)
                if success_overlay:
                    print(f"FFmpegオーバーレイビデオが {processed_video_path} に保存されました")
                    # 後続の処理でこのパスを使いたい場合は、overlay_processor インスタンスに保存させるか、ここで変数に保持
                    # overlay_processor.last_processed_output_path = processed_video_path (PredictionOverlay側で設定する想定)
                else:
                    print("FFmpegオーバーレイ処理に失敗しました。")
            elif args.overlay_mode == "realtime":
                print("ビデオと予測をリアルタイムで表示します...")
                overlay_processor.display_video_with_predictions_realtime(video_path, predictions_df)
            else:
                print(f"不明なオーバーレイモード: {args.overlay_mode}")

        step_5_end_time = time.time()
        print(f"ステップ 5 (予測結果のオーバーレイ) 処理時間: {step_5_end_time - step_5_start_time:.2f} 秒")
    else:
        print("\nℹ️ インターバルカットモードまたはRally抽出モードが有効なため、ステップ5 (予測結果のオーバーレイ) はスキップされました。")

    # --- ステップ 6: 長いインターバルのカット (オプション) ---
    # overlay_input_csv_path はステップ4.5の出力 (hmm_processed_csv_path) を引き続き使用
    # HMM処理がスキップされた場合はLSTMの予測結果CSVを指す
    csv_for_cutting_path = hmm_processed_csv_path 

    if args.cut_interval_mode:
        print("\n--- ステップ 6: 長いインターバルのカット ---")
        step_6_start_time = time.time()
        
        input_video_for_cutting_path = video_path # 常に元のビデオをカット対象とする
        print(f"カット処理の入力として元のビデオを使用: {input_video_for_cutting_path}")

        cut_video_filename = f"{input_video_for_cutting_path.stem}_cut_intervals.mp4"
        output_cut_video_path = cut_video_output_dir / cut_video_filename

        print(f"カット処理に使用するCSV: {csv_for_cutting_path}")
        print(f"カットされたビデオの出力先: {output_cut_video_path}")
        print(f"カット閾値: {args.cut_interval_threshold} 秒")

        success_cut = cut_video_by_point_interval(
            video_path_str=str(input_video_for_cutting_path),
            csv_path_str=str(csv_for_cutting_path),
            output_video_path_str=str(output_cut_video_path),
            # fps=video_fps, # FFmpeg版では不要
            threshold_seconds=args.cut_interval_threshold,
            interval_phase_name="point_interval" 
        )

        if success_cut:
            print(f"インターバルカット処理が完了しました。出力ビデオ: {output_cut_video_path}")
        else:
            print("インターバルカット処理に失敗しました。")
        
        step_6_end_time = time.time()
        print(f"ステップ 6 (長いインターバルのカット) 処理時間: {step_6_end_time - step_6_start_time:.2f} 秒")
    # このelseブロックは、cut_interval_modeがfalseの場合のメッセージなので、ステップ5のスキップメッセージとは別に維持
    # else:
    #     print("\nℹ️ インターバルカットモードが無効なため、ステップ6はスキップされました。") # このメッセージは不要になるか、cut_interval_modeがFalseの時のステップ5の後に移動

    # --- ステップ 7: Rally区間の抽出 (オプション) ---
    if args.extract_rally_mode:
        print("\n--- ステップ 7: Rally区間の抽出 ---")
        step_7_start_time = time.time()
        
        input_video_for_rally_extraction = video_path # 常に元のビデオを対象とする
        print(f"Rally抽出処理の入力として元のビデオを使用: {input_video_for_rally_extraction}")

        rally_video_filename = f"{input_video_for_rally_extraction.stem}_rally_only.mp4"
        output_rally_video_path = rally_extract_output_dir / rally_video_filename

        print(f"Rally抽出処理に使用するCSV: {csv_for_cutting_path}")
        print(f"Rally区間抽出ビデオの出力先: {output_rally_video_path}")
        print(f"Rally前後のバッファ: {args.rally_buffer_seconds} 秒")

        success_rally_extract = cut_non_rally_segments(
            video_path_str=str(input_video_for_rally_extraction),
            csv_path_str=str(csv_for_cutting_path),
            output_video_path_str=str(output_rally_video_path),
            rally_phase_name="rally",
            buffer_seconds=args.rally_buffer_seconds
        )

        if success_rally_extract:
            print(f"Rally区間抽出処理が完了しました。出力ビデオ: {output_rally_video_path}")
        else:
            print("Rally区間抽出処理に失敗しました。")
        
        step_7_end_time = time.time()
        print(f"ステップ 7 (Rally区間の抽出) 処理時間: {step_7_end_time - step_7_start_time:.2f} 秒")

    pipeline_end_time = time.time() # パイプライン全体の終了時刻
    total_pipeline_duration = pipeline_end_time - pipeline_start_time
    print(f"\nパイプラインが完了しました。")
    print(f"総処理時間: {total_pipeline_duration:.2f} 秒")

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

    hmm_model_path_str = ""
    hmm_models_base_dir = Path("./training_data/hmm_models")
    hmm_model_files = []

    if hmm_models_base_dir.exists() and hmm_models_base_dir.is_dir():
        print(f"\n利用可能なHMMモデルファイル ({hmm_models_base_dir}):")
        for item in hmm_models_base_dir.iterdir():
            if item.is_file() and item.suffix.lower() == ".joblib":
                hmm_model_files.append(item)
                print(f"  {len(hmm_model_files)}. {item.name}")
        
        if hmm_model_files:
            while True:
                try:
                    # HMMモデルはオプションなので、空の入力を許可
                    choice = input(f"HMMモデルファイルを選択してください (番号を入力、'm'で手動入力、Enterでスキップ): ").strip().lower()
                    if not choice: # Enterでスキップ
                        hmm_model_path_str = ""
                        print("HMM後処理はスキップされます。")
                        break
                    if choice == 'm':
                        break # 手動入力へ
                    selected_index = int(choice) - 1
                    if 0 <= selected_index < len(hmm_model_files):
                        hmm_model_path_str = str(hmm_model_files[selected_index].resolve())
                        print(f"選択されたHMMモデル: {hmm_model_path_str}")
                        break
                    else:
                        print("無効な番号です。リストから選択するか、'm' を入力するか、Enterでスキップしてください。")
                except ValueError:
                    print("無効な入力です。番号を入力するか、'm' を入力するか、Enterでスキップしてください。")
        else:
            print(f"{hmm_models_base_dir} にHMMモデルファイル (.joblib) が見つかりませんでした。手動でパスを入力するか、スキップしてください。")
    else:
        print(f"ディレクトリ {hmm_models_base_dir} が見つかりません。手動でHMMモデルパスを入力するか、スキップしてください。")

    if not hmm_model_path_str and choice != "" and choice != "m": # リスト選択でも手動選択でもなく、かつスキップでもない場合（つまり手動入力が期待される状況）
         # choiceが 'm' だった場合、またはリストが空だった場合など
        hmm_model_path_str = input("HMMモデルファイルのパス (拡張子.joblib, スキップする場合はEnter): ").strip()
        if hmm_model_path_str and not Path(hmm_model_path_str).exists():
            print(f"警告: 指定されたHMMモデルパス '{hmm_model_path_str}' が存在しません。HMM処理はスキップされる可能性があります。")
        elif not hmm_model_path_str:
            print("HMM後処理はスキップされます。")


    overlay_mode_str = ""
    valid_overlay_modes = ["ffmpeg", "realtime"]
    while overlay_mode_str not in valid_overlay_modes:
        overlay_mode_str = input(f"予測オーバーレイモード ({'/'.join(valid_overlay_modes)}, デフォルト: ffmpeg): ").strip().lower() or "ffmpeg"
        if overlay_mode_str not in valid_overlay_modes:
            print(f"無効なオーバーレイモードです。{', '.join(valid_overlay_modes)} のいずれかを入力してください。")

    cut_interval_mode_input = input("長いインターバルをカットしますか？ (yes/no, デフォルト: no): ").strip().lower()
    cut_interval_mode_bool = cut_interval_mode_input == 'yes'
    
    cut_interval_threshold_float = 2.0 # デフォルト値
    if cut_interval_mode_bool:
        threshold_str = input("インターバルカットの閾値（秒、デフォルト: 2.0）: ").strip()
        if threshold_str:
            try:
                cut_interval_threshold_float = float(threshold_str)
                if cut_interval_threshold_float <= 0:
                    print("閾値は正の数である必要があります。デフォルトの2.0秒を使用します。")
                    cut_interval_threshold_float = 2.0
            except ValueError:
                print("無効な閾値です。デフォルトの2.0秒を使用します。")
                cut_interval_threshold_float = 2.0

    extract_rally_mode_input = input("Rally区間のみを抽出しますか？ (yes/no, デフォルト: no): ").strip().lower()
    extract_rally_mode_bool = extract_rally_mode_input == 'yes'
    
    rally_buffer_seconds_float = 2.0 # デフォルト値
    if extract_rally_mode_bool:
        buffer_str = input("Rally区間の前後に保持する秒数（デフォルト: 2.0）: ").strip()
        if buffer_str:
            try:
                rally_buffer_seconds_float = float(buffer_str)
                if rally_buffer_seconds_float < 0:
                    print("バッファ秒数は0以上である必要があります。デフォルトの2.0秒を使用します。")
                    rally_buffer_seconds_float = 2.0
            except ValueError:
                print("無効なバッファ秒数です。デフォルトの2.0秒を使用します。")
                rally_buffer_seconds_float = 2.0    # インターバルカットとRally抽出の両方が選択された場合の警告
    if cut_interval_mode_bool and extract_rally_mode_bool:
        print("警告: インターバルカットとRally抽出の両方が選択されています。")
        print("両方の処理が実行されますが、通常はどちらか一方を選択することを推奨します。")
        confirm = input("続行しますか？ (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("処理を中止します。")
            exit()


    # PipelineArgsオブジェクトを作成
    pipeline_args = PipelineArgs(
        video_path=video_path_str,
        output_dir=output_dir_str,
        frame_skip=frame_skip_int,
        imgsz=imgsz_int,
        yolo_model=yolo_model_str,
        lstm_model=lstm_model_str,
        hmm_model_path=hmm_model_path_str, # HMMモデルパスを渡す
        overlay_mode=overlay_mode_str,
        cut_interval_mode=cut_interval_mode_bool, # インターバルカットモード
        cut_interval_threshold=cut_interval_threshold_float, # インターバルカット閾値
        extract_rally_mode=extract_rally_mode_bool, # Rally区間抽出モード
        rally_buffer_seconds=rally_buffer_seconds_float # Rally前後のバッファ秒数
    )
    
    run_pipeline(pipeline_args)
