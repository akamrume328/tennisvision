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
from cut_non_rally_segments import cut_rally_segments # Rally区間抽出用にインポート

# argparse.Namespaceの代わりに使用するシンプルなクラス
class PipelineArgs:
    def __init__(self, video_path, output_dir, frame_skip, imgsz, yolo_model, lstm_model, hmm_model_path, overlay_mode, extract_rally_mode, rally_buffer_before_seconds, rally_buffer_after_seconds, min_rally_duration_seconds, min_phase_duration_seconds): # rally抽出パラメータを更新
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_skip = frame_skip
        self.imgsz = imgsz
        self.yolo_model = yolo_model
        self.lstm_model = lstm_model
        self.hmm_model_path = hmm_model_path # HMMモデルパスを追加
        self.overlay_mode = overlay_mode
        self.extract_rally_mode = extract_rally_mode # Rally区間抽出モードを追加
        self.rally_buffer_before_seconds = rally_buffer_before_seconds # Rally前のバッファ秒数
        self.rally_buffer_after_seconds = rally_buffer_after_seconds # Rally後のバッファ秒数
        self.min_rally_duration_seconds = min_rally_duration_seconds # 最小Rally区間長を追加
        self.min_phase_duration_seconds = min_phase_duration_seconds # 最小局面長を追加

def run_pipeline(args):
    pipeline_start_time = time.time() # パイプライン全体の開始時刻
    
    # 処理時間記録用の辞書を初期化
    step_times = {}

    video_path = Path(args.video_path)
    video_stem = video_path.stem
    output_dir = Path(args.output_dir)

    calibration_output_dir = output_dir / "00_calibration_data"
    tracking_output_dir = output_dir / "01_tracking_data"
    features_output_dir = output_dir / "02_extracted_features"
    predictions_output_dir = output_dir / "03_lstm_predictions"
    hmm_output_dir = output_dir / "03a_hmm_processed_predictions" # HMM処理結果用ディレクトリ
    final_output_dir = output_dir / "04_final_output"
    rally_extract_output_dir = output_dir / "06_rally_extract" # Rally区間抽出されたビデオ用ディレクトリ

    for p_dir in [calibration_output_dir, tracking_output_dir, features_output_dir, predictions_output_dir, hmm_output_dir, final_output_dir, rally_extract_output_dir]:
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
    step_1_duration = step_1_end_time - step_1_start_time
    step_times["ステップ 1 (コートキャリブレーション)"] = step_1_duration
    print(f"ステップ 1 (コートキャリブレーション) 処理時間: {step_1_duration:.2f} 秒")


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

    # フレーム最適化をframe_skip>1なら必ず有効にする
    use_optimized_reader = True if args.frame_skip > 1 else False
    if use_optimized_reader:
        print(f"⚡ 最適化フレーム読み込みを使用（{args.frame_skip}フレームに1回処理）")
    else:
        print("🖥️ 全フレーム処理モード（標準読み込み）")

    # BallTrackerを最新仕様で初期化
    tracker = BallTracker(
        model_path=args.yolo_model,
        imgsz=args.imgsz,
        save_training_data=True,
        data_dir=str(tracking_output_dir),
        frame_skip=args.frame_skip,
        enable_profiling=False,  # パイプラインでは無効
        use_optimized_reader=use_optimized_reader
    )

    # BallTrackerの統合フレームリーダーを使用
    try:
        fps, width, height, total_video_frames = tracker.initialize_video_processing(str(video_path))
        print(f"ビデオ情報: FPS={fps}, 総フレーム数={total_video_frames}")
        print(f"処理モード: {'⚡ 最適化フレーム読み込み' if use_optimized_reader else '🖥️ 標準フレーム読み込み'}")
    except Exception as e:
        print(f"エラー: 動画初期化に失敗しました - {e}")
        return

    # 進捗表示用
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("tqdmが見つかりません。進捗バーなしで実行します。")

    # フレーム処理ループ
    processed_frames = 0
    current_frame_idx = 0
    
    if use_tqdm and total_video_frames > 0:
        # 最適化使用時は処理予定フレーム数、標準時は全フレーム数
        expected_frames = total_video_frames // args.frame_skip if use_optimized_reader else total_video_frames
        progress_bar = tqdm(total=expected_frames, desc=f"トラッキング中 {video_stem}")
    else:
        progress_bar = None

    try:
        while True:
            # BallTrackerの統合フレーム読み込み
            ret, frame, frame_number = tracker.read_next_frame()
            if not ret:
                break
            
            current_frame_idx = frame_number

            # フレーム処理（最適化時は常に処理、標準時はスキップ判定）
            if use_optimized_reader:
                # 最適化リーダーは処理対象フレームのみ返すので直接処理
                tracker.process_frame_core(frame, frame_number, is_lightweight=True)
                processed_frames += 1
            else:
                # 標準処理でのフレームスキップ対応
                result_frame, was_processed = tracker.process_frame_optimized(
                    frame, frame_number, training_data_only=True
                )
                if was_processed:
                    processed_frames += 1

            # 進捗更新
            if progress_bar:
                progress_bar.update(1)

    except Exception as e:
        print(f"フレーム処理中にエラーが発生しました: {e}")
    finally:
        if progress_bar:
            progress_bar.close()
        
        # BallTrackerのリソース解放
        tracker.release_video_resources()

    # 処理結果の表示
    actual_processed_frames = processed_frames
    processing_efficiency = (processed_frames / current_frame_idx * 100) if current_frame_idx > 0 else 0
    expected_efficiency = (100 / args.frame_skip) if args.frame_skip > 1 else 100

    print(f"フレーム処理完了:")
    print(f"  総読み込みフレーム: {current_frame_idx}")
    print(f"  実処理フレーム: {actual_processed_frames}")
    print(f"  処理効率: {processing_efficiency:.1f}% (期待値: {expected_efficiency:.1f}%)")
    
    if use_optimized_reader:
        print(f"  ⚡ 最適化効果: 約{args.frame_skip}倍高速化")

    # total_video_framesが0または不正確だった場合、actual_processed_framesで更新
    if total_video_frames == 0 or abs(total_video_frames - current_frame_idx) > 5:
        print(f"警告: OpenCVの総フレーム数 ({total_video_frames}) と実読み込みフレーム数 ({current_frame_idx}) が異なります。")
        total_video_frames = current_frame_idx

    # トラッキング結果保存
    tracker.save_tracking_features_with_video_info(video_stem, fps, total_video_frames)
    
    step_2_end_time = time.time()
    step_2_duration = step_2_end_time - step_2_start_time
    step_times["ステップ 2 (ボールトラッキング)"] = step_2_duration
    print(f"ステップ 2 (ボールトラッキング) 処理時間: {step_2_duration:.2f} 秒")

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
    step_3_duration = step_3_end_time - step_3_start_time
    step_times["ステップ 3 (特徴量抽出)"] = step_3_duration
    print(f"ステップ 3 (特徴量抽出) 処理時間: {step_3_duration:.2f} 秒")

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
    step_4_duration = step_4_end_time - step_4_start_time
    step_times["ステップ 4 (LSTM予測)"] = step_4_duration
    print(f"ステップ 4 (LSTM予測) 処理時間: {step_4_duration:.2f} 秒")


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
    step_4_5_duration = step_4_5_end_time - step_4_5_start_time
    if args.hmm_model_path or hmm_processing_done: # HMMパスが指定されたか、実際に処理が試みられた場合のみ時間表示
        step_times["ステップ 4.5 (HMM後処理)"] = step_4_5_duration
        print(f"ステップ 4.5 (HMM後処理) 処理時間: {step_4_5_duration:.2f} 秒")

    # --- ステップ 5: 予測結果のオーバーレイ (Rally抽出モードが無効な場合のみ) ---
    if not args.extract_rally_mode:
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
            video_fps=fps,  # video_fps -> fps に修正
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
        step_5_duration = step_5_end_time - step_5_start_time
        step_times["ステップ 5 (予測結果のオーバーレイ)"] = step_5_duration
        print(f"ステップ 5 (予測結果のオーバーレイ) 処理時間: {step_5_duration:.2f} 秒")
    else:
        print("\nℹ️ Rally抽出モードが有効なため、ステップ5 (予測結果のオーバーレイ) はスキップされました。")

    # --- ステップ 6: Rally区間の抽出 (オプション) ---
    csv_for_rally_extraction_path = hmm_processed_csv_path 

    if args.extract_rally_mode:
        print("\n--- ステップ 6: Rally区間の抽出 ---")
        step_6_start_time = time.time()
        
        input_video_for_rally_extraction = video_path # 常に元のビデオを対象とする
        print(f"Rally抽出処理の入力として元のビデオを使用: {input_video_for_rally_extraction}")

        rally_video_filename = f"{input_video_for_rally_extraction.stem}_rally_only.mp4"
        output_rally_video_path = rally_extract_output_dir / rally_video_filename

        print(f"Rally抽出処理に使用するCSV: {csv_for_rally_extraction_path}")
        print(f"Rally区間抽出ビデオの出力先: {output_rally_video_path}")
        print(f"Rally前バッファ: {args.rally_buffer_before_seconds} 秒")
        print(f"Rally後バッファ: {args.rally_buffer_after_seconds} 秒")

        success_rally_extract = cut_rally_segments(
            video_path=Path(input_video_for_rally_extraction),
            csv_path=Path(csv_for_rally_extraction_path),
            output_path=output_rally_video_path,
            buffer_before=args.rally_buffer_before_seconds,
            buffer_after=args.rally_buffer_after_seconds,
            min_rally_duration=args.min_rally_duration_seconds,
            min_phase_duration=args.min_phase_duration_seconds
        )

        if success_rally_extract:
            print(f"Rally区間抽出処理が完了しました。出力ビデオ: {output_rally_video_path}")
        else:
            print("Rally区間抽出処理に失敗しました。")
        
        step_6_end_time = time.time()
        step_6_duration = step_6_end_time - step_6_start_time
        step_times["ステップ 6 (Rally区間の抽出)"] = step_6_duration
        print(f"ステップ 6 (Rally区間の抽出) 処理時間: {step_6_duration:.2f} 秒")

    pipeline_end_time = time.time() # パイプライン全体の終了時刻
    total_pipeline_duration = pipeline_end_time - pipeline_start_time
    step_times["全体パイプライン"] = total_pipeline_duration
    print(f"\nパイプラインが完了しました。")
    print(f"総処理時間: {total_pipeline_duration:.2f} 秒")

    # --- 処理時間レポートの保存 ---
    print("\n--- 処理時間レポートの保存 ---")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"processing_time_report_{video_stem}_{timestamp}.txt"
        report_path = output_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("テニスビデオ分析パイプライン 処理時間レポート\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"ビデオファイル: {video_path}\n")
            f.write(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
            f.write(f"フレームスキップ: {args.frame_skip}\n")
            f.write(f"画像サイズ: {args.imgsz}\n")
            f.write(f"YOLOモデル: {args.yolo_model}\n")
            f.write(f"LSTMモデル: {args.lstm_model}\n")
            f.write(f"HMMモデル: {args.hmm_model_path if args.hmm_model_path else '使用なし'}\n")
            f.write(f"オーバーレイモード: {args.overlay_mode}\n")
            f.write(f"Rally抽出モード: {'有効' if args.extract_rally_mode else '無効'}\n")
            if args.extract_rally_mode:
                f.write(f"Rally前バッファ: {args.rally_buffer_before_seconds} 秒\n")
                f.write(f"Rally後バッファ: {args.rally_buffer_after_seconds} 秒\n")
            f.write("\n" + "-" * 60 + "\n")
            f.write("各ステップの処理時間\n")
            f.write("-" * 60 + "\n\n")
            
            for step_name, duration in step_times.items():
                if step_name == "全体パイプライン":
                    continue
                f.write(f"{step_name:<30}: {duration:>8.2f} 秒\n")
            
            f.write("\n" + "-" * 60 + "\n")
            f.write(f"{'総処理時間':<30}: {total_pipeline_duration:>8.2f} 秒\n")
            f.write("=" * 60 + "\n")
        
        print(f"処理時間レポートを保存しました: {report_path}")
        
    except Exception as e:
        print(f"処理時間レポートの保存中にエラーが発生しました: {e}")

if __name__ == "__main__":
    print("テニスビデオ分析パイプラインへようこそ！")
    print("いくつかの情報を入力してください。デフォルト値を使用する場合はEnterキーを押してください。")

    video_path_str = ""
    raw_data_dir = Path("./data/raw")
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
    
    frame_skip_str = input("フレームスキップ (デフォルト: 10): ").strip() or "10"
    try:
        frame_skip_int = int(frame_skip_str)
        if frame_skip_int < 1:
            print("フレームスキップは1以上である必要があります。デフォルトの10を使用します。")
            frame_skip_int = 1
    except ValueError:
        print("無効なフレームスキップ値です。デフォルトの1を使用します。")
        frame_skip_int = 1

    imgsz_str = input("YOLOモデル推論時の画像サイズ (デフォルト: 1920): ").strip() or "1920"
    try:
        imgsz_int = int(imgsz_str)
    except ValueError:
        print("無効な画像サイズです。デフォルトの1920を使用します。")
        imgsz_int = 1920

    yolo_model_str = ""
    models_weights_dir = Path("./models/yolo_model")
    pt_files = []

    if models_weights_dir.exists() and models_weights_dir.is_dir():
        print(f"\n利用可能なYOLOモデルファイル ({models_weights_dir}):")
        for i, file_path in enumerate(models_weights_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() == ".engine":
                pt_files.append(file_path)
                print(f"  {len(pt_files)}. {file_path.name}")
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
    lstm_models_base_dir = Path("./models")
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
    hmm_models_base_dir = Path("./models/hmm_model")
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

    extract_rally_mode_input = input("Rally区間のみを抽出しますか？ (yes/no, デフォルト: no): ").strip().lower()
    extract_rally_mode_bool = extract_rally_mode_input == 'yes'
    
    rally_buffer_before_seconds_float = 2.0 # デフォルト値
    rally_buffer_after_seconds_float = 2.0 # デフォルト値
    min_rally_duration_seconds_float = 2.0 # 最小Rally区間長のデフォルト値
    min_phase_duration_seconds_float = 0.5 # 最小局面長のデフォルト値
    
    if extract_rally_mode_bool:
        buffer_before_str = input("Rally区間の前に保持する秒数（デフォルト: 2.0）: ").strip()
        if buffer_before_str:
            try:
                rally_buffer_before_seconds_float = float(buffer_before_str)
                if rally_buffer_before_seconds_float < 0:
                    print("Rally前バッファ秒数は0以上である必要があります。デフォルトの2.0秒を使用します。")
                    rally_buffer_before_seconds_float = 2.0
            except ValueError:
                print("無効なRally前バッファ秒数です。デフォルトの2.0秒を使用します。")
                rally_buffer_before_seconds_float = 2.0

        buffer_after_str = input("Rally区間の後に保持する秒数（デフォルト: 2.0）: ").strip()
        if buffer_after_str:
            try:
                rally_buffer_after_seconds_float = float(buffer_after_str)
                if rally_buffer_after_seconds_float < 0:
                    print("Rally後バッファ秒数は0以上である必要があります。デフォルトの2.0秒を使用します。")
                    rally_buffer_after_seconds_float = 2.0
            except ValueError:
                print("無効なRally後バッファ秒数です。デフォルトの2.0秒を使用します。")
                rally_buffer_after_seconds_float = 2.0

        min_rally_duration_str = input("最小Rally区間長（秒）（デフォルト: 2.0）: ").strip()
        if min_rally_duration_str:
            try:
                min_rally_duration_seconds_float = float(min_rally_duration_str)
                if min_rally_duration_seconds_float < 0:
                    print("最小Rally区間長は0以上である必要があります。デフォルトの2.0秒を使用します。")
                    min_rally_duration_seconds_float = 2.0
            except ValueError:
                print("無効な最小Rally区間長です。デフォルトの2.0秒を使用します。")
                min_rally_duration_seconds_float = 2.0

        min_phase_duration_str = input("最小局面長（秒）（デフォルト: 0.5）: ").strip()
        if min_phase_duration_str:
            try:
                min_phase_duration_seconds_float = float(min_phase_duration_str)
                if min_phase_duration_seconds_float < 0:
                    print("最小局面長は0以上である必要があります。デフォルトの0.5秒を使用します。")
                    min_phase_duration_seconds_float = 0.5
            except ValueError:
                print("無効な最小局面長です。デフォルトの0.5秒を使用します。")
                min_phase_duration_seconds_float = 0.5

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
        extract_rally_mode=extract_rally_mode_bool, # Rally区間抽出モード
        rally_buffer_before_seconds=rally_buffer_before_seconds_float, # Rally前のバッファ秒数
        rally_buffer_after_seconds=rally_buffer_after_seconds_float, # Rally後のバッファ秒数
        min_rally_duration_seconds=min_rally_duration_seconds_float, # 最小Rally区間長を追加
        min_phase_duration_seconds=min_phase_duration_seconds_float # 最小局面長を追加
    )
    
    run_pipeline(pipeline_args)
