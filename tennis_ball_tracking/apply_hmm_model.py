import argparse
from pathlib import Path
import numpy as np
import pandas as pd # load_data内でdf_loadedが使われるためインポートが必要になる場合がある
from hmm_postprocessor import HMMSupervisedPostprocessor

def main(hmm_model_path: Path, 
         input_csv_path: Path, 
         output_dir: Path, 
         pred_col_name: str,
         verbose: bool):

    print("=== HMM 後処理適用開始 ===")
    
    postprocessor = HMMSupervisedPostprocessor(verbose=verbose, random_state=42)

    # 1. HMMモデルの読み込み
    print(f"\n--- HMMモデル読み込み: {hmm_model_path} ---")
    if not postprocessor.load_hmm_model(hmm_model_path):
        print(f"❌ HMMモデルの読み込みに失敗しました: {hmm_model_path}")
        return
    print(f"✅ HMMモデル読み込み完了。状態数: {postprocessor.n_states}, ラベル: {postprocessor.states}")

    # 2. データの読み込み (真ラベルなし、メタデータはモデルから復元されたものを使用)
    print(f"\n--- 入力CSVデータ読み込み: {input_csv_path.name} ---")
    if not postprocessor.load_data(data_csv_path=input_csv_path,
                                   pred_col_name=pred_col_name,
                                   true_col_name=None, # 真ラベル列は指定しない
                                   metadata_json_path=None): # モデルのラベル情報を使うのでNone
        print("❌ データ読み込みに失敗しました。処理を中断します。")
        return
    
    if postprocessor.valid_observations_int is None:
        print("❌ 処理対象の観測シーケンスが読み込めませんでした。")
        return

    # 3. HMMによる平滑化
    print(f"\n--- HMMによる平滑化 ---")
    smoothed_sequence_int = postprocessor.smooth() # self.valid_observations_int を使う
    
    if smoothed_sequence_int is None:
        print("❌ HMMによる平滑化に失敗しました。")
        return

    # 4. 結果をDataFrameに追加
    print(f"\n--- 平滑化結果をDataFrameに追加 ---")
    if not postprocessor.int_to_label:
        print("❌ ラベル->整数 マッピング (int_to_label) がHMMモデルから復元されていません。")
        return

    smoothed_sequence_labels = np.array([postprocessor.int_to_label.get(s, "UNKNOWN_STATE") for s in smoothed_sequence_int])
    
    if not postprocessor.add_smoothed_results_to_df(smoothed_sequence_labels):
        print("⚠️ 平滑化結果のDataFrameへの追加に失敗しました。結果は保存されません。")
        return

    # 5. 結果の保存
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- 結果保存 ---")
    saved_path = postprocessor.save_results(postprocessor.df_loaded, 
                                            input_csv_path, 
                                            output_base_dir=output_dir)
    if saved_path:
        print(f"✅ HMM後処理済み予測結果を保存しました: {saved_path}")
    else:
        print("⚠️ 結果の保存に失敗しました。")

    print("\n=== HMM 後処理適用終了 ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済みHMMモデルを使用してLSTM予測結果を後処理します。")
    parser.add_argument("--hmm_model_path", type=Path, required=True, 
                        help="学習済みHMMモデルファイル (.joblib) のパス。例: ./training_data/hmm_models/hmm_model_supervised.joblib")
    parser.add_argument("--input_csv_path", type=Path, required=True, 
                        help="後処理対象のLSTM予測結果を含むCSVファイルのパス。例: ./training_data/merged_predictions/merged_some_video_predictions.csv")
    parser.add_argument("--output_dir", type=Path, default=Path("./hmm_applied_predictions"), 
                        help="処理結果CSVの出力先ディレクトリ。デフォルト: ./hmm_applied_predictions")
    parser.add_argument("--pred_col_name", type=str, default="predicted_phase", 
                        help="CSVファイル内のLSTM予測ラベルが含まれる列名。デフォルト: predicted_phase")
    parser.add_argument("--verbose", action="store_true", help="詳細情報を表示する。")
    
    args = parser.parse_args()

    main(args.hmm_model_path, args.input_csv_path, args.output_dir, args.pred_col_name, args.verbose)

    # 使用例:
    # python tennis_ball_tracking/apply_hmm_model.py \
    #   --hmm_model_path ./training_data/hmm_models/hmm_model_supervised.joblib \
    #   --input_csv_path ./training_data/merged_predictions/your_lstm_predictions.csv \
    #   --output_dir ./my_hmm_results \
    #   --verbose
