import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from datetime import datetime
import argparse

def predict_from_saved_model(model_path: str, input_csv_path: str, output_csv_path: str = None):
    """
    保存されたLightGBMモデルを読み込み、新しいデータに対して予測を行う。
    """
    model_path = Path(model_path)
    input_csv_path = Path(input_csv_path)

    # 1. モデルの読み込み
    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
    
    print(f"モデルを読み込んでいます: {model_path}")
    model = joblib.load(model_path)

    # 2. 予測用データの読み込み
    if not input_csv_path.exists():
        raise FileNotFoundError(f"入力CSVファイルが見つかりません: {input_csv_path}")

    print(f"予測用データを読み込んでいます: {input_csv_path}")
    df_inference = pd.read_csv(input_csv_path)

    # 3. 特徴量の整合性を確認
    # モデルが学習した際の特徴量名リストを取得
    model_features = model.feature_name_
    
    # 入力データから、モデルが学習した特徴量のみを、同じ順序で抽出
    # もし入力データに存在しない特徴量があれば、エラーを出すか、デフォルト値で埋めるなどの処理も可能
    missing_features = [f for f in model_features if f not in df_inference.columns]
    if missing_features:
        raise ValueError(f"入力データに必須の特徴量が不足しています: {missing_features}")

    X_inference = df_inference[model_features]
    print(f"{len(model_features)}個の特徴量を使って予測を実行します。")

    # 4. 予測の実行
    print("予測を開始します...")
    # 各クラスに属する確率を予測
    predicted_probas = model.predict_proba(X_inference)
    # 最も確率が高いクラスを予測結果とする
    predicted_ids = np.argmax(predicted_probas, axis=1)
    # 予測したクラスの確信度（確率）を取得
    prediction_confidence = np.max(predicted_probas, axis=1)
    print("予測が完了しました。")

    # 5. 結果の整形と保存
    # IDからラベル名への変換マップ
    phase_labels_map = {
        0: "point_interval", 1: "rally", 2: "serve_front_deuce", 3: "serve_front_ad",
        4: "serve_back_deuce", 5: "serve_back_ad", 6: "changeover"
    }
    
    # 結果を格納する新しいDataFrameを作成
    results_df = pd.DataFrame()
    
    # 識別用に元のフレーム情報などをコピー
    id_columns = ['video_name', 'frame_number', 'original_frame_number']
    for col in id_columns:
        if col in df_inference.columns:
            results_df[col] = df_inference[col]

    results_df['predicted_label_id'] = predicted_ids
    results_df['predicted_phase'] = results_df['predicted_label_id'].map(phase_labels_map)
    results_df['prediction_confidence'] = prediction_confidence
    
    # 出力ファイルパスの決定
    if not output_csv_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = input_csv_path.parent.parent / "predictions"
        output_dir.mkdir(exist_ok=True)
        output_csv_path = output_dir / f"predictions_{input_csv_path.stem}_{timestamp}.csv"

    # CSVファイルとして保存
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n予測結果を保存しました: {output_csv_path}")
    print(results_df.head().to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="学習済みLightGBMモデルによる推論スクリプト")
    parser.add_argument('--model_path', type=str, required=True, 
                        help="学習済みモデルのパス (.pkl)")
    parser.add_argument('--input_csv', type=str, required=True, 
                        help="予測対象のデータCSV（特徴量抽出済み）")
    parser.add_argument('--output_csv', type=str, default=None,
                        help="(オプション) 予測結果を保存するCSVのパス")
    
    args = parser.parse_args()
    
    predict_from_saved_model(
        model_path=args.model_path,
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv
    )