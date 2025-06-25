import joblib
import pandas as pd
import argparse
from pathlib import Path

def extract_and_save_top_features(model_path: str, output_path: str, top_n: int = 100):
    """
    学習済みLightGBMモデルから特徴量の重要度を抽出し、
    上位N個の特徴量名をテキストファイルに保存する。
    """
    print(f"モデルを読み込んでいます: {model_path}")
    model = joblib.load(model_path)
    
    # 特徴量名と重要度をデータフレームにまとめる
    feature_importances = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n--- 特徴量の重要度 (上位10) ---")
    print(feature_importances.head(10).to_string(index=False))
    
    # 上位N個の特徴量名を取得
    top_features = feature_importances.head(top_n)['feature'].tolist()
    
    # ファイルに保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for feature in top_features:
            f.write(f"{feature}\n")
            
    print(f"\n上位{top_n}個の特徴量をファイルに保存しました: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済みモデルから上位の特徴量を抽出するツール")
    
    # ユーザーがモデルファイルのパスを指定できるように引数を追加
    parser.add_argument('--model_path', type=str, required=True, 
                        help="特徴量を抽出する学習済みモデルのパス（.pklファイル）")
    parser.add_argument('--output_path', type=str, default='lgbm_models/top_100_features.txt',
                        help="抽出した特徴量リストを保存するファイルのパス")
    parser.add_argument('--top_n', type=int, default=100,
                        help="抽出する上位特徴量の数")
    
    args = parser.parse_args()
    
    extract_and_save_top_features(
        model_path=args.model_path,
        output_path=args.output_path,
        top_n=args.top_n
    )