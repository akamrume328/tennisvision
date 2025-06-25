import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import yaml
import joblib
import json
from pathlib import Path
from datetime import datetime
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_config(path: str) -> dict:
    """YAML設定ファイルを読み込む"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"設定ファイルを読み込みました: {path}")
        return config
    except Exception as e:
        print(f"エラー: 設定ファイルの読み込み中にエラーが発生しました: {e}")
        raise

def load_dataset(data_dir: Path, csv_path: str = None) -> pd.DataFrame:
    """データセットを読み込む"""
    if not csv_path:
        # featuresディレクトリも含めて最新のCSVを探す
        all_files = list(data_dir.glob("features/*.csv")) + list(data_dir.glob("*.csv"))
        if not all_files:
            raise FileNotFoundError(f"データディレクトリにCSVファイルが見つかりません: {data_dir}")
        csv_path = str(max(all_files, key=lambda p: p.stat().st_mtime))

    dataset_path = Path(csv_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"指定されたファイルが見つかりません: {dataset_path}")

    print(f"データセット読み込み: {dataset_path.name}")
    df = pd.read_csv(dataset_path)
    print(f"読み込み完了: {len(df):,} 行")
    return df, csv_path

def main():
    """メインの学習パイプライン"""
    parser = argparse.ArgumentParser(description="LightGBMによるテニス局面分類モデル学習ツール")
    parser.add_argument('--config', type=str, default='config_lightgbm.yaml', help='設定ファイルのパス')
    parser.add_argument('--csv_path', type=str, default=None, help='(オプション) 使用するデータセットCSVファイルのパス')
    args = parser.parse_args()

    # 1. 設定とデータの読み込み
    config = load_config(args.config)
    df, loaded_csv_path = load_dataset(Path(config['training_data_dir']), args.csv_path)

    # 2. データの前処理
    # ラベルの欠損値除去と型変換
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    # 特徴量(X)とラベル(y)の定義
    exclude_cols = ['label', 'video_name', 'frame_number', 'original_frame_number', 'timestamp', 'interpolated']
    feature_columns = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_columns]
    y = df['label']
    
    # 設定ファイルから特徴量リストのパスを取得
    feature_list_path = config.get('feature_list_path')
    if feature_list_path and Path(feature_list_path).exists():
        print(f"\n特徴量リストを読み込んでいます: {feature_list_path}")
        with open(feature_list_path, 'r', encoding='utf-8') as f:
            top_features = [line.strip() for line in f]
        
        # 実際にデータに存在する特徴量のみに絞る
        top_features_existing = [f for f in top_features if f in X.columns]
        
        print(f"上位{len(top_features_existing)}個の特徴量のみを使用して学習します。")
        X = X[top_features_existing]
        feature_columns = top_features_existing # 後で使うために更新
    else:
        print("\n全特徴量を使用して学習します。")

    # ラベル名の定義（分類レポート用）
    phase_labels_map = {
        0: "point_interval", 1: "rally", 2: "serve_front_deuce", 3: "serve_front_ad",
        4: "serve_back_deuce", 5: "serve_back_ad", 6: "changeover"
    }
    active_labels_indices = sorted(y.unique())
    active_phase_labels = [phase_labels_map[i] for i in active_labels_indices]

    # 3. データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['test_size'],
        random_state=config['random_state'],
        stratify=y  # ラベルの比率を保って分割
    )
    print(f"データを分割しました: 学習データ {len(X_train)}件, テストデータ {len(X_test)}件")

    # 4. モデルの学習
    print("\nLightGBMモデルの学習を開始します...")
    model = lgb.LGBMClassifier(**config['lgbm_params'])
    
    # 早期停止（Early Stopping）を用いて学習
    callbacks = [
        lgb.early_stopping(stopping_rounds=config['early_stopping_rounds'], verbose=True),
        lgb.log_evaluation(period=100)
    ]
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=config['lgbm_params']['metric'],
        callbacks=callbacks
    )
    print("モデルの学習が完了しました。")

    # 5. モデルの評価
    print("\n--- モデル評価 ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"テストデータ精度 (Accuracy): {accuracy:.4f}")
    
    print("\n分類レポート:")
    report = classification_report(y_test, y_pred, target_names=active_phase_labels, zero_division=0)
    print(report)

    # 6. 結果の保存
    # 出力ディレクトリの作成
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 混同行列のプロットと保存
    cm = confusion_matrix(y_test, y_pred, labels=active_labels_indices)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=active_phase_labels, yticklabels=active_phase_labels)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
    plt.xlabel('Predicted Phase')
    plt.ylabel('True Phase')
    plt.tight_layout()
    cm_path = output_dir / f'confusion_matrix_{timestamp}.png'
    plt.savefig(cm_path)
    print(f"\n混同行列を保存しました: {cm_path}")
    
    # 特徴量の重要度のプロットと保存
    lgb.plot_importance(model, figsize=(12, 18), max_num_features=50, importance_type='gain')
    plt.title('Feature Importance (Top 50)')
    plt.tight_layout()
    fi_path = output_dir / f'feature_importance_{timestamp}.png'
    plt.savefig(fi_path)
    print(f"特徴量の重要度グラフを保存しました: {fi_path}")

    # 学習済みモデルの保存
    model_path = output_dir / f'lgbm_model_{timestamp}.pkl'
    joblib.dump(model, model_path)
    print(f"学習済みモデルを保存しました: {model_path}")

    # メタデータの保存
    metadata = {
        'timestamp': timestamp,
        'model_type': 'LightGBM',
        'dataset_path': loaded_csv_path,
        'config': config,
        'evaluation': {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, target_names=active_phase_labels, output_dict=True, zero_division=0)
        },
        'best_iteration': model.best_iteration_,
        'feature_names': feature_columns
    }
    metadata_path = output_dir / f'metadata_{timestamp}.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"メタデータを保存しました: {metadata_path}")


if __name__ == '__main__':
    main()