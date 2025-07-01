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
import warnings
import shap

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

warnings.filterwarnings('ignore', category=UserWarning)

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
    """メインの学習パイプライン（交差検証対応）"""
    parser = argparse.ArgumentParser(description="LightGBMによるテニス局面分類モデル学習ツール（交差検証対応）")
    parser.add_argument('--config', type=str, default='config_lightgbm.yaml', help='設定ファイルのパス')
    parser.add_argument('--csv_path', type=str, default=None, help='(オプション) 使用するデータセットCSVファイルのパス')
    args = parser.parse_args()

    # 1. 設定とデータの読み込み
    config = load_config(args.config)
    df, loaded_csv_path = load_dataset(Path(config['training_data_dir']), args.csv_path)
    final_model = None  # 最終モデルを保存するための変数

    # 2. データの前処理
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    exclude_cols = ['label', 'video_name', 'frame_number', 'original_frame_number', 'timestamp', 'interpolated']
    feature_columns = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_columns]
    y = df['label']
    groups = df['video_name']

    feature_list_path = config.get('feature_list_path')
    if feature_list_path and Path(feature_list_path).exists():
        print(f"\n特徴量リストを読み込んでいます: {feature_list_path}")
        with open(feature_list_path, 'r', encoding='utf-8') as f:
            top_features = [line.strip() for line in f]
        top_features_existing = [f for f in top_features if f in X.columns]
        print(f"上位{len(top_features_existing)}個の特徴量のみを使用して学習します。")
        X = X[top_features_existing]
        feature_columns = top_features_existing
    else:
        print("\n全特徴量を使用して学習します。")

    phase_labels_map = {
        0: "point_interval", 1: "rally", 2: "serve_front_deuce", 3: "serve_front_ad",
        4: "serve_back_deuce", 5: "serve_back_ad", 6: "changeover"
    }
    active_labels_indices = sorted(y.unique())
    active_phase_labels = [phase_labels_map[i] for i in active_labels_indices]
    
    # 3. 交差検証の準備
    n_splits = config.get('n_splits', 5)
    gkf = GroupKFold(n_splits=n_splits)

    df['oof_prediction_id'] = -1  # OOF予測用の列を追加
    
    all_preds = []
    all_true = []
    fold_scores = {'accuracy': [], 'f1_score': []}
    # 特徴量の重要度を全Fold分保存するためのDataFrame
    feature_importances = pd.DataFrame(index=feature_columns)

    print(f"\n===== {n_splits}分割グループ交差検証を開始します =====")

    # 4. 交差検証ループ
    for fold, (train_val_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n--- FOLD {fold + 1}/{n_splits} ---")
        
        # --- データ分割 ---
        # Foldのテストデータをまず確保
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # 訓練・検証データとグループを確保
        X_train_val, y_train_val = X.iloc[train_val_idx], y.iloc[train_val_idx]
        groups_train_val = groups.iloc[train_val_idx]

        # ★★★ 早期停止のための検証データを訓練データから分割（情報リーク防止） ★★★
        gss_val = GroupShuffleSplit(n_splits=1, test_size=config['validation_size'], random_state=config['random_state'])
        train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))
        
        X_train, y_train = X_train_val.iloc[train_idx], y_train_val.iloc[train_idx]
        X_val, y_val = X_train_val.iloc[val_idx], y_train_val.iloc[val_idx]

        print(f"学習データ: {len(X_train)}件, 検証データ: {len(X_val)}件, テストデータ: {len(X_test)}件")

        # --- モデル学習 ---
        model = lgb.LGBMClassifier(**config['lgbm_params'])
        callbacks = [
            lgb.early_stopping(stopping_rounds=config['early_stopping_rounds'], verbose=True),
            lgb.log_evaluation(period=100)
        ]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)], # テストデータではなく、検証用データで監視
            eval_metric=config['lgbm_params'].get('metric', 'multi_logloss'),
            callbacks=callbacks
        )
        
        # --- 評価と結果の集計 ---
        y_pred = model.predict(X_test)
        df.loc[test_idx, 'oof_prediction_id'] = y_pred  # OOF予測結果を保存
        all_preds.extend(y_pred)
        all_true.extend(y_test)
        
        fold_acc = accuracy_score(y_test, y_pred)
        fold_f1 = f1_score(y_test, y_pred, average='weighted')
        fold_scores['accuracy'].append(fold_acc)
        fold_scores['f1_score'].append(fold_f1)
        print(f"FOLD {fold + 1} 結果: Accuracy={fold_acc:.4f}, F1-score={fold_f1:.4f}")

        # 特徴量の重要度を保存
        feature_importances[f'fold_{fold+1}'] = model.feature_importances_

    # 5. 交差検証の最終結果評価
    print("\n===== 交差検証 最終結果 =====")
    avg_acc = np.mean(fold_scores['accuracy'])
    std_acc = np.std(fold_scores['accuracy'])
    avg_f1 = np.mean(fold_scores['f1_score'])
    std_f1 = np.std(fold_scores['f1_score'])
    
    print(f"平均精度: {avg_acc:.4f} (+/- {std_acc:.4f})")
    print(f"平均F1スコア: {avg_f1:.4f} (+/- {std_f1:.4f})")
    
    print("\n総合分類レポート (全Fold):")
    report = classification_report(all_true, all_preds, target_names=active_phase_labels, zero_division=0)
    print(report)

    # 6. 結果の保存
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 混同行列
    cm = confusion_matrix(all_true, all_preds, labels=active_labels_indices)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=active_phase_labels, yticklabels=active_phase_labels)
    plt.title(f'Overall Confusion Matrix (Avg Acc: {avg_acc:.4f})')
    plt.xlabel('Predicted Phase'); plt.ylabel('True Phase'); plt.tight_layout()
    cm_path = output_dir / f'cv_confusion_matrix_{timestamp}.png'
    plt.savefig(cm_path)
    print(f"\n総合混同行列を保存しました: {cm_path}")

    # 特徴量の重要度（平均）
    feature_importances['mean'] = feature_importances.mean(axis=1)
    feature_importances.sort_values('mean', ascending=False, inplace=True)
    
    plt.figure(figsize=(12, 18))
    sns.barplot(x='mean', y=feature_importances.index[:50], data=feature_importances.head(50))
    plt.title('Average Feature Importance over Folds (Top 50)')
    plt.tight_layout()
    fi_path = output_dir / f'cv_feature_importance_{timestamp}.png'
    plt.savefig(fi_path)
    print(f"平均特徴量重要度グラフを保存しました: {fi_path}")
        # 分類レポートをテキストファイルに保存
    report_path = output_dir / f'cv_classification_report_{timestamp}.txt'
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("===== 交差検証 最終結果 =====\n\n")
            f.write(f"平均精度: {avg_acc:.4f} (+/- {std_acc:.4f})\n")
            f.write(f"平均F1スコア: {avg_f1:.4f} (+/- {std_f1:.4f})\n\n")
            f.write("総合分類レポート (全Fold):\n")
            f.write(report) # classification_report の結果を書き込む
        print(f"分類レポートをテキストファイルに保存しました: {report_path}")
    except Exception as e:
        print(f"❌ 分類レポートの保存中にエラーが発生しました: {e}")

    # 7. (オプション) 全データで最終モデルを再学習・保存
    if config.get('retrain_final_model', False):
        print("\n全データを使用して最終モデルを再学習します...")
        final_model = lgb.LGBMClassifier(**config['lgbm_params'])
        # ここでは早期停止なしで、交差検証で得られた最適なイテレーション数などを参考に学習する
        # best_iterationは各Foldで異なるため、ここでは単純に全データでfitさせる
        final_model.fit(X, y)
        print("最終モデルの学習が完了しました。")
        
        model_path = output_dir / f'lgbm_final_model_{timestamp}.pkl'
        joblib.dump(final_model, model_path)
        print(f"最終モデルを保存しました: {model_path}")
    
    # メタデータの保存
    metadata = {
        'timestamp': timestamp,
        'model_type': 'LightGBM_CV',
        'dataset_path': loaded_csv_path,
        'config': config,
        'phase_labels': active_phase_labels,
        'cv_evaluation': {
            'avg_accuracy': avg_acc, 'std_accuracy': std_acc,
            'avg_f1_score': avg_f1, 'std_f1_score': std_f1,
            'classification_report': classification_report(all_true, all_preds, target_names=active_phase_labels, output_dict=True, zero_division=0)
        },
        'feature_names': feature_columns
    }
    metadata_path = output_dir / f'metadata_cv_{timestamp}.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"メタデータを保存しました: {metadata_path}")

    # 8. SHAPによる局面ごとの特徴量重要度の分析
    if final_model is not None:
        print("\n===== SHAPによる局面ごとの特徴量重要度分析を開始します =====")
        
        sample_size = min(config.get('shap_sample_size', 5000), len(X))
        X_sample = X.sample(n=sample_size, random_state=config.get('random_state', 42))
        
        print(f"{sample_size}件のサンプルデータでSHAP値を計算します...")
        
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_sample)
        
        print("SHAP値の計算が完了しました。要約プロットを生成します...")
        
        # --- SHAPプロットの保存（変更なし） ---
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_sample,
            plot_type="bar",
            class_names=active_phase_labels,
            max_display=50,
            show=False
        )
        fig = plt.gcf()
        fig.set_size_inches(15, 25)
        plt.title('SHAP Feature Importance by Phase', fontsize=16)
        plt.tight_layout()
        
        shap_plot_path = output_dir / f'shap_summary_plot_{timestamp}.png'
        plt.savefig(shap_plot_path)
        plt.close()
        
        print(f"✅ SHAPの要約プロットを保存しました: {shap_plot_path}")

        # --- ▼▼▼ ここからが今回の修正箇所です ▼▼▼ ---
        
        print("SHAPの数値をCSVファイルに保存します...")
        try:
            # 1. 空のDataFrameを、正しいインデックス（特徴量名）で作成
            df_shap_values = pd.DataFrame(index=X_sample.columns)

            # 2. ループでクラスごとに計算し、列として追加していく
            for i, class_name in enumerate(active_phase_labels):
                # クラスiに対する全サンプルのSHAP値の絶対値をとり、特徴量ごとに平均を計算
                # この結果は (特徴量数,) の形状の配列になる
                mean_abs_shap = np.abs(shap_values[i]).mean(axis=0)
                df_shap_values[class_name] = mean_abs_shap

            # 3. 全体的な重要度を計算してソート
            df_shap_values['global_importance'] = df_shap_values.sum(axis=1)
            df_shap_values.sort_values('global_importance', ascending=False, inplace=True)
            
            # 4. CSVファイルとして保存
            shap_csv_path = output_dir / f'shap_values_{timestamp}.csv'
            df_shap_values.to_csv(shap_csv_path, encoding='utf-8-sig')
            
            print(f"✅ SHAPの数値データを保存しました: {shap_csv_path}")

        except Exception as e:
            print(f"❌ SHAPの数値データ保存中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc() # 詳細なエラー情報を表示

    # OOF予測結果をCSVに保存
    if config.get('save_oof_predictions', True):
        print("\nOOF予測結果を保存します...")
        df['predicted_phase'] = df['oof_prediction_id'].map(phase_labels_map)
        oof_output_df = df[[
            'video_name', 
            'frame_number', 
            'label', # 正解ラベル
            'predicted_phase' # OOF予測ラベル
        ]].copy()
        
        # oof_prediction_id が -1 のままの行（何らかの理由で予測されなかった行）を除外
        oof_output_df = oof_output_df[df['oof_prediction_id'] != -1].copy()

        oof_csv_path = output_dir / f'lgbm_oof_predictions_{timestamp}.csv'
        oof_output_df.to_csv(oof_csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ HMM学習用のOOF予測データを保存しました: {oof_csv_path}")

    print("\n🎉 LightGBMの交差検証とOOF予測の生成が完了しました。")


if __name__ == '__main__':
    main()