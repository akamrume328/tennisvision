# compare_feature_pipelines.py (最終版)

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
import warnings

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ===== GPU設定 =====
def setup_gpu_config():
    if torch.cuda.is_available():
        print("GPU検出: CUDAが利用可能です。")
        return torch.device('cuda')
    else:
        print("警告: GPU未検出: CPUで実行します")
        return torch.device('cpu')
DEVICE = setup_gpu_config()

# ===== モデル定義 =====
class TennisLSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, 
                 dropout_rate: float, model_type: str, use_batch_norm: bool, 
                 enable_confidence_weighting: bool):
        super(TennisLSTMModel, self).__init__()
        self.input_size, self.hidden_sizes, self.num_classes, self.dropout_rate, self.model_type, self.use_batch_norm, self.enable_confidence_weighting = input_size, hidden_sizes, num_classes, dropout_rate, model_type, use_batch_norm, enable_confidence_weighting
        if self.enable_confidence_weighting:
            self.confidence_attention = nn.Sequential(nn.Linear(input_size, hidden_sizes[0] // 4), nn.ReLU(), nn.Linear(hidden_sizes[0] // 4, 1), nn.Sigmoid())
        self._build_lstm_layers()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1, self.fc2, self.fc_out = nn.Linear(self.lstm_output_size, 128), nn.Linear(128, 64), nn.Linear(64, num_classes)
        if self.use_batch_norm:
            self.bn1, self.bn2 = nn.BatchNorm1d(128), nn.BatchNorm1d(64)
        else:
            self.ln1, self.ln2 = nn.LayerNorm(128), nn.LayerNorm(64)
        self._init_weights()
    
    def _build_lstm_layers(self):
        num_layers_to_build = len(self.hidden_sizes)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_sizes[0], num_layers=num_layers_to_build, batch_first=True, dropout=self.dropout_rate if num_layers_to_build > 1 else 0, bidirectional=(self.model_type == "bidirectional"))
        self.lstm_output_size = self.hidden_sizes[0] * 2 if self.model_type == "bidirectional" else self.hidden_sizes[0]

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name: nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name: nn.init.orthogonal_(param)
                elif 'bias' in name: nn.init.zeros_(param)
            elif 'fc' in name or 'confidence_attention' in name:
                if 'weight' in name: nn.init.xavier_uniform_(param)
                elif 'bias' in name: nn.init.zeros_(param)

    def forward(self, x, confidence_scores=None):
        if self.enable_confidence_weighting and confidence_scores is not None:
            attention_weights = self.confidence_attention(x).squeeze(-1)
            combined_weights = attention_weights * confidence_scores
            x = x * combined_weights.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        out = F.relu(self.fc1(lstm_out))
        out = self._apply_normalization(out, self.bn1 if self.use_batch_norm else self.ln1)
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self._apply_normalization(out, self.bn2 if self.use_batch_norm else self.ln2)
        out = self.dropout(out)
        return self.fc_out(out)

    def _apply_normalization(self, x, norm_layer):
        return norm_layer(x) if x.size(0) > 1 else x if self.use_batch_norm else norm_layer(x)

# ===== パイプライン比較クラス =====
class PipelineComparator:
    def __init__(self, models_dir: str, v2_features_dir: str, predict_features_dir: str):
        self.models_dir = Path(models_dir)
        self.v2_features_dir = Path(v2_features_dir)
        self.predict_features_dir = Path(predict_features_dir)
        self.model, self.scaler, self.metadata = None, None, None
        self.device = DEVICE
        self.phase_labels, self.label_map_inv = [], None

    def _select_file(self, directory: Path, pattern: str, description: str) -> Optional[Path]:
        print(f"\n=== {description} ===")
        print(f"'{directory}' ディレクトリから探します...")
        files = sorted(list(directory.glob(pattern)), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            print(f"❌ ファイル ({pattern}) が見つかりません。")
            return None
        for i, f_path in enumerate(files, 1):
            try:
                display_path = f_path.relative_to(Path.cwd())
            except ValueError:
                display_path = f_path
            print(f"  {i}. {display_path} (更新日時: {datetime.fromtimestamp(f_path.stat().st_mtime):%Y-%m-%d %H:%M})")
        try:
            choice = input(f"選択してください (1-{len(files)}): ").strip()
            return files[int(choice) - 1]
        except (ValueError, IndexError):
            print("無効な入力です。")
            return None
            
    def _run_inference_on_file(self, df: pd.DataFrame, batch_size: int = 512) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not all([self.model, self.scaler, self.metadata, 'feature_names' in self.metadata]):
            print("❌ モデル、スケーラー、またはメタデータが正しく読み込まれていません。")
            return None, None
        missing_features = [f for f in self.metadata['feature_names'] if f not in df.columns]
        if missing_features:
            print(f"❌ 特徴量が不足しています: {missing_features[:5]}...")
            return None, None
        print("入力データをスケーリングしています...")
        X_df = df[self.metadata['feature_names']].copy().fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X_df).astype(np.float32)
        sequence_length = self.metadata['model_config'].get('sequence_length', 30)
        num_frames = X_scaled.shape[0]
        if num_frames < sequence_length:
            print("データがシーケンス長より短いため、推論をスキップします。")
            return np.array([]), np.array([])
        all_preds, all_indices = [], []
        with torch.no_grad():
            num_sequences_to_process = num_frames - sequence_length + 1
            for i in tqdm(range(0, num_sequences_to_process, batch_size), desc="バッチ推論中"):
                batch_start_index, batch_end_index = i, min(i + batch_size, num_sequences_to_process)
                batch_sequences_list = [X_scaled[j : j + sequence_length] for j in range(batch_start_index, batch_end_index)]
                batch_indices_list = [j + sequence_length - 1 for j in range(batch_start_index, batch_end_index)]
                if not batch_sequences_list: continue
                X_tensor_batch = torch.from_numpy(np.array(batch_sequences_list)).to(self.device)
                outputs = self.model(X_tensor_batch)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_indices.extend(batch_indices_list)
        return np.array(all_preds), np.array(all_indices)
    
    def _perform_analysis(self, final_df: pd.DataFrame):
        """比較分析を行い、結果を表示する共通メソッド"""
        print("\n===== 検証結果 =====")
        
        # --- 分析1 ---
        print("\n--- 【分析1】 精度検証 (正解ラベル vs v2パイプラインの予測) ---")
        y_true_labels_full = final_df['label_str'].values
        y_pred_v2_labels = final_df['predicted_phase_v2'].values
        accuracy = accuracy_score(y_true_labels_full, y_pred_v2_labels)
        print(f"\n正解率 (Accuracy): {accuracy:.4f}")
        print("\n分類レポート:")
        print(classification_report(y_true_labels_full, y_pred_v2_labels, labels=self.phase_labels, zero_division=0))
        
        # --- 分析2 ---
        print("\n--- 【分析2】 一致率検証 (v2パイプライン vs predictパイプライン) ---")
        y_pred_predict_labels = final_df['predicted_phase_predict'].values
        valid_mask_consistency = pd.notna(y_pred_v2_labels) & pd.notna(y_pred_predict_labels)
        if np.sum(valid_mask_consistency) > 0:
            consistency_accuracy = accuracy_score(np.array(y_pred_v2_labels)[valid_mask_consistency], y_pred_predict_labels[valid_mask_consistency])
            print(f"予測の一致率: {consistency_accuracy:.4f} ({np.sum(valid_mask_consistency)}フレーム中)")
            if consistency_accuracy < 1.0:
                mismatch_df = final_df[final_df['predicted_phase_v2'] != final_df['predicted_phase_predict']].dropna(subset=['predicted_phase_predict'])
                mismatch_count = len(mismatch_df)
                if mismatch_count > 0:
                    print(f"予測が一致しなかったフレーム数: {mismatch_count}")
                    print("不一致の内訳 (上位5件):")
                    print(mismatch_df.groupby(['predicted_phase_v2', 'predicted_phase_predict']).size().nlargest(5))
        else:
            print("予測の一致率: nan (0フレーム中)")
        
        # --- 分析3 ---
        print("\n--- 【分析3】 精度検証 (正解ラベル vs predictパイプラインの予測) ---")
        valid_mask_predict_accuracy = pd.notna(y_true_labels_full) & pd.notna(y_pred_predict_labels)
        if np.sum(valid_mask_predict_accuracy) > 0:
            true_labels_for_predict = np.array(y_true_labels_full)[valid_mask_predict_accuracy]
            pred_labels_for_predict = y_pred_predict_labels[valid_mask_predict_accuracy]
            predict_accuracy = accuracy_score(true_labels_for_predict, pred_labels_for_predict)
            print(f"\n正解率 (Accuracy): {predict_accuracy:.4f}")
            print("\n分類レポート:")
            print(classification_report(true_labels_for_predict, pred_labels_for_predict, labels=self.phase_labels, zero_division=0))
        else:
            print("比較可能なフレームが見つかりませんでした。")

    def _perform_hybrid_analysis(self, final_df: pd.DataFrame):
        """【モード3用】正解ラベル、予測A、予測C の3者を比較分析する"""
        print("\n" + "="*25 + " 検証結果 " + "="*25)

        # 比較対象の列を準備
        y_true = final_df['label_str'].values
        y_pred_A = final_df['predicted_phase_A'].values
        y_pred_C = final_df['predicted_phase_C'].values
        
        # --- 分析1: ファイルAの精度検証 ---
        print("\n--- 【分析1】 ファイルAの精度検証 (正解ラベル vs ファイルAの推論結果) ---")
        accuracy_A = accuracy_score(y_true, y_pred_A)
        print(f"\n正解率 (Accuracy): {accuracy_A:.4f}")
        print("\n分類レポート:")
        print(classification_report(y_true, y_pred_A, labels=self.phase_labels, zero_division=0))

        # --- 分析2: 予測A vs 予測C の一致率検証 ---
        print("\n--- 【分析2】 ファイルA vs Cの一致率検証 (ファイルAの推論結果 vs ファイルCの推論結果) ---")
        consistency_accuracy = accuracy_score(y_pred_A, y_pred_C)
        print(f"\n予測の一致率: {consistency_accuracy:.4f} ({len(final_df)}フレーム中)")
        if consistency_accuracy < 1.0:
            mismatch_df = final_df[final_df['predicted_phase_A'] != final_df['predicted_phase_C']]
            mismatch_count = len(mismatch_df)
            print(f"予測が一致しなかったフレーム数: {mismatch_count}")
            print("不一致の内訳 (上位5件):")
            print(mismatch_df.groupby(['predicted_phase_A', 'predicted_phase_C']).size().nlargest(5))

        # --- 分析3: ファイルCの精度検証 ---
        print("\n--- 【分析3】 ファイルCの精度検証 (正解ラベル vs ファイルCの推論結果) ---")
        accuracy_C = accuracy_score(y_true, y_pred_C)
        print(f"\n正解率 (Accuracy): {accuracy_C:.4f}")
        print("\n分類レポート:")
        print(classification_report(y_true, y_pred_C, labels=self.phase_labels, zero_division=0))
        print("="*62)

    def run_comparison(self):
            """モデル推論を実行し、2つの特徴量パイプラインの結果を比較・検証します。"""
            model_path = self._select_file(self.models_dir, "**/tennis_pytorch*_model.pth", "1. 学習済みモデル選択")
            if not model_path: return
            
            base_name = model_path.name.removesuffix("_model.pth")
            scaler_path = model_path.parent / f"{base_name}_scaler.pkl"
            metadata_path = model_path.parent / f"{base_name}_metadata.json"
            
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f: self.metadata = json.load(f)
                self.scaler = joblib.load(scaler_path)
                config = self.metadata['model_config']

                model_params = {
                    'hidden_sizes': config['lstm_units'],
                    'dropout_rate': config['dropout_rate'],
                    'model_type': config['model_type'],
                    'use_batch_norm': config['batch_size'] > 1,
                    'enable_confidence_weighting': config.get('enable_confidence_weighting', False) # getで安全にアクセス
                }
                self.model = TennisLSTMModel(
                    input_size=len(self.metadata['feature_names']),
                    num_classes=len(self.metadata['phase_labels']),
                    **model_params
                ).to(DEVICE)
                self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                self.model.eval()
                self.phase_labels = self.metadata['phase_labels']
                self.label_map_inv = {i: name for i, name in enumerate(self.phase_labels)}
                print("✅ モデルと関連ファイルの読み込み完了")
            except Exception as e:
                print(f"❌ モデル読み込みエラー: {e}")
                import traceback
                traceback.print_exc()
                return

            v2_file_path = self._select_file(self.v2_features_dir, "tennis_features_*.csv", "2. ラベル付き特徴量ファイル選択 (v2)")
            if not v2_file_path: return
            
            predict_file_path = self._select_file(self.predict_features_dir, "tennis_inference_features_*.csv", "3. ラベルなし特徴量ファイル選択 (predict)")
            if not predict_file_path: return

            print("\n--- 推論処理開始 ---")
            df_v2 = pd.read_csv(v2_file_path)
            y_pred_v2_ids, indices_v2 = self._run_inference_on_file(df_v2)
            print(f"✅ v2ファイル推論完了: {len(y_pred_v2_ids)}件")

            df_predict = pd.read_csv(predict_file_path)
            
            print("\n'predict'側のvideo_nameからタイムスタンプなどの接尾辞を正規化します...")
            def normalize_video_name(name):
                # 例: "video1_20240101_123456" -> "video1"
                parts = name.split('_')
                # 数字のみのパートで分割が繰り返されることを考慮し、最初の非数字パートまでを結合
                base_parts = []
                for part in parts:
                    if part.isdigit():
                        break
                    base_parts.append(part)
                return '_'.join(base_parts) if base_parts else name

            df_predict['video_name'] = df_predict['video_name'].apply(normalize_video_name)
            print("✅ video_nameの正規化完了")
            
            y_pred_predict_ids, indices_predict = self._run_inference_on_file(df_predict)
            print(f"✅ predictファイル推論完了: {len(y_pred_predict_ids)}件")
            
            print("\n--- 結果の集計と分析 ---")
            results_df = df_v2.iloc[indices_v2].copy()
            if 'label' not in results_df.columns:
                print("❌ v2ファイルに'label'列がありません。")
                return
                
            results_df['predicted_phase_v2'] = [self.label_map_inv.get(i) for i in y_pred_v2_ids]
            
            predict_results_df = pd.DataFrame({
                'video_name': df_predict.iloc[indices_predict]['video_name'],
                'original_frame_number': df_predict.iloc[indices_predict]['original_frame_number'],
                'predicted_phase_predict': [self.label_map_inv.get(i) for i in y_pred_predict_ids]
            })
            
            try:
                for df in [results_df, predict_results_df]:
                    df['video_name'] = df['video_name'].astype(str)
                    df['original_frame_number'] = pd.to_numeric(df['original_frame_number'])
            except Exception as e:
                print(f"❌ データ型変換中にエラー: {e}")
                return
                
            final_df = pd.merge(results_df, predict_results_df, on=['video_name', 'original_frame_number'], how='inner')
            print(f"✅ 結合完了。比較対象フレーム数: {len(final_df)}")
            
            if final_df.empty:
                print("⚠️ 結合後のデータが空です。video_nameとframe_numberが一致するデータがありませんでした。")
                return

            # ★★★ ここからが変更箇所 ★★★
            # 'label'列を、モデルのメタデータから取得したラベル名に変換します
            final_df['label_str'] = final_df['label'].apply(lambda x: self.label_map_inv.get(int(x), "Unknown"))

            # 共通の分析メソッドを呼び出して結果を表示します
            self._perform_analysis(final_df)
            # ★★★ 変更箇所ここまで ★★★

    def run_comparison_from_precomputed(self):
        """推論済みのCSVファイルを読み込んで比較する"""
        print("\n--- 推論済みCSVファイルの比較モード ---")
        
        # 1. ファイル選択
        v2_results_path = self._select_file(Path("./training_data/features"), "*.csv", "1. 正解ラベルとv2予測を含むCSVを選択")
        if not v2_results_path: return
        
        predict_results_path = self._select_file(Path("./training_data/predictions"), "*.csv", "2. predictパイプラインの予測CSVを選択")
        if not predict_results_path: return
        
        # 2. CSV読み込み
        try:
            df_v2 = pd.read_csv(v2_results_path)
            df_predict = pd.read_csv(predict_results_path)
            print("✅ 2つのCSVファイルを読み込みました。")
        except Exception as e:
            print(f"❌ CSV読み込みエラー: {e}")
            return
            
        # 3. 必要な列の存在チェック
        required_v2_cols = ['label', 'predicted_phase', 'video_name', 'original_frame_number']
        if not all(col in df_v2.columns for col in required_v2_cols):
            print(f"❌ v2結果ファイルに必要な列がありません。必要な列: {required_v2_cols}")
            return
            
        required_predict_cols = ['predicted_phase', 'video_name', 'original_frame_number']
        if not all(col in df_predict.columns for col in required_predict_cols):
            print(f"❌ predict結果ファイルに必要な列がありません。必要な列: {required_predict_cols}")
            return

        # 4. マージのために列名を変更
        df_v2_renamed = df_v2.rename(columns={'predicted_phase': 'predicted_phase_v2'})
        df_predict_renamed = df_predict.rename(columns={'predicted_phase': 'predicted_phase_predict'})

        # 5. データフレームをマージ
        print("読み込んだ2つのファイルを 'video_name' と 'original_frame_number' で結合します...")
        final_df = pd.merge(
            df_v2_renamed,
            df_predict_renamed[['video_name', 'original_frame_number', 'predicted_phase_predict']],
            on=['video_name', 'original_frame_number'],
            how='inner' # 両方に存在するフレームのみを対象とする
        )
        print(f"✅ 結合完了。比較対象フレーム数: {len(final_df)}")
        
        if final_df.empty:
            print("⚠️ 結合後のデータが空です。video_nameとframe_numberが一致するデータがありませんでした。")
            return

        # 6. ラベル情報をデータから生成
        # 'label' 列のユニークな値からラベル辞書を作成
        self.label_map_inv = {int(i): name for i, name in enumerate(pd.unique(df_v2_renamed['label']))}
        self.phase_labels = list(self.label_map_inv.values())
        final_df['label_str'] = final_df['label'].apply(lambda x: self.label_map_inv.get(int(x), "Unknown"))

        # 7. 共通の分析メソッドを呼び出す
        self._perform_analysis(final_df)

    def run_hybrid_comparison(self):
        """【モード3】特徴量ファイル(A)に推論を実行し、別の推論済みCSV(C)と比較"""
        print("\n--- ハイブリッド比較モード ---")
        
        # 1. モデル選択
        model_path = self._select_file(self.models_dir, "**/tennis_pytorch*_model.pth", "1. 推論に使用する学習済みモデルを選択")
        if not model_path: return
        
        # モデルと関連ファイルを読み込み
        try:
            base_name = model_path.name.removesuffix("_model.pth")
            scaler_path = model_path.parent / f"{base_name}_scaler.pkl"
            metadata_path = model_path.parent / f"{base_name}_metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f: self.metadata = json.load(f)
            self.scaler = joblib.load(scaler_path)
            config = self.metadata['model_config']
            model_params = { 'hidden_sizes': config['lstm_units'], 'dropout_rate': config['dropout_rate'], 'model_type': config['model_type'], 'use_batch_norm': config['batch_size'] > 1, 'enable_confidence_weighting': config.get('enable_confidence_weighting', False) }
            self.model = TennisLSTMModel(input_size=len(self.metadata['feature_names']), num_classes=len(self.metadata['phase_labels']), **model_params).to(DEVICE)
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.model.eval()
            self.phase_labels = self.metadata['phase_labels']
            self.label_map_inv = {i: name for i, name in enumerate(self.phase_labels)}
            print("✅ モデルと関連ファイルの読み込み完了")
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            import traceback; traceback.print_exc()
            return

        # 2. ファイル選択
        file_A_path = self._select_file(self.v2_features_dir, "tennis_features_*.csv", "2. ラベル付き特徴量ファイル(A)を選択 (これに推論を実行します)")
        if not file_A_path: return

        file_C_path = self._select_file(Path("./training_data/predictions"), "*.csv", "3. 推論済みの予測結果CSV(C)を選択 (比較対象)")
        if not file_C_path: return

        # 3. データの読み込みと推論実行
        try:
            df_A = pd.read_csv(file_A_path)
            df_C = pd.read_csv(file_C_path)
            print("✅ 2つのファイルを読み込みました。ファイルAに推論を実行します...")
            y_pred_A_ids, indices_A = self._run_inference_on_file(df_A)
            print(f"✅ ファイルAの推論完了: {len(y_pred_A_ids)}件")
        except Exception as e:
            print(f"❌ データ処理または推論中にエラー: {e}")
            return

        # 4. データの準備とマージ
        # ファイルAの結果をまとめる
        results_A = df_A.iloc[indices_A].copy()
        results_A['predicted_phase_A'] = [self.label_map_inv.get(i) for i in y_pred_A_ids]

        # ファイルCの列名を変更
        df_C_renamed = df_C.rename(columns={'predicted_phase': 'predicted_phase_C'})

        # ファイルAの結果とファイルCの結果をマージ
        final_df = pd.merge(
            results_A,
            df_C_renamed[['video_name', 'original_frame_number', 'predicted_phase_C']],
            on=['video_name', 'original_frame_number'],
            how='inner'
        )
        print(f"✅ 結合完了。比較対象フレーム数: {len(final_df)}")

        if final_df.empty:
            print("⚠️ 結合後のデータが空です。video_nameとframe_numberが一致するデータがありませんでした。")
            return

        # 正解ラベルを数値からラベル名に変換
        final_df['label_str'] = final_df['label'].apply(lambda x: self.label_map_inv.get(int(x), "Unknown"))

        # 5. 新しい分析メソッドを呼び出す
        self._perform_hybrid_analysis(final_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2つの特徴量抽出パイプラインの推論結果を比較・検証するツール")
    parser.add_argument('--models_dir', type=str, default="./training_data/lstm_models")
    parser.add_argument('--v2_features_dir', type=str, default="./training_data/features")
    parser.add_argument('--predict_features_dir', type=str, default="./training_data/predict_features")
    args = parser.parse_args()
    
    comparator = PipelineComparator(models_dir=args.models_dir, v2_features_dir=args.v2_features_dir, predict_features_dir=args.predict_features_dir)
    
    # --- ★★★ 実行モード選択メニューを更新 ★★★ ---
    while True:
        print("\n" + "="*50)
        print("=== パイプライン比較ツール メインメニュー ===")
        print("1.【推論実行】2つの特徴量ファイルを推論して比較")
        print("2.【読込比較】3つの推論済みCSV/特徴量ファイルを比較")
        print("3.【ﾊｲﾌﾞﾘｯﾄﾞ】特徴量ファイル(A)に推論し、推論済みCSV(C)と比較")
        print("q. 終了")
        print("="*50)
        
        choice = input("実行したいモードを選択してください (1/2/3/q): ").strip().lower()

        if choice == '1':
            comparator.run_comparison()
            break
        elif choice == '2':
            comparator.run_comparison_from_precomputed()
            break
        elif choice == '3':
            comparator.run_hybrid_comparison() # 新しいモードを呼び出し
            break
        elif choice == 'q':
            print("ツールを終了します。")
            break
        else:
            print("無効な選択です。1, 2, 3, または q を入力してください。")