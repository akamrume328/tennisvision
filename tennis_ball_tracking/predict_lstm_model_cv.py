# predict_compatible.py

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Generator
import argparse
from model import TennisLSTMModel  # モデル定義をインポート

# ===== GPU設定 =====
def setup_gpu_config():
    """GPU設定とCUDA最適化"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"GPU検出: {device_count}台")
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({memory_total:.1f}GB)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        return torch.device('cuda')
    else:
        print("警告: GPU未検出: CPUで実行します")
        return torch.device('cpu')

DEVICE = setup_gpu_config()

# ===== 推論クラス =====
class TennisLSTMPredictor:
    def __init__(self, models_dir: str = "./training_data/lstm_models", 
                 input_features_dir: str = "./training_data/predict_features"):
        self.models_dir = Path(models_dir)
        self.input_features_dir = Path(input_features_dir)
        self.predictions_output_dir = Path("./training_data/predictions")
        self.predictions_output_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[TennisLSTMModel] = None
        self.scaler = None # 型ヒントは joblib のバージョンに依存するため省略
        self.metadata: Optional[Dict] = None
        self.device = DEVICE
        
        self.phase_labels: List[str] = []
        self.feature_names: List[str] = []
        self.sequence_length: int = 30
        self.label_map_inv: Optional[Dict[int, str]] = None

    def select_model_files(self) -> Optional[Tuple[Path, Path, Path]]:
            print(f"\n=== 学習済みモデルファイル選択 ===")
            
            # '**/...' を使って、現在のディレクトリと全てのサブディレクトリを再帰的に検索
            all_model_files = sorted(
                list(self.models_dir.glob("**/tennis_pytorch*.pth")), 
                key=lambda p: p.stat().st_mtime, 
                reverse=True
            )

            # --- ▼▼▼ デバッグ用プリント ▼▼▼ ---
            print(f"【デバッグ】 発見した.pthファイルの数: {len(all_model_files)}")
            # --- ▲▲▲ デバッグ用プリント ▲▲▲ ---

            if not all_model_files:
                print(f"❌ モデルファイル (*.pth) が見つかりません in {self.models_dir} およびそのサブディレクトリ。")
                return None

            valid_sets = []
            for mf_path in all_model_files:
                # --- ▼▼▼ デバッグ用プリント ▼▼▼ ---
                print(f"\n【デバッグ】 チェック中のモデルファイル: {mf_path.name}")
                # --- ▲▲▲ デバッグ用プリント ▲▲▲ ---

                parent_dir = mf_path.parent
                
                if not mf_path.name.endswith("_model.pth"):
                    continue

                base_name = mf_path.name.removesuffix("_model.pth")
                scaler_path = parent_dir / f"{base_name}_scaler.pkl"
                meta_path = parent_dir / f"{base_name}_metadata.json"

                # --- ▼▼▼ デバッグ用プリント ▼▼▼ ---
                print(f"【デバッグ】 探しているスケーラー: {scaler_path.name}")
                print(f"【デバッグ】 探しているメタデータ: {meta_path.name}")
                print(f"【デバッグ】 スケーラー存在?: {scaler_path.exists()}, メタデータ存在?: {meta_path.exists()}")
                # --- ▲▲▲ デバッグ用プリント ▲▲▲ ---
                
                if scaler_path.exists() and meta_path.exists():
                    valid_sets.append((mf_path, scaler_path, meta_path))

            # --- ▼▼▼ デバッグ用プリント ▼▼▼ ---
            print(f"\n【デバッグ】 発見した有効なモデルセットの数: {len(valid_sets)}")
            # --- ▲▲▲ デバッグ用プリント ▲▲▲ ---

            if not valid_sets:
                print(f"❌ 完全なモデルセット（scaler, metadataが揃っているもの）が見つかりません。")
                return None

            for i, (mf_path, _, _) in enumerate(valid_sets, 1):
                try:
                    display_path = mf_path.relative_to(self.models_dir)
                except ValueError:
                    display_path = mf_path
                
                print(f"  {i}. {display_path} (更新日時: {datetime.fromtimestamp(mf_path.stat().st_mtime):%Y-%m-%d %H:%M})")
            
            try:
                choice = input(f"選択してください (1-{len(valid_sets)}): ").strip()
                choice_num = int(choice)
                return valid_sets[choice_num - 1] if 1 <= choice_num <= len(valid_sets) else None
            except (ValueError, IndexError):
                print("無効な入力です。")
                return None

    def load_model_and_metadata(self, model_path: Path, scaler_path: Path, metadata_path: Path) -> bool:
        print(f"\n--- モデルとメタデータの読み込み ---")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f: self.metadata = json.load(f)
            print(f"✅ メタデータ読み込み: {metadata_path.name}")

            self.scaler = joblib.load(scaler_path)
            print(f"✅ スケーラー読み込み: {scaler_path.name}")
            
            # メタデータ内のモデル設定を使用してモデルを再構築
            model_config = self.metadata['model_config']
            
            self.model = TennisLSTMModel(
                input_size=len(self.metadata['feature_names']), # 特徴量数からinput_sizeを取得
                num_classes=len(self.metadata['phase_labels']), # ラベル数からnum_classesを取得
                hidden_sizes=model_config['lstm_units'],
                dropout_rate=model_config.get('dropout_rate', 0.3),
                model_type=model_config.get('model_type', 'bidirectional'),
                use_batch_norm=model_config.get('batch_size', 64) > 1,
                enable_confidence_weighting=model_config.get('enable_confidence_weighting', False)
            )

            # 学習済み重みを読み込み
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ モデル読み込み完了: {model_path.name}")

            # 推論に必要な情報をメタデータから取得
            self.phase_labels = self.metadata['phase_labels']
            self.feature_names = self.metadata['feature_names']
            self.sequence_length = self.metadata.get('sequence_length', model_config.get('sequence_length', 30))
            self.label_map_inv = {i: label_name for i, label_name in enumerate(self.phase_labels)}
            
            return True
        except Exception as e:
            print(f"❌ モデル/メタデータ読み込みエラー: {e}")
            import traceback; traceback.print_exc()
            return False

    def select_input_feature_file(self) -> Optional[Path]:
        print(f"\n=== 推論用 特徴量ファイル選択 ===")
        search_dirs = [self.input_features_dir, Path("./training_data")]
        feature_files = []
        for sdir in search_dirs:
            feature_files.extend(list(sdir.glob("tennis_inference_features_*.csv")))
        
        feature_files = sorted(list(set(feature_files)), key=lambda p: p.stat().st_mtime, reverse=True)

        if not feature_files:
            print(f"❌ 推論用特徴量ファイル (*.csv) が見つかりません。")
            return None

        for i, f_path in enumerate(feature_files, 1):
            print(f"  {i}. {f_path.name} (更新日時: {datetime.fromtimestamp(f_path.stat().st_mtime):%Y-%m-%d %H:%M})")
        
        try:
            choice = input(f"選択してください (1-{len(feature_files)}): ").strip()
            choice_num = int(choice)
            return feature_files[choice_num - 1] if 1 <= choice_num <= len(feature_files) else None
        except (ValueError, IndexError):
            print("無効な入力です。")
            return None

    def _generate_sequences_for_inference(self, X_scaled: np.ndarray, confidence_scores: Optional[np.ndarray]) -> Generator:
        num_frames = X_scaled.shape[0]
        if num_frames < self.sequence_length:
            return
        for i in range(num_frames - self.sequence_length + 1):
            seq_X = X_scaled[i : i + self.sequence_length]
            original_idx = i + self.sequence_length - 1
            seq_conf = confidence_scores[i : i + self.sequence_length] if confidence_scores is not None else None
            yield seq_X, original_idx, seq_conf

    def prepare_input_data(self, csv_path: Path) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[np.ndarray]]:
        print(f"\n--- 入力データ準備: {csv_path.name} ---")
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ CSV読み込み完了: {len(df)}行")

            missing_features = [f for f in self.feature_names if f not in df.columns]
            if missing_features:
                print(f"❌ 必要な特徴量がCSVにありません: {missing_features[:5]}...")
                return None, None, None
            
            X_df = df[self.feature_names].copy().fillna(0).replace([np.inf, -np.inf], 0)
            X_scaled = self.scaler.transform(X_df).astype(np.float32)

            confidence_scores = None
            if self.model and self.model.enable_confidence_weighting:
                if 'interpolation_ratio' in df.columns:
                    confidence_scores = (1.0 - df['interpolation_ratio'].fillna(0)).astype(np.float32).values
                else:
                    confidence_scores = np.ones(len(df), dtype=np.float32)
                    print("⚠️ 'interpolation_ratio'列が見つからないため、信頼度重み付けなしで処理します。")
            
            print(f"✅ データ前処理完了。フレーム数: {len(df)}")
            return X_scaled, df, confidence_scores
        except Exception as e:
            print(f"❌ データ準備エラー: {e}")
            return None, None, None

    def predict(self, X_scaled: np.ndarray, confidence_scores: Optional[np.ndarray], batch_size: int = 256) -> Optional[Tuple]:
        if not self.model: return None
        print(f"\n--- 推論実行 (バッチ処理) ---")
        self.model.eval()
        
        all_preds, all_probas, all_indices = [], [], []
        seq_generator = self._generate_sequences_for_inference(X_scaled, confidence_scores)
        
        batch_seq, batch_conf, batch_idx = [], [], []
        for seq_X, original_idx, seq_conf in seq_generator:
            batch_seq.append(seq_X)
            batch_idx.append(original_idx)
            if seq_conf is not None: batch_conf.append(seq_conf)

            if len(batch_seq) == batch_size:
                self._process_batch(batch_seq, batch_conf, batch_idx, all_preds, all_probas, all_indices)
                batch_seq, batch_conf, batch_idx = [], [], []

        if batch_seq: # 残りのバッチを処理
            self._process_batch(batch_seq, batch_conf, batch_idx, all_preds, all_probas, all_indices)

        if not all_preds:
            print("⚠️  推論対象のシーケンスがありませんでした。")
            return np.array([]), np.array([]), []

        print(f"✅ 推論完了: {len(all_preds)}件")
        return np.array(all_preds), np.array(all_probas), all_indices

    def _process_batch(self, batch_seq, batch_conf, batch_idx, all_preds, all_probas, all_indices):
        batch_X_tensor = torch.from_numpy(np.array(batch_seq, dtype=np.float32)).to(self.device)
        batch_conf_tensor = torch.from_numpy(np.array(batch_conf, dtype=np.float32)).to(self.device) if batch_conf else None
        
        with torch.no_grad():
            outputs = self.model(batch_X_tensor, batch_conf_tensor)
            probas = F.softmax(outputs, dim=1)
            _, preds = torch.max(probas, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probas.extend(probas.cpu().numpy())
        all_indices.extend(batch_idx)

    def format_predictions(self, predictions: np.ndarray, probabilities: np.ndarray, original_df: pd.DataFrame, original_indices: List[int]) -> pd.DataFrame:
        if not self.label_map_inv: return original_df
        
        results_df = original_df.copy()
        results_df['predicted_phase'] = ""
        results_df['prediction_confidence'] = np.nan

        # 予測結果を対応するフレームにマッピング
        pred_series = pd.Series([self.label_map_inv.get(p, f"Unknown_{p}") for p in predictions], index=original_indices)
        conf_series = pd.Series(np.max(probabilities, axis=1), index=original_indices)
        
        results_df.loc[original_indices, 'predicted_phase'] = pred_series
        results_df.loc[original_indices, 'prediction_confidence'] = conf_series
        
        print("✅ 予測結果のフォーマット完了")
        return results_df

    def save_predictions(self, predictions_df: pd.DataFrame, input_filename: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(input_filename).stem
        output_filename = f"{base_name}_predictions_{timestamp}.csv"
        output_path = self.predictions_output_dir / output_filename
        
        predictions_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 予測結果を保存しました: {output_path}")
        return output_path

    def run_prediction_pipeline(self):
        """対話的に推論を実行するパイプライン"""
        selected_files = self.select_model_files()
        if not selected_files: return
        
        if not self.load_model_and_metadata(*selected_files): return

        input_csv_path = self.select_input_feature_file()
        if not input_csv_path: return

        X_scaled, original_df, confidence_scores = self.prepare_input_data(input_csv_path)
        if X_scaled is None or original_df is None: return

        prediction_results = self.predict(X_scaled, confidence_scores)
        if prediction_results is None: return
        
        raw_preds, raw_probas, all_original_indices = prediction_results
        formatted_df = self.format_predictions(
            predictions=raw_preds, 
            probabilities=raw_probas, 
            original_df=original_df, 
            original_indices=all_original_indices
        )
        self.save_predictions(formatted_df, input_csv_path.name)

        print("\n🎉 推論パイプライン完了！")

if __name__ == "__main__":
    print("=== テニス動画局面分類PyTorch LSTM 推論ツール ===")
    predictor = TennisLSTMPredictor()
    
    try:
        predictor.run_prediction_pipeline()
    except KeyboardInterrupt:
        print("\n操作がキャンセルされました。")
    except Exception as e:
        print(f"❌ 予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()