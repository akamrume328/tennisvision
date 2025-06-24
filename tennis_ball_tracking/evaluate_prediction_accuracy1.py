# evaluate_prediction_accuracy.py

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
import warnings

# 機械学習・評価ライブラリ
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# グラフ描画ライブラリ
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ===== GPU設定 (predict_lstm_model_cv.py から流用) =====
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

# ===== モデル定義 (predict_lstm_model_cv.py と完全に同一) =====
class TennisLSTMModel(nn.Module):
    """PyTorch LSTM モデル（学習時と完全互換）"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, 
                 dropout_rate: float, model_type: str, use_batch_norm: bool, 
                 enable_confidence_weighting: bool):
        super(TennisLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model_type = model_type
        self.use_batch_norm = use_batch_norm
        self.enable_confidence_weighting = enable_confidence_weighting
        
        if self.enable_confidence_weighting:
            self.confidence_attention = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0] // 4),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0] // 4, 1),
                nn.Sigmoid()
            )
        
        self._build_lstm_layers()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.lstm_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_classes)
        
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(64)
        else:
            self.ln1 = nn.LayerNorm(128)
            self.ln2 = nn.LayerNorm(64)
        
        self._init_weights()
    
    def _build_lstm_layers(self):
            num_layers_to_build = len(self.hidden_sizes)
            self.lstm = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_sizes[0],
                num_layers=num_layers_to_build,
                batch_first=True,
                dropout=self.dropout_rate if num_layers_to_build > 1 else 0,
                bidirectional=(self.model_type == "bidirectional")
            )
            if self.model_type == "bidirectional":
                self.lstm_output_size = self.hidden_sizes[0] * 2
            else:
                self.lstm_output_size = self.hidden_sizes[0]

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
        if self.use_batch_norm:
            return norm_layer(x) if x.size(0) > 1 else x
        return norm_layer(x)

# ===== 精度検証クラス =====
class PredictionEvaluator:
    def __init__(self, models_dir: str = "./training_data/lstm_models", 
                 features_dir: str = "./training_data/features"):
        self.models_dir = Path(models_dir)
        self.features_dir = Path(features_dir)
        self.output_dir = Path("./evaluation_results")
        self.output_dir.mkdir(exist_ok=True)

        self.model: Optional[TennisLSTMModel] = None
        self.scaler = None
        self.metadata: Optional[Dict] = None
        self.device = DEVICE
        
        self.phase_labels: List[str] = []
        self.feature_names: List[str] = []
        self.sequence_length: int = 30
        self.label_map: Optional[Dict[str, int]] = None
        self.label_map_inv: Optional[Dict[int, str]] = None
        
        print(f"精度検証ツールを初期化しました。結果は '{self.output_dir}' に保存されます。")

    def select_model_files(self) -> Optional[Tuple[Path, Path, Path]]:
        """対話形式でモデルファイルを選択 (predict_lstm_model_cv.pyから流用)"""
        print(f"\n=== 1. 学習済みモデルファイル選択 ===")
        all_model_files = sorted(list(self.models_dir.glob("**/tennis_pytorch*.pth")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not all_model_files:
            print(f"❌ モデルファイル (*.pth) が見つかりません in {self.models_dir}")
            return None

        valid_sets = []
        for mf_path in all_model_files:
            if not mf_path.name.endswith("_model.pth"): continue
            base_name = mf_path.name.removesuffix("_model.pth")
            scaler_path = mf_path.parent / f"{base_name}_scaler.pkl"
            meta_path = mf_path.parent / f"{base_name}_metadata.json"
            if scaler_path.exists() and meta_path.exists():
                valid_sets.append((mf_path, scaler_path, meta_path))

        if not valid_sets:
            print(f"❌ 完全なモデルセット（scaler, metadataが揃っているもの）が見つかりません。")
            return None

        for i, (mf_path, _, _) in enumerate(valid_sets, 1):
            print(f"  {i}. {mf_path.relative_to(self.models_dir)} (更新日時: {datetime.fromtimestamp(mf_path.stat().st_mtime):%Y-%m-%d %H:%M})")
        
        try:
            choice = input(f"選択してください (1-{len(valid_sets)}): ").strip()
            return valid_sets[int(choice) - 1]
        except (ValueError, IndexError):
            print("無効な入力です。")
            return None

    def load_model_and_metadata(self, model_path: Path, scaler_path: Path, metadata_path: Path) -> bool:
        """モデルと関連ファイルを読み込み (predict_lstm_model_cv.pyから流用・修正)"""
        print(f"\n--- モデルとメタデータの読み込み ---")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f: self.metadata = json.load(f)
            self.scaler = joblib.load(scaler_path)
            
            model_config = self.metadata['model_config']
            self.model = TennisLSTMModel(
                input_size=len(self.metadata['feature_names']),
                num_classes=len(self.metadata['phase_labels']),
                hidden_sizes=model_config['lstm_units'],
                dropout_rate=model_config.get('dropout_rate', 0.3),
                model_type=model_config.get('model_type', 'bidirectional'),
                use_batch_norm=model_config.get('batch_size', 64) > 1,
                enable_confidence_weighting=model_config.get('enable_confidence_weighting', False)
            )
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.phase_labels = self.metadata['phase_labels']
            self.feature_names = self.metadata['feature_names']
            self.sequence_length = model_config.get('sequence_length', 30)
            
            # ラベルとIDの対応辞書を作成
            self.label_map = {name: i for i, name in enumerate(self.phase_labels)}
            self.label_map_inv = {i: name for i, name in enumerate(self.phase_labels)}
            
            print(f"✅ モデル読み込み完了: {model_path.name}")
            return True
        except Exception as e:
            print(f"❌ モデル/メタデータ読み込みエラー: {e}")
            import traceback; traceback.print_exc()
            return False

    def select_feature_file(self) -> Optional[Path]:
        """対話形式で特徴量ファイルを選択"""
        print(f"\n=== 2. 精度検証に使用する特徴量ファイル選択 ===")
        print(f"'{self.features_dir}' ディレクトリから探します...")
        
        # 'tennis_features_*.csv' (ラベル付き) を優先的に探す
        feature_files = sorted(list(self.features_dir.glob("tennis_features_*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not feature_files:
            print(f"❌ 正解ラベル付きの特徴量ファイル (tennis_features_*.csv) が見つかりません。")
            return None

        for i, f_path in enumerate(feature_files, 1):
            print(f"  {i}. {f_path.name} (更新日時: {datetime.fromtimestamp(f_path.stat().st_mtime):%Y-%m-%d %H:%M})")
        
        try:
            choice = input(f"選択してください (1-{len(feature_files)}): ").strip()
            return feature_files[int(choice) - 1]
        except (ValueError, IndexError):
            print("無効な入力です。")
            return None

    def _generate_sequences(self, X_scaled: np.ndarray, confidence_scores: Optional[np.ndarray]) -> Generator:
        """推論用のシーケンスを生成"""
        num_frames = X_scaled.shape[0]
        if num_frames < self.sequence_length:
            return
        for i in range(num_frames - self.sequence_length + 1):
            seq_X = X_scaled[i : i + self.sequence_length]
            original_idx = i + self.sequence_length - 1
            seq_conf = confidence_scores[i : i + self.sequence_length] if confidence_scores is not None else None
            yield seq_X, original_idx, seq_conf

    def predict(self, X_scaled: np.ndarray, confidence_scores: Optional[np.ndarray], batch_size: int = 256) -> Optional[Tuple]:
        """推論を実行 (predict_lstm_model_cv.pyから流用)"""
        if not self.model: return None
        print(f"\n--- 推論実行 ---")
        
        all_preds, all_probas, all_indices = [], [], []
        seq_generator = self._generate_sequences(X_scaled, confidence_scores)
        
        batch_seq, batch_conf, batch_idx = [], [], []
        for seq_X, original_idx, seq_conf in seq_generator:
            batch_seq.append(seq_X)
            batch_idx.append(original_idx)
            if seq_conf is not None: batch_conf.append(seq_conf)

            if len(batch_seq) >= batch_size:
                self._process_batch(batch_seq, batch_conf, all_preds, all_probas)
                all_indices.extend(batch_idx)
                batch_seq, batch_conf, batch_idx = [], [], []

        if batch_seq:
            self._process_batch(batch_seq, batch_conf, all_preds, all_probas)
            all_indices.extend(batch_idx)

        print(f"✅ 推論完了: {len(all_preds)}件")
        return np.array(all_preds), np.array(all_probas), all_indices

    def _process_batch(self, batch_seq, batch_conf, all_preds, all_probas):
        """バッチ単位で推論処理"""
        X_tensor = torch.from_numpy(np.array(batch_seq, dtype=np.float32)).to(self.device)
        conf_tensor = torch.from_numpy(np.array(batch_conf, dtype=np.float32)).to(self.device) if batch_conf else None
        
        with torch.no_grad():
            outputs = self.model(X_tensor, conf_tensor)
            probas = F.softmax(outputs, dim=1)
            _, preds = torch.max(probas, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probas.extend(probas.cpu().numpy())

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]):
        """精度の評価と結果表示"""
        print("\n===== 3. 精度検証結果 =====")
        
        # 正解率
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n正解率 (Accuracy): {accuracy:.4f}")
        
        # 分類レポート
        print("\n分類レポート:")
        report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
        print(report)
        
        # 混同行列
        self.plot_confusion_matrix(y_true, y_pred, labels)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]):
        """混同行列の可視化 (train_lstm_model_cv.pyから流用)"""
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Phase')
        plt.ylabel('True Phase')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = self.output_dir / f'evaluation_confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(save_path, dpi=300)
        print(f"\n混同行列を保存しました: {save_path}")
        plt.show()

    def run_evaluation_pipeline(self):
        """精度検証のパイプラインを実行"""
        # 1. モデル選択
        selected_files = self.select_model_files()
        if not selected_files: return
        
        if not self.load_model_and_metadata(*selected_files): return

        # 2. 特徴量ファイル選択
        input_csv_path = self.select_feature_file()
        if not input_csv_path: return

        # 3. データ準備
        print(f"\n--- データ準備: {input_csv_path.name} ---")
        try:
            df = pd.read_csv(input_csv_path)
            # 正解ラベル列の存在チェック
            if 'label' not in df.columns:
                print(f"❌ エラー: 特徴量ファイルに正解ラベルを示す 'label' 列がありません。")
                return
            
            X_df = df[self.feature_names].copy().fillna(0).replace([np.inf, -np.inf], 0)
            X_scaled = self.scaler.transform(X_df).astype(np.float32)
            
            confidence_scores = None
            if self.model and self.model.enable_confidence_weighting:
                confidence_scores = df.get('interpolation_confidence', pd.Series(np.ones(len(df)))).fillna(1.0).astype(np.float32).values

        except Exception as e:
            print(f"❌ データ準備エラー: {e}")
            return
            
        # 4. 推論実行
        pred_ids, _, pred_indices = self.predict(X_scaled, confidence_scores)
        if pred_ids is None: return

        # 5. 正解ラベルと予測ラベルの準備
        # 推論が行われたフレームに対応する正解ラベルを取得
        true_ids = df.loc[pred_indices, 'label'].values
        
        # IDをラベル名に変換
        y_true_labels = [self.label_map_inv.get(int(i), "Unknown") for i in true_ids]
        y_pred_labels = [self.label_map_inv.get(i, "Unknown") for i in pred_ids]

        # 6. 精度評価
        self.evaluate(y_true_labels, y_pred_labels, self.phase_labels)
        
        print("\n🎉 精度検証パイプライン完了！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済みモデルの精度検証ツール")
    parser.add_argument('--models_dir', type=str, default="./training_data/lstm_models", help="モデルが保存されているディレクトリ")
    parser.add_argument('--features_dir', type=str, default="./training_data/features", help="特徴量ファイルが保存されているディレクトリ")
    args = parser.parse_args()

    try:
        evaluator = PredictionEvaluator(models_dir=args.models_dir, features_dir=args.features_dir)
        evaluator.run_evaluation_pipeline()
    except KeyboardInterrupt:
        print("\n操作がキャンセルされました。")
    except Exception as e:
        print(f"\n❌ 予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()