# evaluate_on_training_data.py

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from sklearn.metrics import classification_report, accuracy_score

# ===== モデル定義 (推論スクリプトと同一) =====
# （学習時と完全に互換性のあるモデル定義をここに含めます）
def setup_gpu_config():
    if torch.cuda.is_available(): return torch.device('cuda')
    else: return torch.device('cpu')
DEVICE = setup_gpu_config()

class TennisLSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, 
                 dropout_rate: float, model_type: str, use_batch_norm: bool, 
                 enable_confidence_weighting: bool):
        super(TennisLSTMModel, self).__init__()
        self.input_size, self.hidden_sizes, self.num_classes = input_size, hidden_sizes, num_classes
        self.dropout_rate, self.model_type, self.use_batch_norm = dropout_rate, model_type, use_batch_norm
        self.enable_confidence_weighting = enable_confidence_weighting
        if self.enable_confidence_weighting:
            self.confidence_attention = nn.Sequential(nn.Linear(input_size, hidden_sizes[0] // 4), nn.ReLU(), nn.Linear(hidden_sizes[0] // 4, 1), nn.Sigmoid())
        self._build_lstm_layers()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.lstm_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_classes)
        if self.use_batch_norm:
            self.bn1, self.bn2 = nn.BatchNorm1d(128), nn.BatchNorm1d(64)
        else:
            self.ln1, self.ln2 = nn.LayerNorm(128), nn.LayerNorm(64)
        self._init_weights()
    def _build_lstm_layers(self):
        num_layers_to_build = len(self.hidden_sizes)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_sizes[0], num_layers=num_layers_to_build, batch_first=True, dropout=self.dropout_rate if num_layers_to_build > 1 else 0, bidirectional=(self.model_type == "bidirectional"))
        if self.model_type == "bidirectional": self.lstm_output_size = self.hidden_sizes[0] * 2
        else: self.lstm_output_size = self.hidden_sizes[0]
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
        out = self.dropout(lstm_out)
        out = F.relu(self.fc1(out)); out = self._apply_normalization(out, self.bn1 if self.use_batch_norm else self.ln1); out = self.dropout(out)
        out = F.relu(self.fc2(out)); out = self._apply_normalization(out, self.bn2 if self.use_batch_norm else self.ln2); out = self.dropout(out)
        return self.fc_out(out)
    def _apply_normalization(self, x, norm_layer):
        if self.use_batch_norm: return norm_layer(x) if x.size(0) > 1 else x
        return norm_layer(x)

# ===== 検証クラス =====
class ModelEvaluator:
    def __init__(self):
        self.model: Optional[TennisLSTMModel] = None
        self.scaler = None
        self.metadata: Optional[Dict] = None
        self.device = DEVICE

    def load_model_and_metadata(self, model_set_dir: Path) -> bool:
        print(f"\n--- 1. モデルとメタデータの読み込み ---")
        model_path = next(model_set_dir.glob("*_model.pth"), None)
        scaler_path = next(model_set_dir.glob("*_scaler.pkl"), None)
        metadata_path = next(model_set_dir.glob("*_metadata.json"), None)

        if not all([model_path, scaler_path, metadata_path]):
            print(f"❌ モデルセットが見つかりません in {model_set_dir}")
            return False

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
            print(f"✅ モデル読み込み完了: {model_path.name}")
            return True
        except Exception as e:
            print(f"❌ 読み込みエラー: {e}")
            return False

    def evaluate(self, training_csv_path: Path):
            print(f"\n--- 2. 検証データ準備 ---")
            df = pd.read_csv(training_csv_path)
            print(f"✅ CSV読み込み完了: {len(df)}行")

            feature_names = self.metadata['feature_names']
            feature_names = [f for f in feature_names if f in df.columns]
            X = df[feature_names].fillna(0).values
            y_true = df['label'].values
            
            print("   スケーラーを適用中...")
            X_scaled = self.scaler.transform(X)

            print(f"\n--- 3. 推論と評価 (メモリ効率化・バッチ処理版) ---")
            sequence_length = self.metadata['model_config']['sequence_length']
            label_map = {int(k): int(v) for k, v in self.metadata['label_map'].items()}

            all_preds = []
            all_true_mapped_for_eval = []
            batch_size = 512  # メモリに応じて調整可能なバッチサイズ
            num_sequences = len(X_scaled) - sequence_length + 1

            with torch.no_grad():
                for i in range(0, num_sequences, batch_size):
                    # 現在のバッチの終わりを計算
                    end_index = min(i + batch_size, num_sequences)
                    
                    # このバッチに対応するシーケンスとラベルを動的に作成
                    batch_sequences_X = np.array([X_scaled[j:j+sequence_length] for j in range(i, end_index)])
                    batch_labels_y = np.array([y_true[j+sequence_length-1] for j in range(i, end_index)])
                    
                    # 正解ラベルをマッピングして評価リストに追加
                    all_true_mapped_for_eval.extend([label_map[label] for label in batch_labels_y])

                    # バッチをテンソルに変換して推論実行
                    batch_X_tensor = torch.FloatTensor(batch_sequences_X).to(self.device)
                    outputs = self.model(batch_X_tensor, None) # 信頼度重み付けはオフで検証
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())

                    # 進捗表示
                    if (i // batch_size) % 10 == 0:
                        print(f"  処理中... {end_index} / {num_sequences} シーケンス")

            print(f"✅ 推論完了")

            print(f"\n--- 4. 性能評価 ---")
            accuracy = accuracy_score(all_true_mapped_for_eval, all_preds)
            report = classification_report(
                all_true_mapped_for_eval, 
                all_preds, 
                target_names=self.metadata['phase_labels'],
                zero_division=0
            )
            
            print("\n=====【最終検証結果】=====")
            print(f"対象データ: {training_csv_path.name}")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(report)

def main():
    parser = argparse.ArgumentParser(description="学習済みモデルを、学習時に使用したCSV全体で再評価するスクリプト")
    parser.add_argument('--model_set_dir', type=str, required=True, help='評価したいモデルセット（.pth, .pkl, .jsonを含む）のディレクトリパス')
    parser.add_argument('--training_csv', type=str, required=True, help='学習時に使用した完全な特徴量CSVファイルのパス')
    args = parser.parse_args()

    print("===== モデル性能再検証プロセス開始 =====")
    evaluator = ModelEvaluator()
    if evaluator.load_model_and_metadata(Path(args.model_set_dir)):
        evaluator.evaluate(Path(args.training_csv))
    print("\n===== プロセス完了 =====")

if __name__ == "__main__":
    main()