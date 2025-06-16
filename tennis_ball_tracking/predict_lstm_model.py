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
from typing import Dict, List, Tuple, Optional, Union, Generator

# train_lstm_model.py から必要な定義をコピーまたはインポート
# (ここでは簡略化のため、主要なクラスと関数を直接記述します)

def setup_gpu_config():
    """GPU設定とCUDA最適化"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"🚀 GPU検出: {device_count}台")
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({memory_total:.1f}GB)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        return torch.device('cuda')
    else:
        print("⚠️  GPU未検出: CPUで実行します")
        return torch.device('cpu')

DEVICE = setup_gpu_config()

class TennisLSTMModel(nn.Module):
    """PyTorch LSTM モデル（信頼度重み付け対応）"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, 
                 dropout_rate: float = 0.3, model_type: str = "bidirectional", 
                 use_batch_norm: bool = True, enable_confidence_weighting: bool = True):
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
        if self.model_type == "simple":
            self.lstm1 = nn.LSTM(self.input_size, self.hidden_sizes[0], batch_first=True, dropout=self.dropout_rate)
            self.lstm2 = nn.LSTM(self.hidden_sizes[0], self.hidden_sizes[1], batch_first=True, dropout=self.dropout_rate)
            self.lstm_output_size = self.hidden_sizes[1]
        elif self.model_type == "bidirectional":
            self.lstm1 = nn.LSTM(self.input_size, self.hidden_sizes[0], batch_first=True, 
                                bidirectional=True, dropout=self.dropout_rate)
            self.lstm2 = nn.LSTM(self.hidden_sizes[0] * 2, self.hidden_sizes[1], batch_first=True,
                                bidirectional=True, dropout=self.dropout_rate)
            self.lstm_output_size = self.hidden_sizes[1] * 2
        elif self.model_type == "stacked":
            self.lstm1 = nn.LSTM(self.input_size, self.hidden_sizes[0], batch_first=True, dropout=self.dropout_rate)
            self.lstm2 = nn.LSTM(self.hidden_sizes[0], self.hidden_sizes[1], batch_first=True, dropout=self.dropout_rate)
            self.lstm3 = nn.LSTM(self.hidden_sizes[1], 64, batch_first=True, dropout=self.dropout_rate)
            self.lstm_output_size = 64
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name: torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name: torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name: param.data.fill_(0)
            elif isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, confidence_scores=None):
        if self.enable_confidence_weighting and confidence_scores is not None:
            # confidence_scores は (batch, seq_len) の形状を期待
            # attention_weights は (batch, seq_len, 1)
            attention_weights = self.confidence_attention(x) 
            # combined_weights は (batch, seq_len)
            combined_weights = attention_weights.squeeze(-1) * confidence_scores
            x = x * combined_weights.unsqueeze(-1) # (batch, seq_len, features)
        
        lstm_out = self._forward_lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        
        out = F.relu(self.fc1(lstm_out))
        out = self._apply_normalization(out, self.bn1 if self.use_batch_norm else self.ln1)
        out = self.dropout(out)
        
        out = F.relu(self.fc2(out))
        out = self._apply_normalization(out, self.bn2 if self.use_batch_norm else self.ln2)
        out = self.dropout(out)
        
        return self.fc_out(out)

    def _forward_lstm(self, x):
        if self.model_type == "simple":
            lstm_out, _ = self.lstm1(x)
            lstm_out = self.dropout(lstm_out)
            lstm_out, _ = self.lstm2(lstm_out)
        elif self.model_type == "bidirectional":
            lstm_out, _ = self.lstm1(x)
            lstm_out = self.dropout(lstm_out)
            lstm_out, _ = self.lstm2(lstm_out)
        elif self.model_type == "stacked":
            lstm_out, _ = self.lstm1(x)
            lstm_out = self.dropout(lstm_out)
            lstm_out, _ = self.lstm2(lstm_out)
            lstm_out = self.dropout(lstm_out)
            lstm_out, _ = self.lstm3(lstm_out)
        return lstm_out

    def _apply_normalization(self, x, norm_layer):
        if self.use_batch_norm and x.size(0) > 1:
            return norm_layer(x)
        elif not self.use_batch_norm:
            return norm_layer(x)
        return x

class TennisLSTMPredictor:
    def __init__(self, models_dir: str = "./training_data/lstm_models", 
                 input_features_dir: str = "./training_data/predict_features"):
        self.models_dir = Path(models_dir)
        self.input_features_dir = Path(input_features_dir)
        self.predictions_output_dir = Path("./training_data/predictions")
        self.predictions_output_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[TennisLSTMModel] = None
        self.scaler: Optional[joblib.numpy_pickle.NumpyPickler] = None # sklearn.preprocessing.StandardScaler
        self.metadata: Optional[Dict] = None
        self.device = DEVICE
        
        self.phase_labels: List[str] = []
        self.feature_names: List[str] = []
        self.sequence_length: int = 30
        self.label_map_inv: Optional[Dict[int, str]] = None


    def select_model_files(self) -> Optional[Tuple[Path, Path, Path]]:
        print(f"\n=== 学習済みモデルファイル選択 ===")
        
        potential_model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
        if not potential_model_dirs:
            # サブディレクトリがない場合は、models_dir 直下を検索（従来互換）
            potential_model_dirs = [self.models_dir]

        print("利用可能なモデルセット:")
        valid_sets = []
        
        for model_subdir in potential_model_dirs:
            model_files = sorted(list(model_subdir.glob("tennis_pytorch_model_*.pth")))
            if not model_files:
                continue

            for mf_path in model_files:
                base_name = mf_path.name.replace("tennis_pytorch_model_", "").replace(".pth", "")
                scaler_path = model_subdir / f"tennis_pytorch_scaler_{base_name}.pkl"
                meta_path = model_subdir / f"tennis_pytorch_metadata_{base_name}.json"

                if scaler_path.exists() and meta_path.exists():
                    # サブディレクトリ名も表示に含める
                    display_name = f"{model_subdir.name}/{base_name}" if model_subdir != self.models_dir else base_name
                    valid_sets.append((mf_path, scaler_path, meta_path, display_name))
        
        if not valid_sets:
            print(f"❌ 完全なモデルセット（model, scaler, metadata）が見つかりません in {self.models_dir} およびそのサブディレクトリ。")
            return None

        # 更新日時でソート (新しいものが上に来るように)
        valid_sets.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)

        for i, (mf_path, _, _, display_name) in enumerate(valid_sets, 1):
            print(f"  {i}. {display_name} (更新日時: {datetime.fromtimestamp(mf_path.stat().st_mtime):%Y-%m-%d %H:%M})")
        
        try:
            choice = input(f"選択してください (1-{len(valid_sets)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(valid_sets):
                # display_name を除いたタプルを返す
                selected_set = valid_sets[choice_num - 1]
                return selected_set[0], selected_set[1], selected_set[2]
            else:
                print("無効な選択です。")
                return None
        except ValueError:
            print("無効な入力です。")
            return None

    def load_model_and_metadata(self, model_path: Path, scaler_path: Path, metadata_path: Path) -> bool:
        print(f"\n--- モデルとメタデータの読み込み ---")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"✅ メタデータ読み込み: {metadata_path.name}")

            self.scaler = joblib.load(scaler_path)
            print(f"✅ スケーラー読み込み: {scaler_path.name}")

            model_checkpoint = torch.load(model_path, map_location=self.device)
            model_config = model_checkpoint['model_config']
            
            self.model = TennisLSTMModel(
                input_size=model_config['input_size'],
                hidden_sizes=model_config['hidden_sizes'],
                num_classes=model_config['num_classes'],
                dropout_rate=model_config.get('dropout_rate', 0.3), # 後方互換性
                model_type=model_config.get('model_type', 'bidirectional'),
                use_batch_norm=model_config.get('use_batch_norm', True),
                enable_confidence_weighting=model_config.get('enable_confidence_weighting', False)
            )
            self.model.load_state_dict(model_checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ モデル読み込み: {model_path.name}")

            # メタデータから情報を取得
            self.phase_labels = self.metadata.get('phase_labels', [])
            self.feature_names = self.metadata.get('feature_names', [])
            self.sequence_length = self.metadata.get('sequence_length', 30)
            
            # ラベルマッピングの逆を作成 (予測結果をラベル名に戻すため)
            # metadata['label_map'] は {original_label_int: remapped_idx}
            # metadata['phase_labels'] は remapped_idx に対応するラベル名リスト
            # 必要なのは remapped_idx -> phase_label_name
            if 'label_map' in self.metadata and self.phase_labels:
                 # phase_labels は学習時にアクティブだったラベルのリスト
                self.label_map_inv = {i: label_name for i, label_name in enumerate(self.phase_labels)}
            else: # 古いメタデータの場合のフォールバック
                self.label_map_inv = {i: label for i, label in enumerate(self.phase_labels)}

            if not self.feature_names:
                print("⚠️  メタデータに特徴量名リスト (feature_names) がありません。")
                return False
            return True
        except Exception as e:
            print(f"❌ モデル/メタデータ読み込みエラー: {e}")
            return False

    def find_model_files_in_set_dir(self, model_set_dir: Path) -> Optional[Tuple[Path, Path, Path]]:
        """指定されたモデルセットディレクトリから最新のモデルファイル群を検索する"""
        print(f"--- モデルセットディレクトリ検索: {model_set_dir} ---")
        model_files = sorted(list(model_set_dir.glob("tennis_pytorch_model_*.pth")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not model_files:
            print(f"❌ {model_set_dir} にモデルファイル (*.pth) が見つかりません。")
            return None
        
        latest_model_file = model_files[0]
        base_name = latest_model_file.name.replace("tennis_pytorch_model_", "").replace(".pth", "")
        
        scaler_path = model_set_dir / f"tennis_pytorch_scaler_{base_name}.pkl"
        meta_path = model_set_dir / f"tennis_pytorch_metadata_{base_name}.json"

        if scaler_path.exists() and meta_path.exists():
            print(f"✅ モデルファイル発見: {latest_model_file.name}")
            print(f"✅ スケーラーファイル発見: {scaler_path.name}")
            print(f"✅ メタデータファイル発見: {meta_path.name}")
            return latest_model_file, scaler_path, meta_path
        else:
            print(f"❌ {model_set_dir} に完全なモデルセット (scaler or metadata) が見つかりません。")
            if not scaler_path.exists(): print(f"   - スケーラーが見つかりません: {scaler_path}")
            if not meta_path.exists(): print(f"   - メタデータが見つかりません: {meta_path}")
            return None

    def select_input_feature_file(self) -> Optional[Path]:
        print(f"\n=== 推論用 特徴量ファイル選択 ===")
        feature_files = sorted(list(self.input_features_dir.glob("tennis_inference_features_*.csv")))
        
        if not feature_files:
            # training_data 直下も検索 (feature_extractor_predict.py のデフォルト保存先変更前の互換性)
            feature_files.extend(sorted(list(Path("./training_data").glob("tennis_inference_features_*.csv"))))
            if not feature_files:
                 print(f"❌ 推論用特徴量ファイル (*.csv) が見つかりません in {self.input_features_dir} or ./training_data")
                 return None

        for i, f_path in enumerate(feature_files, 1):
            print(f"  {i}. {f_path.name} (更新日時: {datetime.fromtimestamp(f_path.stat().st_mtime):%Y-%m-%d %H:%M})")
        
        try:
            choice = input(f"選択してください (1-{len(feature_files)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(feature_files):
                return feature_files[choice_num - 1]
            else:
                print("無効な選択です。")
                return None
        except ValueError:
            print("無効な入力です。")
            return None

    def _generate_sequences_for_inference(self, X_scaled: np.ndarray, 
                                          confidence_scores_full: Optional[np.ndarray] = None
                                          ) -> Generator[Tuple[np.ndarray, int, Optional[np.ndarray]], None, None]:
        """
        推論用のシーケンス、元のインデックス、信頼度シーケンスを生成するジェネレータ。
        """
        num_frames = X_scaled.shape[0]
        if num_frames < self.sequence_length:
            print(f"⚠️  データ長 ({num_frames}) がシーケンス長 ({self.sequence_length}) より短いため、シーケンスを生成できません。")
            return

        for i in range(num_frames - self.sequence_length + 1):
            seq_X = X_scaled[i : i + self.sequence_length]
            original_idx = i + self.sequence_length - 1 # シーケンスの最後のフレームのインデックス

            seq_conf = None
            if confidence_scores_full is not None:
                seq_conf = confidence_scores_full[i : i + self.sequence_length]
            
            yield seq_X, original_idx, seq_conf

    def prepare_input_data(self, csv_path: Path) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[np.ndarray]]:
        print(f"\n--- 入力データ準備: {csv_path.name} ---")
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ CSV読み込み完了: {len(df)}行, {len(df.columns)}列")

            # 特徴量選択
            missing_features = [f for f in self.feature_names if f not in df.columns]
            if missing_features:
                print(f"❌ 必要な特徴量がCSVにありません: {missing_features}")
                return None, None, None
            
            X_df = df[self.feature_names].copy()

            # 欠損値処理
            X_df = X_df.fillna(0)
            X_df = X_df.replace([np.inf, -np.inf], 0)

            X_scaled = self.scaler.transform(X_df)
            X_scaled = X_scaled.astype(np.float32) # メモリ使用量削減

            confidence_scores_full = None
            if self.model and self.model.enable_confidence_weighting:
                if 'interpolated' in df.columns:
                    base_confidence = 1.0 - df['interpolated'].astype(np.float32) * 0.3
                    confidence_scores_full = base_confidence.astype(np.float32).values
                elif 'data_quality' in df.columns:
                     confidence_scores_full = df['data_quality'].fillna(1.0).astype(np.float32).values
                else:
                    print("⚠️  信頼度重み付けが有効なモデルですが、信頼度関連列が見つかりません。信頼度1.0で処理します。")
                    confidence_scores_full = np.ones(len(df), dtype=np.float32)
            
            print(f"✅ データ前処理完了。フレーム数: {len(df)}")
            return X_scaled, df, confidence_scores_full

        except Exception as e:
            print(f"❌ データ準備エラー: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def predict(self, X_scaled: np.ndarray,
                original_df: pd.DataFrame,
                confidence_scores_full: Optional[np.ndarray] = None,
                batch_size: int = 256  # バッチサイズを調整可能にする
                ) -> Optional[Tuple[np.ndarray, np.ndarray, List[int]]]:
        if self.model is None:
            print("❌ モデルがロードされていません。")
            return None
        
        print(f"\n--- 推論実行 (バッチ処理) ---")
        self.model.eval()
        
        all_preds_list = []
        all_probas_list = []
        all_original_indices_list = []

        current_batch_sequences = []
        current_batch_confidences = [] if self.model.enable_confidence_weighting and confidence_scores_full is not None else None
        current_batch_indices = []

        num_sequences_processed = 0
        total_sequences_to_generate = max(0, X_scaled.shape[0] - self.sequence_length + 1)

        seq_generator = self._generate_sequences_for_inference(X_scaled, confidence_scores_full)

        for seq_X, original_idx, seq_conf in seq_generator:
            current_batch_sequences.append(seq_X)
            current_batch_indices.append(original_idx)
            if current_batch_confidences is not None and seq_conf is not None:
                current_batch_confidences.append(seq_conf)

            if len(current_batch_sequences) == batch_size:
                batch_X_np = np.array(current_batch_sequences, dtype=np.float32)
                batch_X_tensor = torch.from_numpy(batch_X_np).to(self.device)
                
                batch_conf_tensor = None
                if current_batch_confidences:
                    batch_conf_np = np.array(current_batch_confidences, dtype=np.float32)
                    batch_conf_tensor = torch.from_numpy(batch_conf_np).to(self.device)

                with torch.no_grad():
                    outputs = self.model(batch_X_tensor, batch_conf_tensor)
                    probas = F.softmax(outputs, dim=1)
                    _, preds = torch.max(probas, 1)
                
                all_preds_list.extend(preds.cpu().numpy())
                all_probas_list.extend(probas.cpu().numpy())
                all_original_indices_list.extend(current_batch_indices)
                num_sequences_processed += len(current_batch_sequences)

                current_batch_sequences = []
                current_batch_indices = []
                if current_batch_confidences is not None:
                    current_batch_confidences = []
                
                if num_sequences_processed % (batch_size * 10) == 0: # 進捗表示
                     print(f"   処理済みシーケンス: {num_sequences_processed} / {total_sequences_to_generate}")
        
        # 残りのシーケンスを処理
        if current_batch_sequences:
            batch_X_np = np.array(current_batch_sequences, dtype=np.float32)
            batch_X_tensor = torch.from_numpy(batch_X_np).to(self.device)
            
            batch_conf_tensor = None
            if current_batch_confidences:
                batch_conf_np = np.array(current_batch_confidences, dtype=np.float32)
                batch_conf_tensor = torch.from_numpy(batch_conf_np).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_X_tensor, batch_conf_tensor)
                probas = F.softmax(outputs, dim=1)
                _, preds = torch.max(probas, 1)

            all_preds_list.extend(preds.cpu().numpy())
            all_probas_list.extend(probas.cpu().numpy())
            all_original_indices_list.extend(current_batch_indices)
            num_sequences_processed += len(current_batch_sequences)

        if not all_preds_list:
            print("⚠️  推論対象のシーケンスがありませんでした。")
            return np.array([]), np.array([]), []

        print(f"✅ 推論完了: {num_sequences_processed}件 (全シーケンス)")
        return np.array(all_preds_list), np.array(all_probas_list), all_original_indices_list

    def format_predictions(self, predictions: np.ndarray, probabilities: np.ndarray, 
                           original_df: pd.DataFrame, original_indices: List[int]) -> pd.DataFrame:
        if self.label_map_inv is None:
            print("❌ ラベルマッピングが未定義です。")
            return original_df
        
        # 結果を格納するための新しい列を準備
        num_frames = len(original_df)
        predicted_label_names = [""] * num_frames
        max_probabilities = np.zeros(num_frames)
        
        # 各クラスの確率も保存する場合
        # proba_columns = {f"proba_{self.label_map_inv.get(i, f'class_{i}')}": np.zeros(num_frames) for i in range(probabilities.shape[1])}

        for i, pred_idx in enumerate(predictions):
            frame_idx = original_indices[i] # この予測が対応する元のフレームのインデックス
            if 0 <= frame_idx < num_frames:
                predicted_label_names[frame_idx] = self.label_map_inv.get(pred_idx, f"Unknown_{pred_idx}")
                max_probabilities[frame_idx] = probabilities[i, pred_idx]
                # for class_idx in range(probabilities.shape[1]):
                #    proba_columns[f"proba_{self.label_map_inv.get(class_idx, f'class_{class_idx}')}"][frame_idx] = probabilities[i, class_idx]


        results_df = original_df.copy()
        results_df['predicted_phase'] = predicted_label_names
        results_df['prediction_confidence'] = max_probabilities
        
        # for col_name, prob_values in proba_columns.items():
        #    results_df[col_name] = prob_values
            
        # シーケンスの最初の数フレームは予測がないため、それらをどう扱うか
        # ここでは空文字またはNaNのまま
        results_df.loc[results_df['predicted_phase'] == "", 'prediction_confidence'] = np.nan

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
        # 1. モデルファイル選択
        selected_files = self.select_model_files()
        if not selected_files: return
        model_p, scaler_p, meta_p = selected_files

        # 2. モデルとメタデータ読み込み
        if not self.load_model_and_metadata(model_p, scaler_p, meta_p): return

        # 3. 入力特徴量ファイル選択
        input_csv_path = self.select_input_feature_file()
        if not input_csv_path: return

        # 4. データ準備
        X_scaled, original_df, confidence_scores_full = self.prepare_input_data(input_csv_path)
        if X_scaled is None or original_df is None:
            print("❌ データ準備に失敗しました。処理を中断します。")
            return

        # 5. 推論実行
        prediction_results = self.predict(X_scaled, original_df, confidence_scores_full)
        if prediction_results is None:
            print("❌ 推論に失敗しました。処理を中断します。")
            return
        raw_preds, raw_probas, all_original_indices = prediction_results
        
        # 6. 結果フォーマット
        formatted_df = self.format_predictions(raw_preds, raw_probas, original_df, all_original_indices)

        # 7. 結果保存
        self.save_predictions(formatted_df, input_csv_path.name)

        print("\n🎉 推論パイプライン完了！")

    def run_prediction_for_file(self, model_set_path: Path, feature_csv_path: Path) -> Optional[Path]:
        """
        指定されたモデルセットと特徴量ファイルを使用して予測を実行し、結果のCSVパスを返す。
        パイプラインからの呼び出し用。
        """
        print(f"\n=== 非対話的推論開始 ===")
        print(f"モデルセットパス: {model_set_path}")
        print(f"特徴量CSVパス: {feature_csv_path}")

        # 1. モデルファイル特定と読み込み
        model_files_tuple = self.find_model_files_in_set_dir(model_set_path)
        if not model_files_tuple:
            print(f"❌ 指定されたパス {model_set_path} でモデルファイル群を特定できませんでした。")
            return None
        model_p, scaler_p, meta_p = model_files_tuple

        if not self.load_model_and_metadata(model_p, scaler_p, meta_p):
            print(f"❌ モデルとメタデータの読み込みに失敗しました: {model_p.name}")
            return None

        # 2. データ準備
        X_scaled, original_df, confidence_scores_full = self.prepare_input_data(feature_csv_path)
        if X_scaled is None or original_df is None:
            print(f"❌ データ準備に失敗しました: {feature_csv_path.name}")
            return None

        # 3. 推論実行
        prediction_results = self.predict(X_scaled, original_df, confidence_scores_full)
        if prediction_results is None:
            print("❌ 推論に失敗しました。")
            return None
        raw_preds, raw_probas, all_original_indices = prediction_results
        
        # 4. 結果フォーマット
        formatted_df = self.format_predictions(raw_preds, raw_probas, original_df, all_original_indices)

        # 5. 結果保存
        output_csv_path = self.save_predictions(formatted_df, feature_csv_path.name)
        
        print(f"\n🎉 非対話的推論完了！結果: {output_csv_path}")
        return output_csv_path

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

